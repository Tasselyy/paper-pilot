"""Multi-hop strategy: Plan-and-Execute sub-graph.

Implements the Plan-and-Execute pattern for multi-hop reasoning:

1. **Plan** — decompose the original question into sub-questions.
2. **Execute** — search RAG for each sub-question sequentially.
3. **Replan** — after each execution, decide: continue / revise plan /
   synthesize early.
4. **Synthesize** — combine all retrieved contexts into a final answer.

The loop terminates when all sub-questions are executed, when the LLM
decides enough information is gathered (``synthesize``), or when
``max_plan_steps`` is reached (safety limit).

Design reference: PAPER_PILOT_DESIGN.md §5.5, §6.4, DEV_SPEC C4.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.agent.state import AgentState, ReasoningStep, RetrievedContext
from src.prompts.strategies import (
    MULTI_HOP_DECIDE_SYSTEM_PROMPT,
    MULTI_HOP_DECIDE_USER_TEMPLATE,
    MULTI_HOP_DECOMPOSE_SYSTEM_PROMPT,
    MULTI_HOP_DECOMPOSE_USER_TEMPLATE,
    MULTI_HOP_REPLAN_SYSTEM_PROMPT,
    MULTI_HOP_REPLAN_USER_TEMPLATE,
    MULTI_HOP_SYNTHESIS_SYSTEM_PROMPT,
    MULTI_HOP_SYNTHESIS_USER_TEMPLATE,
    format_retrieved_contexts,
)
from src.tools.tool_wrapper import RAGToolWrapper

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default parameters (configurable via factory or function args)
# ---------------------------------------------------------------------------

DEFAULT_TOP_K = 3
"""Default number of RAG results to retrieve per sub-question."""

DEFAULT_MAX_PLAN_STEPS = 5
"""Safety limit on the maximum number of plan steps to execute."""

# ---------------------------------------------------------------------------
# Structured-output models (used with LLM.with_structured_output)
# ---------------------------------------------------------------------------


class SubQuestionPlan(BaseModel):
    """LLM structured output: ordered list of sub-questions."""

    questions: list[str] = Field(
        description="Ordered list of sub-questions to investigate",
    )


class ReplanDecision(BaseModel):
    """LLM structured output: decision after observing execution results."""

    action: Literal["next_step", "replan", "synthesize"] = Field(
        description=(
            "next_step = continue executing next sub-question; "
            "replan = revise the sub-question plan; "
            "synthesize = enough information gathered, produce final answer"
        ),
    )
    reason: str = Field(
        description="Brief explanation for the chosen action",
    )


# ---------------------------------------------------------------------------
# Internal step functions (testable helpers)
# ---------------------------------------------------------------------------


def _format_sub_questions(questions: list[str]) -> str:
    """Format a list of sub-questions as a numbered list string."""
    if not questions:
        return "(none)"
    return "\n".join(f"  {i + 1}. {q}" for i, q in enumerate(questions))


async def _decompose(
    question: str,
    constraints: dict[str, Any],
    llm: BaseChatModel,
) -> list[str]:
    """Decompose the original question into sub-questions.

    Args:
        question: The reformulated user question.
        constraints: Intent constraints (e.g. technique, time_range).
        llm: Cloud LLM instance.

    Returns:
        Ordered list of sub-questions (2–4 items).
    """
    structured_llm = llm.with_structured_output(SubQuestionPlan)
    result: SubQuestionPlan = await structured_llm.ainvoke([
        SystemMessage(content=MULTI_HOP_DECOMPOSE_SYSTEM_PROMPT),
        HumanMessage(content=MULTI_HOP_DECOMPOSE_USER_TEMPLATE.format(
            question=question,
            constraints=constraints or {},
        )),
    ])
    return result.questions


async def _revise_plan(
    question: str,
    constraints: dict[str, Any],
    original_plan: list[str],
    executed_queries: list[str],
    retrieved_contexts: list[RetrievedContext],
    llm: BaseChatModel,
) -> list[str]:
    """Revise the sub-question plan based on results obtained so far.

    Args:
        question: The reformulated user question.
        constraints: Intent constraints.
        original_plan: Current list of sub-questions.
        executed_queries: Sub-questions already executed.
        retrieved_contexts: All contexts retrieved so far.
        llm: Cloud LLM instance.

    Returns:
        Revised ordered list of sub-questions (includes executed ones first).
    """
    structured_llm = llm.with_structured_output(SubQuestionPlan)
    result: SubQuestionPlan = await structured_llm.ainvoke([
        SystemMessage(content=MULTI_HOP_REPLAN_SYSTEM_PROMPT),
        HumanMessage(content=MULTI_HOP_REPLAN_USER_TEMPLATE.format(
            question=question,
            original_plan=_format_sub_questions(original_plan),
            executed_queries=_format_sub_questions(executed_queries),
            results_so_far=format_retrieved_contexts(retrieved_contexts),
            constraints=constraints or {},
        )),
    ])
    return result.questions


async def _decide_next(
    question: str,
    plan: list[str],
    executed_count: int,
    all_contexts: list[RetrievedContext],
    llm: BaseChatModel,
) -> ReplanDecision:
    """Decide the next action: continue, replan, or synthesize.

    Args:
        question: The reformulated user question.
        plan: Current sub-question plan.
        executed_count: Number of sub-questions already executed.
        all_contexts: All contexts retrieved so far.
        llm: Cloud LLM instance.

    Returns:
        A ``ReplanDecision`` with ``action`` and ``reason``.
    """
    remaining = plan[executed_count:]
    latest_contexts = all_contexts[-3:] if all_contexts else []

    structured_llm = llm.with_structured_output(ReplanDecision)
    result: ReplanDecision = await structured_llm.ainvoke([
        SystemMessage(content=MULTI_HOP_DECIDE_SYSTEM_PROMPT),
        HumanMessage(content=MULTI_HOP_DECIDE_USER_TEMPLATE.format(
            question=question,
            plan=_format_sub_questions(plan),
            executed_count=executed_count,
            total_steps=len(plan),
            latest_results=format_retrieved_contexts(latest_contexts),
            remaining=_format_sub_questions(remaining),
        )),
    ])
    return result


async def _synthesize(
    question: str,
    sub_questions: list[str],
    all_contexts: list[RetrievedContext],
    llm: BaseChatModel,
) -> str:
    """Synthesize the final answer from all retrieved contexts.

    Args:
        question: The reformulated user question.
        sub_questions: All sub-questions (final plan).
        all_contexts: All contexts retrieved across all steps.
        llm: Cloud LLM instance.

    Returns:
        The synthesized draft answer text.
    """
    messages = [
        SystemMessage(content=MULTI_HOP_SYNTHESIS_SYSTEM_PROMPT),
        HumanMessage(content=MULTI_HOP_SYNTHESIS_USER_TEMPLATE.format(
            question=question,
            sub_questions=_format_sub_questions(sub_questions),
            contexts=format_retrieved_contexts(all_contexts),
        )),
    ]
    response = await llm.ainvoke(messages)
    return (
        response.content
        if isinstance(response.content, str)
        else str(response.content)
    )


# ---------------------------------------------------------------------------
# Core implementation (testable with explicit deps)
# ---------------------------------------------------------------------------


async def run_multi_hop_strategy(
    state: AgentState,
    rag: RAGToolWrapper,
    llm: BaseChatModel,
    *,
    top_k: int = DEFAULT_TOP_K,
    max_plan_steps: int = DEFAULT_MAX_PLAN_STEPS,
    collection: str | None = None,
) -> dict[str, Any]:
    """Execute multi-hop strategy: Plan -> Execute -> Replan loop -> Synthesize.

    This is the core implementation that accepts explicit dependencies for
    easy unit testing.

    Args:
        state: Current agent state (must have ``intent`` populated).
        rag: RAG tool wrapper for knowledge-base search.
        llm: Cloud LLM instance for planning, replanning, and synthesis.
        top_k: Number of RAG results per sub-question search.
        max_plan_steps: Safety limit on execution steps.

    Returns:
        Partial state update dict with ``sub_questions``, ``draft_answer``,
        ``retrieved_contexts``, ``retrieval_queries``, and
        ``reasoning_trace`` entries.

    Raises:
        ValueError: If ``state.intent`` is ``None``.
    """
    if state.intent is None:
        raise ValueError("multi_hop_strategy requires state.intent to be set")

    # -- Resolve the query --------------------------------------------------
    question = state.intent.reformulated_query or state.question
    if not question:
        logger.warning("No query available — using raw question as fallback")
        question = state.question or "No question provided"

    constraints = state.intent.constraints

    logger.info("Multi-hop strategy: decomposing question=%r", question[:80])

    # -- 1. Plan: decompose into sub-questions ------------------------------
    sub_questions = await _decompose(question, constraints, llm)

    trace_steps: list[ReasoningStep] = [
        ReasoningStep(
            step_type="thought",
            content=f"Plan: decomposed into {len(sub_questions)} sub-question(s): {sub_questions}",
            metadata={"sub_questions": sub_questions},
        ),
    ]

    logger.info("Multi-hop plan: %d sub-questions", len(sub_questions))

    # -- 2. Execute + Replan loop -------------------------------------------
    all_contexts: list[RetrievedContext] = []
    all_queries: list[str] = []
    executed = 0

    while executed < len(sub_questions) and executed < max_plan_steps:
        current_sub_q = sub_questions[executed]

        # Execute: RAG search for current sub-question
        logger.info("Multi-hop execute step %d/%d: %r", executed + 1, len(sub_questions), current_sub_q[:60])
        results: list[RetrievedContext] = await rag.search(
            current_sub_q, top_k=top_k, collection=collection,
        )
        all_contexts.extend(results)
        all_queries.append(current_sub_q)

        trace_steps.append(ReasoningStep(
            step_type="action",
            content=f"Execute step {executed + 1}/{len(sub_questions)}: {current_sub_q}",
            metadata={"step_index": executed, "num_results": len(results)},
        ))
        trace_steps.append(ReasoningStep(
            step_type="observation",
            content=f"Retrieved {len(results)} chunk(s) for: {current_sub_q}",
            metadata={"num_results": len(results)},
        ))

        executed += 1

        # Replan decision (skip if this was the last sub-question)
        if executed < len(sub_questions):
            decision = await _decide_next(
                question, sub_questions, executed, all_contexts, llm,
            )

            trace_steps.append(ReasoningStep(
                step_type="thought",
                content=f"Replan decision: {decision.action} — {decision.reason}",
                metadata={"action": decision.action},
            ))

            if decision.action == "synthesize":
                logger.info("Multi-hop: early synthesis after %d step(s)", executed)
                break

            if decision.action == "replan":
                logger.info("Multi-hop: revising plan after %d step(s)", executed)
                sub_questions = await _revise_plan(
                    question, constraints, sub_questions,
                    all_queries, all_contexts, llm,
                )
                trace_steps.append(ReasoningStep(
                    step_type="thought",
                    content=f"Revised plan ({len(sub_questions)} sub-question(s)): {sub_questions}",
                    metadata={"revised_sub_questions": sub_questions},
                ))

    # -- 3. Synthesize final answer -----------------------------------------
    logger.info(
        "Multi-hop: synthesizing from %d context(s) across %d step(s)",
        len(all_contexts), executed,
    )
    draft_answer = await _synthesize(question, sub_questions, all_contexts, llm)

    trace_steps.append(ReasoningStep(
        step_type="thought",
        content=f"Synthesized answer ({len(draft_answer)} chars) from {len(all_contexts)} context(s)",
        metadata={"answer_length": len(draft_answer), "total_contexts": len(all_contexts)},
    ))

    logger.info(
        "Multi-hop strategy complete: draft_answer=%d chars, contexts=%d, queries=%d",
        len(draft_answer), len(all_contexts), len(all_queries),
    )

    return {
        "sub_questions": sub_questions,
        "draft_answer": draft_answer,
        "retrieved_contexts": all_contexts,
        "retrieval_queries": all_queries,
        "reasoning_trace": trace_steps,
    }


# ---------------------------------------------------------------------------
# Factory for creating a graph-compatible node with bound dependencies
# ---------------------------------------------------------------------------


def create_multi_hop_strategy_node(
    rag: RAGToolWrapper,
    llm: BaseChatModel,
    *,
    top_k: int = DEFAULT_TOP_K,
    max_plan_steps: int = DEFAULT_MAX_PLAN_STEPS,
    rag_default_collection: str | None = None,
):
    """Create a multi-hop strategy node function with bound dependencies.

    Use this factory when wiring the node into the LangGraph graph with
    real or test dependencies.

    Args:
        rag: RAG tool wrapper instance.
        llm: Cloud LLM instance.
        top_k: Number of RAG results per sub-question.
        max_plan_steps: Safety limit on execution steps.
        rag_default_collection: Optional RAG collection name to restrict search to.

    Returns:
        An async callable ``(state: AgentState) -> dict`` suitable for
        ``graph.add_node()``.
    """

    async def _node(state: AgentState) -> dict[str, Any]:
        return await run_multi_hop_strategy(
            state, rag, llm,
            top_k=top_k,
            max_plan_steps=max_plan_steps,
            collection=rag_default_collection,
        )

    _node.__name__ = "multi_hop_strategy_node"
    _node.__doc__ = "Multi-hop strategy node (Plan-and-Execute)."
    return _node


# ---------------------------------------------------------------------------
# Default export — placeholder until graph.py wires real dependencies
# ---------------------------------------------------------------------------


def multi_hop_strategy_node(state) -> dict:
    """Execute multi-hop strategy via plan / execute / replan / synthesize.

    .. note::

        This is a **synchronous placeholder** used by ``graph.py`` before
        real RAG/LLM dependencies are wired via
        ``create_multi_hop_strategy_node()``.  It returns a stub
        ``draft_answer`` so the graph can compile and invoke without errors.

    Args:
        state: The current AgentState.

    Returns:
        Partial state update with ``draft_answer``.
    """
    return {"draft_answer": "[multi_hop placeholder] No RAG/LLM configured yet."}
