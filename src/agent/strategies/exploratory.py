"""Exploratory strategy: ReAct sub-graph (think/act/observe/synthesize).

Implements the ReAct (Reasoning + Acting) pattern for open-ended
exploratory questions:

1. **Think** — LLM decides the next action: search for more info or
   synthesize the answer.
2. **Act** — execute a RAG search with the LLM-chosen query.
3. **Observe** — record the retrieval results as observations.
4. **Loop** — repeat think/act/observe until the LLM decides to
   synthesize or ``max_react_steps`` is reached.
5. **Synthesize** — combine all observations into a final draft answer.

The loop always terminates: either the LLM chooses ``"synthesize"`` or
the ``max_react_steps`` safety limit forces synthesis.

Design reference: PAPER_PILOT_DESIGN.md §5.6, §6.6, DEV_SPEC C6.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.agent.state import AgentState, ReasoningStep, RetrievedContext
from src.prompts.strategies import (
    EXPLORATORY_SYNTHESIS_SYSTEM_PROMPT,
    EXPLORATORY_SYNTHESIS_USER_TEMPLATE,
    EXPLORATORY_THINK_SYSTEM_PROMPT,
    EXPLORATORY_THINK_USER_TEMPLATE,
    format_observations,
    format_retrieved_contexts,
)
from src.tools.tool_wrapper import RAGToolWrapper

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default parameters (configurable via factory or function args)
# ---------------------------------------------------------------------------

DEFAULT_TOP_K = 3
"""Default number of RAG results to retrieve per search step."""

DEFAULT_MAX_REACT_STEPS = 5
"""Safety limit on the maximum number of ReAct loop iterations."""

# ---------------------------------------------------------------------------
# Structured-output models (used with LLM.with_structured_output)
# ---------------------------------------------------------------------------


class ReactDecision(BaseModel):
    """LLM structured output: next action in the ReAct loop."""

    action: Literal["search", "synthesize"] = Field(
        description=(
            "search = perform a knowledge-base search with the given query; "
            "synthesize = enough information gathered, produce final answer"
        ),
    )
    query: str = Field(
        default="",
        description=(
            "The search query to execute (only used when action='search'). "
            "Should be focused and different from previous queries."
        ),
    )
    reasoning: str = Field(
        description="Brief explanation for choosing this action",
    )


# ---------------------------------------------------------------------------
# Internal step functions (testable helpers)
# ---------------------------------------------------------------------------


def _format_query_list(queries: list[str]) -> str:
    """Format a list of queries as a numbered list string."""
    if not queries:
        return "(none)"
    return "\n".join(f"  {i + 1}. {q}" for i, q in enumerate(queries))


async def _think(
    question: str,
    step_count: int,
    max_steps: int,
    previous_queries: list[str],
    observations_text: str,
    llm: BaseChatModel,
) -> ReactDecision:
    """Decide the next action in the ReAct loop.

    Args:
        question: The reformulated user question.
        step_count: Number of steps already completed.
        max_steps: Maximum steps allowed.
        previous_queries: Queries already executed.
        observations_text: Formatted text of all observations so far.
        llm: Cloud LLM instance.

    Returns:
        A ``ReactDecision`` with ``action``, ``query``, and ``reasoning``.
    """
    structured_llm = llm.with_structured_output(ReactDecision)
    result: ReactDecision = await structured_llm.ainvoke([
        SystemMessage(content=EXPLORATORY_THINK_SYSTEM_PROMPT),
        HumanMessage(content=EXPLORATORY_THINK_USER_TEMPLATE.format(
            question=question,
            step_count=step_count,
            max_steps=max_steps,
            previous_queries=_format_query_list(previous_queries),
            observations=observations_text or "(none yet)",
        )),
    ])
    return result


async def _synthesize(
    question: str,
    queries: list[str],
    observations_text: str,
    llm: BaseChatModel,
) -> str:
    """Synthesize the final answer from all observations.

    Args:
        question: The reformulated user question.
        queries: All search queries executed.
        observations_text: Formatted text of all observations.
        llm: Cloud LLM instance.

    Returns:
        The synthesized draft answer text.
    """
    messages = [
        SystemMessage(content=EXPLORATORY_SYNTHESIS_SYSTEM_PROMPT),
        HumanMessage(content=EXPLORATORY_SYNTHESIS_USER_TEMPLATE.format(
            question=question,
            queries=_format_query_list(queries),
            observations=observations_text,
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


async def run_exploratory_strategy(
    state: AgentState,
    rag: RAGToolWrapper,
    llm: BaseChatModel,
    *,
    top_k: int = DEFAULT_TOP_K,
    max_react_steps: int = DEFAULT_MAX_REACT_STEPS,
    collection: str | None = None,
) -> dict[str, Any]:
    """Execute exploratory strategy: ReAct loop (think/act/observe) -> synthesize.

    This is the core implementation that accepts explicit dependencies for
    easy unit testing.

    Args:
        state: Current agent state (must have ``intent`` populated).
        rag: RAG tool wrapper for knowledge-base search.
        llm: Cloud LLM instance for thinking and synthesis.
        top_k: Number of RAG results per search step.
        max_react_steps: Safety limit on loop iterations.

    Returns:
        Partial state update dict with ``draft_answer``,
        ``retrieved_contexts``, ``retrieval_queries``, and
        ``reasoning_trace`` entries.

    Raises:
        ValueError: If ``state.intent`` is ``None``.
    """
    if state.intent is None:
        raise ValueError("exploratory_strategy requires state.intent to be set")

    # -- Resolve the query --------------------------------------------------
    question = state.intent.reformulated_query or state.question
    if not question:
        logger.warning("No query available — using raw question as fallback")
        question = state.question or "No question provided"

    logger.info("Exploratory strategy: question=%r", question[:80])

    trace_steps: list[ReasoningStep] = []
    all_contexts: list[RetrievedContext] = []
    all_queries: list[str] = []
    contexts_per_query: list[list[RetrievedContext]] = []
    step_count = 0

    # -- ReAct loop: think → act → observe ----------------------------------
    while step_count < max_react_steps:
        # Format current observations for the LLM
        observations_text = format_observations(all_queries, contexts_per_query)

        # Think: decide next action
        decision = await _think(
            question=question,
            step_count=step_count,
            max_steps=max_react_steps,
            previous_queries=all_queries,
            observations_text=observations_text,
            llm=llm,
        )

        trace_steps.append(ReasoningStep(
            step_type="thought",
            content=(
                f"ReAct step {step_count + 1}/{max_react_steps}: "
                f"action={decision.action} — {decision.reasoning}"
            ),
            metadata={
                "step": step_count + 1,
                "action": decision.action,
                "query": decision.query,
            },
        ))

        logger.info(
            "Exploratory step %d/%d: action=%s, query=%r",
            step_count + 1, max_react_steps,
            decision.action, decision.query[:60] if decision.query else "",
        )

        # If the LLM decides to synthesize, break out of the loop
        if decision.action == "synthesize":
            logger.info("Exploratory: LLM chose to synthesize at step %d", step_count + 1)
            break

        # Act: execute RAG search
        search_query = decision.query
        if not search_query:
            search_query = question
            logger.warning("Empty search query from LLM — using original question")

        results: list[RetrievedContext] = await rag.search(
            search_query, top_k=top_k, collection=collection,
        )

        trace_steps.append(ReasoningStep(
            step_type="action",
            content=f"RAG search: {search_query}",
            metadata={"step": step_count + 1, "query": search_query, "num_results": len(results)},
        ))

        # Observe: record results
        all_contexts.extend(results)
        all_queries.append(search_query)
        contexts_per_query.append(results)

        trace_steps.append(ReasoningStep(
            step_type="observation",
            content=f"Retrieved {len(results)} chunk(s) for: {search_query}",
            metadata={"step": step_count + 1, "num_results": len(results)},
        ))

        step_count += 1

    # -- Check if we hit the step limit (forced termination) ----------------
    if step_count >= max_react_steps:
        logger.info(
            "Exploratory: max_react_steps (%d) reached — forcing synthesis",
            max_react_steps,
        )
        trace_steps.append(ReasoningStep(
            step_type="thought",
            content=f"Max ReAct steps ({max_react_steps}) reached — forcing synthesis",
            metadata={"forced": True, "max_react_steps": max_react_steps},
        ))

    # -- Synthesize final answer --------------------------------------------
    observations_text = format_observations(all_queries, contexts_per_query)

    logger.info(
        "Exploratory: synthesizing from %d context(s) across %d step(s)",
        len(all_contexts), step_count,
    )

    draft_answer = await _synthesize(question, all_queries, observations_text, llm)

    trace_steps.append(ReasoningStep(
        step_type="thought",
        content=(
            f"Synthesized answer ({len(draft_answer)} chars) "
            f"from {len(all_contexts)} context(s) across {step_count} step(s)"
        ),
        metadata={
            "answer_length": len(draft_answer),
            "total_contexts": len(all_contexts),
            "total_steps": step_count,
        },
    ))

    logger.info(
        "Exploratory strategy complete: draft_answer=%d chars, "
        "contexts=%d, queries=%d, steps=%d",
        len(draft_answer), len(all_contexts), len(all_queries), step_count,
    )

    return {
        "draft_answer": draft_answer,
        "retrieved_contexts": all_contexts,
        "retrieval_queries": all_queries,
        "reasoning_trace": trace_steps,
    }


# ---------------------------------------------------------------------------
# Factory for creating a graph-compatible node with bound dependencies
# ---------------------------------------------------------------------------


def create_exploratory_strategy_node(
    rag: RAGToolWrapper,
    llm: BaseChatModel,
    *,
    top_k: int = DEFAULT_TOP_K,
    max_react_steps: int = DEFAULT_MAX_REACT_STEPS,
    rag_default_collection: str | None = None,
):
    """Create an exploratory strategy node function with bound dependencies.

    Use this factory when wiring the node into the LangGraph graph with
    real or test dependencies.

    Args:
        rag: RAG tool wrapper instance.
        llm: Cloud LLM instance.
        top_k: Number of RAG results per search step.
        max_react_steps: Safety limit on ReAct loop iterations.
        rag_default_collection: Optional RAG collection name to restrict search to.

    Returns:
        An async callable ``(state: AgentState) -> dict`` suitable for
        ``graph.add_node()``.
    """

    async def _node(state: AgentState) -> dict[str, Any]:
        return await run_exploratory_strategy(
            state, rag, llm,
            top_k=top_k,
            max_react_steps=max_react_steps,
            collection=rag_default_collection,
        )

    _node.__name__ = "exploratory_strategy_node"
    _node.__doc__ = "Exploratory strategy node (ReAct: think/act/observe loop)."
    return _node


# ---------------------------------------------------------------------------
# Default export — placeholder until graph.py wires real dependencies
# ---------------------------------------------------------------------------


def exploratory_strategy_node(state) -> dict:
    """Execute exploratory strategy via ReAct loop.

    .. note::

        This is a **synchronous placeholder** used by ``graph.py`` before
        real RAG/LLM dependencies are wired via
        ``create_exploratory_strategy_node()``.  It returns a stub
        ``draft_answer`` so the graph can compile and invoke without errors.

    Args:
        state: The current AgentState.

    Returns:
        Partial state update with ``draft_answer``.
    """
    return {"draft_answer": "[exploratory placeholder] No retrieval performed yet."}
