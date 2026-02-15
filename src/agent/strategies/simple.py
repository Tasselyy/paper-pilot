"""Simple strategy: single retrieval + LLM synthesis.

Given ``state.intent.reformulated_query``, performs a single RAG search via
``RAGToolWrapper.search()`` and synthesizes an answer using a Cloud LLM.

Writes ``draft_answer``, ``retrieved_contexts``, and ``retrieval_queries``
back into the agent state.

Design reference: PAPER_PILOT_DESIGN.md §6.3, DEV_SPEC B3.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from src.agent.state import AgentState, ReasoningStep, RetrievedContext
from src.prompts.strategies import (
    SIMPLE_SYSTEM_PROMPT,
    SIMPLE_USER_TEMPLATE,
    format_simple_contexts,
)
from src.tools.tool_wrapper import RAGToolWrapper

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default retrieval parameters
# ---------------------------------------------------------------------------

DEFAULT_TOP_K = 5
"""Default number of RAG results to retrieve."""


# ---------------------------------------------------------------------------
# Core implementation (testable with explicit deps)
# ---------------------------------------------------------------------------


async def run_simple_strategy(
    state: AgentState,
    rag: RAGToolWrapper,
    llm: BaseChatModel,
    *,
    top_k: int = DEFAULT_TOP_K,
) -> dict[str, Any]:
    """Execute the simple strategy: single retrieval + LLM synthesis.

    This is the core implementation that accepts explicit dependencies for
    easy unit testing.

    Args:
        state: Current agent state (must have ``intent`` populated).
        rag: RAG tool wrapper for knowledge-base search.
        llm: Cloud LLM instance for answer synthesis.
        top_k: Maximum number of retrieval results.

    Returns:
        Partial state update dict with ``draft_answer``,
        ``retrieved_contexts``, ``retrieval_queries``, and
        ``reasoning_trace`` entries.

    Raises:
        ValueError: If ``state.intent`` is ``None`` (should not happen in
            normal graph execution).
    """
    if state.intent is None:
        raise ValueError("simple_strategy_node requires state.intent to be set")

    # -- 1. Determine the query -------------------------------------------
    query = state.intent.reformulated_query or state.question
    if not query:
        logger.warning("No query available — using raw question as fallback")
        query = state.question or "No question provided"

    logger.info("Simple strategy: searching with query=%r (top_k=%d)", query[:80], top_k)

    # -- 2. Retrieve contexts via RAG ------------------------------------
    retrieved: list[RetrievedContext] = await rag.search(query, top_k=top_k)

    retrieval_queries = [query]

    # Record reasoning step for observability
    trace_steps: list[ReasoningStep] = [
        ReasoningStep(
            step_type="action",
            content=f"RAG search: query={query!r}, returned {len(retrieved)} result(s)",
            metadata={"top_k": top_k, "num_results": len(retrieved)},
        ),
    ]

    # -- 3. Format contexts for the LLM prompt ----------------------------
    context_dicts = [
        {"content": ctx.content, "source": ctx.source}
        for ctx in retrieved
    ]
    formatted_contexts = format_simple_contexts(context_dicts)

    user_prompt = SIMPLE_USER_TEMPLATE.format(
        question=query,
        contexts=formatted_contexts,
    )

    # -- 4. Synthesize answer via LLM -------------------------------------
    messages = [
        SystemMessage(content=SIMPLE_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]

    logger.info("Simple strategy: invoking LLM for synthesis")
    llm_response = await llm.ainvoke(messages)
    draft_answer = (
        llm_response.content
        if isinstance(llm_response.content, str)
        else str(llm_response.content)
    )

    trace_steps.append(
        ReasoningStep(
            step_type="thought",
            content=f"Synthesized answer ({len(draft_answer)} chars) from {len(retrieved)} context(s)",
            metadata={"answer_length": len(draft_answer)},
        ),
    )

    logger.info(
        "Simple strategy complete: draft_answer=%d chars, contexts=%d",
        len(draft_answer),
        len(retrieved),
    )

    return {
        "draft_answer": draft_answer,
        "retrieved_contexts": retrieved,
        "retrieval_queries": retrieval_queries,
        "reasoning_trace": trace_steps,
    }


# ---------------------------------------------------------------------------
# Factory for creating a graph-compatible node with bound dependencies
# ---------------------------------------------------------------------------


def create_simple_strategy_node(
    rag: RAGToolWrapper,
    llm: BaseChatModel,
    *,
    top_k: int = DEFAULT_TOP_K,
):
    """Create a simple strategy node function with bound dependencies.

    Use this factory when wiring the node into the LangGraph graph with
    real or test dependencies.

    Args:
        rag: RAG tool wrapper instance.
        llm: Cloud LLM instance.
        top_k: Default number of retrieval results.

    Returns:
        An async callable ``(state: AgentState) -> dict`` suitable for
        ``graph.add_node()``.
    """

    async def _node(state: AgentState) -> dict[str, Any]:
        return await run_simple_strategy(state, rag, llm, top_k=top_k)

    _node.__name__ = "simple_strategy_node"
    _node.__doc__ = "Simple strategy node (retrieve + synthesize)."
    return _node


# ---------------------------------------------------------------------------
# Default export — placeholder until B4 wires real dependencies
# ---------------------------------------------------------------------------


def simple_strategy_node(state) -> dict:
    """Execute simple strategy: retrieve once, synthesize answer.

    .. note::

        This is a **synchronous placeholder** used by ``graph.py`` before B4
        wires real RAG/LLM dependencies via ``create_simple_strategy_node()``.
        It returns a stub ``draft_answer`` so the graph can compile and
        invoke (both sync and async) without errors.

    Args:
        state: The current AgentState.

    Returns:
        Partial state update with ``draft_answer``.
    """
    return {"draft_answer": "[simple placeholder] No RAG/LLM configured yet."}
