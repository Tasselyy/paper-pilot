"""Comparative strategy: parallel retrieval by entities/dimensions (ReWOO).

Implements the ReWOO (Reasoning WithOut Observation) pattern for
comparative analysis:

1. **Entities & dimensions** — prioritize ``intent.entities`` and
   ``intent.dimensions`` from slot filling; fallback to LLM extraction
   when missing.
2. **Parallel retrieval** — use ``asyncio.gather`` to search RAG for
   each entity concurrently (no intermediate LLM calls).
3. **Structured comparison** — a single LLM call synthesizes all
   per-entity contexts into a structured comparison answer.

Design reference: PAPER_PILOT_DESIGN.md §6.5, DEV_SPEC C5.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.agent.state import AgentState, ReasoningStep, RetrievedContext
from src.prompts.strategies import (
    COMPARATIVE_EXTRACT_SYSTEM_PROMPT,
    COMPARATIVE_EXTRACT_USER_TEMPLATE,
    COMPARATIVE_SYNTHESIS_SYSTEM_PROMPT,
    COMPARATIVE_SYNTHESIS_USER_TEMPLATE,
    format_entity_contexts,
)
from src.tools.tool_wrapper import RAGToolWrapper

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default parameters
# ---------------------------------------------------------------------------

DEFAULT_TOP_K = 3
"""Default number of RAG results to retrieve per entity."""

# ---------------------------------------------------------------------------
# Structured-output model (LLM fallback extraction)
# ---------------------------------------------------------------------------


class CompareEntities(BaseModel):
    """LLM structured output: entities and dimensions for comparison."""

    entities: list[str] = Field(
        description="Items being compared (e.g. paper names, techniques)",
    )
    dimensions: list[str] = Field(
        default_factory=list,
        description="Comparison dimensions (e.g. memory usage, accuracy)",
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


async def _extract_entities_dimensions(
    question: str,
    llm: BaseChatModel,
) -> CompareEntities:
    """Fallback: extract entities and dimensions from the question via LLM.

    Called only when ``intent.entities`` is empty — the slot-filling step
    should normally have populated them already.

    Args:
        question: The reformulated user question.
        llm: Cloud LLM instance.

    Returns:
        A ``CompareEntities`` with extracted entities and dimensions.
    """
    structured_llm = llm.with_structured_output(CompareEntities)
    result: CompareEntities = await structured_llm.ainvoke([
        SystemMessage(content=COMPARATIVE_EXTRACT_SYSTEM_PROMPT),
        HumanMessage(content=COMPARATIVE_EXTRACT_USER_TEMPLATE.format(
            question=question,
        )),
    ])
    return result


async def _retrieve_for_entity(
    entity: str,
    question: str,
    rag: RAGToolWrapper,
    top_k: int,
    *,
    collection: str | None = None,
) -> list[RetrievedContext]:
    """Search RAG for a single entity in the context of the question.

    Args:
        entity: The entity name (e.g. "LoRA").
        question: The user's reformulated question for context.
        rag: RAG tool wrapper.
        top_k: Number of results to retrieve.
        collection: Optional RAG collection name to restrict search to.

    Returns:
        A list of ``RetrievedContext`` objects for this entity.
    """
    query = f"{entity}: {question}"
    return await rag.search(query, top_k=top_k, collection=collection)


# ---------------------------------------------------------------------------
# Core implementation (testable with explicit deps)
# ---------------------------------------------------------------------------


async def run_comparative_strategy(
    state: AgentState,
    rag: RAGToolWrapper,
    llm: BaseChatModel,
    *,
    top_k: int = DEFAULT_TOP_K,
    collection: str | None = None,
) -> dict[str, Any]:
    """Execute the comparative strategy: parallel retrieval + structured comparison.

    This is the core implementation that accepts explicit dependencies for
    easy unit testing.

    Args:
        state: Current agent state (must have ``intent`` populated).
        rag: RAG tool wrapper for knowledge-base search.
        llm: Cloud LLM instance for entity extraction fallback and synthesis.
        top_k: Maximum number of retrieval results per entity.

    Returns:
        Partial state update dict with ``draft_answer``,
        ``retrieved_contexts``, ``retrieval_queries``, and
        ``reasoning_trace`` entries.

    Raises:
        ValueError: If ``state.intent`` is ``None``.
    """
    if state.intent is None:
        raise ValueError("comparative_strategy requires state.intent to be set")

    # -- 1. Resolve query ---------------------------------------------------
    question = state.intent.reformulated_query or state.question
    if not question:
        logger.warning("No query available — using raw question as fallback")
        question = state.question or "No question provided"

    logger.info("Comparative strategy: question=%r", question[:80])

    # -- 2. Entities & dimensions -------------------------------------------
    # Prioritize intent.entities/dimensions from slot filling; fallback to LLM
    intent = state.intent
    if intent.entities:
        entities = intent.entities
        dimensions = intent.dimensions
        logger.info(
            "Using intent entities=%r, dimensions=%r",
            entities, dimensions,
        )
    else:
        logger.info("Intent entities empty — falling back to LLM extraction")
        extracted = await _extract_entities_dimensions(question, llm)
        entities = extracted.entities
        dimensions = extracted.dimensions

    trace_steps: list[ReasoningStep] = [
        ReasoningStep(
            step_type="thought",
            content=(
                f"Comparative plan: entities={entities}, dimensions={dimensions}"
            ),
            metadata={"entities": entities, "dimensions": dimensions},
        ),
    ]

    # -- 3. Parallel retrieval (ReWOO: no intermediate LLM calls) ----------
    retrieval_tasks = [
        _retrieve_for_entity(entity, question, rag, top_k, collection=collection)
        for entity in entities
    ]

    logger.info(
        "Comparative strategy: parallel retrieval for %d entity/entities",
        len(entities),
    )
    all_results: list[list[RetrievedContext]] = await asyncio.gather(
        *retrieval_tasks,
    )

    # Collect all contexts and per-entity mapping
    all_contexts: list[RetrievedContext] = []
    entity_context_map: dict[str, list[RetrievedContext]] = {}
    retrieval_queries: list[str] = []

    for entity, results in zip(entities, all_results):
        entity_context_map[entity] = results
        all_contexts.extend(results)
        retrieval_queries.append(f"{entity}: {question}")

    trace_steps.append(
        ReasoningStep(
            step_type="action",
            content=(
                f"Parallel RAG search: {len(entities)} entity/entities, "
                f"retrieved {len(all_contexts)} total context(s)"
            ),
            metadata={
                "num_entities": len(entities),
                "results_per_entity": {
                    e: len(r) for e, r in entity_context_map.items()
                },
            },
        ),
    )

    # -- 4. Structured comparison synthesis (single LLM call) ---------------
    formatted_contexts = format_entity_contexts(entity_context_map)

    user_prompt = COMPARATIVE_SYNTHESIS_USER_TEMPLATE.format(
        question=question,
        entities=", ".join(entities),
        dimensions=", ".join(dimensions) if dimensions else "(not specified)",
        entity_contexts=formatted_contexts,
    )

    messages = [
        SystemMessage(content=COMPARATIVE_SYNTHESIS_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]

    logger.info("Comparative strategy: invoking LLM for structured comparison")
    llm_response = await llm.ainvoke(messages)
    draft_answer = (
        llm_response.content
        if isinstance(llm_response.content, str)
        else str(llm_response.content)
    )

    trace_steps.append(
        ReasoningStep(
            step_type="thought",
            content=(
                f"Synthesized comparison ({len(draft_answer)} chars) "
                f"across {len(entities)} entities and "
                f"{len(dimensions)} dimension(s)"
            ),
            metadata={
                "answer_length": len(draft_answer),
                "num_entities": len(entities),
                "num_dimensions": len(dimensions),
            },
        ),
    )

    logger.info(
        "Comparative strategy complete: draft_answer=%d chars, "
        "contexts=%d, entities=%d",
        len(draft_answer),
        len(all_contexts),
        len(entities),
    )

    return {
        "draft_answer": draft_answer,
        "retrieved_contexts": all_contexts,
        "retrieval_queries": retrieval_queries,
        "reasoning_trace": trace_steps,
    }


# ---------------------------------------------------------------------------
# Factory for creating a graph-compatible node with bound dependencies
# ---------------------------------------------------------------------------


def create_comparative_strategy_node(
    rag: RAGToolWrapper,
    llm: BaseChatModel,
    *,
    top_k: int = DEFAULT_TOP_K,
    rag_default_collection: str | None = None,
):
    """Create a comparative strategy node function with bound dependencies.

    Use this factory when wiring the node into the LangGraph graph with
    real or test dependencies.

    Args:
        rag: RAG tool wrapper instance.
        llm: Cloud LLM instance.
        top_k: Default number of retrieval results per entity.
        rag_default_collection: Optional RAG collection name to restrict search to.

    Returns:
        An async callable ``(state: AgentState) -> dict`` suitable for
        ``graph.add_node()``.
    """

    async def _node(state: AgentState) -> dict[str, Any]:
        return await run_comparative_strategy(
            state, rag, llm, top_k=top_k, collection=rag_default_collection,
        )

    _node.__name__ = "comparative_strategy_node"
    _node.__doc__ = "Comparative strategy node (ReWOO: parallel retrieval + synthesis)."
    return _node


# ---------------------------------------------------------------------------
# Default export — placeholder until graph.py wires real dependencies
# ---------------------------------------------------------------------------


def comparative_strategy_node(state) -> dict:
    """Execute comparative strategy with parallel retrieval.

    .. note::

        This is a **synchronous placeholder** used by ``graph.py`` before
        real RAG/LLM dependencies are wired via
        ``create_comparative_strategy_node()``.  It returns a stub
        ``draft_answer`` so the graph can compile and invoke without errors.

    Args:
        state: The current AgentState.

    Returns:
        Partial state update with ``draft_answer``.
    """
    return {"draft_answer": "[comparative placeholder] No retrieval performed yet."}
