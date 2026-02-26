"""Slot Filling node: extract entities, dimensions, constraints, and reformulated query.

Takes the partial ``Intent`` (type + confidence) written by the Router and
uses a Cloud LLM with structured output to populate the remaining slots:
``entities``, ``dimensions``, ``constraints``, and ``reformulated_query``.

When a Cloud LLM is provided (via ``create_slot_filling_node``), the node
uses LangChain's ``with_structured_output`` to extract a
``SlotFillingOutput`` Pydantic model.  Without an LLM, the sync placeholder
passes the intent through with the original question as the reformulated
query.

Design reference: PAPER_PILOT_DESIGN.md §6.2, DEV_SPEC C2.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.agent.state import Intent, IntentType, ReasoningStep
from src.prompts.slot_filling import (
    SLOT_FILLING_SYSTEM_PROMPT,
    SLOT_FILLING_USER_TEMPLATE,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Structured output model
# ---------------------------------------------------------------------------

# List of {key, value} is used instead of dict[str, str] so that the JSON
# schema has explicit properties; OpenAI structured output rejects nested
# objects with only additionalProperties (see "Extra required key 'constraints'"
# 400 error).


class ConstraintEntry(BaseModel):
    """Single key-value constraint for Slot Filling (schema-friendly for OpenAI)."""

    key: str = Field(description="Constraint name (e.g. time_range, model_scale)")
    value: str = Field(description="Constraint value")


class SlotFillingOutput(BaseModel):
    """Structured output schema for the Slot Filling LLM call.

    Attributes:
        entities: Key entities extracted from the question.
        dimensions: Comparison dimensions (non-empty for comparative intent).
        constraints: Implicit constraints as list of key-value pairs.
        reformulated_query: Rewritten query optimised for retrieval.
    """

    entities: list[str] = Field(
        description="Key entities (paper names, technique names, model names)",
    )
    dimensions: list[str] = Field(
        default_factory=list,
        description="Comparison dimensions (for comparative intent only)",
    )
    constraints: list[ConstraintEntry] = Field(
        default_factory=list,
        description="Implicit constraints (time_range, model_scale, domain, etc.) as key-value pairs",
    )
    reformulated_query: str = Field(
        description="Rewritten clear query optimised for retrieval",
    )


# ---------------------------------------------------------------------------
# Core implementation (testable with explicit deps)
# ---------------------------------------------------------------------------


async def run_slot_filling(
    state: Any,
    llm: BaseChatModel,
) -> dict[str, Any]:
    """Fill intent slots via Cloud LLM structured output.

    Reads the partial ``Intent`` (from the Router) out of *state*, invokes
    the LLM to extract entities / dimensions / constraints /
    reformulated_query, and returns an updated ``Intent`` with all slots
    populated.

    Args:
        state: The current AgentState (dict-like or object).
        llm: Cloud LLM instance supporting ``with_structured_output``.

    Returns:
        Partial state update dict with ``intent`` (fully populated) and
        ``reasoning_trace`` entries.
    """
    # -- Extract fields from state ------------------------------------------
    if isinstance(state, dict):
        intent: Intent | None = state.get("intent")
        question: str = state.get("question", "")
    else:
        intent = getattr(state, "intent", None)
        question = getattr(state, "question", "")

    # Guard: if Router hasn't set an intent, create a fallback
    if intent is None:
        logger.warning("SlotFilling received no intent — defaulting to factual")
        intent = Intent(
            type="factual",
            confidence=0.0,
            reformulated_query=question,
        )

    logger.info(
        "SlotFilling: filling slots for intent_type=%s, question=%r",
        intent.type,
        question[:80],
    )

    # -- Build messages -----------------------------------------------------
    user_prompt = SLOT_FILLING_USER_TEMPLATE.format(
        intent_type=intent.type,
        confidence=intent.confidence,
        question=question,
    )
    messages = [
        SystemMessage(content=SLOT_FILLING_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]

    # -- Invoke LLM with structured output ----------------------------------
    structured_llm = llm.with_structured_output(SlotFillingOutput)
    sf_output: SlotFillingOutput = await structured_llm.ainvoke(messages)

    constraints_dict = {e.key: e.value for e in sf_output.constraints}

    logger.info(
        "SlotFilling result: entities=%s, dimensions=%s, "
        "constraints=%s, reformulated_query=%r",
        sf_output.entities,
        sf_output.dimensions,
        constraints_dict,
        sf_output.reformulated_query[:80],
    )

    # -- Build fully-populated Intent ---------------------------------------
    filled_intent = Intent(
        type=intent.type,
        confidence=intent.confidence,
        entities=sf_output.entities,
        dimensions=sf_output.dimensions,
        constraints=constraints_dict,
        reformulated_query=sf_output.reformulated_query,
    )

    trace_step = ReasoningStep(
        step_type="route",
        content=(
            f"SlotFilling: extracted {len(sf_output.entities)} entities, "
            f"{len(sf_output.dimensions)} dimensions, "
            f"{len(sf_output.constraints)} constraints; "
            f"reformulated_query={sf_output.reformulated_query!r}"
        ),
        metadata={
            "entities": sf_output.entities,
            "dimensions": sf_output.dimensions,
            "constraints": constraints_dict,
            "reformulated_query": sf_output.reformulated_query,
        },
    )

    return {
        "intent": filled_intent,
        "reasoning_trace": [trace_step],
    }


# ---------------------------------------------------------------------------
# Factory for creating a graph-compatible node with bound dependencies
# ---------------------------------------------------------------------------


def create_slot_filling_node(llm: BaseChatModel):
    """Create a Slot Filling node function with a bound LLM dependency.

    Use this factory when wiring the node into the LangGraph graph with
    a real or test LLM instance.

    Args:
        llm: Cloud LLM instance.

    Returns:
        An async callable ``(state) -> dict`` suitable for
        ``graph.add_node()``.
    """

    async def _node(state: Any) -> dict[str, Any]:
        return await run_slot_filling(state, llm)

    _node.__name__ = "slot_filling_node"
    _node.__doc__ = "Slot Filling node (entity/dimension/constraint extraction via Cloud LLM)."
    return _node


# ---------------------------------------------------------------------------
# Default export — sync placeholder when no LLM is configured
# ---------------------------------------------------------------------------


def slot_filling_node(state) -> dict:
    """Fill intent slots (placeholder — passes intent through).

    .. note::

        This is a **synchronous placeholder** used when no Cloud LLM is
        configured.  It preserves the Router's partial intent and sets
        the ``reformulated_query`` to the original question if empty.
        The real async implementation is created via
        ``create_slot_filling_node(llm)``.

    Args:
        state: The current AgentState.

    Returns:
        Partial state update — intent with reformulated_query backfilled
        if it was empty.
    """
    if isinstance(state, dict):
        intent: Intent | None = state.get("intent")
        question: str = state.get("question", "")
    else:
        intent = getattr(state, "intent", None)
        question = getattr(state, "question", "")

    if intent is None:
        return {}

    # If reformulated_query is already set (by Router), keep it.
    # Otherwise, fall back to the original question.
    if not intent.reformulated_query:
        return {
            "intent": Intent(
                type=intent.type,
                confidence=intent.confidence,
                entities=intent.entities,
                dimensions=intent.dimensions,
                constraints=intent.constraints,
                reformulated_query=question or "placeholder query",
            ),
        }

    return {}
