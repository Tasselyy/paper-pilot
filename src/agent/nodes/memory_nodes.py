"""Memory nodes: load_memory and save_memory.

``load_memory_node`` — runs at the start of each agent turn, retrieving
relevant historical facts from long-term memory and writing them to
``state.accumulated_facts``.

``save_memory_node`` — runs after Critic passes, extracting notable
facts from the completed Q&A turn (via LLM structured output) and
persisting them to long-term memory.

Both nodes are available as:
- Sync placeholder functions (``load_memory_node`` / ``save_memory_node``)
  that work without external dependencies.
- Factory-created async functions (``create_load_memory_node`` /
  ``create_save_memory_node``) that accept a ``LongTermMemory`` instance
  and, for save, a Cloud LLM.

Design reference: PAPER_PILOT_DESIGN.md §8.3, DEV_SPEC D4.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.agent.state import ReasoningStep
from src.memory.long_term import LongTermMemory
from src.prompts.memory import (
    EXTRACT_FACTS_SYSTEM_PROMPT,
    EXTRACT_FACTS_USER_TEMPLATE,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Structured output model for fact extraction
# ---------------------------------------------------------------------------


class ExtractedFacts(BaseModel):
    """Structured output schema for the fact-extraction LLM call.

    Attributes:
        facts: List of factual statements worth remembering (0–5 items).
    """

    facts: list[str] = Field(
        default_factory=list,
        description="Key facts extracted from the Q&A pair (0–5 items)",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get(state: Any, key: str, default: Any = None) -> Any:
    """Extract a value from *state* regardless of dict or model form."""
    if isinstance(state, dict):
        return state.get(key, default)
    return getattr(state, key, default)


# ---------------------------------------------------------------------------
# Core implementations (testable with explicit deps)
# ---------------------------------------------------------------------------


async def run_load_memory(
    state: Any,
    memory: LongTermMemory,
) -> dict[str, Any]:
    """Load relevant facts from long-term memory into state.

    Reads ``state.question`` and calls ``memory.recall`` to retrieve
    historically relevant facts.

    Args:
        state: The current AgentState (dict-like or object).
        memory: ``LongTermMemory`` instance.

    Returns:
        Partial state update with ``accumulated_facts`` and a
        ``reasoning_trace`` entry.
    """
    question: str = _get(state, "question", "")

    if not question.strip():
        logger.warning("load_memory: question is empty — skipping recall")
        return {
            "accumulated_facts": [],
            "reasoning_trace": [
                ReasoningStep(
                    step_type="action",
                    content="load_memory: skipped (empty question)",
                    metadata={},
                ),
            ],
        }

    relevant_facts = await memory.recall(question)

    logger.info(
        "load_memory: recalled %d facts for question=%r",
        len(relevant_facts),
        question[:80],
    )

    trace_step = ReasoningStep(
        step_type="action",
        content=(
            f"Loaded {len(relevant_facts)} fact(s) from long-term memory"
        ),
        metadata={
            "fact_count": len(relevant_facts),
            "facts_preview": [f[:100] for f in relevant_facts[:3]],
        },
    )

    return {
        "accumulated_facts": relevant_facts,
        "reasoning_trace": [trace_step],
    }


async def run_save_memory(
    state: Any,
    memory: LongTermMemory,
    llm: BaseChatModel | None = None,
) -> dict[str, Any]:
    """Extract and persist notable facts from the current Q&A turn.

    When an LLM is provided, uses structured output to extract facts.
    Otherwise, falls back to a simple heuristic (stores the first
    sentence of the draft answer if it is substantive).

    Args:
        state: The current AgentState (dict-like or object).
        memory: ``LongTermMemory`` instance.
        llm: Optional Cloud LLM for fact extraction.

    Returns:
        Partial state update with a ``reasoning_trace`` entry.
    """
    question: str = _get(state, "question", "")
    draft_answer: str = _get(state, "draft_answer", "")

    if not draft_answer.strip():
        logger.info("save_memory: draft_answer is empty — nothing to memorize")
        return {
            "reasoning_trace": [
                ReasoningStep(
                    step_type="action",
                    content="save_memory: skipped (empty draft_answer)",
                    metadata={},
                ),
            ],
        }

    # -- Extract facts -------------------------------------------------------
    facts_to_store: list[str] = []

    if llm is not None:
        try:
            user_prompt = EXTRACT_FACTS_USER_TEMPLATE.format(
                question=question,
                answer=draft_answer,
            )
            messages = [
                SystemMessage(content=EXTRACT_FACTS_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ]
            structured_llm = llm.with_structured_output(ExtractedFacts)
            extraction: ExtractedFacts = await structured_llm.ainvoke(messages)
            facts_to_store = [f for f in extraction.facts if f.strip()]
        except Exception as exc:
            logger.warning(
                "save_memory: LLM fact extraction failed — falling back to heuristic: %s",
                exc,
            )
            facts_to_store = _heuristic_extract(draft_answer)
    else:
        facts_to_store = _heuristic_extract(draft_answer)

    # -- Persist facts -------------------------------------------------------
    new_facts = await memory.memorize(
        question=question,
        answer=draft_answer,
        facts_to_store=facts_to_store,
    )

    logger.info(
        "save_memory: stored %d fact(s) for question=%r",
        len(new_facts),
        question[:80],
    )

    trace_step = ReasoningStep(
        step_type="action",
        content=(
            f"Saved {len(new_facts)} fact(s) to long-term memory"
        ),
        metadata={
            "stored_count": len(new_facts),
            "facts_preview": [f.content[:100] for f in new_facts[:3]],
        },
    )

    return {
        "reasoning_trace": [trace_step],
    }


def _heuristic_extract(answer: str) -> list[str]:
    """Simple heuristic fact extraction (no LLM required).

    Splits the answer into sentences and returns the first substantive
    sentence (length > 20 chars) as a single fact.

    Args:
        answer: The draft answer text.

    Returns:
        List with 0 or 1 fact strings.
    """
    sentences = answer.replace("\n", " ").split(".")
    for sent in sentences:
        sent = sent.strip()
        if len(sent) > 20:
            return [sent + "."]
    return []


# ---------------------------------------------------------------------------
# Factories for creating graph-compatible nodes with bound dependencies
# ---------------------------------------------------------------------------


def create_load_memory_node(memory: LongTermMemory):
    """Create a load_memory node function with a bound ``LongTermMemory``.

    Args:
        memory: ``LongTermMemory`` instance.

    Returns:
        An async callable ``(state) -> dict`` suitable for
        ``graph.add_node()``.
    """

    async def _node(state: Any) -> dict[str, Any]:
        return await run_load_memory(state, memory)

    _node.__name__ = "load_memory_node"
    _node.__doc__ = "Load relevant facts from long-term memory."
    return _node


def create_save_memory_node(
    memory: LongTermMemory,
    llm: BaseChatModel | None = None,
):
    """Create a save_memory node function with bound dependencies.

    Args:
        memory: ``LongTermMemory`` instance.
        llm: Optional Cloud LLM for structured fact extraction.

    Returns:
        An async callable ``(state) -> dict`` suitable for
        ``graph.add_node()``.
    """

    async def _node(state: Any) -> dict[str, Any]:
        return await run_save_memory(state, memory, llm)

    _node.__name__ = "save_memory_node"
    _node.__doc__ = "Extract and persist facts to long-term memory."
    return _node


# ---------------------------------------------------------------------------
# Default exports — sync placeholders when no dependencies are configured
# ---------------------------------------------------------------------------


def load_memory_node(state: Any) -> dict[str, Any]:
    """Load relevant facts from long-term memory into state.

    .. note::

        This is a **synchronous placeholder** used when no
        ``LongTermMemory`` is configured.  It returns an empty
        ``accumulated_facts`` list.  The real async implementation
        is created via ``create_load_memory_node(memory)``.

    Args:
        state: The current AgentState.

    Returns:
        Partial state update with empty ``accumulated_facts``.
    """
    return {
        "accumulated_facts": [],
        "reasoning_trace": [
            ReasoningStep(
                step_type="action",
                content="load_memory: placeholder (no LongTermMemory configured)",
                metadata={},
            ),
        ],
    }


def save_memory_node(state: Any) -> dict[str, Any]:
    """Persist notable facts from the current turn to long-term memory.

    .. note::

        This is a **synchronous placeholder** used when no
        ``LongTermMemory`` is configured.  It is a no-op.  The real
        async implementation is created via
        ``create_save_memory_node(memory, llm)``.

    Args:
        state: The current AgentState.

    Returns:
        Partial state update with a reasoning trace entry.
    """
    return {
        "reasoning_trace": [
            ReasoningStep(
                step_type="action",
                content="save_memory: placeholder (no LongTermMemory configured)",
                metadata={},
            ),
        ],
    }
