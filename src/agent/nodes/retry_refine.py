"""Retry-refine node: rewrite question based on critic feedback.

When the Critic rejects a draft answer, this node uses a Cloud LLM to
produce a refined question that incorporates the feedback.  The refined
question replaces ``state.question`` so the next strategy cycle (route →
slot_fill → strategy → critic) retrieves and synthesizes a better answer.

The node also:
- Increments ``retry_count``.
- Clears ``draft_answer`` so the next cycle starts fresh.
- Records a reasoning trace step of type ``"critique"``.

Design reference: PAPER_PILOT_DESIGN.md §6.8, DEV_SPEC D3.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.agent.state import ReasoningStep
from src.prompts.retry_refine import (
    RETRY_REFINE_SYSTEM_PROMPT,
    RETRY_REFINE_USER_TEMPLATE,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Structured output model
# ---------------------------------------------------------------------------


class RefineOutput(BaseModel):
    """Structured output schema for the retry-refine LLM call.

    Attributes:
        refined_question: The rewritten question incorporating critic feedback.
        refinement_summary: Brief summary of what was changed and why.
    """

    refined_question: str = Field(
        description="Rewritten question that addresses the critic's feedback",
    )
    refinement_summary: str = Field(
        description="Brief summary of what was changed and why",
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
# Core implementation (testable with explicit deps)
# ---------------------------------------------------------------------------


async def run_retry_refine(
    state: Any,
    llm: BaseChatModel,
) -> dict[str, Any]:
    """Refine the question based on critic feedback via Cloud LLM.

    Reads the original question, draft answer, critic feedback, and intent
    from *state*, invokes the LLM to produce a refined question, then
    returns a partial state update.

    Args:
        state: The current AgentState (dict-like or object).
        llm: Cloud LLM instance supporting ``with_structured_output``.

    Returns:
        Partial state update dict with:
        - ``question``: the refined question.
        - ``retry_count``: incremented by 1.
        - ``draft_answer``: cleared (empty string).
        - ``reasoning_trace``: a new trace step.
    """
    # -- Extract fields from state ------------------------------------------
    question: str = _get(state, "question", "")
    draft_answer: str = _get(state, "draft_answer", "")
    retry_count: int = _get(state, "retry_count", 0)
    intent = _get(state, "intent")
    critic_verdict = _get(state, "critic_verdict")

    strategy = "unknown"
    if intent is not None:
        strategy = getattr(intent, "to_strategy", lambda: "unknown")()

    feedback = ""
    if critic_verdict is not None:
        feedback = getattr(critic_verdict, "feedback", "")

    new_retry_count = retry_count + 1

    logger.info(
        "retry_refine: refining question (retry %d→%d), feedback=%r",
        retry_count,
        new_retry_count,
        feedback[:120],
    )

    # Guard: no feedback available — fall back to appending a generic hint
    if not feedback.strip():
        logger.warning("retry_refine: no critic feedback — appending generic hint")
        refined_question = f"{question} (Please provide a more comprehensive and detailed answer.)"
        summary = "No specific feedback; added generic improvement hint."
    else:
        # -- Build messages -------------------------------------------------
        user_prompt = RETRY_REFINE_USER_TEMPLATE.format(
            question=question,
            strategy=strategy,
            draft_answer=draft_answer,
            feedback=feedback,
        )
        messages = [
            SystemMessage(content=RETRY_REFINE_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]

        # -- Invoke LLM with structured output ------------------------------
        structured_llm = llm.with_structured_output(RefineOutput)
        refine_output: RefineOutput = await structured_llm.ainvoke(messages)

        refined_question = refine_output.refined_question
        summary = refine_output.refinement_summary

    logger.info(
        "retry_refine: refined question=%r (summary=%r)",
        refined_question[:120],
        summary[:120],
    )

    # -- Build reasoning trace step -----------------------------------------
    trace_step = ReasoningStep(
        step_type="critique",
        content=(
            f"Retry-refine (retry {new_retry_count}): "
            f"rewrote question based on critic feedback. "
            f"Summary: {summary}"
        ),
        metadata={
            "retry_count": new_retry_count,
            "original_question": question,
            "refined_question": refined_question,
            "feedback": feedback[:200],
            "summary": summary,
        },
    )

    return {
        "question": refined_question,
        "retry_count": new_retry_count,
        "draft_answer": "",
        "reasoning_trace": [trace_step],
    }


# ---------------------------------------------------------------------------
# Factory for creating a graph-compatible node with bound dependencies
# ---------------------------------------------------------------------------


def create_retry_refine_node(llm: BaseChatModel):
    """Create a retry_refine node function with a bound LLM dependency.

    Use this factory when wiring the node into the LangGraph graph with
    a real or test LLM instance.

    Args:
        llm: Cloud LLM instance.

    Returns:
        An async callable ``(state) -> dict`` suitable for
        ``graph.add_node()``.
    """

    async def _node(state: Any) -> dict[str, Any]:
        return await run_retry_refine(state, llm)

    _node.__name__ = "retry_refine_node"
    _node.__doc__ = "Retry-refine node (question rewrite based on critic feedback)."
    return _node


# ---------------------------------------------------------------------------
# Default export — sync placeholder when no LLM is configured
# ---------------------------------------------------------------------------


def retry_refine_node(state: Any) -> dict[str, Any]:
    """Refine the question or plan based on critic feedback.

    .. note::

        This is a **synchronous placeholder** used when no Cloud LLM is
        configured.  It simply increments ``retry_count`` and appends the
        critic feedback to the question.  The real async implementation
        is created via ``create_retry_refine_node(llm)``.

    Args:
        state: The current AgentState.

    Returns:
        Partial state update with refined ``question``,
        incremented ``retry_count``, and cleared ``draft_answer``.
    """
    question: str = _get(state, "question", "")
    retry_count: int = _get(state, "retry_count", 0)
    critic_verdict = _get(state, "critic_verdict")

    feedback = ""
    if critic_verdict is not None:
        feedback = getattr(critic_verdict, "feedback", "")

    new_retry_count = retry_count + 1

    if feedback.strip():
        refined_question = (
            f"{question} (Feedback: {feedback})"
        )
    else:
        refined_question = (
            f"{question} (Please provide a more comprehensive and detailed answer.)"
        )

    return {
        "question": refined_question,
        "retry_count": new_retry_count,
        "draft_answer": "",
        "reasoning_trace": [
            ReasoningStep(
                step_type="critique",
                content=(
                    f"Retry-refine placeholder (retry {new_retry_count}): "
                    f"appended feedback to question."
                ),
                metadata={
                    "retry_count": new_retry_count,
                    "feedback": feedback[:200],
                },
            ),
        ],
    }
