"""Critic node: evaluate answer quality via Cloud LLM (structured output).

Evaluates the draft answer against the original question and retrieved
contexts.  Outputs a ``CriticVerdict`` Pydantic model with quality scores
(completeness, faithfulness), an overall score (0–10), a pass/fail flag,
and actionable feedback.

When a Cloud LLM is provided (via ``create_critic_node``), the node uses
LangChain's ``with_structured_output`` to extract a ``CriticOutput``
Pydantic model from the LLM response.  Without an LLM, the sync
placeholder always passes with a fixed score.

Design reference: PAPER_PILOT_DESIGN.md §6.7, DEV_SPEC D1.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Protocol

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.agent.state import CriticVerdict, ReasoningStep
from src.prompts.critic import CRITIC_SYSTEM_PROMPT, CRITIC_USER_TEMPLATE
from src.prompts.strategies import format_retrieved_contexts

logger = logging.getLogger(__name__)

class LocalCriticEvaluator(Protocol):
    """Protocol for local Critic inference providers."""

    def evaluate_answer(
        self,
        *,
        question: str,
        draft_answer: str,
        retrieved_contexts: list[Any] | None = None,
        strategy: str = "unknown",
        pass_threshold: float = 7.0,
    ) -> dict[str, float | bool | str]:
        """Return quality metrics for the current draft answer."""


# ---------------------------------------------------------------------------
# Structured output model
# ---------------------------------------------------------------------------

CRITIC_PASS_THRESHOLD = 7.0
"""Score threshold for the pass/fail decision (score >= 7.0 → pass)."""


class CriticOutput(BaseModel):
    """Structured output schema for the Critic LLM call.

    Attributes:
        score: Overall answer quality score in [0, 10].
        completeness: How completely the answer addresses the question (0–1).
        faithfulness: How well the answer is grounded in contexts (0–1).
        feedback: Actionable improvement suggestions or positive remarks.
    """

    score: float = Field(
        ge=0.0,
        le=10.0,
        description="Overall answer quality score (0–10)",
    )
    completeness: float = Field(
        ge=0.0,
        le=1.0,
        description="Completeness rating (0–1)",
    )
    faithfulness: float = Field(
        ge=0.0,
        le=1.0,
        description="Faithfulness rating (0–1)",
    )
    feedback: str = Field(
        description="Specific feedback — improvement suggestions or positive remarks",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get(state: Any, key: str, default: Any = None) -> Any:
    """Extract a value from *state* regardless of dict or model form."""
    if isinstance(state, dict):
        return state.get(key, default)
    return getattr(state, key, default)


def _derive_passed(score: float) -> bool:
    """Determine pass/fail from the overall score."""
    return score >= CRITIC_PASS_THRESHOLD


# ---------------------------------------------------------------------------
# Core implementation (testable with explicit deps)
# ---------------------------------------------------------------------------


async def run_critic(
    state: Any,
    llm: BaseChatModel | None,
    local_critic: LocalCriticEvaluator | None = None,
) -> dict[str, Any]:
    """Evaluate the draft answer via Cloud LLM structured output.

    Reads the question, draft_answer, retrieved_contexts, and intent from
    *state*. The node prefers local Critic inference when ``local_critic`` is
    provided. If local inference fails, it falls back to Cloud LLM structured
    output. If neither local nor cloud is available, a deterministic default
    verdict is returned.

    Args:
        state: The current AgentState (dict-like or object).
        llm: Optional Cloud LLM instance supporting ``with_structured_output``.
        local_critic: Optional local Critic inference provider.

    Returns:
        Partial state update dict with ``critic_verdict`` and
        ``reasoning_trace`` entries.
    """
    # -- Extract fields from state ------------------------------------------
    question: str = _get(state, "question", "")
    draft_answer: str = _get(state, "draft_answer", "")
    retrieved_contexts: list = _get(state, "retrieved_contexts", [])
    intent = _get(state, "intent")

    strategy = "unknown"
    if intent is not None:
        strategy = getattr(intent, "to_strategy", lambda: "unknown")()

    # Guard: empty draft answer
    if not draft_answer.strip():
        logger.warning("Critic received empty draft_answer — auto-fail")
        verdict = CriticVerdict(
            passed=False,
            score=0.0,
            completeness=0.0,
            faithfulness=0.0,
            feedback="Draft answer is empty. The strategy must produce a substantive answer.",
        )
        return {
            "critic_verdict": verdict,
            "reasoning_trace": [
                ReasoningStep(
                    step_type="critique",
                    content="Critic: draft_answer is empty — auto-fail (score=0.0)",
                    metadata={"score": 0.0, "passed": False},
                ),
            ],
        }

    logger.info(
        "Critic: evaluating draft_answer (len=%d) for question=%r",
        len(draft_answer),
        question[:80],
    )

    source = "cloud"
    score: float
    completeness: float
    faithfulness: float
    feedback: str

    if local_critic is not None:
        try:
            t0 = time.perf_counter()
            local_result = local_critic.evaluate_answer(
                question=question,
                draft_answer=draft_answer,
                retrieved_contexts=retrieved_contexts,
                strategy=strategy,
                pass_threshold=CRITIC_PASS_THRESHOLD,
            )
            llm_call_ms = (time.perf_counter() - t0) * 1000
            local_output = CriticOutput(
                score=float(local_result.get("score", 5.0)),
                completeness=float(local_result.get("completeness", 0.5)),
                faithfulness=float(local_result.get("faithfulness", 0.5)),
                feedback=str(
                    local_result.get(
                        "feedback",
                        "Local critic did not provide feedback.",
                    )
                ),
            )
            score = local_output.score
            completeness = local_output.completeness
            faithfulness = local_output.faithfulness
            feedback = local_output.feedback
            source = "local"
        except Exception as exc:
            logger.warning(
                "Critic local evaluation failed, falling back to cloud: %s",
                exc,
            )
            source = "cloud_fallback"
        else:
            passed = _derive_passed(score)
            logger.info(
                "Critic result(local): score=%.1f, completeness=%.2f, "
                "faithfulness=%.2f, passed=%s",
                score,
                completeness,
                faithfulness,
                passed,
            )
            verdict = CriticVerdict(
                passed=passed,
                score=score,
                completeness=completeness,
                faithfulness=faithfulness,
                feedback=feedback,
            )
            trace_step = ReasoningStep(
                step_type="critique",
                content=(
                    f"Critic evaluated draft: score={score:.1f}, "
                    f"completeness={completeness:.2f}, "
                    f"faithfulness={faithfulness:.2f}, "
                    f"passed={passed}"
                ),
                metadata={
                    "score": score,
                    "completeness": completeness,
                    "faithfulness": faithfulness,
                    "passed": passed,
                    "feedback_preview": feedback[:200],
                    "source": source,
                    "llm_call_duration_ms": round(llm_call_ms, 1),
                },
            )
            return {
                "critic_verdict": verdict,
                "reasoning_trace": [trace_step],
            }

    if llm is None:
        logger.warning(
            "Critic has no local/cloud model available — returning deterministic "
            "fallback verdict"
        )
        verdict = CriticVerdict(
            passed=True,
            score=7.0,
            completeness=0.8,
            faithfulness=0.9,
            feedback="Fallback verdict — no local/cloud critic model available.",
        )
        trace_step = ReasoningStep(
            step_type="critique",
            content=(
                "Critic returned fallback verdict because no local/cloud model "
                "was available (score=7.0)"
            ),
            metadata={
                "score": 7.0,
                "completeness": 0.8,
                "faithfulness": 0.9,
                "passed": True,
                "feedback_preview": verdict.feedback[:200],
                "source": "default_no_model",
            },
        )
        return {
            "critic_verdict": verdict,
            "reasoning_trace": [trace_step],
        }

    # -- Format contexts for prompt -----------------------------------------
    contexts_text = format_retrieved_contexts(retrieved_contexts)

    # -- Build messages -----------------------------------------------------
    user_prompt = CRITIC_USER_TEMPLATE.format(
        question=question,
        strategy=strategy,
        draft_answer=draft_answer,
        contexts=contexts_text,
    )
    messages = [
        SystemMessage(content=CRITIC_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]

    # -- Invoke LLM with structured output ----------------------------------
    structured_llm = llm.with_structured_output(CriticOutput)
    t0 = time.perf_counter()
    critic_output: CriticOutput = await structured_llm.ainvoke(messages)
    llm_call_ms = (time.perf_counter() - t0) * 1000

    score = critic_output.score
    completeness = critic_output.completeness
    faithfulness = critic_output.faithfulness
    feedback = critic_output.feedback
    passed = _derive_passed(score)

    logger.info(
        "Critic result: score=%.1f, completeness=%.2f, faithfulness=%.2f, "
        "passed=%s",
        score,
        completeness,
        faithfulness,
        passed,
    )

    # -- Build CriticVerdict ------------------------------------------------
    verdict = CriticVerdict(
        passed=passed,
        score=score,
        completeness=completeness,
        faithfulness=faithfulness,
        feedback=feedback,
    )

    trace_step = ReasoningStep(
        step_type="critique",
        content=(
            f"Critic evaluated draft: score={score:.1f}, "
            f"completeness={completeness:.2f}, "
            f"faithfulness={faithfulness:.2f}, "
            f"passed={passed}"
        ),
        metadata={
            "score": score,
            "completeness": completeness,
            "faithfulness": faithfulness,
            "passed": passed,
            "feedback_preview": feedback[:200],
            "source": source,
            "llm_call_duration_ms": round(llm_call_ms, 1),
        },
    )

    return {
        "critic_verdict": verdict,
        "reasoning_trace": [trace_step],
    }


# ---------------------------------------------------------------------------
# Factory for creating a graph-compatible node with bound dependencies
# ---------------------------------------------------------------------------


def create_critic_node(
    llm: BaseChatModel | None,
    *,
    local_critic: LocalCriticEvaluator | None = None,
):
    """Create a Critic node function with a bound LLM dependency.

    Use this factory when wiring the node into the LangGraph graph with
    a real or test LLM instance.

    Args:
        llm: Optional Cloud LLM instance.
        local_critic: Optional local Critic evaluator.

    Returns:
        An async callable ``(state) -> dict`` suitable for
        ``graph.add_node()``.
    """

    async def _node(state: Any) -> dict[str, Any]:
        return await run_critic(state, llm, local_critic=local_critic)

    _node.__name__ = "critic_node"
    _node.__doc__ = "Critic node (answer quality evaluation via Cloud LLM)."
    return _node


# ---------------------------------------------------------------------------
# Default export — sync placeholder when no LLM is configured
# ---------------------------------------------------------------------------


def critic_node(state) -> dict:
    """Evaluate the draft answer and produce a CriticVerdict.

    .. note::

        This is a **synchronous placeholder** used when no Cloud LLM is
        configured.  It always passes with a good score.  The real async
        implementation is created via ``create_critic_node(llm)``.

    Args:
        state: The current AgentState.

    Returns:
        Partial state update with a placeholder ``critic_verdict``.
    """
    return {
        "critic_verdict": CriticVerdict(
            passed=True,
            score=7.0,
            completeness=0.8,
            faithfulness=0.9,
            feedback="Placeholder — critic not yet implemented.",
        ),
    }
