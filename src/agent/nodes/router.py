"""Router node: intent classification via Cloud LLM (structured output).

Classifies the user question into one of five intent types (factual,
comparative, multi_hop, exploratory, follow_up) and outputs a partial
``Intent`` object with ``type`` and ``confidence``.

When a Cloud LLM is provided (via ``create_router_node``), the node uses
LangChain's ``with_structured_output`` to extract a ``RouterOutput``
Pydantic model from the LLM response.  Without an LLM, the sync
placeholder defaults to ``factual``.

Design reference: PAPER_PILOT_DESIGN.md §6.1, DEV_SPEC C1.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Protocol

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.agent.state import Intent, IntentType, ReasoningStep
from src.prompts.router import ROUTER_SYSTEM_PROMPT, ROUTER_USER_TEMPLATE

logger = logging.getLogger(__name__)


class LocalRouterClassifier(Protocol):
    """Protocol for local Router inference providers."""

    def classify_question(self, question: str) -> tuple[IntentType, float]:
        """Return ``(intent_type, confidence)`` for a question."""

# ---------------------------------------------------------------------------
# Structured output model
# ---------------------------------------------------------------------------


class RouterOutput(BaseModel):
    """Structured output schema for the Router LLM call.

    Attributes:
        type: The classified intent type.
        confidence: Classification confidence in [0, 1].
    """

    type: IntentType = Field(description="The classified intent type")
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Classification confidence score between 0 and 1",
    )


# ---------------------------------------------------------------------------
# Core implementation (testable with explicit deps)
# ---------------------------------------------------------------------------


async def run_router(
    state: Any,
    llm: BaseChatModel | None,
    local_router: LocalRouterClassifier | None = None,
) -> dict[str, Any]:
    """Classify the user question into an intent type.

    The node prefers a local Router model when ``local_router`` is provided.
    If local inference fails, it gracefully falls back to Cloud LLM
    structured output. If neither local nor cloud is available, the node
    returns a deterministic factual default.

    Args:
        state: The current AgentState (dict-like or object).
        llm: Optional Cloud LLM instance supporting
            ``with_structured_output``.
        local_router: Optional local Router inference provider.

    Returns:
        Partial state update dict with ``intent`` (partial — only ``type``
        and ``confidence`` populated) and ``reasoning_trace`` entries.
    """
    # -- Extract question from state ---
    question: str = (
        state.get("question", "") if isinstance(state, dict)
        else getattr(state, "question", "")
    )

    if not question:
        logger.warning("Router received empty question — defaulting to factual")
        return {
            "intent": Intent(
                type="factual",
                confidence=0.0,
                reformulated_query="",
            ),
            "reasoning_trace": [
                ReasoningStep(
                    step_type="route",
                    content="Router: empty question — defaulted to factual (confidence=0.0)",
                ),
            ],
        }

    logger.info("Router: classifying question=%r", question[:80])

    intent_type: IntentType
    confidence: float
    source = "cloud"

    if local_router is not None:
        try:
            t0 = time.perf_counter()
            local_intent_type, local_confidence = local_router.classify_question(question)
            llm_call_ms = (time.perf_counter() - t0) * 1000
            validated = RouterOutput(type=local_intent_type, confidence=local_confidence)
            intent_type = validated.type
            confidence = validated.confidence
            source = "local"
        except Exception as exc:
            logger.warning(
                "Router local classification failed, falling back to cloud: %s",
                exc,
            )
            source = "cloud_fallback"
        else:
            logger.info(
                "Router result(local): type=%s, confidence=%.2f",
                intent_type,
                confidence,
            )
            intent = Intent(
                type=intent_type,
                confidence=confidence,
                reformulated_query=question,
            )
            trace_step = ReasoningStep(
                step_type="route",
                content=(
                    f"Router classified question as '{intent_type}' "
                    f"(confidence={confidence:.2f})"
                ),
                metadata={
                    "intent_type": intent_type,
                    "confidence": confidence,
                    "question_preview": question[:100],
                    "source": source,
                    "llm_call_duration_ms": round(llm_call_ms, 1),
                },
            )
            return {
                "intent": intent,
                "reasoning_trace": [trace_step],
            }

    if llm is None:
        logger.warning(
            "Router has no cloud LLM available — defaulting to factual "
            "(confidence=0.5)"
        )
        intent = Intent(
            type="factual",
            confidence=0.5,
            reformulated_query=question,
        )
        trace_step = ReasoningStep(
            step_type="route",
            content=(
                "Router defaulted to factual because no local/cloud model "
                "was available (confidence=0.50)"
            ),
            metadata={
                "intent_type": "factual",
                "confidence": 0.5,
                "question_preview": question[:100],
                "source": "default_no_model",
            },
        )
        return {
            "intent": intent,
            "reasoning_trace": [trace_step],
        }

    # -- Build messages ---
    user_prompt = ROUTER_USER_TEMPLATE.format(question=question)
    messages = [
        SystemMessage(content=ROUTER_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]

    # -- Invoke LLM with structured output ---
    structured_llm = llm.with_structured_output(RouterOutput)
    t0 = time.perf_counter()
    router_output: RouterOutput = await structured_llm.ainvoke(messages)
    llm_call_ms = (time.perf_counter() - t0) * 1000

    intent_type = router_output.type
    confidence = router_output.confidence

    logger.info(
        "Router result: type=%s, confidence=%.2f",
        intent_type,
        confidence,
    )

    # -- Build partial Intent (Slot Filling completes the rest in C2) ---
    intent = Intent(
        type=intent_type,
        confidence=confidence,
        reformulated_query=question,  # preserve original until Slot Filling
    )

    trace_step = ReasoningStep(
        step_type="route",
        content=(
            f"Router classified question as '{intent_type}' "
            f"(confidence={confidence:.2f})"
        ),
        metadata={
            "intent_type": intent_type,
            "confidence": confidence,
            "question_preview": question[:100],
            "source": source,
            "llm_call_duration_ms": round(llm_call_ms, 1),
        },
    )

    return {
        "intent": intent,
        "reasoning_trace": [trace_step],
    }


# ---------------------------------------------------------------------------
# Factory for creating a graph-compatible node with bound dependencies
# ---------------------------------------------------------------------------


def create_router_node(
    llm: BaseChatModel | None,
    *,
    local_router: LocalRouterClassifier | None = None,
):
    """Create a Router node function with a bound LLM dependency.

    Use this factory when wiring the node into the LangGraph graph with
    a real or test LLM instance.

    Args:
        llm: Optional Cloud LLM instance.
        local_router: Optional local Router classifier.

    Returns:
        An async callable ``(state) -> dict`` suitable for
        ``graph.add_node()``.
    """

    async def _node(state: Any) -> dict[str, Any]:
        return await run_router(state, llm, local_router=local_router)

    _node.__name__ = "router_node"
    _node.__doc__ = "Router node (intent classification via Cloud LLM)."
    return _node


# ---------------------------------------------------------------------------
# Default export — sync placeholder when no LLM is configured
# ---------------------------------------------------------------------------


def router_node(state) -> dict:
    """Classify the user question into an intent type.

    .. note::

        This is a **synchronous placeholder** used when no Cloud LLM is
        configured.  It defaults to ``factual`` intent with full confidence.
        The real async implementation is created via
        ``create_router_node(llm)``.

    Args:
        state: The current AgentState.

    Returns:
        Partial state update with a default ``factual`` intent.
    """
    question = (
        state.get("question", "") if isinstance(state, dict)
        else getattr(state, "question", "")
    )
    return {
        "intent": Intent(
            type="factual",
            confidence=1.0,
            reformulated_query=question or "placeholder query",
        ),
    }
