"""Integration tests for the retry path (D2).

Verifies the ``critic_gate`` conditional edge routes correctly:

    critic → [critic_gate] → "pass"  → save_memory → format_output → END
                            → "retry" → retry_refine → route (loop)

The retry loop should increment ``retry_count`` on each iteration and
terminate after ``max_retries`` is reached (forced pass).

Expected flow with max_retries=2 and a constantly-failing critic:

    Iter 1: critic (fail) → gate (retry, count=0<2) → retry_refine → count=1
    Iter 2: critic (fail) → gate (retry, count=1<2) → retry_refine → count=2
    Iter 3: critic (fail) → gate (pass,  count=2≥2) → save_memory → END

All external dependencies (RAG MCP, Cloud LLM) are replaced by
lightweight stubs.  The graph is built from scratch to inject a
custom always-failing critic while reusing the real edge functions.

Design reference: DEV_SPEC D2.
"""

from __future__ import annotations

from typing import Any

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.agent.edges import critic_gate, route_by_intent
from src.agent.nodes.format_output import format_output_node
from src.agent.nodes.memory_nodes import load_memory_node, save_memory_node
from src.agent.nodes.retry_refine import retry_refine_node
from src.agent.nodes.router import router_node
from src.agent.nodes.slot_filling import slot_filling_node
from src.agent.state import AgentState, CriticVerdict, ReasoningStep
from src.agent.strategies.simple import simple_strategy_node


# ---------------------------------------------------------------------------
# Stub nodes
# ---------------------------------------------------------------------------


def _always_failing_critic(state: Any) -> dict[str, Any]:
    """Critic stub that always returns ``passed=False`` with a low score.

    This forces the ``critic_gate`` to evaluate the retry logic on every
    iteration rather than short-circuiting on a passing verdict.
    """
    return {
        "critic_verdict": CriticVerdict(
            passed=False,
            score=3.0,
            completeness=0.3,
            faithfulness=0.4,
            feedback="Answer lacks depth and misses key details.",
        ),
        "reasoning_trace": [
            ReasoningStep(
                step_type="critique",
                content="Critic stub: always-fail (score=3.0)",
                metadata={"score": 3.0, "passed": False},
            ),
        ],
    }


def _always_passing_critic(state: Any) -> dict[str, Any]:
    """Critic stub that always returns ``passed=True``."""
    return {
        "critic_verdict": CriticVerdict(
            passed=True,
            score=8.5,
            completeness=0.85,
            faithfulness=0.90,
            feedback="Good answer, well-grounded.",
        ),
    }


# ---------------------------------------------------------------------------
# Graph builder helper
# ---------------------------------------------------------------------------


def _build_retry_graph(
    critic_fn=_always_failing_critic,
    max_retries: int = 2,
) -> StateGraph:
    """Build a compiled graph with a custom critic for retry path testing.

    Uses placeholder / stub nodes for all components except the critic,
    which is injected via *critic_fn*.  The real ``critic_gate`` and
    ``route_by_intent`` conditional edges are used so the test exercises
    the actual routing logic.

    Args:
        critic_fn: Callable ``(state) -> dict`` used as the critic node.
        max_retries: Value to verify against (the state default is used).

    Returns:
        Compiled LangGraph StateGraph with MemorySaver checkpointer.
    """
    graph = StateGraph(AgentState)

    graph.add_node("load_memory", load_memory_node)
    graph.add_node("route", router_node)
    graph.add_node("slot_fill", slot_filling_node)
    graph.add_node("simple", simple_strategy_node)
    graph.add_node("critic", critic_fn)
    graph.add_node("retry_refine", retry_refine_node)
    graph.add_node("save_memory", save_memory_node)
    graph.add_node("format_output", format_output_node)

    # Entry flow
    graph.add_edge(START, "load_memory")
    graph.add_edge("load_memory", "route")
    graph.add_edge("route", "slot_fill")

    # Route by intent — only "simple" strategy wired (sufficient for retry test)
    graph.add_conditional_edges(
        "slot_fill",
        route_by_intent,
        {
            "simple": "simple",
        },
    )

    # Strategy → critic
    graph.add_edge("simple", "critic")

    # Critic → pass / retry (the edge under test)
    graph.add_conditional_edges(
        "critic",
        critic_gate,
        {
            "pass": "save_memory",
            "retry": "retry_refine",
        },
    )

    # Retry loop back to route
    graph.add_edge("retry_refine", "route")

    # Pass path → END
    graph.add_edge("save_memory", "format_output")
    graph.add_edge("format_output", END)

    return graph.compile(checkpointer=MemorySaver())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get(result: Any, key: str, default: Any = None) -> Any:
    """Extract a value from the invoke result (dict or model)."""
    if isinstance(result, dict):
        return result.get(key, default)
    return getattr(result, key, default)


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestRetryPath:
    """Integration tests for the critic_gate retry loop."""

    @pytest.fixture()
    def config(self) -> dict[str, Any]:
        """Provide a unique graph invocation config."""
        return {"configurable": {"thread_id": "test-retry-path"}}

    # -- Core retry behaviour ------------------------------------------------

    async def test_retry_count_reaches_max_retries(
        self,
        config: dict[str, Any],
    ) -> None:
        """With an always-failing critic, retry_count should reach max_retries (2)."""
        graph = _build_retry_graph(critic_fn=_always_failing_critic)
        result = await graph.ainvoke({"question": "What is LoRA?"}, config=config)

        retry_count = _get(result, "retry_count", 0)
        assert retry_count == 2, (
            f"Expected retry_count=2 after exhausting max_retries, got {retry_count}"
        )

    async def test_retry_count_increments_from_zero(
        self,
        config: dict[str, Any],
    ) -> None:
        """retry_count should start at 0 and increment by 1 each retry."""
        graph = _build_retry_graph(critic_fn=_always_failing_critic)
        result = await graph.ainvoke({"question": "What is LoRA?"}, config=config)

        retry_count = _get(result, "retry_count", 0)
        assert retry_count >= 1, "At least one retry should have occurred"
        assert retry_count == 2, "Exactly 2 retries expected with max_retries=2"

    async def test_graph_terminates_after_max_retries(
        self,
        config: dict[str, Any],
    ) -> None:
        """The graph should terminate (reach END) even with a failing critic."""
        graph = _build_retry_graph(critic_fn=_always_failing_critic)
        result = await graph.ainvoke({"question": "What is LoRA?"}, config=config)

        assert result is not None, "Graph should complete and return a result"
        final_answer = _get(result, "final_answer", "")
        assert final_answer, "final_answer should be set after reaching format_output"

    # -- Pass path (no retries) ---------------------------------------------

    async def test_passing_critic_skips_retry(self) -> None:
        """When the critic passes, retry_count should stay 0."""
        graph = _build_retry_graph(critic_fn=_always_passing_critic)
        result = await graph.ainvoke(
            {"question": "What is LoRA?"},
            config={"configurable": {"thread_id": "test-pass-path"}},
        )

        retry_count = _get(result, "retry_count", 0)
        assert retry_count == 0, "No retries expected when critic passes"

    async def test_passing_critic_sets_final_answer(self) -> None:
        """Pass path should produce a non-empty final_answer."""
        graph = _build_retry_graph(critic_fn=_always_passing_critic)
        result = await graph.ainvoke(
            {"question": "What is LoRA?"},
            config={"configurable": {"thread_id": "test-pass-final"}},
        )

        final_answer = _get(result, "final_answer", "")
        assert final_answer, "final_answer should be set on pass path"

    # -- Critic verdict in final state --------------------------------------

    async def test_critic_verdict_present_after_retry(
        self,
        config: dict[str, Any],
    ) -> None:
        """critic_verdict should be present in the final state."""
        graph = _build_retry_graph(critic_fn=_always_failing_critic)
        result = await graph.ainvoke({"question": "What is LoRA?"}, config=config)

        verdict = _get(result, "critic_verdict")
        assert verdict is not None, "critic_verdict should be in the final state"
        assert isinstance(verdict, CriticVerdict)

    async def test_forced_pass_still_has_failing_verdict(
        self,
        config: dict[str, Any],
    ) -> None:
        """When forced-pass by max_retries, the verdict should still show passed=False.

        The ``critic_gate`` forces the "pass" route but doesn't modify the
        verdict itself — the original failing verdict remains in state.
        """
        graph = _build_retry_graph(critic_fn=_always_failing_critic)
        result = await graph.ainvoke({"question": "What is LoRA?"}, config=config)

        verdict = _get(result, "critic_verdict")
        assert verdict is not None
        assert verdict.passed is False, (
            "Forced-pass should not alter the critic's verdict"
        )
        assert verdict.score < 7.0

    # -- Feedback preserved --------------------------------------------------

    async def test_feedback_nonempty_after_retry(
        self,
        config: dict[str, Any],
    ) -> None:
        """Critic feedback should be non-empty in the final state."""
        graph = _build_retry_graph(critic_fn=_always_failing_critic)
        result = await graph.ainvoke({"question": "What is LoRA?"}, config=config)

        verdict = _get(result, "critic_verdict")
        assert verdict is not None
        assert len(verdict.feedback) > 0, "Feedback should be non-empty"

    # -- Reasoning trace records retry activity ------------------------------

    async def test_reasoning_trace_records_retry_critiques(
        self,
        config: dict[str, Any],
    ) -> None:
        """Reasoning trace should contain critique steps from each retry iteration.

        With 2 retries + 1 forced-pass iteration = 3 total critic invocations,
        so at least 3 critique-type trace steps should be present.
        """
        graph = _build_retry_graph(critic_fn=_always_failing_critic)
        result = await graph.ainvoke({"question": "What is LoRA?"}, config=config)

        trace = _get(result, "reasoning_trace", [])
        critique_steps = [s for s in trace if s.step_type == "critique"]
        assert len(critique_steps) >= 3, (
            f"Expected >= 3 critique trace steps (3 critic invocations), "
            f"got {len(critique_steps)}"
        )

    # -- Custom max_retries --------------------------------------------------

    async def test_single_retry_with_max_retries_one(self) -> None:
        """With max_retries=1, only one retry should occur before forced pass."""
        graph = _build_retry_graph(critic_fn=_always_failing_critic)
        result = await graph.ainvoke(
            {"question": "What is LoRA?", "max_retries": 1},
            config={"configurable": {"thread_id": "test-max1"}},
        )

        retry_count = _get(result, "retry_count", 0)
        assert retry_count == 1, (
            f"Expected retry_count=1 with max_retries=1, got {retry_count}"
        )

    async def test_zero_retries_with_max_retries_zero(self) -> None:
        """With max_retries=0, the critic_gate should force pass immediately."""
        graph = _build_retry_graph(critic_fn=_always_failing_critic)
        result = await graph.ainvoke(
            {"question": "What is LoRA?", "max_retries": 0},
            config={"configurable": {"thread_id": "test-max0"}},
        )

        retry_count = _get(result, "retry_count", 0)
        assert retry_count == 0, (
            f"Expected retry_count=0 with max_retries=0, got {retry_count}"
        )

    # -- Intent preserved across retries -------------------------------------

    async def test_intent_preserved_through_retries(
        self,
        config: dict[str, Any],
    ) -> None:
        """The intent should remain valid after multiple retry loops."""
        graph = _build_retry_graph(critic_fn=_always_failing_critic)
        result = await graph.ainvoke({"question": "What is LoRA?"}, config=config)

        intent = _get(result, "intent")
        assert intent is not None, "Intent should be preserved through retries"
        assert intent.type == "factual", (
            "Placeholder router should maintain factual intent"
        )

    # -- Draft answer present ------------------------------------------------

    async def test_draft_answer_present_after_retry(
        self,
        config: dict[str, Any],
    ) -> None:
        """draft_answer should be set even after retry exhaustion."""
        graph = _build_retry_graph(critic_fn=_always_failing_critic)
        result = await graph.ainvoke({"question": "What is LoRA?"}, config=config)

        draft = _get(result, "draft_answer", "")
        assert draft, "draft_answer should be present after retries"
