"""Unit tests for the Router node (C1).

Verifies that the Router correctly:
- Classifies 5 different question types to the correct intent type.
- Writes ``intent.type`` and ``intent.confidence`` into partial state.
- Maps each intent type to the correct strategy via ``to_strategy()``.
- Records a reasoning trace step of type ``route``.
- Handles empty questions gracefully (defaults to ``factual``).
- Works through the ``create_router_node`` factory.
- Falls back to the synchronous ``router_node`` placeholder.

All tests use a mocked LLM — no real API calls.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agent.nodes.router import (
    RouterOutput,
    create_router_node,
    router_node,
    run_router,
)
from src.agent.graph import build_main_graph
from src.agent.state import AgentState, Intent, ReasoningStep
from src.prompts.router import ROUTER_SYSTEM_PROMPT, ROUTER_USER_TEMPLATE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(
    *,
    question: str = "What is LoRA?",
) -> AgentState:
    """Create a minimal AgentState for testing."""
    return AgentState(question=question)


def _make_mock_llm(
    intent_type: str = "factual",
    confidence: float = 0.92,
) -> MagicMock:
    """Create a mock LLM that returns a ``RouterOutput`` via structured output.

    Simulates ``llm.with_structured_output(RouterOutput).ainvoke(...)``
    returning a ``RouterOutput`` Pydantic instance.
    """
    router_output = RouterOutput(type=intent_type, confidence=confidence)

    structured_llm = MagicMock()
    structured_llm.ainvoke = AsyncMock(return_value=router_output)

    llm = MagicMock()
    llm.with_structured_output = MagicMock(return_value=structured_llm)

    return llm


class _MockLocalRouter:
    """Simple local router stub for fallback tests."""

    def __init__(
        self,
        *,
        intent_type: str = "factual",
        confidence: float = 0.8,
        should_raise: bool = False,
    ) -> None:
        self.intent_type = intent_type
        self.confidence = confidence
        self.should_raise = should_raise
        self.calls: list[str] = []

    def classify_question(self, question: str) -> tuple[str, float]:
        """Return a deterministic local classification or raise."""
        self.calls.append(question)
        if self.should_raise:
            raise RuntimeError("local router unavailable")
        return self.intent_type, self.confidence


# ---------------------------------------------------------------------------
# RouterOutput model
# ---------------------------------------------------------------------------


class TestRouterOutput:
    """Tests for the RouterOutput Pydantic model."""

    def test_valid_output(self) -> None:
        """Should accept valid intent type and confidence."""
        output = RouterOutput(type="factual", confidence=0.95)
        assert output.type == "factual"
        assert output.confidence == 0.95

    def test_all_intent_types_valid(self) -> None:
        """All five intent types should be valid."""
        for intent_type in ("factual", "comparative", "multi_hop", "exploratory", "follow_up"):
            output = RouterOutput(type=intent_type, confidence=0.8)
            assert output.type == intent_type

    def test_confidence_bounds(self) -> None:
        """Confidence must be in [0, 1]."""
        RouterOutput(type="factual", confidence=0.0)
        RouterOutput(type="factual", confidence=1.0)

        with pytest.raises(Exception):
            RouterOutput(type="factual", confidence=-0.1)
        with pytest.raises(Exception):
            RouterOutput(type="factual", confidence=1.1)

    def test_invalid_intent_type_rejected(self) -> None:
        """Unknown intent type should be rejected."""
        with pytest.raises(Exception):
            RouterOutput(type="unknown", confidence=0.5)


# ---------------------------------------------------------------------------
# run_router — core logic with 5 question types
# ---------------------------------------------------------------------------


class TestRunRouter:
    """Tests for the core router implementation."""

    async def test_factual_question(self) -> None:
        """A direct factual question should be classified as 'factual'."""
        state = _make_state(question="What is LoRA?")
        mock_llm = _make_mock_llm("factual", 0.95)

        result = await run_router(state, mock_llm)

        intent = result["intent"]
        assert isinstance(intent, Intent)
        assert intent.type == "factual"
        assert intent.confidence == 0.95
        assert intent.to_strategy() == "simple"

    async def test_comparative_question(self) -> None:
        """A comparison question should be classified as 'comparative'."""
        state = _make_state(
            question="Compare LoRA and QLoRA in terms of memory efficiency"
        )
        mock_llm = _make_mock_llm("comparative", 0.88)

        result = await run_router(state, mock_llm)

        intent = result["intent"]
        assert intent.type == "comparative"
        assert intent.confidence == 0.88
        assert intent.to_strategy() == "comparative"

    async def test_multi_hop_question(self) -> None:
        """A multi-hop reasoning question should be classified as 'multi_hop'."""
        state = _make_state(
            question=(
                "How does the attention mechanism in Transformers relate to "
                "the improvements proposed in FlashAttention?"
            )
        )
        mock_llm = _make_mock_llm("multi_hop", 0.82)

        result = await run_router(state, mock_llm)

        intent = result["intent"]
        assert intent.type == "multi_hop"
        assert intent.confidence == 0.82
        assert intent.to_strategy() == "multi_hop"

    async def test_exploratory_question(self) -> None:
        """A broad survey-style question should be classified as 'exploratory'."""
        state = _make_state(
            question="What are the latest trends in efficient fine-tuning?"
        )
        mock_llm = _make_mock_llm("exploratory", 0.79)

        result = await run_router(state, mock_llm)

        intent = result["intent"]
        assert intent.type == "exploratory"
        assert intent.confidence == 0.79
        assert intent.to_strategy() == "exploratory"

    async def test_follow_up_question(self) -> None:
        """A follow-up question should be classified as 'follow_up'."""
        state = _make_state(question="Can you elaborate on that?")
        mock_llm = _make_mock_llm("follow_up", 0.91)

        result = await run_router(state, mock_llm)

        intent = result["intent"]
        assert intent.type == "follow_up"
        assert intent.confidence == 0.91
        # follow_up maps to "simple" strategy
        assert intent.to_strategy() == "simple"

    async def test_intent_has_reformulated_query(self) -> None:
        """Partial intent should preserve the original question as reformulated_query."""
        question = "What is the role of attention in Transformers?"
        state = _make_state(question=question)
        mock_llm = _make_mock_llm("factual", 0.90)

        result = await run_router(state, mock_llm)

        intent = result["intent"]
        assert intent.reformulated_query == question

    async def test_reasoning_trace_recorded(self) -> None:
        """Should include a reasoning trace step of type 'route'."""
        state = _make_state(question="What is LoRA?")
        mock_llm = _make_mock_llm("factual", 0.95)

        result = await run_router(state, mock_llm)

        trace = result["reasoning_trace"]
        assert len(trace) == 1
        step = trace[0]
        assert isinstance(step, ReasoningStep)
        assert step.step_type == "route"
        assert "factual" in step.content
        assert "0.95" in step.content

    async def test_reasoning_trace_metadata(self) -> None:
        """Reasoning trace should contain intent_type and confidence in metadata."""
        state = _make_state(question="Compare LoRA and Adapter")
        mock_llm = _make_mock_llm("comparative", 0.87)

        result = await run_router(state, mock_llm)

        metadata = result["reasoning_trace"][0].metadata
        assert metadata["intent_type"] == "comparative"
        assert metadata["confidence"] == 0.87
        assert "question_preview" in metadata
        assert "llm_call_duration_ms" in metadata
        assert isinstance(metadata["llm_call_duration_ms"], float)

    async def test_llm_receives_correct_prompts(self) -> None:
        """LLM should receive system prompt and formatted user message."""
        state = _make_state(question="What is LoRA?")
        mock_llm = _make_mock_llm("factual", 0.95)

        await run_router(state, mock_llm)

        # Verify with_structured_output was called with RouterOutput
        mock_llm.with_structured_output.assert_called_once_with(RouterOutput)

        # Verify ainvoke was called with correct messages
        structured_llm = mock_llm.with_structured_output.return_value
        call_args = structured_llm.ainvoke.call_args[0][0]

        assert len(call_args) == 2  # system + user
        system_msg = call_args[0]
        user_msg = call_args[1]

        assert system_msg.content == ROUTER_SYSTEM_PROMPT
        assert "What is LoRA?" in user_msg.content

    async def test_empty_question_defaults_to_factual(self) -> None:
        """Empty question should default to factual with zero confidence."""
        state = _make_state(question="")
        mock_llm = _make_mock_llm()

        result = await run_router(state, mock_llm)

        intent = result["intent"]
        assert intent.type == "factual"
        assert intent.confidence == 0.0

        # LLM should NOT be called for empty question
        mock_llm.with_structured_output.assert_not_called()

    async def test_dict_state_access(self) -> None:
        """Should work when state is a plain dict (LangGraph compatibility)."""
        state = {"question": "What is LoRA?"}
        mock_llm = _make_mock_llm("factual", 0.93)

        result = await run_router(state, mock_llm)

        intent = result["intent"]
        assert intent.type == "factual"
        assert intent.confidence == 0.93

    async def test_fallback_local_router_preferred_when_available(self) -> None:
        """Local router result should be used when local inference succeeds."""
        state = _make_state(question="What are recent PEFT trends?")
        mock_llm = _make_mock_llm("factual", 0.11)
        local_router = _MockLocalRouter(intent_type="exploratory", confidence=0.84)

        result = await run_router(state, mock_llm, local_router=local_router)

        intent = result["intent"]
        assert intent.type == "exploratory"
        assert intent.confidence == 0.84
        assert local_router.calls == ["What are recent PEFT trends?"]
        mock_llm.with_structured_output.assert_not_called()
        assert result["reasoning_trace"][0].metadata["source"] == "local"
        assert "llm_call_duration_ms" in result["reasoning_trace"][0].metadata

    async def test_fallback_to_cloud_when_local_router_fails(self) -> None:
        """Router should fall back to cloud classification on local errors."""
        state = _make_state(question="Compare LoRA and adapters")
        mock_llm = _make_mock_llm("comparative", 0.91)
        local_router = _MockLocalRouter(should_raise=True)

        result = await run_router(state, mock_llm, local_router=local_router)

        intent = result["intent"]
        assert intent.type == "comparative"
        assert intent.confidence == 0.91
        assert local_router.calls == ["Compare LoRA and adapters"]
        mock_llm.with_structured_output.assert_called_once_with(RouterOutput)
        assert result["reasoning_trace"][0].metadata["source"] == "cloud_fallback"

    async def test_fallback_default_when_no_local_and_no_cloud(self) -> None:
        """Router should return deterministic default when no model is available."""
        state = _make_state(question="Any question")

        result = await run_router(state, llm=None, local_router=None)

        intent = result["intent"]
        assert intent.type == "factual"
        assert intent.confidence == 0.5
        assert result["reasoning_trace"][0].metadata["source"] == "default_no_model"


# ---------------------------------------------------------------------------
# create_router_node (factory)
# ---------------------------------------------------------------------------


class TestCreateRouterNode:
    """Tests for the node factory function."""

    async def test_factory_creates_callable_node(self) -> None:
        """Factory should return a callable that processes AgentState."""
        mock_llm = _make_mock_llm("factual", 0.90)

        node = create_router_node(mock_llm)
        state = _make_state(question="What is LoRA?")

        result = await node(state)

        assert "intent" in result
        assert result["intent"].type == "factual"

    async def test_factory_node_has_correct_name(self) -> None:
        """Factory-produced node should have the expected __name__."""
        mock_llm = _make_mock_llm()

        node = create_router_node(mock_llm)

        assert node.__name__ == "router_node"

    async def test_factory_delegates_to_run_router(self) -> None:
        """Factory node should produce same result as run_router."""
        mock_llm = _make_mock_llm("exploratory", 0.85)

        node = create_router_node(mock_llm)
        state = _make_state(question="What are the latest trends?")

        result = await node(state)

        assert result["intent"].type == "exploratory"
        assert result["intent"].confidence == 0.85
        assert len(result["reasoning_trace"]) == 1


# ---------------------------------------------------------------------------
# router_node (default sync placeholder)
# ---------------------------------------------------------------------------


class TestRouterNodePlaceholder:
    """Tests for the default synchronous placeholder node."""

    def test_placeholder_returns_factual_intent(self) -> None:
        """Placeholder should return factual intent."""
        state = _make_state(question="Some question")
        result = router_node(state)

        assert "intent" in result
        intent = result["intent"]
        assert isinstance(intent, Intent)
        assert intent.type == "factual"
        assert intent.confidence == 1.0

    def test_placeholder_preserves_question(self) -> None:
        """Placeholder should set reformulated_query to the question."""
        state = _make_state(question="What is attention?")
        result = router_node(state)

        assert result["intent"].reformulated_query == "What is attention?"

    def test_placeholder_handles_dict_state(self) -> None:
        """Placeholder should work with dict-like state."""
        state = {"question": "What is LoRA?"}
        result = router_node(state)

        assert result["intent"].type == "factual"
        assert result["intent"].reformulated_query == "What is LoRA?"

    def test_placeholder_handles_missing_question(self) -> None:
        """Placeholder should handle missing question gracefully."""
        state = {}
        result = router_node(state)

        assert result["intent"].type == "factual"
        assert result["intent"].reformulated_query == "placeholder query"


# ---------------------------------------------------------------------------
# Integration: intent.to_strategy() mapping for all 5 types
# ---------------------------------------------------------------------------


class TestIntentStrategyMapping:
    """Verify that all 5 intent types route to the correct strategy."""

    @pytest.mark.parametrize(
        "intent_type,expected_strategy",
        [
            ("factual", "simple"),
            ("comparative", "comparative"),
            ("multi_hop", "multi_hop"),
            ("exploratory", "exploratory"),
            ("follow_up", "simple"),
        ],
    )
    async def test_intent_to_strategy(
        self,
        intent_type: str,
        expected_strategy: str,
    ) -> None:
        """Each intent type should map to its expected strategy."""
        state = _make_state(question="Test question")
        mock_llm = _make_mock_llm(intent_type, 0.90)

        result = await run_router(state, mock_llm)

        assert result["intent"].to_strategy() == expected_strategy


def test_local_router_wired_through_graph() -> None:
    """Graph should use local_router when provided to build_main_graph()."""

    class _LocalRouter:
        def classify_question(self, question: str) -> tuple[str, float]:
            return "comparative", 0.91

    class _LocalCritic:
        def evaluate_answer(self, **_: object) -> dict[str, object]:
            return {
                "score": 8.0,
                "completeness": 0.8,
                "faithfulness": 0.9,
                "feedback": "ok",
            }

    graph = build_main_graph(local_router=_LocalRouter(), local_critic=_LocalCritic())
    result = graph.invoke(
        {"question": "Compare LoRA and QLoRA"},
        config={"configurable": {"thread_id": "router-local-wire-1"}},
    )
    trace = result.get("reasoning_trace", [])
    route_steps = [step for step in trace if getattr(step, "step_type", None) == "route"]
    assert route_steps, "Expected route reasoning step in graph trace"
    metadata = route_steps[0].metadata
    assert metadata.get("source") == "local"
    assert "llm_call_duration_ms" in metadata
