"""Unit tests for the Slot Filling node (C2).

Verifies that the Slot Filling node correctly:
- Extracts entities, dimensions, constraints, and reformulated_query via
  structured output from a mocked Cloud LLM.
- Populates intent with all slots (entities/dimensions/reformulated_query
  are non-empty).
- Preserves intent type and confidence from the Router.
- Handles comparative intent (dimensions non-empty).
- Handles non-comparative intents (dimensions can be empty).
- Records a reasoning trace step.
- Works through the ``create_slot_filling_node`` factory.
- Falls back to the synchronous ``slot_filling_node`` placeholder.
- Handles missing intent gracefully.

All tests use a mocked LLM — no real API calls.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agent.nodes.slot_filling import (
    SlotFillingOutput,
    create_slot_filling_node,
    run_slot_filling,
    slot_filling_node,
)
from src.agent.state import AgentState, Intent, ReasoningStep
from src.prompts.slot_filling import (
    SLOT_FILLING_SYSTEM_PROMPT,
    SLOT_FILLING_USER_TEMPLATE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(
    *,
    question: str = "What is LoRA?",
    intent_type: str = "factual",
    confidence: float = 0.92,
    include_intent: bool = True,
) -> AgentState:
    """Create a minimal AgentState with a partial intent from the Router."""
    intent = None
    if include_intent:
        intent = Intent(
            type=intent_type,
            confidence=confidence,
            reformulated_query=question,
        )
    return AgentState(question=question, intent=intent)


def _make_mock_llm(
    *,
    entities: list[str] | None = None,
    dimensions: list[str] | None = None,
    constraints: dict[str, str] | None = None,
    reformulated_query: str = "What is Low-Rank Adaptation (LoRA)?",
) -> MagicMock:
    """Create a mock LLM that returns a ``SlotFillingOutput`` via structured output.

    Simulates ``llm.with_structured_output(SlotFillingOutput).ainvoke(...)``
    returning a ``SlotFillingOutput`` Pydantic instance.
    """
    sf_output = SlotFillingOutput(
        entities=entities or ["LoRA"],
        dimensions=dimensions or [],
        constraints=constraints or {},
        reformulated_query=reformulated_query,
    )

    structured_llm = MagicMock()
    structured_llm.ainvoke = AsyncMock(return_value=sf_output)

    llm = MagicMock()
    llm.with_structured_output = MagicMock(return_value=structured_llm)

    return llm


# ---------------------------------------------------------------------------
# SlotFillingOutput model
# ---------------------------------------------------------------------------


class TestSlotFillingOutput:
    """Tests for the SlotFillingOutput Pydantic model."""

    def test_valid_output(self) -> None:
        """Should accept valid entities and reformulated query."""
        output = SlotFillingOutput(
            entities=["LoRA"],
            reformulated_query="What is LoRA?",
        )
        assert output.entities == ["LoRA"]
        assert output.reformulated_query == "What is LoRA?"
        assert output.dimensions == []
        assert output.constraints == {}

    def test_full_output(self) -> None:
        """Should accept all fields populated."""
        output = SlotFillingOutput(
            entities=["LoRA", "QLoRA"],
            dimensions=["memory efficiency", "training speed"],
            constraints={"model_scale": "7B", "time_range": "2023"},
            reformulated_query="Compare LoRA and QLoRA memory efficiency",
        )
        assert len(output.entities) == 2
        assert len(output.dimensions) == 2
        assert output.constraints["model_scale"] == "7B"

    def test_empty_entities_allowed_by_model(self) -> None:
        """Pydantic model itself allows empty entities (prompt enforces >=1)."""
        output = SlotFillingOutput(
            entities=[],
            reformulated_query="test",
        )
        assert output.entities == []


# ---------------------------------------------------------------------------
# run_slot_filling — core logic
# ---------------------------------------------------------------------------


class TestRunSlotFilling:
    """Tests for the core slot filling implementation."""

    async def test_factual_question_fills_slots(self) -> None:
        """A factual question should have entities and reformulated_query filled."""
        state = _make_state(
            question="What is LoRA?",
            intent_type="factual",
            confidence=0.95,
        )
        mock_llm = _make_mock_llm(
            entities=["LoRA"],
            reformulated_query="What is Low-Rank Adaptation (LoRA)?",
        )

        result = await run_slot_filling(state, mock_llm)

        intent = result["intent"]
        assert isinstance(intent, Intent)
        assert intent.type == "factual"
        assert intent.confidence == 0.95
        assert intent.entities == ["LoRA"]
        assert intent.reformulated_query == "What is Low-Rank Adaptation (LoRA)?"
        assert intent.reformulated_query != ""

    async def test_comparative_question_fills_dimensions(self) -> None:
        """A comparative question should have non-empty dimensions."""
        state = _make_state(
            question="Compare LoRA and QLoRA in terms of memory efficiency",
            intent_type="comparative",
            confidence=0.88,
        )
        mock_llm = _make_mock_llm(
            entities=["LoRA", "QLoRA"],
            dimensions=["memory efficiency"],
            reformulated_query=(
                "Compare Low-Rank Adaptation (LoRA) and Quantized LoRA (QLoRA) "
                "in terms of memory efficiency"
            ),
        )

        result = await run_slot_filling(state, mock_llm)

        intent = result["intent"]
        assert intent.type == "comparative"
        assert len(intent.entities) >= 2
        assert len(intent.dimensions) >= 1
        assert "memory efficiency" in intent.dimensions
        assert intent.reformulated_query != ""

    async def test_multi_hop_question_fills_entities(self) -> None:
        """A multi-hop question should have multiple entities extracted."""
        state = _make_state(
            question=(
                "How does the attention mechanism in Transformers relate to "
                "the improvements proposed in FlashAttention?"
            ),
            intent_type="multi_hop",
            confidence=0.82,
        )
        mock_llm = _make_mock_llm(
            entities=["Transformers", "FlashAttention", "attention mechanism"],
            reformulated_query=(
                "How does the attention mechanism in Transformers relate to "
                "FlashAttention improvements?"
            ),
        )

        result = await run_slot_filling(state, mock_llm)

        intent = result["intent"]
        assert intent.type == "multi_hop"
        assert len(intent.entities) >= 2
        assert intent.reformulated_query != ""

    async def test_exploratory_question_fills_slots(self) -> None:
        """An exploratory question should have entities and reformulated_query."""
        state = _make_state(
            question="What are the latest trends in efficient fine-tuning?",
            intent_type="exploratory",
            confidence=0.79,
        )
        mock_llm = _make_mock_llm(
            entities=["efficient fine-tuning"],
            reformulated_query=(
                "What are the latest trends and methods in "
                "parameter-efficient fine-tuning of large language models?"
            ),
        )

        result = await run_slot_filling(state, mock_llm)

        intent = result["intent"]
        assert intent.type == "exploratory"
        assert len(intent.entities) >= 1
        assert intent.reformulated_query != ""

    async def test_follow_up_question_fills_slots(self) -> None:
        """A follow-up question should still have entities and reformulated_query."""
        state = _make_state(
            question="Can you elaborate on that?",
            intent_type="follow_up",
            confidence=0.91,
        )
        mock_llm = _make_mock_llm(
            entities=["previous topic"],
            reformulated_query="Please provide more details on the previous topic.",
        )

        result = await run_slot_filling(state, mock_llm)

        intent = result["intent"]
        assert intent.type == "follow_up"
        assert len(intent.entities) >= 1
        assert intent.reformulated_query != ""

    async def test_constraints_extracted(self) -> None:
        """Constraints should be extracted when present in the question."""
        state = _make_state(
            question="What LoRA papers were published in 2023 for models under 7B?",
            intent_type="factual",
            confidence=0.85,
        )
        mock_llm = _make_mock_llm(
            entities=["LoRA"],
            constraints={"time_range": "2023", "model_scale": "under 7B"},
            reformulated_query=(
                "LoRA papers published in 2023 for models under 7 billion parameters"
            ),
        )

        result = await run_slot_filling(state, mock_llm)

        intent = result["intent"]
        assert len(intent.constraints) >= 1
        assert "time_range" in intent.constraints
        assert intent.constraints["time_range"] == "2023"

    async def test_preserves_intent_type_and_confidence(self) -> None:
        """Slot filling should preserve the Router's type and confidence."""
        state = _make_state(
            question="Compare LoRA and Adapter",
            intent_type="comparative",
            confidence=0.87,
        )
        mock_llm = _make_mock_llm(
            entities=["LoRA", "Adapter"],
            dimensions=["performance", "parameter count"],
            reformulated_query="Compare LoRA and Adapter methods",
        )

        result = await run_slot_filling(state, mock_llm)

        intent = result["intent"]
        assert intent.type == "comparative"
        assert intent.confidence == 0.87

    async def test_reasoning_trace_recorded(self) -> None:
        """Should include a reasoning trace step with slot filling details."""
        state = _make_state(question="What is LoRA?")
        mock_llm = _make_mock_llm(
            entities=["LoRA"],
            reformulated_query="What is Low-Rank Adaptation (LoRA)?",
        )

        result = await run_slot_filling(state, mock_llm)

        trace = result["reasoning_trace"]
        assert len(trace) == 1
        step = trace[0]
        assert isinstance(step, ReasoningStep)
        assert step.step_type == "route"
        assert "1 entities" in step.content
        assert "SlotFilling" in step.content

    async def test_reasoning_trace_metadata(self) -> None:
        """Reasoning trace metadata should contain extracted slot details."""
        state = _make_state(
            question="Compare LoRA and QLoRA",
            intent_type="comparative",
        )
        mock_llm = _make_mock_llm(
            entities=["LoRA", "QLoRA"],
            dimensions=["memory efficiency"],
            constraints={"domain": "NLP"},
            reformulated_query="Compare LoRA and QLoRA",
        )

        result = await run_slot_filling(state, mock_llm)

        metadata = result["reasoning_trace"][0].metadata
        assert metadata["entities"] == ["LoRA", "QLoRA"]
        assert metadata["dimensions"] == ["memory efficiency"]
        assert metadata["constraints"] == {"domain": "NLP"}
        assert metadata["reformulated_query"] == "Compare LoRA and QLoRA"

    async def test_llm_receives_correct_prompts(self) -> None:
        """LLM should receive system prompt and formatted user message."""
        state = _make_state(
            question="What is LoRA?",
            intent_type="factual",
            confidence=0.95,
        )
        mock_llm = _make_mock_llm()

        await run_slot_filling(state, mock_llm)

        # Verify with_structured_output was called with SlotFillingOutput
        mock_llm.with_structured_output.assert_called_once_with(SlotFillingOutput)

        # Verify ainvoke was called with correct messages
        structured_llm = mock_llm.with_structured_output.return_value
        call_args = structured_llm.ainvoke.call_args[0][0]

        assert len(call_args) == 2  # system + user
        system_msg = call_args[0]
        user_msg = call_args[1]

        assert system_msg.content == SLOT_FILLING_SYSTEM_PROMPT
        assert "What is LoRA?" in user_msg.content
        assert "factual" in user_msg.content

    async def test_missing_intent_defaults_to_factual(self) -> None:
        """Missing intent should default to factual with zero confidence."""
        state = _make_state(
            question="What is LoRA?",
            include_intent=False,
        )
        mock_llm = _make_mock_llm(
            entities=["LoRA"],
            reformulated_query="What is Low-Rank Adaptation (LoRA)?",
        )

        result = await run_slot_filling(state, mock_llm)

        intent = result["intent"]
        assert intent.type == "factual"
        assert intent.confidence == 0.0
        assert intent.entities == ["LoRA"]

    async def test_dict_state_access(self) -> None:
        """Should work when state is a plain dict (LangGraph compatibility)."""
        state = {
            "question": "What is LoRA?",
            "intent": Intent(
                type="factual",
                confidence=0.93,
                reformulated_query="What is LoRA?",
            ),
        }
        mock_llm = _make_mock_llm(
            entities=["LoRA"],
            reformulated_query="What is Low-Rank Adaptation (LoRA)?",
        )

        result = await run_slot_filling(state, mock_llm)

        intent = result["intent"]
        assert intent.type == "factual"
        assert intent.confidence == 0.93
        assert intent.entities == ["LoRA"]


# ---------------------------------------------------------------------------
# create_slot_filling_node (factory)
# ---------------------------------------------------------------------------


class TestCreateSlotFillingNode:
    """Tests for the node factory function."""

    async def test_factory_creates_callable_node(self) -> None:
        """Factory should return a callable that processes AgentState."""
        mock_llm = _make_mock_llm(
            entities=["LoRA"],
            reformulated_query="What is Low-Rank Adaptation (LoRA)?",
        )

        node = create_slot_filling_node(mock_llm)
        state = _make_state(question="What is LoRA?")

        result = await node(state)

        assert "intent" in result
        assert result["intent"].entities == ["LoRA"]

    async def test_factory_node_has_correct_name(self) -> None:
        """Factory-produced node should have the expected __name__."""
        mock_llm = _make_mock_llm()

        node = create_slot_filling_node(mock_llm)

        assert node.__name__ == "slot_filling_node"

    async def test_factory_delegates_to_run_slot_filling(self) -> None:
        """Factory node should produce same result as run_slot_filling."""
        mock_llm = _make_mock_llm(
            entities=["Transformers", "FlashAttention"],
            dimensions=[],
            reformulated_query="Transformers attention vs FlashAttention",
        )

        node = create_slot_filling_node(mock_llm)
        state = _make_state(
            question="How does attention relate to FlashAttention?",
            intent_type="multi_hop",
            confidence=0.85,
        )

        result = await node(state)

        assert result["intent"].type == "multi_hop"
        assert result["intent"].confidence == 0.85
        assert result["intent"].entities == ["Transformers", "FlashAttention"]
        assert len(result["reasoning_trace"]) == 1


# ---------------------------------------------------------------------------
# slot_filling_node (default sync placeholder)
# ---------------------------------------------------------------------------


class TestSlotFillingNodePlaceholder:
    """Tests for the default synchronous placeholder node."""

    def test_placeholder_returns_empty_when_intent_present(self) -> None:
        """Placeholder should return empty dict when intent has reformulated_query."""
        state = _make_state(question="What is LoRA?")
        result = slot_filling_node(state)

        # Intent already has reformulated_query set by Router, so no-op
        assert result == {}

    def test_placeholder_backfills_reformulated_query(self) -> None:
        """Placeholder should backfill reformulated_query when it's empty."""
        state = AgentState(
            question="What is LoRA?",
            intent=Intent(
                type="factual",
                confidence=0.95,
                reformulated_query="",
            ),
        )
        result = slot_filling_node(state)

        assert "intent" in result
        assert result["intent"].reformulated_query == "What is LoRA?"

    def test_placeholder_handles_dict_state(self) -> None:
        """Placeholder should work with dict-like state."""
        state = {
            "question": "What is LoRA?",
            "intent": Intent(
                type="factual",
                confidence=0.90,
                reformulated_query="What is LoRA?",
            ),
        }
        result = slot_filling_node(state)

        # Intent already has reformulated_query, so no-op
        assert result == {}

    def test_placeholder_handles_missing_intent(self) -> None:
        """Placeholder should handle missing intent gracefully."""
        state = {"question": "What is LoRA?"}
        result = slot_filling_node(state)

        assert result == {}

    def test_placeholder_handles_no_question(self) -> None:
        """Placeholder should handle empty state gracefully."""
        state = {}
        result = slot_filling_node(state)

        assert result == {}


# ---------------------------------------------------------------------------
# Integration: entities/dimensions/reformulated_query all non-empty
# ---------------------------------------------------------------------------


class TestSlotFillingCompleteness:
    """Verify that all required slots are filled after slot filling."""

    @pytest.mark.parametrize(
        "intent_type,question,entities,dimensions,reformulated_query",
        [
            (
                "factual",
                "What is LoRA?",
                ["LoRA"],
                [],
                "What is Low-Rank Adaptation (LoRA)?",
            ),
            (
                "comparative",
                "Compare LoRA and QLoRA",
                ["LoRA", "QLoRA"],
                ["performance", "memory efficiency"],
                "Compare LoRA and QLoRA in performance and memory efficiency",
            ),
            (
                "multi_hop",
                "How does attention in Transformers relate to FlashAttention?",
                ["Transformers", "FlashAttention", "attention"],
                [],
                "Relationship between Transformer attention and FlashAttention",
            ),
            (
                "exploratory",
                "What are the latest trends in efficient fine-tuning?",
                ["efficient fine-tuning"],
                [],
                "Latest trends in parameter-efficient fine-tuning methods",
            ),
            (
                "follow_up",
                "Can you elaborate on that?",
                ["previous topic"],
                [],
                "Provide more details on the previously discussed topic",
            ),
        ],
    )
    async def test_all_required_slots_non_empty(
        self,
        intent_type: str,
        question: str,
        entities: list[str],
        dimensions: list[str],
        reformulated_query: str,
    ) -> None:
        """After slot filling, entities and reformulated_query must be non-empty."""
        state = _make_state(
            question=question,
            intent_type=intent_type,
            confidence=0.90,
        )
        mock_llm = _make_mock_llm(
            entities=entities,
            dimensions=dimensions,
            reformulated_query=reformulated_query,
        )

        result = await run_slot_filling(state, mock_llm)

        intent = result["intent"]
        assert len(intent.entities) > 0, "entities must be non-empty"
        assert intent.reformulated_query != "", "reformulated_query must be non-empty"

        # For comparative, dimensions must also be non-empty
        if intent_type == "comparative":
            assert len(intent.dimensions) > 0, (
                "dimensions must be non-empty for comparative"
            )
