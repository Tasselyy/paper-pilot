"""Unit tests for src/agent/state.py — AgentState, Intent, sub-types."""

from __future__ import annotations

import pytest

from src.agent.state import (
    AgentState,
    CriticVerdict,
    Intent,
    IntentType,
    ReasoningStep,
    RetrievedContext,
    StrategyName,
)


# ---------------------------------------------------------------------------
# Intent tests
# ---------------------------------------------------------------------------


class TestIntent:
    """Tests for Intent model and to_strategy() mapping."""

    def test_intent_factual_to_simple(self) -> None:
        intent = Intent(type="factual", confidence=0.95, reformulated_query="What is LoRA?")
        assert intent.to_strategy() == "simple"

    def test_intent_follow_up_to_simple(self) -> None:
        intent = Intent(type="follow_up", confidence=0.8, reformulated_query="Tell me more")
        assert intent.to_strategy() == "simple"

    def test_intent_comparative_to_comparative(self) -> None:
        intent = Intent(
            type="comparative",
            confidence=0.9,
            entities=["BERT", "GPT"],
            dimensions=["architecture", "performance"],
            reformulated_query="Compare BERT and GPT",
        )
        assert intent.to_strategy() == "comparative"

    def test_intent_multi_hop_to_multi_hop(self) -> None:
        intent = Intent(
            type="multi_hop",
            confidence=0.85,
            reformulated_query="How does LoRA improve fine-tuning efficiency?",
        )
        assert intent.to_strategy() == "multi_hop"

    def test_intent_exploratory_to_exploratory(self) -> None:
        intent = Intent(
            type="exploratory",
            confidence=0.7,
            reformulated_query="What are recent advances in efficient LLM training?",
        )
        assert intent.to_strategy() == "exploratory"

    def test_all_five_intent_types_mapped(self) -> None:
        """Verify all 5 IntentType values produce a valid StrategyName."""
        intent_types: list[IntentType] = [
            "factual",
            "comparative",
            "multi_hop",
            "exploratory",
            "follow_up",
        ]
        expected_strategies: list[StrategyName] = [
            "simple",
            "comparative",
            "multi_hop",
            "exploratory",
            "simple",
        ]
        for itype, expected in zip(intent_types, expected_strategies):
            intent = Intent(type=itype, confidence=0.9, reformulated_query="q")
            assert intent.to_strategy() == expected, f"{itype} → {intent.to_strategy()}"

    def test_intent_defaults(self) -> None:
        intent = Intent(type="factual", confidence=0.5, reformulated_query="q")
        assert intent.entities == []
        assert intent.dimensions == []
        assert intent.constraints == {}

    def test_intent_confidence_bounds(self) -> None:
        with pytest.raises(ValueError):
            Intent(type="factual", confidence=-0.1, reformulated_query="q")
        with pytest.raises(ValueError):
            Intent(type="factual", confidence=1.1, reformulated_query="q")

    def test_intent_invalid_type_raises(self) -> None:
        with pytest.raises(ValueError):
            Intent(type="invalid_type", confidence=0.5, reformulated_query="q")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# RetrievedContext tests
# ---------------------------------------------------------------------------


class TestRetrievedContext:
    """Tests for RetrievedContext model."""

    def test_retrieved_context_creation(self) -> None:
        ctx = RetrievedContext(
            content="LoRA is a parameter-efficient method.",
            source="LoRA paper",
            doc_id="doc_001",
            relevance_score=0.92,
            chunk_index=3,
        )
        assert ctx.content == "LoRA is a parameter-efficient method."
        assert ctx.source == "LoRA paper"
        assert ctx.doc_id == "doc_001"
        assert ctx.relevance_score == 0.92
        assert ctx.chunk_index == 3

    def test_retrieved_context_optional_chunk_index(self) -> None:
        ctx = RetrievedContext(
            content="text",
            source="src",
            doc_id="d1",
            relevance_score=0.5,
        )
        assert ctx.chunk_index is None


# ---------------------------------------------------------------------------
# CriticVerdict tests
# ---------------------------------------------------------------------------


class TestCriticVerdict:
    """Tests for CriticVerdict model."""

    def test_critic_verdict_pass(self) -> None:
        v = CriticVerdict(
            passed=True,
            score=8.5,
            completeness=0.9,
            faithfulness=0.95,
            feedback="Good answer.",
        )
        assert v.passed is True
        assert v.score == 8.5

    def test_critic_verdict_fail(self) -> None:
        v = CriticVerdict(
            passed=False,
            score=3.0,
            completeness=0.4,
            faithfulness=0.3,
            feedback="Missing key details about architecture.",
        )
        assert v.passed is False
        assert v.feedback != ""

    def test_critic_verdict_score_bounds(self) -> None:
        with pytest.raises(ValueError):
            CriticVerdict(
                passed=True, score=-1, completeness=0.5,
                faithfulness=0.5, feedback="x",
            )
        with pytest.raises(ValueError):
            CriticVerdict(
                passed=True, score=11, completeness=0.5,
                faithfulness=0.5, feedback="x",
            )


# ---------------------------------------------------------------------------
# ReasoningStep tests
# ---------------------------------------------------------------------------


class TestReasoningStep:
    """Tests for ReasoningStep model."""

    def test_reasoning_step_creation(self) -> None:
        step = ReasoningStep(
            step_type="thought",
            content="Analyzing the question structure.",
            timestamp=1000.0,
        )
        assert step.step_type == "thought"
        assert step.content == "Analyzing the question structure."
        assert step.metadata == {}

    def test_reasoning_step_with_metadata(self) -> None:
        step = ReasoningStep(
            step_type="action",
            content="RAG query: What is LoRA?",
            timestamp=1001.0,
            metadata={"query": "What is LoRA?", "tool": "query_knowledge_hub"},
        )
        assert step.metadata["tool"] == "query_knowledge_hub"

    def test_reasoning_step_invalid_type_raises(self) -> None:
        with pytest.raises(ValueError):
            ReasoningStep(step_type="invalid", content="x", timestamp=0.0)  # type: ignore[arg-type]

    def test_reasoning_step_valid_types(self) -> None:
        for stype in ("thought", "action", "observation", "critique", "route"):
            step = ReasoningStep(step_type=stype, content="c", timestamp=0.0)  # type: ignore[arg-type]
            assert step.step_type == stype


# ---------------------------------------------------------------------------
# AgentState tests
# ---------------------------------------------------------------------------


class TestAgentState:
    """Tests for the main AgentState model."""

    def test_agent_state_defaults(self) -> None:
        state = AgentState()
        assert state.question == ""
        assert state.intent is None
        assert state.sub_questions == []
        assert state.retrieved_contexts == []
        assert state.retrieval_queries == []
        assert state.draft_answer == ""
        assert state.final_answer == ""
        assert state.critic_verdict is None
        assert state.retry_count == 0
        assert state.max_retries == 2
        assert state.current_react_step == 0
        assert state.max_react_steps == 5
        assert state.reasoning_trace == []
        assert state.accumulated_facts == []
        assert state.messages == []

    def test_agent_state_with_intent(self) -> None:
        intent = Intent(type="factual", confidence=0.9, reformulated_query="q")
        state = AgentState(question="What is LoRA?", intent=intent)
        assert state.intent is not None
        assert state.intent.to_strategy() == "simple"

    def test_agent_state_with_contexts(self) -> None:
        ctx = RetrievedContext(
            content="text", source="src", doc_id="d1", relevance_score=0.8
        )
        state = AgentState(retrieved_contexts=[ctx])
        assert len(state.retrieved_contexts) == 1
        assert state.retrieved_contexts[0].source == "src"

    def test_agent_state_with_verdict(self) -> None:
        verdict = CriticVerdict(
            passed=True, score=8.0, completeness=0.9,
            faithfulness=0.95, feedback="Good.",
        )
        state = AgentState(critic_verdict=verdict)
        assert state.critic_verdict is not None
        assert state.critic_verdict.passed is True

    def test_agent_state_with_reasoning_trace(self) -> None:
        steps = [
            ReasoningStep(step_type="route", content="factual", timestamp=1.0),
            ReasoningStep(step_type="action", content="search", timestamp=2.0),
        ]
        state = AgentState(reasoning_trace=steps)
        assert len(state.reasoning_trace) == 2
