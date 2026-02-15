"""Unit tests for src/tracing/tracer.py — AgentTrace, NodeTraceEntry, AgentTracer.

Verifies that:
- ``NodeTraceEntry`` records node_name, duration, and output correctly.
- ``AgentTrace`` can be constructed with all fields from the design spec (§10.1).
- ``AgentTrace.from_state()`` populates trace fields from a completed ``AgentState``.
- Trace records contain ``node_name``, ``duration``, and ``output`` (acceptance criteria).
- ``AgentTracer`` tracks per-node timing via ``start_node``/``end_node``.
- ``AgentTracer.build_trace()`` produces a valid ``AgentTrace`` with correct durations.
- Latency breakdown (router, retrieval, llm, critic) is computed from node traces.
- Token usage is accumulated and included in the trace.
- Edge cases: end_node without start raises ValueError, empty state produces valid trace.

No real API calls or external dependencies.
"""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from src.agent.state import (
    AgentState,
    CriticVerdict,
    Intent,
    ReasoningStep,
    RetrievedContext,
)
from src.tracing.tracer import AgentTrace, AgentTracer, NodeTraceEntry


# ---------------------------------------------------------------------------
# NodeTraceEntry tests
# ---------------------------------------------------------------------------


class TestNodeTraceEntry:
    """Tests for the NodeTraceEntry model."""

    def test_creation_with_all_fields(self) -> None:
        entry = NodeTraceEntry(
            node_name="route",
            duration_ms=42.5,
            output={"intent_type": "factual", "confidence": 0.95},
            timestamp=1000.0,
        )
        assert entry.node_name == "route"
        assert entry.duration_ms == 42.5
        assert entry.output["intent_type"] == "factual"
        assert entry.timestamp == 1000.0

    def test_default_output_is_empty_dict(self) -> None:
        entry = NodeTraceEntry(
            node_name="critic",
            duration_ms=10.0,
            timestamp=2000.0,
        )
        assert entry.output == {}

    def test_default_timestamp_is_auto_generated(self) -> None:
        before = time.time()
        entry = NodeTraceEntry(node_name="slot_fill", duration_ms=5.0)
        after = time.time()
        assert before <= entry.timestamp <= after

    def test_duration_must_be_non_negative(self) -> None:
        with pytest.raises(ValueError):
            NodeTraceEntry(node_name="route", duration_ms=-1.0)

    def test_output_contains_arbitrary_keys(self) -> None:
        entry = NodeTraceEntry(
            node_name="simple",
            duration_ms=100.0,
            output={"draft_answer": "LoRA is...", "num_contexts": 3},
        )
        assert entry.output["draft_answer"] == "LoRA is..."
        assert entry.output["num_contexts"] == 3


# ---------------------------------------------------------------------------
# AgentTrace tests
# ---------------------------------------------------------------------------


class TestAgentTrace:
    """Tests for the AgentTrace model."""

    def test_defaults_produce_valid_trace(self) -> None:
        trace = AgentTrace()
        assert trace.trace_id  # non-empty UUID hex
        assert len(trace.trace_id) == 32  # UUID4 hex is 32 chars
        assert trace.question == ""
        assert trace.duration_ms == 0.0
        assert trace.intent is None
        assert trace.strategy_executed == ""
        assert trace.reasoning_steps == []
        assert trace.retrieval_queries == []
        assert trace.contexts_retrieved == 0
        assert trace.tokens_used == {}
        assert trace.critic_verdict is None
        assert trace.retry_count == 0
        assert trace.final_answer == ""
        assert trace.node_traces == []
        assert trace.router_latency_ms == 0.0
        assert trace.retrieval_latency_ms == 0.0
        assert trace.llm_latency_ms == 0.0
        assert trace.critic_latency_ms == 0.0

    def test_full_construction(self) -> None:
        intent = Intent(type="factual", confidence=0.95, reformulated_query="What is LoRA?")
        verdict = CriticVerdict(
            passed=True, score=8.5, completeness=0.9,
            faithfulness=0.95, feedback="Good answer.",
        )
        node_entry = NodeTraceEntry(
            node_name="route", duration_ms=30.0, timestamp=1000.0,
            output={"intent_type": "factual"},
        )
        step = ReasoningStep(
            step_type="route", content="Classified as factual", timestamp=1000.0,
        )
        trace = AgentTrace(
            trace_id="abc123",
            question="What is LoRA?",
            timestamp=1000.0,
            duration_ms=500.0,
            intent=intent,
            strategy_executed="simple",
            reasoning_steps=[step],
            retrieval_queries=["What is LoRA?"],
            contexts_retrieved=3,
            tokens_used={"input": 100, "output": 50},
            critic_verdict=verdict,
            retry_count=0,
            final_answer="LoRA is a parameter-efficient fine-tuning method.",
            node_traces=[node_entry],
            router_latency_ms=30.0,
            retrieval_latency_ms=200.0,
            llm_latency_ms=100.0,
            critic_latency_ms=50.0,
        )
        assert trace.trace_id == "abc123"
        assert trace.question == "What is LoRA?"
        assert trace.duration_ms == 500.0
        assert trace.intent is not None
        assert trace.intent.type == "factual"
        assert trace.strategy_executed == "simple"
        assert len(trace.reasoning_steps) == 1
        assert trace.reasoning_steps[0].step_type == "route"
        assert trace.retrieval_queries == ["What is LoRA?"]
        assert trace.contexts_retrieved == 3
        assert trace.tokens_used["input"] == 100
        assert trace.critic_verdict is not None
        assert trace.critic_verdict.passed is True
        assert trace.final_answer == "LoRA is a parameter-efficient fine-tuning method."
        assert len(trace.node_traces) == 1
        assert trace.node_traces[0].node_name == "route"
        assert trace.node_traces[0].duration_ms == 30.0
        assert trace.node_traces[0].output["intent_type"] == "factual"

    def test_trace_contains_node_name_duration_output(self) -> None:
        """Acceptance: trace records contain node_name, duration, and output."""
        entries = [
            NodeTraceEntry(
                node_name="route",
                duration_ms=25.0,
                output={"intent_type": "comparative"},
            ),
            NodeTraceEntry(
                node_name="slot_fill",
                duration_ms=50.0,
                output={"entities": ["BERT", "GPT"]},
            ),
            NodeTraceEntry(
                node_name="comparative",
                duration_ms=200.0,
                output={"num_contexts": 5},
            ),
            NodeTraceEntry(
                node_name="critic",
                duration_ms=40.0,
                output={"passed": True, "score": 8.0},
            ),
        ]
        trace = AgentTrace(node_traces=entries)
        for entry in trace.node_traces:
            assert entry.node_name, "node_name must be non-empty"
            assert entry.duration_ms >= 0, "duration must be non-negative"
            assert isinstance(entry.output, dict), "output must be a dict"


# ---------------------------------------------------------------------------
# AgentTrace.from_state tests
# ---------------------------------------------------------------------------


class TestAgentTraceFromState:
    """Tests for the AgentTrace.from_state() factory method."""

    def _make_state(self, **overrides: object) -> AgentState:
        """Create a minimal AgentState with optional overrides."""
        defaults: dict = {
            "question": "What is LoRA?",
            "intent": Intent(
                type="factual", confidence=0.9, reformulated_query="What is LoRA?"
            ),
            "retrieval_queries": ["What is LoRA?"],
            "retrieved_contexts": [
                RetrievedContext(
                    content="LoRA is ...",
                    source="LoRA paper",
                    doc_id="doc1",
                    relevance_score=0.9,
                ),
            ],
            "draft_answer": "LoRA is a parameter-efficient method.",
            "final_answer": "LoRA is a parameter-efficient fine-tuning method.",
            "critic_verdict": CriticVerdict(
                passed=True, score=8.0, completeness=0.9,
                faithfulness=0.95, feedback="Good.",
            ),
            "retry_count": 0,
            "reasoning_trace": [
                ReasoningStep(
                    step_type="route", content="factual", timestamp=1.0,
                ),
                ReasoningStep(
                    step_type="action", content="RAG search", timestamp=2.0,
                ),
            ],
        }
        defaults.update(overrides)
        return AgentState(**defaults)

    def test_from_state_populates_all_fields(self) -> None:
        state = self._make_state()
        trace = AgentTrace.from_state(state, duration_ms=500.0)

        assert trace.question == "What is LoRA?"
        assert trace.strategy_executed == "simple"
        assert trace.duration_ms == 500.0
        assert len(trace.reasoning_steps) == 2
        assert trace.reasoning_steps[0].step_type == "route"
        assert trace.reasoning_steps[1].step_type == "action"
        assert trace.retrieval_queries == ["What is LoRA?"]
        assert trace.contexts_retrieved == 1
        assert trace.critic_verdict is not None
        assert trace.critic_verdict.passed is True
        assert trace.retry_count == 0
        assert trace.final_answer == "LoRA is a parameter-efficient fine-tuning method."

    def test_from_state_with_custom_trace_id(self) -> None:
        state = self._make_state()
        trace = AgentTrace.from_state(state, trace_id="custom-id-123")
        assert trace.trace_id == "custom-id-123"

    def test_from_state_auto_generates_trace_id(self) -> None:
        state = self._make_state()
        trace = AgentTrace.from_state(state)
        assert trace.trace_id  # non-empty
        assert len(trace.trace_id) == 32  # UUID4 hex

    def test_from_state_empty_state(self) -> None:
        state = AgentState()
        trace = AgentTrace.from_state(state)
        assert trace.question == ""
        assert trace.strategy_executed == ""
        assert trace.intent is None
        assert trace.reasoning_steps == []
        assert trace.contexts_retrieved == 0
        assert trace.final_answer == ""

    def test_from_state_with_node_traces_computes_latency(self) -> None:
        state = self._make_state()
        node_traces = [
            NodeTraceEntry(node_name="route", duration_ms=25.0),
            NodeTraceEntry(node_name="slot_fill", duration_ms=50.0),
            NodeTraceEntry(node_name="simple", duration_ms=200.0),
            NodeTraceEntry(node_name="critic", duration_ms=40.0),
        ]
        trace = AgentTrace.from_state(state, node_traces=node_traces)

        assert trace.router_latency_ms == 25.0
        assert trace.retrieval_latency_ms == 200.0
        assert trace.llm_latency_ms == 50.0
        assert trace.critic_latency_ms == 40.0
        assert len(trace.node_traces) == 4

    def test_from_state_with_tokens_used(self) -> None:
        state = self._make_state()
        trace = AgentTrace.from_state(
            state, tokens_used={"input": 200, "output": 80},
        )
        assert trace.tokens_used == {"input": 200, "output": 80}

    def test_from_state_comparative_strategy(self) -> None:
        intent = Intent(
            type="comparative",
            confidence=0.9,
            entities=["BERT", "GPT"],
            dimensions=["architecture"],
            reformulated_query="Compare BERT and GPT",
        )
        state = self._make_state(intent=intent)
        trace = AgentTrace.from_state(state)
        assert trace.strategy_executed == "comparative"

    def test_from_state_reasoning_steps_are_copied(self) -> None:
        """Verify reasoning_steps is a copy, not a reference."""
        state = self._make_state()
        trace = AgentTrace.from_state(state)
        assert trace.reasoning_steps == list(state.reasoning_trace)
        assert trace.reasoning_steps is not state.reasoning_trace


# ---------------------------------------------------------------------------
# AgentTracer tests
# ---------------------------------------------------------------------------


class TestAgentTracer:
    """Tests for the AgentTracer stateful builder."""

    def test_init(self) -> None:
        tracer = AgentTracer(question="What is LoRA?")
        assert tracer.question == "What is LoRA?"
        assert tracer.node_traces == []

    def test_start_and_end_node(self) -> None:
        tracer = AgentTracer(question="test")

        tracer.start_node("route")
        time.sleep(0.01)  # small delay to ensure measurable duration
        entry = tracer.end_node("route", output={"intent_type": "factual"})

        assert entry.node_name == "route"
        assert entry.duration_ms > 0
        assert entry.output == {"intent_type": "factual"}
        assert len(tracer.node_traces) == 1

    def test_multiple_nodes_tracked(self) -> None:
        tracer = AgentTracer(question="test")

        tracer.start_node("route")
        tracer.end_node("route", output={"type": "factual"})

        tracer.start_node("slot_fill")
        tracer.end_node("slot_fill", output={"entities": []})

        tracer.start_node("simple")
        tracer.end_node("simple", output={"draft": "answer"})

        tracer.start_node("critic")
        tracer.end_node("critic", output={"passed": True})

        assert len(tracer.node_traces) == 4
        names = [e.node_name for e in tracer.node_traces]
        assert names == ["route", "slot_fill", "simple", "critic"]

    def test_end_node_without_start_raises(self) -> None:
        tracer = AgentTracer(question="test")
        with pytest.raises(ValueError, match="without a matching start_node"):
            tracer.end_node("route")

    def test_end_node_default_output_is_empty(self) -> None:
        tracer = AgentTracer(question="test")
        tracer.start_node("route")
        entry = tracer.end_node("route")
        assert entry.output == {}

    def test_add_token_usage(self) -> None:
        tracer = AgentTracer(question="test")
        tracer.add_token_usage(input_tokens=100, output_tokens=50)
        tracer.add_token_usage(input_tokens=200, output_tokens=80)

        state = AgentState(question="test")
        trace = tracer.build_trace(state)
        assert trace.tokens_used == {"input": 300, "output": 130}

    def test_build_trace_from_state(self) -> None:
        tracer = AgentTracer(question="What is LoRA?")

        tracer.start_node("route")
        tracer.end_node("route", output={"intent_type": "factual"})

        tracer.start_node("simple")
        tracer.end_node("simple", output={"num_contexts": 2})

        tracer.start_node("critic")
        tracer.end_node("critic", output={"passed": True, "score": 8.5})

        intent = Intent(
            type="factual", confidence=0.9, reformulated_query="What is LoRA?",
        )
        state = AgentState(
            question="What is LoRA?",
            intent=intent,
            final_answer="LoRA is a parameter-efficient fine-tuning method.",
            reasoning_trace=[
                ReasoningStep(step_type="route", content="factual", timestamp=1.0),
                ReasoningStep(step_type="action", content="search", timestamp=2.0),
            ],
            retrieved_contexts=[
                RetrievedContext(
                    content="text", source="src", doc_id="d1", relevance_score=0.9,
                ),
            ],
            retrieval_queries=["What is LoRA?"],
            critic_verdict=CriticVerdict(
                passed=True, score=8.5, completeness=0.9,
                faithfulness=0.95, feedback="Good.",
            ),
        )

        trace = tracer.build_trace(state)

        assert trace.question == "What is LoRA?"
        assert trace.duration_ms > 0
        assert trace.strategy_executed == "simple"
        assert len(trace.reasoning_steps) == 2
        assert trace.contexts_retrieved == 1
        assert trace.final_answer == "LoRA is a parameter-efficient fine-tuning method."
        assert len(trace.node_traces) == 3
        assert trace.router_latency_ms > 0
        assert trace.critic_latency_ms > 0

    def test_build_trace_with_custom_trace_id(self) -> None:
        tracer = AgentTracer(question="test")
        state = AgentState(question="test")
        trace = tracer.build_trace(state, trace_id="my-trace-id")
        assert trace.trace_id == "my-trace-id"

    def test_build_trace_computes_total_duration(self) -> None:
        with patch("src.tracing.tracer.time") as mock_time:
            mock_time.time.side_effect = [
                1000.0,   # __init__ start_time
                1000.050, # start_node route
                1000.100, # end_node route -> 50ms
                1000.500, # build_trace: (1000.5 - 1000.0) * 1000 = 500ms
            ]
            tracer = AgentTracer(question="test")
            tracer.start_node("route")
            tracer.end_node("route", output={"type": "factual"})
            state = AgentState(question="test")
            trace = tracer.build_trace(state)

            assert trace.duration_ms == pytest.approx(500.0)
            assert trace.node_traces[0].duration_ms == pytest.approx(50.0)

    def test_node_traces_property_returns_copy(self) -> None:
        """Verify node_traces property returns a copy, not internal list."""
        tracer = AgentTracer(question="test")
        tracer.start_node("route")
        tracer.end_node("route")
        traces = tracer.node_traces
        assert len(traces) == 1
        traces.clear()  # mutating copy
        assert len(tracer.node_traces) == 1  # internal list unaffected
