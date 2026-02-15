"""Unit tests for the main graph skeleton (A4).

Verifies that ``build_main_graph()`` compiles successfully and that a
single-turn invoke returns a valid state with ``final_answer`` populated.
"""

from __future__ import annotations

import pytest

from src.agent.graph import build_main_graph
from src.agent.state import AgentState


class TestGraphSkeleton:
    """Tests for the graph skeleton with placeholder nodes."""

    def test_graph_compiles(self) -> None:
        """build_main_graph() should return a compiled graph without error."""
        graph = build_main_graph()
        assert graph is not None

    def test_graph_has_expected_nodes(self) -> None:
        """The compiled graph should contain all registered nodes."""
        graph = build_main_graph()
        # Access the underlying graph structure
        node_names = set(graph.get_graph().nodes.keys())
        expected = {
            "__start__",
            "__end__",
            "load_memory",
            "route",
            "slot_fill",
            "simple",
            "multi_hop",
            "comparative",
            "exploratory",
            "critic",
            "retry_refine",
            "save_memory",
            "format_output",
        }
        assert expected.issubset(node_names), f"Missing nodes: {expected - node_names}"

    def test_invoke_simple_path_returns_state(self) -> None:
        """Invoke with a basic question should flow through the simple
        strategy path and return a state dict with final_answer."""
        graph = build_main_graph()
        config = {"configurable": {"thread_id": "test-skeleton-1"}}
        result = graph.invoke(
            {"question": "What is LoRA?"},
            config=config,
        )
        # Result should be a dict (or state-like object) with final_answer
        assert result is not None
        final_answer = result.get("final_answer") if isinstance(result, dict) else result.final_answer
        assert final_answer, "final_answer should be non-empty"

    def test_invoke_returns_intent(self) -> None:
        """The router placeholder should set an intent on the state."""
        graph = build_main_graph()
        config = {"configurable": {"thread_id": "test-skeleton-2"}}
        result = graph.invoke(
            {"question": "Compare BERT and GPT"},
            config=config,
        )
        intent = result.get("intent") if isinstance(result, dict) else result.intent
        assert intent is not None
        assert intent.type == "factual"  # placeholder always returns factual

    def test_invoke_returns_critic_verdict(self) -> None:
        """The critic placeholder should set a passing verdict."""
        graph = build_main_graph()
        config = {"configurable": {"thread_id": "test-skeleton-3"}}
        result = graph.invoke(
            {"question": "What is attention?"},
            config=config,
        )
        verdict = result.get("critic_verdict") if isinstance(result, dict) else result.critic_verdict
        assert verdict is not None
        assert verdict.passed is True

    def test_invoke_default_state_fields(self) -> None:
        """State should retain default values for fields not modified."""
        graph = build_main_graph()
        config = {"configurable": {"thread_id": "test-skeleton-4"}}
        result = graph.invoke(
            {"question": "Test defaults"},
            config=config,
        )
        assert result.get("retry_count", 0) == 0
        assert result.get("current_react_step", 0) == 0

    def test_multiple_invocations_independent(self) -> None:
        """Different thread_ids should produce independent results."""
        graph = build_main_graph()
        r1 = graph.invoke(
            {"question": "Q1"},
            config={"configurable": {"thread_id": "test-ind-1"}},
        )
        r2 = graph.invoke(
            {"question": "Q2"},
            config={"configurable": {"thread_id": "test-ind-2"}},
        )
        assert r1.get("final_answer") is not None
        assert r2.get("final_answer") is not None
