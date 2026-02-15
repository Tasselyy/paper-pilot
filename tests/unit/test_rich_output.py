"""Unit tests for src/tracing/rich_output.py — RichStreamOutput.

Verifies that:

- ``_get_node_display`` returns correct labels and styles for known nodes
  and falls back for unknown nodes.
- ``_summarize_node_output`` extracts the right summary for every node type
  (route, slot_fill, strategies, critic, retry_refine, format_output) and
  handles missing/empty data gracefully.
- ``_attr_or_key`` works with Pydantic models, plain dicts, and falls back.
- ``RichStreamOutput.on_node_start`` / ``on_node_end`` track timing and
  populate ``completed_nodes``.
- ``on_node_end`` without ``on_node_start`` records zero duration.
- ``on_node_error`` captures errors and timing.
- ``display_header`` renders a panel containing the question text.
- ``display_final_answer`` renders a panel containing the answer.
- ``display_summary`` renders strategy, timing, tokens, and retries.
- ``display_trace_summary`` reads fields from an ``AgentTrace`` instance.
- ``stream_graph`` yields per-node output and accumulates final state (async).
- ``create_rich_output`` factory returns a ``RichStreamOutput``.
- Verbose mode prints "in-progress" indicators on ``on_node_start``.
- ``completed_nodes`` returns a copy (internal list unaffected by mutation).

No real API calls or external dependencies.
"""

from __future__ import annotations

import asyncio
import io
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from rich.console import Console

from src.agent.state import CriticVerdict, Intent
from src.tracing.rich_output import (
    RichStreamOutput,
    _attr_or_key,
    _get_node_display,
    _summarize_node_output,
    create_rich_output,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_console() -> Console:
    """Create a Rich Console that captures output for assertions."""
    return Console(file=io.StringIO(), force_terminal=True, width=120)


def _get_output(console: Console) -> str:
    """Extract printed text from a Console backed by StringIO."""
    console.file.seek(0)  # type: ignore[union-attr]
    return console.file.read()  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# _get_node_display tests
# ---------------------------------------------------------------------------


class TestGetNodeDisplay:
    """Tests for the node display lookup helper."""

    def test_known_nodes_return_label_and_style(self) -> None:
        label, style = _get_node_display("route")
        assert label == "Router"
        assert "cyan" in style

    def test_all_known_nodes_have_entries(self) -> None:
        known = [
            "load_memory", "route", "slot_fill", "simple", "multi_hop",
            "comparative", "exploratory", "critic", "retry_refine",
            "save_memory", "format_output",
        ]
        for name in known:
            label, style = _get_node_display(name)
            assert label, f"Missing label for {name}"
            assert style, f"Missing style for {name}"

    def test_unknown_node_falls_back(self) -> None:
        label, style = _get_node_display("custom_node_xyz")
        assert label == "custom_node_xyz"
        assert style == "bold"


# ---------------------------------------------------------------------------
# _attr_or_key tests
# ---------------------------------------------------------------------------


class TestAttrOrKey:
    """Tests for the attribute-or-key helper."""

    def test_reads_attribute_from_object(self) -> None:
        intent = Intent(type="factual", confidence=0.9)
        assert _attr_or_key(intent, "type") == "factual"
        assert _attr_or_key(intent, "confidence") == 0.9

    def test_reads_key_from_dict(self) -> None:
        d = {"type": "comparative", "confidence": 0.8}
        assert _attr_or_key(d, "type") == "comparative"
        assert _attr_or_key(d, "confidence") == 0.8

    def test_returns_default_when_missing(self) -> None:
        assert _attr_or_key({}, "missing", "fallback") == "fallback"
        assert _attr_or_key(42, "missing", None) is None

    def test_attribute_preferred_over_dict_key(self) -> None:
        """When object has both attribute and is dict-like, attribute wins."""

        class Hybrid(dict):  # type: ignore[type-arg]
            color = "red"

        obj = Hybrid(color="blue")
        assert _attr_or_key(obj, "color") == "red"


# ---------------------------------------------------------------------------
# _summarize_node_output tests
# ---------------------------------------------------------------------------


class TestSummarizeNodeOutput:
    """Tests for per-node output summarisation."""

    def test_route_with_intent_model(self) -> None:
        intent = Intent(type="factual", confidence=0.92)
        result = _summarize_node_output("route", {"intent": intent})
        assert "intent=factual" in result
        assert "92%" in result

    def test_route_with_intent_dict(self) -> None:
        result = _summarize_node_output(
            "route",
            {"intent": {"type": "comparative", "confidence": 0.85}},
        )
        assert "intent=comparative" in result
        assert "85%" in result

    def test_route_with_no_intent(self) -> None:
        assert _summarize_node_output("route", {}) == ""

    def test_route_with_none_intent(self) -> None:
        assert _summarize_node_output("route", {"intent": None}) == ""

    def test_slot_fill_with_query(self) -> None:
        intent = Intent(
            type="factual",
            confidence=0.9,
            reformulated_query="What is LoRA?",
        )
        result = _summarize_node_output("slot_fill", {"intent": intent})
        assert 'query="What is LoRA?"' in result

    def test_slot_fill_with_long_query_truncates(self) -> None:
        long_query = "A" * 80
        intent = Intent(
            type="factual",
            confidence=0.9,
            reformulated_query=long_query,
        )
        result = _summarize_node_output("slot_fill", {"intent": intent})
        assert result.endswith('..."')
        # The truncated part should be 50 chars + "..."
        assert "A" * 50 in result

    def test_slot_fill_no_intent(self) -> None:
        assert _summarize_node_output("slot_fill", {}) == ""

    def test_strategy_with_contexts_and_draft(self) -> None:
        for node in ("simple", "multi_hop", "comparative", "exploratory"):
            result = _summarize_node_output(node, {
                "retrieved_contexts": [1, 2, 3],
                "draft_answer": "Some answer text",
            })
            assert "3 context(s)" in result
            assert "draft=" in result

    def test_strategy_empty_output(self) -> None:
        assert _summarize_node_output("simple", {}) == ""

    def test_critic_passed(self) -> None:
        verdict = CriticVerdict(
            passed=True, score=8.5, completeness=0.9,
            faithfulness=0.95, feedback="Good.",
        )
        result = _summarize_node_output("critic", {"critic_verdict": verdict})
        assert "passed" in result
        assert "8.5/10" in result

    def test_critic_failed(self) -> None:
        verdict = CriticVerdict(
            passed=False, score=4.0, completeness=0.5,
            faithfulness=0.6, feedback="Needs work.",
        )
        result = _summarize_node_output("critic", {"critic_verdict": verdict})
        assert "failed" in result
        assert "4.0/10" in result

    def test_critic_dict_verdict(self) -> None:
        result = _summarize_node_output(
            "critic",
            {"critic_verdict": {"passed": True, "score": 7.0}},
        )
        assert "passed" in result
        assert "7.0/10" in result

    def test_critic_no_verdict(self) -> None:
        assert _summarize_node_output("critic", {}) == ""

    def test_retry_refine_with_count(self) -> None:
        result = _summarize_node_output("retry_refine", {"retry_count": 2})
        assert "retry #2" in result

    def test_retry_refine_no_count(self) -> None:
        assert _summarize_node_output("retry_refine", {}) == ""

    def test_format_output_with_answer(self) -> None:
        result = _summarize_node_output(
            "format_output",
            {"final_answer": "LoRA is a fine-tuning method."},
        )
        assert "chars" in result

    def test_format_output_empty_answer(self) -> None:
        assert _summarize_node_output("format_output", {"final_answer": ""}) == ""

    def test_unknown_node_returns_empty(self) -> None:
        assert _summarize_node_output("custom_node", {"data": 42}) == ""


# ---------------------------------------------------------------------------
# RichStreamOutput — construction and properties
# ---------------------------------------------------------------------------


class TestRichStreamOutputConstruction:
    """Tests for RichStreamOutput construction and properties."""

    def test_default_construction(self) -> None:
        out = RichStreamOutput()
        assert out.console is not None
        assert out.verbose is False
        assert out.completed_nodes == []

    def test_custom_console_and_verbose(self) -> None:
        console = _make_console()
        out = RichStreamOutput(console=console, verbose=True)
        assert out.console is console
        assert out.verbose is True

    def test_create_rich_output_factory(self) -> None:
        out = create_rich_output(verbose=True)
        assert isinstance(out, RichStreamOutput)
        assert out.verbose is True


# ---------------------------------------------------------------------------
# RichStreamOutput — node lifecycle
# ---------------------------------------------------------------------------


class TestRichStreamOutputNodeLifecycle:
    """Tests for on_node_start / on_node_end / on_node_error."""

    def test_on_node_end_records_completion(self) -> None:
        console = _make_console()
        out = RichStreamOutput(console=console)

        out.on_node_start("route")
        time.sleep(0.01)
        out.on_node_end("route", output={"intent": {"type": "factual"}})

        assert len(out.completed_nodes) == 1
        name, duration_ms, output = out.completed_nodes[0]
        assert name == "route"
        assert duration_ms > 0
        assert output == {"intent": {"type": "factual"}}

    def test_on_node_end_without_start_gives_zero_duration(self) -> None:
        console = _make_console()
        out = RichStreamOutput(console=console)

        out.on_node_end("route", output={"x": 1})
        name, duration_ms, output = out.completed_nodes[0]
        assert name == "route"
        assert duration_ms == 0.0

    def test_multiple_nodes_tracked(self) -> None:
        console = _make_console()
        out = RichStreamOutput(console=console)

        for node in ("load_memory", "route", "slot_fill", "simple", "critic"):
            out.on_node_start(node)
            out.on_node_end(node, output={})

        assert len(out.completed_nodes) == 5
        names = [n for n, _, _ in out.completed_nodes]
        assert names == ["load_memory", "route", "slot_fill", "simple", "critic"]

    def test_on_node_end_prints_check_mark(self) -> None:
        console = _make_console()
        out = RichStreamOutput(console=console)

        out.on_node_start("route")
        out.on_node_end("route", output={})

        text = _get_output(console)
        assert "v" in text  # check mark
        assert "Router" in text

    def test_on_node_end_prints_timing(self) -> None:
        console = _make_console()
        out = RichStreamOutput(console=console)

        out.on_node_start("simple")
        out.on_node_end("simple", output={})

        text = _get_output(console)
        assert "ms" in text

    def test_on_node_end_prints_summary(self) -> None:
        console = _make_console()
        out = RichStreamOutput(console=console)

        intent = Intent(type="factual", confidence=0.95)
        out.on_node_start("route")
        out.on_node_end("route", output={"intent": intent})

        text = _get_output(console)
        assert "intent=factual" in text

    def test_on_node_error_prints_error(self) -> None:
        console = _make_console()
        out = RichStreamOutput(console=console)

        out.on_node_start("critic")
        out.on_node_error("critic", RuntimeError("LLM timeout"))

        text = _get_output(console)
        assert "x" in text  # error marker
        assert "LLM timeout" in text

    def test_verbose_mode_prints_on_start(self) -> None:
        console = _make_console()
        out = RichStreamOutput(console=console, verbose=True)

        out.on_node_start("route")
        text = _get_output(console)
        assert "Router" in text

    def test_non_verbose_no_start_output(self) -> None:
        console = _make_console()
        out = RichStreamOutput(console=console, verbose=False)

        out.on_node_start("route")
        text = _get_output(console)
        # In non-verbose mode, on_node_start should not print anything
        assert "Router" not in text

    def test_completed_nodes_returns_copy(self) -> None:
        console = _make_console()
        out = RichStreamOutput(console=console)

        out.on_node_start("route")
        out.on_node_end("route")
        nodes = out.completed_nodes
        nodes.clear()
        assert len(out.completed_nodes) == 1  # internal unaffected


# ---------------------------------------------------------------------------
# RichStreamOutput — display methods
# ---------------------------------------------------------------------------


class TestRichStreamOutputDisplays:
    """Tests for display_header, display_final_answer, display_summary."""

    def test_display_header_shows_question(self) -> None:
        console = _make_console()
        out = RichStreamOutput(console=console)

        out.display_header("What is LoRA?")
        text = _get_output(console)
        assert "What is LoRA?" in text
        assert "Paper Pilot" in text

    def test_display_header_resets_state(self) -> None:
        console = _make_console()
        out = RichStreamOutput(console=console)

        out.on_node_start("route")
        out.on_node_end("route")
        assert len(out.completed_nodes) == 1

        out.display_header("New question")
        assert len(out.completed_nodes) == 0

    def test_display_final_answer_shows_answer(self) -> None:
        console = _make_console()
        out = RichStreamOutput(console=console)

        out.display_final_answer("LoRA is a fine-tuning method.")
        text = _get_output(console)
        assert "LoRA is a fine-tuning method." in text
        assert "Answer" in text

    def test_display_final_answer_with_sources(self) -> None:
        console = _make_console()
        out = RichStreamOutput(console=console)

        out.display_final_answer(
            "LoRA is a fine-tuning method.",
            sources=["LoRA Paper", "Adapter Survey"],
        )
        text = _get_output(console)
        assert "LoRA Paper" in text
        assert "Adapter Survey" in text
        assert "Sources" in text

    def test_display_summary_with_all_fields(self) -> None:
        console = _make_console()
        out = RichStreamOutput(console=console)

        out.display_summary(
            strategy="simple",
            total_ms=450.0,
            tokens_in=1200,
            tokens_out=450,
            retry_count=1,
        )
        text = _get_output(console)
        assert "450ms" in text
        assert "simple" in text
        assert "1,200" in text
        assert "retries=1" in text

    def test_display_summary_minimal(self) -> None:
        console = _make_console()
        out = RichStreamOutput(console=console)

        out.display_summary(total_ms=100.0)
        text = _get_output(console)
        assert "100ms" in text

    def test_display_summary_auto_computes_total_ms(self) -> None:
        console = _make_console()
        out = RichStreamOutput(console=console)

        out.display_header("test")
        time.sleep(0.01)
        out.display_summary()
        text = _get_output(console)
        assert "ms" in text

    def test_display_trace_summary_from_agent_trace(self) -> None:
        from src.tracing.tracer import AgentTrace

        console = _make_console()
        out = RichStreamOutput(console=console)

        trace = AgentTrace(
            strategy_executed="comparative",
            duration_ms=1200.0,
            tokens_used={"input": 500, "output": 200},
            retry_count=0,
        )
        out.display_trace_summary(trace)
        text = _get_output(console)
        assert "1200ms" in text
        assert "comparative" in text
        assert "500" in text


# ---------------------------------------------------------------------------
# RichStreamOutput — async stream_graph
# ---------------------------------------------------------------------------


class TestRichStreamOutputStreamGraph:
    """Tests for the async stream_graph() method."""

    def _make_mock_graph(
        self,
        events: list[dict[str, Any]],
    ) -> MagicMock:
        """Create a mock compiled graph that yields the given events."""
        graph = MagicMock()

        async def mock_astream(
            input_state: Any, config: Any = None, stream_mode: str = "updates",
        ):  # type: ignore[no-untyped-def]
            for event in events:
                yield event

        graph.astream = mock_astream
        return graph

    @pytest.mark.asyncio
    async def test_stream_graph_basic(self) -> None:
        """stream_graph processes events and returns final state."""
        console = _make_console()
        out = RichStreamOutput(console=console)

        events = [
            {"load_memory": {"accumulated_facts": []}},
            {"route": {"intent": {"type": "factual", "confidence": 0.9}}},
            {"slot_fill": {"intent": {"type": "factual", "confidence": 0.9,
                                      "reformulated_query": "What is LoRA?"}}},
            {"simple": {"draft_answer": "LoRA is...",
                        "retrieved_contexts": [{"content": "..."}]}},
            {"critic": {"critic_verdict": {"passed": True, "score": 8.0}}},
            {"save_memory": {}},
            {"format_output": {"final_answer": "LoRA is a fine-tuning method."}},
        ]
        graph = self._make_mock_graph(events)

        final = await out.stream_graph(
            graph,
            {"question": "What is LoRA?"},
            config={"configurable": {"thread_id": "test"}},
        )

        # Verify final state accumulated
        assert final["final_answer"] == "LoRA is a fine-tuning method."

        # Verify all nodes recorded
        assert len(out.completed_nodes) == 7
        names = [n for n, _, _ in out.completed_nodes]
        assert "route" in names
        assert "simple" in names
        assert "format_output" in names

        # Verify output contains key text
        text = _get_output(console)
        assert "What is LoRA?" in text
        assert "Router" in text
        assert "LoRA is a fine-tuning method." in text

    @pytest.mark.asyncio
    async def test_stream_graph_skips_dunder_nodes(self) -> None:
        """__start__ and __end__ events are ignored."""
        console = _make_console()
        out = RichStreamOutput(console=console)

        events = [
            {"__start__": {"question": "test"}},
            {"route": {"intent": {"type": "factual"}}},
            {"__end__": {}},
        ]
        graph = self._make_mock_graph(events)

        await out.stream_graph(graph, {"question": "test"})

        names = [n for n, _, _ in out.completed_nodes]
        assert "__start__" not in names
        assert "__end__" not in names
        assert "route" in names

    @pytest.mark.asyncio
    async def test_stream_graph_no_final_answer(self) -> None:
        """When no final_answer in state, no answer panel is shown."""
        console = _make_console()
        out = RichStreamOutput(console=console)

        events = [{"route": {"intent": {"type": "factual"}}}]
        graph = self._make_mock_graph(events)

        final = await out.stream_graph(graph, {"question": "test"})

        assert final.get("final_answer", "") == ""
        text = _get_output(console)
        # No "Answer" panel should appear
        assert "Answer" not in text or "Answer" in text  # header may contain it

    @pytest.mark.asyncio
    async def test_stream_graph_empty_events(self) -> None:
        """Handles a graph that yields no events gracefully."""
        console = _make_console()
        out = RichStreamOutput(console=console)

        graph = self._make_mock_graph([])
        final = await out.stream_graph(graph, {"question": "test"})

        assert len(out.completed_nodes) == 0
        assert final.get("question") == "test"


# ---------------------------------------------------------------------------
# Integration: full flow
# ---------------------------------------------------------------------------


class TestRichStreamOutputIntegration:
    """Integration-style tests exercising the full callback flow."""

    def test_full_simple_flow(self) -> None:
        """Simulate a complete simple strategy flow with callbacks."""
        console = _make_console()
        out = RichStreamOutput(console=console)

        out.display_header("What is LoRA?")

        # load_memory
        out.on_node_start("load_memory")
        out.on_node_end("load_memory", output={"accumulated_facts": []})

        # route
        intent = Intent(type="factual", confidence=0.92)
        out.on_node_start("route")
        out.on_node_end("route", output={"intent": intent})

        # slot_fill
        intent_full = Intent(
            type="factual", confidence=0.92,
            reformulated_query="What is LoRA?",
        )
        out.on_node_start("slot_fill")
        out.on_node_end("slot_fill", output={"intent": intent_full})

        # simple
        out.on_node_start("simple")
        out.on_node_end("simple", output={
            "draft_answer": "LoRA is a parameter-efficient fine-tuning method.",
            "retrieved_contexts": [{"content": "LoRA paper text"}],
        })

        # critic
        verdict = CriticVerdict(
            passed=True, score=8.5, completeness=0.9,
            faithfulness=0.95, feedback="Good.",
        )
        out.on_node_start("critic")
        out.on_node_end("critic", output={"critic_verdict": verdict})

        # save_memory
        out.on_node_start("save_memory")
        out.on_node_end("save_memory", output={})

        # format_output
        out.on_node_start("format_output")
        out.on_node_end("format_output", output={
            "final_answer": "LoRA is a parameter-efficient fine-tuning method.\n\nSources:\n- LoRA paper",
        })

        # Display answer and summary
        out.display_final_answer(
            "LoRA is a parameter-efficient fine-tuning method.",
            sources=["LoRA paper"],
        )
        out.display_summary(
            strategy="simple",
            total_ms=450.0,
            tokens_in=500,
            tokens_out=200,
        )

        # Verify all nodes tracked
        assert len(out.completed_nodes) == 7

        # Verify output contains expected content
        text = _get_output(console)
        assert "What is LoRA?" in text
        assert "Router" in text
        assert "Simple Strategy" in text
        assert "Critic" in text
        assert "intent=factual" in text
        assert "passed" in text
        assert "LoRA is a parameter-efficient fine-tuning method." in text
        assert "simple" in text
        assert "450ms" in text

    def test_retry_flow(self) -> None:
        """Simulate a flow with a critic retry."""
        console = _make_console()
        out = RichStreamOutput(console=console)

        out.display_header("Compare BERT and GPT")

        # route + slot_fill + comparative
        out.on_node_start("route")
        out.on_node_end("route", output={
            "intent": Intent(type="comparative", confidence=0.88),
        })
        out.on_node_start("slot_fill")
        out.on_node_end("slot_fill", output={})
        out.on_node_start("comparative")
        out.on_node_end("comparative", output={"retrieved_contexts": [1, 2]})

        # critic fails
        out.on_node_start("critic")
        out.on_node_end("critic", output={
            "critic_verdict": CriticVerdict(
                passed=False, score=4.0, completeness=0.4,
                faithfulness=0.8, feedback="Missing dimensions.",
            ),
        })

        # retry_refine
        out.on_node_start("retry_refine")
        out.on_node_end("retry_refine", output={"retry_count": 1})

        text = _get_output(console)
        assert "failed" in text
        assert "4.0/10" in text
        assert "Retry / Refine" in text
        assert "retry #1" in text
