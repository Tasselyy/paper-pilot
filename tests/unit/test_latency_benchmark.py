"""Unit tests for benchmarks/latency_benchmark.py."""

from __future__ import annotations

from benchmarks.latency_benchmark import _extract_node_llm_ms, _stats


def test_stats_computation() -> None:
    values = [100.0, 200.0, 300.0, 400.0]
    result = _stats(values)
    assert result["mean_ms"] == 250.0
    assert result["p50_ms"] >= 200.0
    assert result["p95_ms"] >= 300.0
    assert result["max_ms"] == 400.0


def test_extract_node_llm_ms_from_dict_trace() -> None:
    trace = [
        {"step_type": "route", "metadata": {"llm_call_duration_ms": 123.4}},
        {"step_type": "critique", "metadata": {"llm_call_duration_ms": 456.7}},
    ]
    assert _extract_node_llm_ms(trace, step_type="route") == 123.4
    assert _extract_node_llm_ms(trace, step_type="critique") == 456.7
