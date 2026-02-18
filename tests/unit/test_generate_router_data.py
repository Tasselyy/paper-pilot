"""Unit tests for synthetic Router training data generation."""

from __future__ import annotations

import json
from pathlib import Path

from training.data.generate_router_data import (
    INTENT_TYPES,
    generate_router_dataset,
    write_jsonl,
)


def test_generate_router_dataset_balanced_counts() -> None:
    """Dataset should be class-balanced with deterministic size."""
    rows = generate_router_dataset(samples_per_intent=10, seed=123)

    assert len(rows) == 50
    counts = {intent: 0 for intent in INTENT_TYPES}
    for row in rows:
        counts[row["intent"]] += 1
    assert all(count == 10 for count in counts.values())


def test_generate_router_dataset_schema_and_output_alignment() -> None:
    """Each row should follow Alpaca fields and aligned output JSON."""
    rows = generate_router_dataset(samples_per_intent=3, seed=42)

    for row in rows:
        assert set(row.keys()) == {"instruction", "input", "output", "intent"}
        assert row["instruction"]
        assert row["input"]
        parsed_output = json.loads(row["output"])
        assert parsed_output["type"] == row["intent"]
        assert 0.0 <= float(parsed_output["confidence"]) <= 1.0


def test_write_jsonl_persists_and_is_recoverable(tmp_path: Path) -> None:
    """JSONL writer should persist valid rows line-by-line."""
    rows = generate_router_dataset(samples_per_intent=2, seed=7)
    output_file = tmp_path / "router_train.jsonl"

    write_jsonl(rows, output_file)

    assert output_file.exists()
    lines = output_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == len(rows)
    first = json.loads(lines[0])
    assert set(first.keys()) == {"instruction", "input", "output", "intent"}
