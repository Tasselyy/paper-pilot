"""Unit tests for synthetic Critic SFT data generation."""

from __future__ import annotations

import json
from pathlib import Path

from training.data.generate_critic_sft_data import (
    generate_critic_sft_dataset,
    write_jsonl,
)


def test_generate_critic_sft_dataset_shape_and_schema() -> None:
    rows = generate_critic_sft_dataset(num_samples=24, seed=7)
    assert len(rows) == 24
    for row in rows:
        assert set(row.keys()) == {"instruction", "input", "output"}
        parsed = json.loads(row["output"])
        assert {"score", "completeness", "faithfulness", "feedback"} <= set(parsed.keys())
        assert 0.0 <= float(parsed["score"]) <= 10.0
        assert 0.0 <= float(parsed["completeness"]) <= 1.0
        assert 0.0 <= float(parsed["faithfulness"]) <= 1.0


def test_generate_critic_sft_dataset_is_deterministic() -> None:
    a = generate_critic_sft_dataset(num_samples=16, seed=123)
    b = generate_critic_sft_dataset(num_samples=16, seed=123)
    assert a == b


def test_write_jsonl_persists_rows(tmp_path: Path) -> None:
    rows = generate_critic_sft_dataset(num_samples=8, seed=11)
    out = tmp_path / "critic_sft_train.jsonl"
    write_jsonl(rows, out)
    assert out.exists()
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 8
    loaded = json.loads(lines[0])
    assert "instruction" in loaded and "input" in loaded and "output" in loaded
