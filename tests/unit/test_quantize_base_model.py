"""Unit tests for training/quantize_base_model.py."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from training.quantize_base_model import (
    QuantizeConfig,
    parse_config,
    run_quantization,
)


def test_parse_config_overrides(tmp_path: Path) -> None:
    out = tmp_path / "awq_out"
    cfg = parse_config(
        [
            "--model-name",
            "Qwen/Qwen2.5-7B-Instruct",
            "--output-dir",
            str(out),
            "--bits",
            "4",
            "--group-size",
            "64",
            "--calib-samples",
            "32",
            "--no-fallback",
        ]
    )
    assert cfg.output_dir == out
    assert cfg.group_size == 64
    assert cfg.calib_samples == 32
    assert cfg.fallback_on_error is False


def test_run_quantization_writes_fallback_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = QuantizeConfig(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        output_dir=tmp_path / "awq_out",
        bits=4,
        group_size=128,
        zero_point=True,
        calib_samples=16,
        seed=42,
        fallback_on_error=True,
    )

    def _raise(_: QuantizeConfig) -> None:
        raise RuntimeError("synthetic quantization error")

    monkeypatch.setattr("training.quantize_base_model._run_real_quantization", _raise)
    result = run_quantization(cfg)
    assert result.used_fallback is True
    meta = cfg.output_dir / "fallback_metadata.json"
    assert meta.exists()
    payload = json.loads(meta.read_text(encoding="utf-8"))
    assert payload["status"] == "fallback"
