"""Unit tests for training/export_quantized.py."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from training.export_quantized import ExportConfig, parse_config, run_export


def _make_config(tmp_path: Path) -> ExportConfig:
    """Build a small config fixture for export script tests."""
    return ExportConfig(
        model_path=tmp_path / "input_model",
        output_dir=tmp_path / "quantized_out",
        base_model_name_or_path=None,
        validate_with_local_manager=True,
        safe_serialization=True,
        fallback_on_error=True,
    )


def test_run_export_writes_fallback_artifact_on_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """run_export should create fallback metadata when export fails."""
    config = _make_config(tmp_path)

    def _raise_export_error(_: ExportConfig) -> None:
        raise RuntimeError("synthetic export failure")

    monkeypatch.setattr("training.export_quantized._run_real_export", _raise_export_error)

    result = run_export(config)
    assert result.used_fallback is True
    assert result.model_dir.exists()

    metadata_file = result.model_dir / "fallback_metadata.json"
    assert metadata_file.exists()
    payload = json.loads(metadata_file.read_text(encoding="utf-8"))
    assert payload["status"] == "fallback"
    assert "synthetic export failure" in payload["reason"]


def test_run_export_success_when_real_export_succeeds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """run_export should report success when underlying export passes."""
    config = _make_config(tmp_path)

    def _fake_real_export(cfg: ExportConfig) -> None:
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        (cfg.output_dir / "export_metadata.json").write_text(
            json.dumps({"status": "ok"}),
            encoding="utf-8",
        )

    monkeypatch.setattr("training.export_quantized._run_real_export", _fake_real_export)

    result = run_export(config)
    assert result.used_fallback is False
    assert result.model_dir == config.output_dir
    assert (result.model_dir / "export_metadata.json").exists()


def test_run_export_raises_when_fallback_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """run_export should raise original error when fallback is disabled."""
    config = _make_config(tmp_path)
    config.fallback_on_error = False

    def _raise_export_error(_: ExportConfig) -> None:
        raise RuntimeError("export hard failure")

    monkeypatch.setattr("training.export_quantized._run_real_export", _raise_export_error)

    with pytest.raises(RuntimeError, match="export hard failure"):
        run_export(config)


def test_parse_config_applies_overrides(tmp_path: Path) -> None:
    """CLI parser should map overrides into typed ExportConfig fields."""
    model_path = tmp_path / "adapter_dir"
    output_dir = tmp_path / "quantized_dir"
    config = parse_config(
        [
            "--model-path",
            str(model_path),
            "--output-dir",
            str(output_dir),
            "--base-model",
            "sshleifer/tiny-gpt2",
            "--no-validate-load",
            "--unsafe-serialization",
            "--no-fallback",
        ]
    )
    assert config.model_path == model_path
    assert config.output_dir == output_dir
    assert config.base_model_name_or_path == "sshleifer/tiny-gpt2"
    assert config.validate_with_local_manager is False
    assert config.safe_serialization is False
    assert config.fallback_on_error is False
