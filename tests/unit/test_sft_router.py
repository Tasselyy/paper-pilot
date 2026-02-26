"""Unit tests for training/sft_router.py."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from training.sft_router import (
    SFTConfig,
    build_prompt,
    load_router_rows,
    parse_config,
    run_sft,
)


def _make_config(tmp_path: Path, dataset_path: Path) -> SFTConfig:
    """Build a small test config for sft router script."""
    return SFTConfig(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        dataset_path=dataset_path,
        output_dir=tmp_path / "adapter_out",
        max_steps=1,
        num_train_epochs=1.0,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=1e-4,
        warmup_ratio=0.0,
        logging_steps=1,
        save_steps=1,
        max_seq_length=128,
        lora_r=4,
        lora_alpha=8,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bf16=False,
        seed=123,
        report_to="none",
        wandb_project="paper-pilot",
        run_name=None,
        fallback_on_error=True,
    )


def test_build_prompt_contains_sections() -> None:
    """Prompt formatter should include instruction/input/response sections."""
    prompt = build_prompt(
        instruction="Classify intent.",
        input_text="What is LoRA?",
        output_text='{"type":"factual","confidence":0.9}',
    )
    assert "### Instruction:" in prompt
    assert "### Input:" in prompt
    assert "### Response:" in prompt
    assert "What is LoRA?" in prompt


def test_load_router_rows_reads_jsonl(tmp_path: Path) -> None:
    """JSONL loader should parse non-empty rows."""
    dataset_file = tmp_path / "router_train.jsonl"
    dataset_file.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "instruction": "Classify intent",
                        "input": "Compare LoRA and QLoRA",
                        "output": '{"type":"comparative","confidence":0.88}',
                        "intent": "comparative",
                    }
                ),
                json.dumps(
                    {
                        "instruction": "Classify intent",
                        "input": "What is LoRA?",
                        "output": '{"type":"factual","confidence":0.95}',
                        "intent": "factual",
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    rows = load_router_rows(dataset_file)
    assert len(rows) == 2
    assert rows[0]["intent"] == "comparative"
    assert rows[1]["intent"] == "factual"


def test_run_sft_writes_fallback_artifact_on_training_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """run_sft should create adapter artifacts in fallback mode on failures."""
    dataset_file = tmp_path / "router_train.jsonl"
    dataset_file.write_text(
        json.dumps(
            {
                "instruction": "Classify intent",
                "input": "What is LoRA?",
                "output": '{"type":"factual","confidence":0.95}',
                "intent": "factual",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config = _make_config(tmp_path, dataset_file)

    def _raise_training_error(*_: object, **__: object) -> None:
        raise RuntimeError("synthetic training failure")

    monkeypatch.setattr("training.sft_router._run_real_training", _raise_training_error)

    result = run_sft(config)
    assert result.used_fallback is True
    assert result.adapter_dir.exists()

    metadata_file = result.adapter_dir / "fallback_metadata.json"
    assert metadata_file.exists()
    payload = json.loads(metadata_file.read_text(encoding="utf-8"))
    assert payload["status"] == "fallback"
    assert "synthetic training failure" in payload["reason"]


def test_parse_config_applies_overrides(tmp_path: Path) -> None:
    """CLI parser should map overrides into typed SFTConfig fields."""
    dataset_file = tmp_path / "router_train.jsonl"
    output_dir = tmp_path / "out_adapter"
    config = parse_config(
        [
            "--dataset",
            str(dataset_file),
            "--output-dir",
            str(output_dir),
            "--max_steps",
            "10",
            "--target-modules",
            "q_proj,k_proj,v_proj,o_proj",
            "--report-to",
            "wandb",
            "--wandb-project",
            "paper-pilot",
            "--run-name",
            "router-sft-test",
            "--no-fallback",
        ]
    )
    assert config.dataset_path == dataset_file
    assert config.output_dir == output_dir
    assert config.max_steps == 10
    assert config.target_modules == ["q_proj", "k_proj", "v_proj", "o_proj"]
    assert config.report_to == "wandb"
    assert config.wandb_project == "paper-pilot"
    assert config.run_name == "router-sft-test"
    assert config.fallback_on_error is False
