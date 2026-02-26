"""Unit tests for training/sft_critic.py."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from training.sft_critic import (
    SFTConfig,
    build_prompt,
    load_rows,
    parse_config,
    run_sft,
)


def _make_config(tmp_path: Path, dataset_path: Path) -> SFTConfig:
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
        max_seq_length=256,
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
    prompt = build_prompt(
        instruction="Evaluate answer quality.",
        input_text="Question: What is LoRA?",
        output_text='{"score":8.0,"completeness":0.8,"faithfulness":0.9,"feedback":"Good."}',
    )
    assert "### Instruction:" in prompt
    assert "### Input:" in prompt
    assert "### Response:" in prompt


def test_load_rows_reads_jsonl(tmp_path: Path) -> None:
    dataset = tmp_path / "critic_sft_train.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "instruction": "Evaluate",
                "input": "Question: ...",
                "output": '{"score":7.0,"completeness":0.7,"faithfulness":0.8,"feedback":"ok"}',
            }
        )
        + "\n",
        encoding="utf-8",
    )
    rows = load_rows(dataset)
    assert len(rows) == 1
    assert "instruction" in rows[0]


def test_run_sft_writes_fallback_on_training_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = tmp_path / "critic_sft_train.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "instruction": "Evaluate",
                "input": "Question: ...",
                "output": '{"score":7.0,"completeness":0.7,"faithfulness":0.8,"feedback":"ok"}',
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config = _make_config(tmp_path, dataset)

    def _raise(*_: object, **__: object) -> None:
        raise RuntimeError("synthetic training failure")

    monkeypatch.setattr("training.sft_critic._run_real_training", _raise)
    result = run_sft(config)
    assert result.used_fallback is True
    metadata_path = result.adapter_dir / "fallback_metadata.json"
    assert metadata_path.exists()


def test_parse_config_overrides(tmp_path: Path) -> None:
    dataset = tmp_path / "critic_sft_train.jsonl"
    output = tmp_path / "critic_adapter"
    config = parse_config(
        [
            "--dataset",
            str(dataset),
            "--output-dir",
            str(output),
            "--target-modules",
            "q_proj,k_proj,v_proj,o_proj",
            "--report-to",
            "wandb",
            "--run-name",
            "critic-sft-test",
            "--no-fallback",
        ]
    )
    assert config.dataset_path == dataset
    assert config.output_dir == output
    assert config.target_modules == ["q_proj", "k_proj", "v_proj", "o_proj"]
    assert config.report_to == "wandb"
    assert config.run_name == "critic-sft-test"
    assert config.fallback_on_error is False
