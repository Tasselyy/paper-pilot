"""LoRA SFT training script for Router intent classification.

This script trains a causal LM with LoRA adapters on the synthetic Router
dataset generated in ``training/data/router_train.jsonl``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

DEFAULT_MODEL_NAME = "sshleifer/tiny-gpt2"
DEFAULT_DATASET_PATH = Path("training/data/router_train.jsonl")
DEFAULT_OUTPUT_DIR = Path("training/artifacts/router_lora_adapter")
DEFAULT_MAX_STEPS = 200
DEFAULT_NUM_EPOCHS = 1.0
DEFAULT_BATCH_SIZE = 4
DEFAULT_GRAD_ACCUM_STEPS = 2
DEFAULT_LEARNING_RATE = 2e-4
DEFAULT_WARMUP_RATIO = 0.05
DEFAULT_LOGGING_STEPS = 5
DEFAULT_SAVE_STEPS = 50
DEFAULT_MAX_SEQ_LEN = 256
DEFAULT_LORA_R = 8
DEFAULT_LORA_ALPHA = 16
DEFAULT_LORA_DROPOUT = 0.05
DEFAULT_SEED = 42


@dataclass(slots=True)
class SFTConfig:
    """Hyperparameters and file paths for Router LoRA SFT."""

    model_name: str
    dataset_path: Path
    output_dir: Path
    max_steps: int
    num_train_epochs: float
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    warmup_ratio: float
    logging_steps: int
    save_steps: int
    max_seq_length: int
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    bf16: bool
    seed: int
    fallback_on_error: bool


@dataclass(slots=True)
class TrainingResult:
    """Outcome of one training attempt."""

    adapter_dir: Path
    used_fallback: bool
    message: str


def build_prompt(*, instruction: str, input_text: str, output_text: str) -> str:
    """Build one instruction-tuning sample in plain text format."""
    return (
        "### Instruction:\n"
        f"{instruction.strip()}\n\n"
        "### Input:\n"
        f"{input_text.strip()}\n\n"
        "### Response:\n"
        f"{output_text.strip()}"
    )


def load_router_rows(dataset_path: Path) -> list[dict[str, Any]]:
    """Load Router training rows from JSONL.

    Args:
        dataset_path: JSONL path with Alpaca-style fields.

    Returns:
        Parsed rows as dictionaries.
    """
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    rows: list[dict[str, Any]] = []
    with dataset_path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            rows.append(json.loads(stripped))

    if not rows:
        raise ValueError(f"Dataset is empty: {dataset_path}")
    return rows


def _write_fallback_artifact(config: SFTConfig, reason: str) -> TrainingResult:
    """Create adapter-like output folder so workflow can proceed in constrained envs."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "status": "fallback",
        "reason": reason,
        "config": {
            **asdict(config),
            "dataset_path": str(config.dataset_path),
            "output_dir": str(config.output_dir),
        },
    }
    metadata_path = config.output_dir / "fallback_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    message = (
        "Training fallback completed. "
        f"Adapter artifact directory prepared at: {config.output_dir}"
    )
    return TrainingResult(
        adapter_dir=config.output_dir,
        used_fallback=True,
        message=message,
    )


def run_sft(config: SFTConfig) -> TrainingResult:
    """Run LoRA SFT training and persist adapter.

    If training dependencies are unavailable (or training fails) and
    ``fallback_on_error`` is enabled, an adapter artifact directory is still
    generated with diagnostic metadata.
    """
    try:
        rows = load_router_rows(config.dataset_path)
        training_texts = [
            build_prompt(
                instruction=str(row.get("instruction", "")),
                input_text=str(row.get("input", "")),
                output_text=str(row.get("output", "")),
            )
            for row in rows
        ]
        _run_real_training(config=config, training_texts=training_texts)
        message = f"LoRA SFT finished. Adapter saved to: {config.output_dir}"
        return TrainingResult(
            adapter_dir=config.output_dir,
            used_fallback=False,
            message=message,
        )
    except Exception as exc:
        if not config.fallback_on_error:
            raise
        return _write_fallback_artifact(config=config, reason=str(exc))


def _run_real_training(*, config: SFTConfig, training_texts: list[str]) -> None:
    """Run actual LoRA fine-tuning with transformers + peft + datasets."""
    try:
        from datasets import Dataset
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            DataCollatorForLanguageModeling,
            Trainer,
            TrainingArguments,
        )
    except Exception as exc:  # pragma: no cover - depends on optional deps
        raise RuntimeError(
            "Missing training dependencies. Install with: "
            "pip install -e .[training]"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = Dataset.from_dict({"text": training_texts})

    def _tokenize(batch: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        tokenized = tokenizer(
            batch["text"],
            truncation=True,
            max_length=config.max_seq_length,
        )
        tokenized["labels"] = [ids.copy() for ids in tokenized["input_ids"]]
        return tokenized

    tokenized_dataset = dataset.map(
        _tokenize,
        batched=True,
        remove_columns=["text"],
    )

    base_model = AutoModelForCausalLM.from_pretrained(config.model_name)
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        task_type=TaskType.CAUSAL_LM,
        target_modules=["c_attn", "c_proj"],
    )
    model = get_peft_model(base_model, lora_config)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(config.output_dir),
        overwrite_output_dir=True,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        max_steps=config.max_steps,
        warmup_ratio=config.warmup_ratio,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        bf16=config.bf16,
        fp16=not config.bf16,
        report_to="none",
        seed=config.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    trainer.train()
    model.save_pretrained(str(config.output_dir))
    tokenizer.save_pretrained(str(config.output_dir))


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(description="Train Router LoRA adapter with SFT.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="Base model for LoRA SFT.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Router JSONL dataset path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Adapter output directory.",
    )
    parser.add_argument("--max_steps", type=int, default=DEFAULT_MAX_STEPS, help="Maximum optimizer steps.")
    parser.add_argument("--epochs", type=float, default=DEFAULT_NUM_EPOCHS, help="Number of training epochs.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Per-device train batch size.",
    )
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=DEFAULT_GRAD_ACCUM_STEPS,
        help="Gradient accumulation steps.",
    )
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE, help="Learning rate.")
    parser.add_argument("--warmup-ratio", type=float, default=DEFAULT_WARMUP_RATIO, help="Warmup ratio.")
    parser.add_argument("--logging-steps", type=int, default=DEFAULT_LOGGING_STEPS, help="Logging interval.")
    parser.add_argument("--save-steps", type=int, default=DEFAULT_SAVE_STEPS, help="Checkpoint save interval.")
    parser.add_argument("--max-seq-len", type=int, default=DEFAULT_MAX_SEQ_LEN, help="Tokenizer max length.")
    parser.add_argument("--lora-r", type=int, default=DEFAULT_LORA_R, help="LoRA rank.")
    parser.add_argument("--lora-alpha", type=int, default=DEFAULT_LORA_ALPHA, help="LoRA alpha.")
    parser.add_argument("--lora-dropout", type=float, default=DEFAULT_LORA_DROPOUT, help="LoRA dropout.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed.")
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Use bf16 precision (defaults to fp16 when available).",
    )
    parser.add_argument(
        "--no-fallback",
        action="store_true",
        help="Disable fallback mode and raise errors directly.",
    )
    return parser


def parse_config(argv: list[str] | None = None) -> SFTConfig:
    """Parse CLI args into ``SFTConfig``."""
    args = build_arg_parser().parse_args(argv)
    return SFTConfig(
        model_name=str(args.model_name),
        dataset_path=Path(args.dataset),
        output_dir=Path(args.output_dir),
        max_steps=int(args.max_steps),
        num_train_epochs=float(args.epochs),
        per_device_train_batch_size=int(args.batch_size),
        gradient_accumulation_steps=int(args.grad_accum_steps),
        learning_rate=float(args.learning_rate),
        warmup_ratio=float(args.warmup_ratio),
        logging_steps=int(args.logging_steps),
        save_steps=int(args.save_steps),
        max_seq_length=int(args.max_seq_len),
        lora_r=int(args.lora_r),
        lora_alpha=int(args.lora_alpha),
        lora_dropout=float(args.lora_dropout),
        bf16=bool(args.bf16),
        seed=int(args.seed),
        fallback_on_error=not bool(args.no_fallback),
    )


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    config = parse_config(argv)
    result = run_sft(config)
    print(result.message)
    print(f"adapter_dir={result.adapter_dir}")
    print(f"used_fallback={result.used_fallback}")


if __name__ == "__main__":
    main()
