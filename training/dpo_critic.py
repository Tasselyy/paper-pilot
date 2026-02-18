"""DPO training script for Critic preference learning.

Trains a causal LM (optionally with LoRA) on preference pairs
(prompt, chosen, rejected) produced by ``training/data/generate_dpo_pairs.py``.
Saves the trained model or adapter to an output directory.

Design reference: DEV_SPEC G3.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

DEFAULT_MODEL_NAME = "sshleifer/tiny-gpt2"
DEFAULT_DATASET_PATH = Path("training/data/dpo_train.jsonl")
DEFAULT_OUTPUT_DIR = Path("training/artifacts/critic_dpo_model")
DEFAULT_MAX_STEPS = 200
DEFAULT_NUM_EPOCHS = 1.0
DEFAULT_BATCH_SIZE = 2
DEFAULT_GRAD_ACCUM_STEPS = 2
DEFAULT_LEARNING_RATE = 5e-5
DEFAULT_BETA = 0.1
DEFAULT_MAX_LENGTH = 384
DEFAULT_LOGGING_STEPS = 5
DEFAULT_SAVE_STEPS = 50
DEFAULT_SEED = 42


@dataclass(slots=True)
class DPOConfig:
    """Hyperparameters and paths for Critic DPO training."""

    model_name: str
    dataset_path: Path
    output_dir: Path
    max_steps: int
    num_train_epochs: float
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    beta: float
    max_length: int
    logging_steps: int
    save_steps: int
    bf16: bool
    seed: int
    fallback_on_error: bool


@dataclass(slots=True)
class DPOResult:
    """Outcome of one DPO training run."""

    model_dir: Path
    used_fallback: bool
    message: str


def load_dpo_rows(dataset_path: Path) -> list[dict[str, Any]]:
    """Load DPO preference rows from JSONL (prompt, chosen, rejected).

    Args:
        dataset_path: Path to JSONL file.

    Returns:
        List of dicts with keys prompt, chosen, rejected.
    """
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    rows: list[dict[str, Any]] = []
    with dataset_path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if "prompt" not in row or "chosen" not in row or "rejected" not in row:
                raise ValueError(
                    f"Each row must have 'prompt', 'chosen', 'rejected'; got keys: {list(row.keys())}"
                )
            rows.append(row)

    if not rows:
        raise ValueError(f"Dataset is empty: {dataset_path}")
    return rows


def _write_fallback_artifact(config: DPOConfig, reason: str) -> DPOResult:
    """Write fallback metadata so workflow can proceed when training fails."""
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
    return DPOResult(
        model_dir=config.output_dir,
        used_fallback=True,
        message=(
            "DPO training fallback completed. "
            f"Artifact directory prepared at: {config.output_dir}"
        ),
    )


def run_dpo(config: DPOConfig) -> DPOResult:
    """Run DPO training and save model.

    If dependencies are missing or training fails and ``fallback_on_error``
    is True, writes fallback metadata to ``output_dir`` instead of raising.
    """
    try:
        rows = load_dpo_rows(config.dataset_path)
        _run_real_dpo(config=config, rows=rows)
        return DPOResult(
            model_dir=config.output_dir,
            used_fallback=False,
            message=f"DPO training finished. Model saved to: {config.output_dir}",
        )
    except Exception as exc:
        if not config.fallback_on_error:
            raise
        return _write_fallback_artifact(config=config, reason=str(exc))


def _run_real_dpo(*, config: DPOConfig, rows: list[dict[str, Any]]) -> None:
    """Run actual DPO training with TRL DPOTrainer."""
    try:
        from datasets import Dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from trl import DPOConfig as TRLDPOConfig
        from trl import DPOTrainer
    except Exception as exc:
        raise RuntimeError(
            "Missing training dependencies. Install with: pip install -e .[training]"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = Dataset.from_list(rows)

    model = AutoModelForCausalLM.from_pretrained(config.model_name)
    training_args = TRLDPOConfig(
        output_dir=str(config.output_dir),
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        max_steps=config.max_steps,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        bf16=config.bf16,
        fp16=not config.bf16,
        report_to="none",
        seed=config.seed,
        beta=config.beta,
        max_length=config.max_length,
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=dataset,
    )
    trainer.train()
    trainer.save_model(str(config.output_dir))
    tokenizer.save_pretrained(str(config.output_dir))


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Train Critic with DPO on preference pairs."
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="Base causal LM for DPO.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="JSONL path with prompt/chosen/rejected.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Model output directory.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help="Maximum optimizer steps.",
    )
    parser.add_argument(
        "--epochs",
        type=float,
        default=DEFAULT_NUM_EPOCHS,
        help="Number of training epochs.",
    )
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
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help="DPO learning rate.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=DEFAULT_BETA,
        help="DPO beta (temperature) parameter.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=DEFAULT_MAX_LENGTH,
        help="Max total sequence length.",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=DEFAULT_LOGGING_STEPS,
        help="Logging interval.",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=DEFAULT_SAVE_STEPS,
        help="Checkpoint save interval.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed.",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Use bf16 (default fp16 when available).",
    )
    parser.add_argument(
        "--no-fallback",
        action="store_true",
        help="Disable fallback; raise on error.",
    )
    return parser


def parse_config(argv: list[str] | None = None) -> DPOConfig:
    """Parse CLI args into DPOConfig."""
    args = build_arg_parser().parse_args(argv)
    return DPOConfig(
        model_name=str(args.model_name),
        dataset_path=Path(args.dataset),
        output_dir=Path(args.output_dir),
        max_steps=int(args.max_steps),
        num_train_epochs=float(args.epochs),
        per_device_train_batch_size=int(args.batch_size),
        gradient_accumulation_steps=int(args.grad_accum_steps),
        learning_rate=float(args.learning_rate),
        beta=float(args.beta),
        max_length=int(args.max_length),
        logging_steps=int(args.logging_steps),
        save_steps=int(args.save_steps),
        bf16=bool(args.bf16),
        seed=int(args.seed),
        fallback_on_error=not bool(args.no_fallback),
    )


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    config = parse_config(argv)
    result = run_dpo(config)
    print(result.message)
    print(f"model_dir={result.model_dir}")
    print(f"used_fallback={result.used_fallback}")


if __name__ == "__main__":
    main()
