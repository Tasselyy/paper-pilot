"""DPO training script for Critic preference learning.

Trains a causal LM (optionally with LoRA) on preference pairs
(prompt, chosen, rejected) produced by ``training/data/generate_dpo_pairs.py``.
Saves the trained model or adapter to an output directory.

Design reference: DEV_SPEC G3.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from training.config_loader import load_training_section

DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_DATASET_PATH = Path("training/data/dpo_train.jsonl")
DEFAULT_OUTPUT_DIR = Path("training/artifacts/critic_dpo_model")
DEFAULT_MAX_STEPS = 200
DEFAULT_NUM_EPOCHS = 1.0
DEFAULT_BATCH_SIZE = 4
DEFAULT_GRAD_ACCUM_STEPS = 2
DEFAULT_LEARNING_RATE = 5e-5
DEFAULT_BETA = 0.1
DEFAULT_MAX_LENGTH = 512
DEFAULT_LOGGING_STEPS = 5
DEFAULT_SAVE_STEPS = 50
DEFAULT_SEED = 42
DEFAULT_LORA_R = 16
DEFAULT_LORA_ALPHA = 32
DEFAULT_LORA_DROPOUT = 0.05
DEFAULT_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")
DEFAULT_REPORT_TO = "wandb"
DEFAULT_WANDB_PROJECT = "paper-pilot"
DEFAULT_TRAINING_CONFIG = Path("training/training_config.yaml")


def _dpo_defaults_from_section(section: dict[str, Any]) -> dict[str, Any]:
    """Convert dpo YAML section to argparse defaults."""
    if not section:
        return {}
    target = section.get("target_modules")
    if isinstance(target, list):
        target = ",".join(str(x) for x in target)
    return {
        "model_name": section.get("model_name"),
        "dataset": Path(section["dataset_path"]) if section.get("dataset_path") else None,
        "output_dir": Path(section["output_dir"]) if section.get("output_dir") else None,
        "max_steps": section.get("max_steps"),
        "epochs": section.get("num_train_epochs"),
        "batch_size": section.get("per_device_train_batch_size"),
        "grad_accum_steps": section.get("gradient_accumulation_steps"),
        "learning_rate": section.get("learning_rate"),
        "beta": section.get("beta"),
        "max_length": section.get("max_length"),
        "lora_r": section.get("lora_r"),
        "lora_alpha": section.get("lora_alpha"),
        "lora_dropout": section.get("lora_dropout"),
        "target_modules": target,
        "logging_steps": section.get("logging_steps"),
        "save_steps": section.get("save_steps"),
        "seed": section.get("seed"),
        "report_to": section.get("report_to"),
        "wandb_project": section.get("wandb_project"),
        "run_name": section.get("run_name"),
        "bf16": section.get("bf16"),
        "no_fallback": not section.get("fallback_on_error", True),
    }


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
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    target_modules: list[str]
    logging_steps: int
    save_steps: int
    bf16: bool
    seed: int
    report_to: str
    wandb_project: str
    run_name: str | None
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
        from peft import LoraConfig, TaskType, get_peft_model
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

    base_model = AutoModelForCausalLM.from_pretrained(config.model_name)
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        task_type=TaskType.CAUSAL_LM,
        target_modules=config.target_modules,
    )
    model = get_peft_model(base_model, lora_config)
    if config.report_to == "wandb":
        os.environ.setdefault("WANDB_PROJECT", config.wandb_project)

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
        report_to=config.report_to,
        run_name=config.run_name,
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
    model.save_pretrained(str(config.output_dir))
    tokenizer.save_pretrained(str(config.output_dir))


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Train Critic with DPO on preference pairs."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_TRAINING_CONFIG,
        help="Path to training_config.yaml (defaults used if file exists).",
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
    parser.add_argument("--lora-r", type=int, default=DEFAULT_LORA_R, help="LoRA rank.")
    parser.add_argument("--lora-alpha", type=int, default=DEFAULT_LORA_ALPHA, help="LoRA alpha.")
    parser.add_argument("--lora-dropout", type=float, default=DEFAULT_LORA_DROPOUT, help="LoRA dropout.")
    parser.add_argument(
        "--target-modules",
        type=str,
        default=",".join(DEFAULT_TARGET_MODULES),
        help="Comma-separated LoRA target modules.",
    )
    parser.add_argument(
        "--report-to",
        type=str,
        choices=("none", "wandb"),
        default=DEFAULT_REPORT_TO,
        help="Training logger backend.",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=DEFAULT_WANDB_PROJECT,
        help="wandb project name when report_to=wandb.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional run name for the tracker.",
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
    """Parse CLI args into DPOConfig. Defaults from training_config.yaml if present."""
    argv = argv if argv is not None else []
    parser = build_arg_parser()
    config_path = DEFAULT_TRAINING_CONFIG
    if "--config" in argv:
        i = argv.index("--config")
        if i + 1 < len(argv):
            config_path = Path(argv[i + 1])
    section = load_training_section(config_path, "dpo")
    defaults = {k: v for k, v in _dpo_defaults_from_section(section).items() if v is not None}
    parser.set_defaults(**defaults)
    args = parser.parse_args(argv)
    target_modules = [part.strip() for part in str(args.target_modules).split(",") if part.strip()]
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
        lora_r=int(args.lora_r),
        lora_alpha=int(args.lora_alpha),
        lora_dropout=float(args.lora_dropout),
        target_modules=target_modules or list(DEFAULT_TARGET_MODULES),
        logging_steps=int(args.logging_steps),
        save_steps=int(args.save_steps),
        bf16=bool(args.bf16),
        seed=int(args.seed),
        report_to=str(args.report_to),
        wandb_project=str(args.wandb_project),
        run_name=str(args.run_name) if args.run_name else None,
        fallback_on_error=not bool(args.no_fallback),
    )


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:
        pass
    config = parse_config(argv)
    result = run_dpo(config)
    print(result.message)
    print(f"model_dir={result.model_dir}")
    print(f"used_fallback={result.used_fallback}")


if __name__ == "__main__":
    main()
