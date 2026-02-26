"""LoRA SFT training script for Critic answer quality evaluation."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from training.config_loader import load_training_section

DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_DATASET_PATH = Path("training/data/critic_sft_train.jsonl")
DEFAULT_OUTPUT_DIR = Path("training/artifacts/critic_sft_adapter")
DEFAULT_MAX_STEPS = 200
DEFAULT_NUM_EPOCHS = 3.0
DEFAULT_BATCH_SIZE = 8
DEFAULT_GRAD_ACCUM_STEPS = 2
DEFAULT_LEARNING_RATE = 2e-4
DEFAULT_WARMUP_RATIO = 0.05
DEFAULT_LOGGING_STEPS = 5
DEFAULT_SAVE_STEPS = 50
DEFAULT_MAX_SEQ_LEN = 512
DEFAULT_LORA_R = 16
DEFAULT_LORA_ALPHA = 32
DEFAULT_LORA_DROPOUT = 0.05
DEFAULT_SEED = 42
DEFAULT_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")
DEFAULT_REPORT_TO = "wandb"
DEFAULT_WANDB_PROJECT = "paper-pilot"
DEFAULT_TRAINING_CONFIG = Path("training/training_config.yaml")


def _critic_sft_defaults_from_section(section: dict[str, Any]) -> dict[str, Any]:
    """Convert critic_sft YAML section to argparse defaults."""
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
        "warmup_ratio": section.get("warmup_ratio"),
        "logging_steps": section.get("logging_steps"),
        "save_steps": section.get("save_steps"),
        "max_seq_len": section.get("max_seq_length"),
        "lora_r": section.get("lora_r"),
        "lora_alpha": section.get("lora_alpha"),
        "lora_dropout": section.get("lora_dropout"),
        "target_modules": target,
        "seed": section.get("seed"),
        "report_to": section.get("report_to"),
        "wandb_project": section.get("wandb_project"),
        "run_name": section.get("run_name"),
        "bf16": section.get("bf16"),
        "no_fallback": not section.get("fallback_on_error", True),
    }


@dataclass(slots=True)
class SFTConfig:
    """Hyperparameters and file paths for Critic LoRA SFT."""

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
    target_modules: list[str]
    bf16: bool
    seed: int
    report_to: str
    wandb_project: str
    run_name: str | None
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


def load_rows(dataset_path: Path) -> list[dict[str, Any]]:
    """Load Critic SFT rows from JSONL."""
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
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return TrainingResult(
        adapter_dir=config.output_dir,
        used_fallback=True,
        message=f"Training fallback (error: {reason}). Artifact dir: {config.output_dir}. Details: {metadata_path}",
    )


def run_sft(config: SFTConfig) -> TrainingResult:
    """Run LoRA SFT training and persist adapter."""
    try:
        rows = load_rows(config.dataset_path)
        training_texts = [
            build_prompt(
                instruction=str(row.get("instruction", "")),
                input_text=str(row.get("input", "")),
                output_text=str(row.get("output", "")),
            )
            for row in rows
        ]
        _run_real_training(config=config, training_texts=training_texts)
        return TrainingResult(
            adapter_dir=config.output_dir,
            used_fallback=False,
            message=f"Critic LoRA SFT finished. Adapter saved to: {config.output_dir}",
        )
    except Exception as exc:
        if not config.fallback_on_error:
            raise
        return _write_fallback_artifact(config=config, reason=str(exc))


def _run_real_training(*, config: SFTConfig, training_texts: list[str]) -> None:
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
    except Exception as exc:
        raise RuntimeError("Missing training dependencies. Install with: pip install -e .[training]") from exc

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = Dataset.from_dict({"text": training_texts})

    def _tokenize(batch: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        tokenized = tokenizer(batch["text"], truncation=True, max_length=config.max_seq_length)
        tokenized["labels"] = [ids.copy() for ids in tokenized["input_ids"]]
        return tokenized

    tokenized_dataset = dataset.map(_tokenize, batched=True, remove_columns=["text"])
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

    config.output_dir.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(config.output_dir),
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
        report_to=config.report_to,
        run_name=config.run_name,
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
    parser = argparse.ArgumentParser(description="Train Critic LoRA adapter with SFT.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_TRAINING_CONFIG,
        help="Path to training_config.yaml (defaults used if file exists).",
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max_steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--epochs", type=float, default=DEFAULT_NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--grad-accum-steps", type=int, default=DEFAULT_GRAD_ACCUM_STEPS)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--warmup-ratio", type=float, default=DEFAULT_WARMUP_RATIO)
    parser.add_argument("--logging-steps", type=int, default=DEFAULT_LOGGING_STEPS)
    parser.add_argument("--save-steps", type=int, default=DEFAULT_SAVE_STEPS)
    parser.add_argument("--max-seq-len", type=int, default=DEFAULT_MAX_SEQ_LEN)
    parser.add_argument("--lora-r", type=int, default=DEFAULT_LORA_R)
    parser.add_argument("--lora-alpha", type=int, default=DEFAULT_LORA_ALPHA)
    parser.add_argument("--lora-dropout", type=float, default=DEFAULT_LORA_DROPOUT)
    parser.add_argument("--target-modules", type=str, default=",".join(DEFAULT_TARGET_MODULES))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--report-to", type=str, choices=("none", "wandb"), default=DEFAULT_REPORT_TO)
    parser.add_argument("--wandb-project", type=str, default=DEFAULT_WANDB_PROJECT)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--no-fallback", action="store_true")
    return parser


def parse_config(argv: list[str] | None = None) -> SFTConfig:
    argv = argv if argv is not None else []
    parser = build_arg_parser()
    config_path = DEFAULT_TRAINING_CONFIG
    if "--config" in argv:
        i = argv.index("--config")
        if i + 1 < len(argv):
            config_path = Path(argv[i + 1])
    section = load_training_section(config_path, "critic_sft")
    defaults = {k: v for k, v in _critic_sft_defaults_from_section(section).items() if v is not None}
    parser.set_defaults(**defaults)
    args = parser.parse_args(argv)
    target_modules = [part.strip() for part in str(args.target_modules).split(",") if part.strip()]
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
        target_modules=target_modules or list(DEFAULT_TARGET_MODULES),
        bf16=bool(args.bf16),
        seed=int(args.seed),
        report_to=str(args.report_to),
        wandb_project=str(args.wandb_project),
        run_name=str(args.run_name) if args.run_name else None,
        fallback_on_error=not bool(args.no_fallback),
    )


def main(argv: list[str] | None = None) -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:
        pass
    config = parse_config(argv)
    result = run_sft(config)
    print(result.message)
    print(f"adapter_dir={result.adapter_dir}")
    print(f"used_fallback={result.used_fallback}")
    if result.used_fallback:
        print("Tip: run with --no-fallback to see the full traceback instead of writing a fallback artifact.")


if __name__ == "__main__":
    main()
