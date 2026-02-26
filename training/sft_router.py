"""Router 意图分类的 LoRA SFT 训练脚本。

在 Alpaca 格式的 Router 数据上微调因果语言模型（加 LoRA），使模型能根据用户问题输出意图类型
（factual / comparative / multi_hop / exploratory / follow_up）及置信度。数据通常由
training/data/generate_router_data.py 生成并写入 router_train.jsonl。支持 fallback 模式。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# 以脚本方式运行时（如 python training/sft_router.py）把项目根加入 path，保证能 import training
_root = Path(__file__).resolve().parent.parent
if _root not in sys.path:
    sys.path.insert(0, str(_root))

from training.config_loader import load_training_section

# ---------- 默认超参与路径 ----------
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_DATASET_PATH = Path("training/data/router_train.jsonl")
DEFAULT_OUTPUT_DIR = Path("training/artifacts/router_lora_adapter")
DEFAULT_MAX_STEPS = 200
DEFAULT_NUM_EPOCHS = 3.0
DEFAULT_BATCH_SIZE = 16
DEFAULT_GRAD_ACCUM_STEPS = 1
DEFAULT_LEARNING_RATE = 2e-4
DEFAULT_WARMUP_RATIO = 0.05
DEFAULT_LOGGING_STEPS = 5
DEFAULT_SAVE_STEPS = 50
DEFAULT_MAX_SEQ_LEN = 256
DEFAULT_LORA_R = 16
DEFAULT_LORA_ALPHA = 32
DEFAULT_LORA_DROPOUT = 0.05
DEFAULT_SEED = 42
DEFAULT_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")
DEFAULT_REPORT_TO = "wandb"
DEFAULT_WANDB_PROJECT = "paper-pilot"
DEFAULT_TRAINING_CONFIG = Path("training/training_config.yaml")


def _router_defaults_from_section(section: dict[str, Any]) -> dict[str, Any]:
    """将 YAML 中 router_sft 节转为 argparse set_defaults 用的字典（key 为 parser 的 dest）。"""
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
    """Router LoRA SFT 的完整配置：模型、数据/输出路径、训练与 LoRA 超参、日志与 fallback。"""

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
    """单次训练结果：适配器目录、是否 fallback、提示信息及可选的失败原因。"""

    adapter_dir: Path
    used_fallback: bool
    message: str
    fallback_reason: str | None = None


def build_prompt(*, instruction: str, input_text: str, output_text: str) -> str:
    """将一条 Alpaca 样本拼成「Instruction / Input / Response」纯文本，供后续 tokenize。"""
    return (
        "### Instruction:\n"
        f"{instruction.strip()}\n\n"
        "### Input:\n"
        f"{input_text.strip()}\n\n"
        "### Response:\n"
        f"{output_text.strip()}"
    )


def load_router_rows(dataset_path: Path) -> list[dict[str, Any]]:
    """从 JSONL 加载 Router 训练行（Alpaca 风格：instruction / input / output 等）。"""
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
    """在训练失败时创建「类适配器」输出目录并写入元数据，便于流水线在受限环境下继续。"""
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
        fallback_reason=reason,
    )


def run_sft(config: SFTConfig) -> TrainingResult:
    """执行 Router LoRA SFT：加载数据、拼 prompt、真实训练并保存 adapter。若开启 fallback_on_error 且失败则写 fallback 元数据。"""
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
            fallback_reason=None,
        )
    except Exception as exc:
        if not config.fallback_on_error:
            raise
        return _write_fallback_artifact(config=config, reason=str(exc))


def _run_real_training(*, config: SFTConfig, training_texts: list[str]) -> None:
    """实际执行 LoRA 微调：tokenize（定长 + labels 掩码）、挂 LoRA、Trainer 训练并保存 adapter 与 tokenizer。"""
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
            padding="max_length",
            return_tensors=None,
        )
        # 因果 LM：labels 与 input_ids 一致，padding 位置填 -100，不参与 loss 计算
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        labels = [
            [tid if m else -100 for tid, m in zip(ids, mask)]
            for ids, mask in zip(input_ids, attention_mask)
        ]
        tokenized["labels"] = labels
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
        target_modules=config.target_modules,
    )
    model = get_peft_model(base_model, lora_config)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    if config.report_to == "wandb":
        os.environ.setdefault("WANDB_PROJECT", config.wandb_project)

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
    """构建 Router SFT 的 CLI 解析器；默认值会与 YAML 在 parse_config 中合并。"""
    parser = argparse.ArgumentParser(description="Train Router LoRA adapter with SFT.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_TRAINING_CONFIG,
        help="Path to training_config.yaml (defaults used if file exists).",
    )
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
    parser.add_argument(
        "--target-modules",
        type=str,
        default=",".join(DEFAULT_TARGET_MODULES),
        help="Comma-separated LoRA target modules.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed.")
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
    """从 argv 解析 SFTConfig；若存在配置文件则先用 router_sft 节覆盖默认值。"""
    argv = argv if argv is not None else []
    parser = build_arg_parser()
    # Resolve --config path from argv so we can load defaults before full parse
    config_path = DEFAULT_TRAINING_CONFIG
    if "--config" in argv:
        i = argv.index("--config")
        if i + 1 < len(argv):
            config_path = Path(argv[i + 1])
    section = load_training_section(config_path, "router_sft")
    defaults = {k: v for k, v in _router_defaults_from_section(section).items() if v is not None}
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
    """CLI 入口：加载 .env、解析配置、执行 SFT 并打印结果；若 fallback 则打印原因。"""
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
    if result.used_fallback and result.fallback_reason:
        print(f"reason={result.fallback_reason}")


if __name__ == "__main__":
    main()
