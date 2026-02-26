"""基座 Qwen 模型的 AWQ INT4 量化脚本。

使用 AutoAWQ 对 HuggingFace 上的基座模型做 4-bit 量化。校准数据优先从 training/data/router_train.jsonl
与 critic_sft_train.jsonl 读取（需先运行 generate_all_with_llm.py）；无文件时使用最小占位文本。
输出目录可被 vLLM 等加载；支持 fallback 模式。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from training.config_loader import load_training_section

# ---------- 默认配置 ----------
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_OUTPUT_DIR = Path("training/artifacts/qwen2.5-7b-awq")
DEFAULT_BITS = 4
DEFAULT_GROUP_SIZE = 128
DEFAULT_ZERO_POINT = True
DEFAULT_CALIB_SAMPLES = 128
DEFAULT_SEED = 42
DEFAULT_TRAINING_CONFIG = Path("training/training_config.yaml")


def _quantize_defaults_from_section(section: dict[str, Any]) -> dict[str, Any]:
    """将 YAML 中 quantize 节转为 argparse set_defaults 用的字典。"""
    if not section:
        return {}
    return {
        "model_name": section.get("model_name"),
        "output_dir": Path(section["output_dir"]) if section.get("output_dir") else None,
        "bits": section.get("bits"),
        "group_size": section.get("group_size"),
        "calib_samples": section.get("calib_samples"),
        "seed": section.get("seed"),
        "no_zero_point": not section.get("zero_point", True),
        "no_fallback": not section.get("fallback_on_error", True),
    }


@dataclass(slots=True)
class QuantizeConfig:
    """量化配置：模型名、输出目录、比特数/分组/零点和校准样本数、随机种子与 fallback 开关。"""

    model_name: str
    output_dir: Path
    bits: int
    group_size: int
    zero_point: bool
    calib_samples: int
    seed: int
    fallback_on_error: bool


@dataclass(slots=True)
class QuantizeResult:
    """单次量化运行结果：输出目录、是否 fallback、提示信息。"""

    output_dir: Path
    used_fallback: bool
    message: str


def run_quantization(config: QuantizeConfig) -> QuantizeResult:
    """执行 AWQ 量化；异常时若开启 fallback_on_error 则写 fallback 元数据并返回，否则抛异常。"""
    try:
        _run_real_quantization(config)
        return QuantizeResult(
            output_dir=config.output_dir,
            used_fallback=False,
            message=f"AWQ quantization finished. Artifacts saved to: {config.output_dir}",
        )
    except Exception as exc:
        if not config.fallback_on_error:
            raise
        return _write_fallback_artifact(config=config, reason=str(exc))


def _run_real_quantization(config: QuantizeConfig) -> None:
    """实际执行 AWQ 量化：加载模型与 tokenizer、用合成数据做校准、量化并保存到 output_dir。"""
    try:
        from awq import AutoAWQForCausalLM
        from transformers import AutoTokenizer
    except Exception as exc:
        raise RuntimeError("Missing AWQ dependencies. Install with: pip install -e .[training]") from exc

    model = AutoAWQForCausalLM.from_pretrained(config.model_name)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    calibration_texts = _build_calibration_texts(config.calib_samples, seed=config.seed)
    quant_config = {
        # w_bit: 权重量化比特数；q_group_size: 分组大小；zero_point: 是否使用零点
        "w_bit": int(config.bits),
        "q_group_size": int(config.group_size),
        "zero_point": bool(config.zero_point),
    }
    model.quantize(tokenizer, quant_config=quant_config, calib_data=calibration_texts)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_quantized(str(config.output_dir))
    tokenizer.save_pretrained(str(config.output_dir))


def _load_calibration_texts_from_jsonl(
    router_path: Path,
    critic_path: Path,
    max_samples: int,
) -> list[str]:
    """从 LLM 生成的 JSONL 中读取 input 文本作为校准数据；文件不存在或为空时返回空列表。"""
    texts: list[str] = []
    for path in (router_path, critic_path):
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    t = row.get("input") or row.get("prompt")
                    if t:
                        texts.append(str(t))
                except json.JSONDecodeError:
                    continue
                if len(texts) >= max_samples:
                    return texts[:max_samples]
    return texts[:max_samples]


def _minimal_calibration_texts(n: int) -> list[str]:
    """无 JSONL 时的最小校准文本（与 Router/Critic 任务句式接近），供量化使用。"""
    pool = [
        "Classify the user question into one intent type. Allowed types: factual, comparative, multi_hop, exploratory, follow_up.",
        "What is LoRA and when would you use it?",
        "Evaluate the answer quality for the given question and retrieved context. Score rubric: score, completeness, faithfulness.",
        "Question: How does RAG improve factual consistency? Draft Answer: RAG retrieves documents and conditions the model.",
    ]
    return (pool * max(1, (n + len(pool) - 1) // len(pool)))[:n]


def _build_calibration_texts(calib_samples: int, *, seed: int) -> list[str]:
    """用 Router/Critic JSONL 的 input 文本做校准；无文件时用最小占位文本。总条数不超过 calib_samples。"""
    root = Path(__file__).resolve().parent
    data_dir = root / "data"
    router_path = data_dir / "router_train.jsonl"
    critic_path = data_dir / "critic_sft_train.jsonl"
    texts = _load_calibration_texts_from_jsonl(router_path, critic_path, calib_samples)
    if not texts:
        texts = _minimal_calibration_texts(calib_samples)
    return texts[:calib_samples]


def _write_fallback_artifact(config: QuantizeConfig, reason: str) -> QuantizeResult:
    """量化失败时在 output_dir 写入 fallback_metadata.json，便于流水线继续。"""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "fallback",
        "reason": reason,
        "config": {**asdict(config), "output_dir": str(config.output_dir)},
    }
    path = config.output_dir / "fallback_metadata.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return QuantizeResult(
        output_dir=config.output_dir,
        used_fallback=True,
        message=f"Quantization fallback completed. Metadata saved to: {path}",
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """构建量化脚本的 CLI 解析器；默认值会与 YAML quantize 节在 parse_config 中合并。"""
    parser = argparse.ArgumentParser(description="Quantize base model with AWQ.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_TRAINING_CONFIG,
        help="Path to training_config.yaml (defaults used if file exists).",
    )
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bits", type=int, default=DEFAULT_BITS)
    parser.add_argument("--group-size", type=int, default=DEFAULT_GROUP_SIZE)
    parser.add_argument("--calib-samples", type=int, default=DEFAULT_CALIB_SAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--no-zero-point",
        action="store_true",
        help="Disable zero-point quantization.",
    )
    parser.add_argument("--no-fallback", action="store_true")
    return parser


def parse_config(argv: list[str] | None = None) -> QuantizeConfig:
    """从 argv 解析 QuantizeConfig；若存在 training_config.yaml 则用其中 quantize 节覆盖默认值。"""
    argv = argv if argv is not None else []
    parser = build_arg_parser()
    config_path = DEFAULT_TRAINING_CONFIG
    if "--config" in argv:
        i = argv.index("--config")
        if i + 1 < len(argv):
            config_path = Path(argv[i + 1])
    section = load_training_section(config_path, "quantize")
    defaults = {k: v for k, v in _quantize_defaults_from_section(section).items() if v is not None}
    parser.set_defaults(**defaults)
    args = parser.parse_args(argv)
    return QuantizeConfig(
        model_name=str(args.model_name),
        output_dir=Path(args.output_dir),
        bits=int(args.bits),
        group_size=int(args.group_size),
        zero_point=not bool(args.no_zero_point),
        calib_samples=int(args.calib_samples),
        seed=int(args.seed),
        fallback_on_error=not bool(args.no_fallback),
    )


def main(argv: list[str] | None = None) -> None:
    """CLI 入口：解析配置、执行量化并打印结果。"""
    config = parse_config(argv)
    result = run_quantization(config)
    print(result.message)
    print(f"output_dir={result.output_dir}")
    print(f"used_fallback={result.used_fallback}")


if __name__ == "__main__":
    main()
