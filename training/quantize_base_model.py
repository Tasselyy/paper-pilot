"""AWQ INT4 quantization script for the base Qwen model."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from training.config_loader import load_training_section
from training.data.generate_critic_sft_data import generate_critic_sft_dataset
from training.data.generate_router_data import generate_router_dataset

DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_OUTPUT_DIR = Path("training/artifacts/qwen2.5-7b-awq")
DEFAULT_BITS = 4
DEFAULT_GROUP_SIZE = 128
DEFAULT_ZERO_POINT = True
DEFAULT_CALIB_SAMPLES = 128
DEFAULT_SEED = 42
DEFAULT_TRAINING_CONFIG = Path("training/training_config.yaml")


def _quantize_defaults_from_section(section: dict[str, Any]) -> dict[str, Any]:
    """Convert quantize YAML section to argparse defaults."""
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
    """Quantization settings and output locations."""

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
    """Outcome of one quantization run."""

    output_dir: Path
    used_fallback: bool
    message: str


def run_quantization(config: QuantizeConfig) -> QuantizeResult:
    """Run AWQ quantization or fallback metadata generation."""
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
    try:
        from awq import AutoAWQForCausalLM
        from transformers import AutoTokenizer
    except Exception as exc:
        raise RuntimeError("Missing AWQ dependencies. Install with: pip install -e .[training]") from exc

    model = AutoAWQForCausalLM.from_pretrained(config.model_name)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    calibration_texts = _build_calibration_texts(config.calib_samples, seed=config.seed)
    quant_config = {
        "w_bit": int(config.bits),
        "q_group_size": int(config.group_size),
        "zero_point": bool(config.zero_point),
    }
    model.quantize(tokenizer, quant_config=quant_config, calib_data=calibration_texts)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_quantized(str(config.output_dir))
    tokenizer.save_pretrained(str(config.output_dir))


def _build_calibration_texts(calib_samples: int, *, seed: int) -> list[str]:
    router_rows = generate_router_dataset(samples_per_intent=max(1, calib_samples // 10), seed=seed)
    critic_rows = generate_critic_sft_dataset(num_samples=max(1, calib_samples), seed=seed + 1)
    texts: list[str] = []
    for row in router_rows:
        texts.append(str(row.get("input", "")))
    for row in critic_rows:
        texts.append(str(row.get("input", "")))
    return texts[:calib_samples]


def _write_fallback_artifact(config: QuantizeConfig, reason: str) -> QuantizeResult:
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
    config = parse_config(argv)
    result = run_quantization(config)
    print(result.message)
    print(f"output_dir={result.output_dir}")
    print(f"used_fallback={result.used_fallback}")


if __name__ == "__main__":
    main()
