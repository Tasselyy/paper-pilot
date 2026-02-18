"""Export trained Router/Critic artifacts into a LocalModelManager-loadable package.

This script supports two common cases:
1. Exporting a full model directory directly.
2. Merging a LoRA adapter into a base model, then exporting merged weights.

The exported directory can be loaded by ``LocalModelManager`` and, on CUDA
machines, can be consumed with 4-bit loading enabled.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

DEFAULT_MODEL_PATH = Path("training/artifacts/router_lora_adapter")
DEFAULT_OUTPUT_DIR = Path("training/artifacts/router_quantized_model")


@dataclass(slots=True)
class ExportConfig:
    """Configuration for quantized export workflow."""

    model_path: Path
    output_dir: Path
    base_model_name_or_path: str | None
    validate_with_local_manager: bool
    safe_serialization: bool
    fallback_on_error: bool


@dataclass(slots=True)
class ExportResult:
    """Result of one export attempt."""

    model_dir: Path
    used_fallback: bool
    message: str


def _write_fallback_artifact(config: ExportConfig, reason: str) -> ExportResult:
    """Write fallback metadata when export cannot proceed."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "status": "fallback",
        "reason": reason,
        "config": {
            **asdict(config),
            "model_path": str(config.model_path),
            "output_dir": str(config.output_dir),
        },
    }
    metadata_path = config.output_dir / "fallback_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return ExportResult(
        model_dir=config.output_dir,
        used_fallback=True,
        message=(
            "Quantized export fallback completed. "
            f"Artifact directory prepared at: {config.output_dir}"
        ),
    )


def _write_export_metadata(
    *,
    output_dir: Path,
    export_config: ExportConfig,
    merged_lora: bool,
) -> None:
    """Persist export metadata for reproducibility."""
    metadata_path = output_dir / "export_metadata.json"
    payload = {
        "status": "ok",
        "model_path": str(export_config.model_path),
        "output_dir": str(export_config.output_dir),
        "base_model_name_or_path": export_config.base_model_name_or_path,
        "merged_lora": merged_lora,
        "validate_with_local_manager": export_config.validate_with_local_manager,
        "safe_serialization": export_config.safe_serialization,
    }
    metadata_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _run_real_export(config: ExportConfig) -> None:
    """Execute model export (and optional LoRA merge) with transformers APIs."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:  # pragma: no cover - depends on optional deps
        raise RuntimeError(
            "Missing export dependencies. Install with: "
            "pip install -e .[training,local-models]"
        ) from exc

    model_source = str(config.model_path)
    merged_lora = False

    if config.base_model_name_or_path:
        try:
            from peft import PeftModel
        except Exception as exc:  # pragma: no cover - depends on optional deps
            raise RuntimeError(
                "LoRA merge requires `peft`. Install with: pip install -e .[training]"
            ) from exc

        base_model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(base_model, model_source)
        model = model.merge_and_unload()
        merged_lora = True
        tokenizer = _load_tokenizer_with_fallback(
            primary=model_source,
            fallback=config.base_model_name_or_path,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_source, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_source, use_fast=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(
        str(config.output_dir),
        safe_serialization=config.safe_serialization,
    )
    tokenizer.save_pretrained(str(config.output_dir))
    _write_export_metadata(
        output_dir=config.output_dir,
        export_config=config,
        merged_lora=merged_lora,
    )

    if config.validate_with_local_manager:
        _validate_export_with_local_model_manager(config.output_dir)


def _load_tokenizer_with_fallback(*, primary: str, fallback: str) -> Any:
    """Load tokenizer from primary path, fallback to base model when missing."""
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(primary, use_fast=True)
    except Exception:
        return AutoTokenizer.from_pretrained(fallback, use_fast=True)


def _validate_export_with_local_model_manager(model_dir: Path) -> None:
    """Validate exported model directory can be loaded by LocalModelManager."""
    from src.models.loader import LocalModelManager

    manager = LocalModelManager(router_model_path=str(model_dir))
    manager.load()


def run_export(config: ExportConfig) -> ExportResult:
    """Run export workflow with fallback mode support."""
    try:
        _run_real_export(config)
        return ExportResult(
            model_dir=config.output_dir,
            used_fallback=False,
            message=(
                "Quantized export finished. "
                f"Model package saved to: {config.output_dir}"
            ),
        )
    except Exception as exc:
        if not config.fallback_on_error:
            raise
        return _write_fallback_artifact(config=config, reason=str(exc))


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Export model artifacts for LocalModelManager 4-bit loading."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to model or adapter directory to export.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write exported model package.",
    )
    parser.add_argument(
        "--base-model",
        default=None,
        help="Optional base model path/id (required when --model-path is LoRA adapter only).",
    )
    parser.add_argument(
        "--no-validate-load",
        action="store_true",
        help="Skip LocalModelManager load validation after export.",
    )
    parser.add_argument(
        "--unsafe-serialization",
        action="store_true",
        help="Disable safetensors serialization and use legacy format.",
    )
    parser.add_argument(
        "--no-fallback",
        action="store_true",
        help="Disable fallback mode and raise export errors directly.",
    )
    return parser


def parse_config(argv: list[str] | None = None) -> ExportConfig:
    """Parse CLI args into ``ExportConfig``."""
    args = build_arg_parser().parse_args(argv)
    return ExportConfig(
        model_path=Path(args.model_path),
        output_dir=Path(args.output_dir),
        base_model_name_or_path=(
            str(args.base_model) if args.base_model else None
        ),
        validate_with_local_manager=not bool(args.no_validate_load),
        safe_serialization=not bool(args.unsafe_serialization),
        fallback_on_error=not bool(args.no_fallback),
    )


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    config = parse_config(argv)
    result = run_export(config)
    print(result.message)
    print(f"model_dir={result.model_dir}")
    print(f"used_fallback={result.used_fallback}")


if __name__ == "__main__":
    main()
