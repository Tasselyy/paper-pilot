"""将训练好的 Router/Critic 产物导出为 LocalModelManager 可加载的模型包。

支持两种用法：
1. 直接导出：--model-path 指向已合并好的完整模型目录，原样保存到 output-dir。
2. 先合并再导出：指定 --base-model 与 --model-path（LoRA 适配器目录），脚本会先 merge LoRA 再保存。

导出目录可被 src.models.loader.LocalModelManager 加载，在 CUDA 环境下可配合 4-bit 加载。支持 fallback 模式。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# ---------- 默认路径 ----------
DEFAULT_MODEL_PATH = Path("training/artifacts/router_lora_adapter")
DEFAULT_OUTPUT_DIR = Path("training/artifacts/router_quantized_model")


@dataclass(slots=True)
class ExportConfig:
    """导出配置：模型/适配器路径、输出目录、可选基座（用于合并 LoRA）、是否校验加载与 fallback。"""

    model_path: Path
    output_dir: Path
    base_model_name_or_path: str | None
    validate_with_local_manager: bool
    safe_serialization: bool
    fallback_on_error: bool


@dataclass(slots=True)
class ExportResult:
    """单次导出结果：模型目录、是否 fallback、提示信息。"""

    model_dir: Path
    used_fallback: bool
    message: str


def _write_fallback_artifact(config: ExportConfig, reason: str) -> ExportResult:
    """导出无法进行时在 output_dir 写入 fallback 元数据，便于流水线继续。"""
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
    """在输出目录写入 export_metadata.json，记录路径、是否合并 LoRA 等，便于复现与排查。"""
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
    """实际执行导出：若指定了 base_model 则先加载基座、挂载 LoRA 并 merge，再保存；否则直接加载 model_path 并保存。"""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:  # pragma: no cover - depends on optional deps
        raise RuntimeError(
            "Missing export dependencies. Install with: "
            "pip install -e .[training,local-models]"
        ) from exc

    model_source = str(config.model_path)
    merged_lora = False

    # 若指定了基座，则按「基座 + LoRA 适配器」合并后导出
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
        # 优先从适配器目录加载 tokenizer，没有则用基座
        tokenizer = _load_tokenizer_with_fallback(
            primary=model_source,
            fallback=config.base_model_name_or_path,
        )
    else:
        # 直接导出：model_path 即为完整模型目录
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
    """优先从 primary（如适配器目录）加载 tokenizer，失败则从 fallback（基座）加载。"""
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(primary, use_fast=True)
    except Exception:
        return AutoTokenizer.from_pretrained(fallback, use_fast=True)


def _validate_export_with_local_model_manager(model_dir: Path) -> None:
    """用 LocalModelManager 加载导出目录，校验是否可被应用侧正常使用。"""
    from src.models.loader import LocalModelManager

    manager = LocalModelManager(router_model_path=str(model_dir))
    manager.load()


def run_export(config: ExportConfig) -> ExportResult:
    """执行导出流程；异常时若开启 fallback_on_error 则写 fallback 元数据并返回，否则抛异常。"""
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
    """构建导出脚本的 CLI 解析器。"""
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
    """从 argv 解析 ExportConfig（本脚本不读 YAML，仅用 CLI 与默认值）。"""
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
    """CLI 入口：解析配置、执行导出并打印结果。"""
    config = parse_config(argv)
    result = run_export(config)
    print(result.message)
    print(f"model_dir={result.model_dir}")
    print(f"used_fallback={result.used_fallback}")


if __name__ == "__main__":
    main()
