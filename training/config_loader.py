"""从 training_config.yaml 加载训练默认配置。

各训练脚本在解析命令行前会先调用本模块，用 YAML 中的值作为 argparse 的默认值；
命令行传入的参数会覆盖配置文件中的对应项。若配置文件不存在或缺少某 section，则返回空字典。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# 与 training_config.yaml 中一级 key 对应，用于按 section 读取配置
SECTION_NAMES = ("router_sft", "critic_sft", "dpo", "quantize")


def load_training_section(config_path: Path | str, section: str) -> dict[str, Any]:
    """从训练配置 YAML 中读取指定 section 的键值对。

    Args:
        config_path: 配置文件路径（如 training/training_config.yaml）。
        section: 节名，如 "critic_sft"、"dpo"。

    Returns:
        该 section 下的配置字典；若文件不存在、无法解析或没有该 section，返回 {}。
    """
    path = Path(config_path)
    if not path.is_file():
        return {}
    try:
        import yaml
    except ImportError:
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {}
    return dict(data.get(section) or {})
