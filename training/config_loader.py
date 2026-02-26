"""Load training defaults from training_config.yaml.

Scripts use these as parser defaults so CLI args override config file values.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

SECTION_NAMES = ("router_sft", "critic_sft", "dpo", "quantize")


def load_training_section(config_path: Path | str, section: str) -> dict[str, Any]:
    """Load one section from training YAML. Returns {} if file/section missing."""
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
