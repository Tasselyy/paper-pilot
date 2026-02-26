"""Configuration loader: reads config/settings.yaml and validates.

Provides Pydantic models for each config section (llm, mcp, agent, memory,
tracing) and a ``load_settings()`` function that reads YAML, resolves
``${ENV_VAR}`` placeholders from the environment, and validates the result.

If a ``.env`` file exists in the project root, it is loaded automatically
before resolving placeholders, so you can put ``OPENAI_API_KEY=sk-...`` etc.
in ``.env`` instead of exporting in the shell.

Raises ``ValueError`` when required fields are missing or invalid.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator

# ---------------------------------------------------------------------------
# Section models
# ---------------------------------------------------------------------------

_ENV_VAR_PATTERN = re.compile(r"\$\{(\w+)\}")


class LLMConfig(BaseModel):
    """Cloud LLM endpoint configuration."""

    provider: str = Field(
        ...,
        description="LLM provider: openai | azure | local",
    )
    model: str = Field(..., description="Model name, e.g. gpt-4o-mini")
    temperature: float = Field(
        default=0.3,
        ge=0.0,
        le=2.0,
        description="Sampling temperature",
    )
    api_key: str | None = Field(
        default=None,
        description="API key (resolved from env at runtime)",
    )


class MCPConnectionConfig(BaseModel):
    """Single MCP server connection."""

    transport: str = Field(
        ...,
        description="Transport type: stdio | streamable-http",
    )
    command: str | None = Field(
        default=None,
        description="Command to start server (stdio transport)",
    )
    args: list[str] = Field(
        default_factory=list,
        description="Arguments for the server command",
    )
    cwd: str | None = Field(
        default=None,
        description="Working directory for the server process (stdio); e.g. RAG project root",
    )
    env: dict[str, str] | None = Field(
        default=None,
        description="Environment variables for the server process (stdio); values support ${VAR}",
    )
    url: str | None = Field(
        default=None,
        description="Server URL (streamable-http transport)",
    )
    mcp_stderr: str | None = Field(
        default=None,
        description=(
            "For stdio: redirect MCP subprocess stderr to avoid pipe blocking. "
            "Use 'devnull' to discard, or a file path (e.g. 'logs/mcp_stderr.log'). "
            "Leave unset for default (parent stderr); set to 'devnull' if the client hangs."
        ),
    )

    @model_validator(mode="after")
    def _check_transport_fields(self) -> "MCPConnectionConfig":
        """Ensure required fields are present for the chosen transport."""
        if self.transport == "stdio" and not self.command:
            raise ValueError(
                "MCP connection with transport='stdio' requires 'command'"
            )
        if self.transport == "streamable-http" and not self.url:
            raise ValueError(
                "MCP connection with transport='streamable-http' requires 'url'"
            )
        return self


class MCPConfig(BaseModel):
    """MCP connections configuration."""

    connections: dict[str, MCPConnectionConfig] = Field(
        ...,
        description="Named MCP server connections",
    )
    rag_default_collection: str | None = Field(
        default=None,
        description="Default RAG collection name for query_knowledge_hub; None = do not restrict",
    )


class AgentConfig(BaseModel):
    """Agent behaviour parameters."""

    max_retries: int = Field(
        default=2,
        ge=0,
        description="Maximum Critic retry rounds",
    )
    max_react_steps: int = Field(
        default=5,
        ge=1,
        description="Maximum ReAct loop steps",
    )
    router_model_path: str | None = Field(
        default=None,
        description="Path to local LoRA router model (optional)",
    )
    critic_model_path: str | None = Field(
        default=None,
        description="Path to local DPO critic model (optional)",
    )


class MemoryConfig(BaseModel):
    """Long-term memory persistence."""

    memory_file: str = Field(
        default="data/long_term_memory.jsonl",
        description="JSONL file for long-term fact memory",
    )


class TracingConfig(BaseModel):
    """Trace output configuration."""

    trace_dir: str = Field(
        default="data/traces",
        description="Directory for trace files",
    )
    trace_file: str = Field(
        default="trace.jsonl",
        description="Trace JSONL filename",
    )


class VLLMConfig(BaseModel):
    """vLLM OpenAI-compatible endpoint configuration."""

    enabled: bool = False
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "token-placeholder"
    router_model: str | None = None
    critic_model: str | None = None


class Settings(BaseModel):
    """Top-level application settings.

    Sections ``llm`` and ``mcp`` are **required**; the rest have defaults.
    """

    llm: LLMConfig
    mcp: MCPConfig
    agent: AgentConfig = Field(default_factory=AgentConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    tracing: TracingConfig = Field(default_factory=TracingConfig)
    vllm: VLLMConfig = Field(default_factory=VLLMConfig)


# ---------------------------------------------------------------------------
# YAML loader helpers
# ---------------------------------------------------------------------------


def _resolve_env_vars(value: Any) -> Any:
    """Recursively resolve ``${VAR}`` placeholders in strings.

    Args:
        value: A scalar, list, or dict (typically parsed from YAML).

    Returns:
        The same structure with environment variables resolved.  Unset
        variables resolve to ``None`` (so Pydantic can apply defaults or
        raise validation errors).
    """
    if isinstance(value, str):
        match = _ENV_VAR_PATTERN.fullmatch(value)
        if match:
            # Whole value is a single env-var reference → return raw
            env_val = os.environ.get(match.group(1))
            return env_val  # may be None
        # Inline substitution (e.g. "prefix_${VAR}_suffix")
        def _replace(m: re.Match) -> str:
            return os.environ.get(m.group(1), "")

        return _ENV_VAR_PATTERN.sub(_replace, value)
    if isinstance(value, dict):
        return {k: _resolve_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_env_vars(item) for item in value]
    return value


def load_settings(
    path: str | Path = "config/settings.yaml",
    *,
    overrides: dict[str, Any] | None = None,
) -> Settings:
    """Load, resolve, validate, and return application ``Settings``.

    Loads ``.env`` from the project root (if present) before resolving
    ``${VAR}`` placeholders in the YAML, so variables like ``OPENAI_API_KEY``
    can be set in ``.env`` instead of the shell.

    Args:
        path: Path to the YAML configuration file.
        overrides: Optional dict merged on top of the YAML data before
            validation (useful for tests or CLI flags).

    Returns:
        A validated ``Settings`` instance.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If required fields are missing or invalid.
    """
    # Load .env from project root so ${OPENAI_API_KEY} etc. are available
    _load_dotenv_once(path)

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    raw_text = config_path.read_text(encoding="utf-8")
    raw_data: dict[str, Any] = yaml.safe_load(raw_text) or {}

    # Resolve environment variable placeholders
    resolved = _resolve_env_vars(raw_data)

    # Apply optional overrides
    if overrides:
        _deep_merge(resolved, overrides)

    # Validate through Pydantic — raises ValidationError (subclass of
    # ValueError) when fields are missing or invalid.
    try:
        return Settings(**resolved)
    except Exception as exc:
        raise ValueError(str(exc)) from exc


def _load_dotenv_once(config_path: str | Path) -> None:
    """Load .env from project root once per process.

    Project root is inferred as the parent of the directory containing
    *config_path* (e.g. when path is ``config/settings.yaml``, project root
    is the repo root). Also tries loading from current working directory.
    """
    if getattr(_load_dotenv_once, "_loaded", False):
        return
    resolved = Path(config_path).resolve()
    # When path is config/settings.yaml, parent is config/, parent.parent is project root
    project_root = resolved.parent.parent
    load_dotenv(project_root / ".env")
    load_dotenv()  # cwd as fallback
    _load_dotenv_once._loaded = True  # type: ignore[attr-defined]


def _deep_merge(base: dict, override: dict) -> None:
    """In-place deep merge *override* into *base*."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
