"""Unit tests for src/config.py — configuration loading and validation."""

from __future__ import annotations

import os
import textwrap
from pathlib import Path

import pytest

from src.config import (
    AgentConfig,
    LLMConfig,
    MCPConfig,
    MCPConnectionConfig,
    MemoryConfig,
    Settings,
    TracingConfig,
    _resolve_env_vars,
    load_settings,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MINIMAL_YAML = textwrap.dedent("""\
    llm:
      provider: openai
      model: gpt-4o-mini
    mcp:
      connections:
        rag_server:
          transport: stdio
          command: python
          args: ["-m", "rag_server"]
""")

FULL_YAML = textwrap.dedent("""\
    llm:
      provider: openai
      model: gpt-4o-mini
      temperature: 0.5
      api_key: ${TEST_API_KEY}
    mcp:
      connections:
        rag_server:
          transport: stdio
          command: python
          args: ["-m", "rag_server"]
    agent:
      max_retries: 3
      max_react_steps: 8
    memory:
      memory_file: custom/memory.jsonl
    tracing:
      trace_dir: custom/traces
      trace_file: custom_trace.jsonl
""")


@pytest.fixture()
def minimal_config_file(tmp_path: Path) -> Path:
    """Write a minimal valid config and return its path."""
    p = tmp_path / "settings.yaml"
    p.write_text(MINIMAL_YAML, encoding="utf-8")
    return p


@pytest.fixture()
def full_config_file(tmp_path: Path) -> Path:
    """Write a full config with env var placeholder and return its path."""
    p = tmp_path / "settings.yaml"
    p.write_text(FULL_YAML, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Pydantic model unit tests
# ---------------------------------------------------------------------------


class TestLLMConfig:
    """Tests for LLMConfig validation."""

    def test_llm_config_valid(self) -> None:
        cfg = LLMConfig(provider="openai", model="gpt-4o-mini")
        assert cfg.provider == "openai"
        assert cfg.model == "gpt-4o-mini"
        assert cfg.temperature == 0.3  # default
        assert cfg.api_key is None  # default

    def test_llm_config_missing_provider_raises(self) -> None:
        with pytest.raises(ValueError):
            LLMConfig(model="gpt-4o-mini")  # type: ignore[call-arg]

    def test_llm_config_missing_model_raises(self) -> None:
        with pytest.raises(ValueError):
            LLMConfig(provider="openai")  # type: ignore[call-arg]

    def test_llm_config_temperature_bounds(self) -> None:
        with pytest.raises(ValueError):
            LLMConfig(provider="openai", model="x", temperature=-0.1)
        with pytest.raises(ValueError):
            LLMConfig(provider="openai", model="x", temperature=2.1)


class TestMCPConnectionConfig:
    """Tests for MCPConnectionConfig validation."""

    def test_stdio_valid(self) -> None:
        cfg = MCPConnectionConfig(transport="stdio", command="python")
        assert cfg.transport == "stdio"
        assert cfg.command == "python"

    def test_stdio_missing_command_raises(self) -> None:
        with pytest.raises(ValueError, match="requires 'command'"):
            MCPConnectionConfig(transport="stdio")

    def test_http_valid(self) -> None:
        cfg = MCPConnectionConfig(
            transport="streamable-http", url="http://localhost:8000"
        )
        assert cfg.url == "http://localhost:8000"

    def test_http_missing_url_raises(self) -> None:
        with pytest.raises(ValueError, match="requires 'url'"):
            MCPConnectionConfig(transport="streamable-http")


class TestSettings:
    """Tests for top-level Settings model."""

    def test_settings_with_defaults(self) -> None:
        settings = Settings(
            llm=LLMConfig(provider="openai", model="gpt-4o-mini"),
            mcp=MCPConfig(
                connections={
                    "rag": MCPConnectionConfig(transport="stdio", command="python")
                }
            ),
        )
        # Defaults applied
        assert settings.agent.max_retries == 2
        assert settings.agent.max_react_steps == 5
        assert settings.memory.memory_file == "data/long_term_memory.jsonl"
        assert settings.tracing.trace_dir == "data/traces"
        assert settings.vllm.enabled is False

    def test_settings_missing_llm_raises(self) -> None:
        with pytest.raises(ValueError):
            Settings(
                mcp=MCPConfig(
                    connections={
                        "rag": MCPConnectionConfig(
                            transport="stdio", command="python"
                        )
                    }
                ),
            )  # type: ignore[call-arg]

    def test_settings_missing_mcp_raises(self) -> None:
        with pytest.raises(ValueError):
            Settings(
                llm=LLMConfig(provider="openai", model="gpt-4o-mini"),
            )  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Env var resolution
# ---------------------------------------------------------------------------


class TestEnvVarResolution:
    """Tests for _resolve_env_vars helper."""

    def test_resolve_single_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MY_KEY", "secret123")
        assert _resolve_env_vars("${MY_KEY}") == "secret123"

    def test_resolve_inline_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HOST", "localhost")
        assert _resolve_env_vars("http://${HOST}:8000") == "http://localhost:8000"

    def test_resolve_missing_env_var_returns_none(self) -> None:
        # Remove the var if it exists
        os.environ.pop("NONEXISTENT_VAR_XYZ", None)
        assert _resolve_env_vars("${NONEXISTENT_VAR_XYZ}") is None

    def test_resolve_nested_dict(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VAL", "42")
        data = {"a": {"b": "${VAL}"}}
        assert _resolve_env_vars(data) == {"a": {"b": "42"}}

    def test_resolve_list(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ITEM", "x")
        assert _resolve_env_vars(["${ITEM}", "fixed"]) == ["x", "fixed"]

    def test_non_string_passthrough(self) -> None:
        assert _resolve_env_vars(42) == 42
        assert _resolve_env_vars(True) is True
        assert _resolve_env_vars(None) is None


# ---------------------------------------------------------------------------
# load_settings integration
# ---------------------------------------------------------------------------


class TestLoadSettings:
    """Tests for the load_settings function."""

    def test_load_minimal_config(self, minimal_config_file: Path) -> None:
        settings = load_settings(minimal_config_file)
        assert settings.llm.provider == "openai"
        assert settings.llm.model == "gpt-4o-mini"
        assert "rag_server" in settings.mcp.connections

    def test_load_full_config_with_env(
        self,
        full_config_file: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("TEST_API_KEY", "sk-test-key")
        settings = load_settings(full_config_file)
        assert settings.llm.api_key == "sk-test-key"
        assert settings.llm.temperature == 0.5
        assert settings.agent.max_retries == 3
        assert settings.agent.max_react_steps == 8
        assert settings.memory.memory_file == "custom/memory.jsonl"
        assert settings.tracing.trace_dir == "custom/traces"

    def test_load_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_settings(tmp_path / "does_not_exist.yaml")

    def test_load_missing_required_section_raises(self, tmp_path: Path) -> None:
        """Config with no llm section should raise ValueError."""
        bad_yaml = tmp_path / "bad.yaml"
        bad_yaml.write_text("agent:\n  max_retries: 1\n", encoding="utf-8")
        with pytest.raises(ValueError):
            load_settings(bad_yaml)

    def test_load_with_overrides(self, minimal_config_file: Path) -> None:
        settings = load_settings(
            minimal_config_file,
            overrides={"agent": {"max_retries": 10}},
        )
        assert settings.agent.max_retries == 10

    def test_load_default_config_path(self) -> None:
        """Verify the real config/settings.yaml loads without error."""
        settings = load_settings("config/settings.yaml")
        assert settings.llm.provider == "openai"
