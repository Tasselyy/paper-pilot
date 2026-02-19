"""Integration test for trace JSONL persistence (H7).

Runs the simple path in dry-run mode and verifies that the configured
trace JSONL file is written with the expected top-level fields.
"""

from __future__ import annotations

import json
from pathlib import Path

from main import run_agent


def _write_test_settings(config_path: Path, trace_dir: Path, trace_file: str) -> None:
    """Write a minimal valid settings YAML for dry-run integration tests."""
    content = f"""
llm:
  provider: openai
  model: gpt-4o-mini
  temperature: 0.3
  api_key: null

mcp:
  rag_default_collection: null
  connections:
    rag_server:
      transport: streamable-http
      url: http://127.0.0.1:8000/mcp

agent:
  max_retries: 2
  max_react_steps: 5
  router_model_path: null
  critic_model_path: null

memory:
  memory_file: "{(trace_dir / "memory.jsonl").as_posix()}"

tracing:
  trace_dir: "{trace_dir.as_posix()}"
  trace_file: "{trace_file}"
"""
    config_path.write_text(content.strip() + "\n", encoding="utf-8")


async def test_trace_jsonl_contains_required_fields_after_simple_run(tmp_path: Path) -> None:
    """Dry-run execution should append a parseable trace with key fields."""
    config_path = tmp_path / "settings.yaml"
    trace_dir = tmp_path / "traces"
    trace_file = "integration_trace.jsonl"
    trace_path = trace_dir / trace_file
    _write_test_settings(config_path, trace_dir, trace_file)

    result = await run_agent(
        question="What is LoRA?",
        config_path=str(config_path),
        dry_run=True,
        verbose=False,
    )
    assert result.get("final_answer"), "Expected non-empty final_answer from dry-run"

    assert trace_path.exists(), f"Expected trace file to exist: {trace_path}"
    lines = [line for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert lines, "Trace JSONL should contain at least one record"

    last_record = json.loads(lines[-1])
    assert "intent" in last_record
    assert "strategy_executed" in last_record
    assert "critic_verdict" in last_record
    assert "final_answer" in last_record

    assert isinstance(last_record["intent"], dict)
    assert last_record["strategy_executed"] == "simple"
    assert isinstance(last_record["critic_verdict"], dict)
    assert isinstance(last_record["final_answer"], str)
    assert last_record["final_answer"], "final_answer in trace should be non-empty"
