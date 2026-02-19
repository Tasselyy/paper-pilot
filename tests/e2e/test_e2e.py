"""End-to-end tests against a live RAG server (H2).

These tests run the real agent entry flow (non-dry-run) and validate that
answers are produced with citations for a small set of questions.

The suite is marked ``e2e`` and requires:
- OPENAI_API_KEY in environment
- A reachable MCP RAG server configured in ``config/settings.yaml``
"""

from __future__ import annotations

import socket
from urllib.parse import urlparse

import pytest

from main import run_agent
from src.config import load_settings


def _host_port_from_url(url: str) -> tuple[str, int] | None:
    """Extract host/port from URL, defaulting port by scheme."""
    parsed = urlparse(url)
    if not parsed.hostname:
        return None
    if parsed.port is not None:
        return parsed.hostname, parsed.port
    if parsed.scheme == "https":
        return parsed.hostname, 443
    if parsed.scheme == "http":
        return parsed.hostname, 80
    return None


def _is_tcp_reachable(host: str, port: int, timeout_sec: float = 2.0) -> bool:
    """Return True if a TCP connection can be opened to host:port."""
    try:
        with socket.create_connection((host, port), timeout=timeout_sec):
            return True
    except OSError:
        return False


@pytest.fixture(scope="module")
def live_config_path() -> str:
    """Return config path if live E2E prerequisites are satisfied.

    Skips module when required external services are not available.
    """
    config_path = "config/settings.yaml"
    settings = load_settings(config_path)
    if not settings.llm.api_key:
        pytest.skip(
            "OPENAI_API_KEY is not resolved from environment/.env; skipping live e2e tests."
        )
    if not settings.mcp.connections:
        pytest.skip("No MCP connections configured; skipping live e2e tests.")

    # H2 targets a real running RAG server; verify configured endpoint/process.
    first_name = next(iter(settings.mcp.connections))
    conn = settings.mcp.connections[first_name]
    if conn.transport == "streamable-http":
        if not conn.url:
            pytest.skip("MCP HTTP connection has no URL; skipping live e2e tests.")
        host_port = _host_port_from_url(conn.url)
        if host_port is None:
            pytest.skip(f"Cannot parse MCP URL: {conn.url}")
        host, port = host_port
        if not _is_tcp_reachable(host, port):
            pytest.skip(
                f"MCP server {conn.url} is unreachable; start RAG server before e2e."
            )
    elif conn.transport == "stdio":
        if not conn.command:
            pytest.skip("MCP stdio connection missing command; skipping live e2e.")
    else:
        pytest.skip(f"Unsupported MCP transport for e2e: {conn.transport}")

    return config_path


@pytest.mark.e2e
@pytest.mark.parametrize(
    ("question", "keyword_hint"),
    [
        ("What is LoRA?", "LoRA"),
        ("Compare LoRA and QLoRA in memory usage.", "QLoRA"),
    ],
)
async def test_live_rag_e2e_answer_has_reasonable_output(
    live_config_path: str,
    question: str,
    keyword_hint: str,
) -> None:
    """Live E2E should return non-empty answer with citations."""
    result = await run_agent(
        question=question,
        config_path=live_config_path,
        dry_run=False,
        verbose=False,
    )

    final_answer = result.get("final_answer", "")
    assert final_answer, "final_answer must be non-empty"
    assert len(final_answer) > 40, "final_answer should be reasonably detailed"
    assert "Sources" in final_answer, "final_answer should include citation section"
    assert keyword_hint.lower() in final_answer.lower()

    retrieved_contexts = result.get("retrieved_contexts", [])
    assert len(retrieved_contexts) > 0, "live retrieval should return contexts"
