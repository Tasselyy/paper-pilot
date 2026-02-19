"""Integration test for long-term memory across multi-turn execution (H5).

Verifies that when the main graph is wired with ``LongTermMemory``:
- First turn persists at least one fact to the configured JSONL memory file.
- Second turn can recall previously stored facts into ``accumulated_facts``.

This test uses placeholder strategy/critic nodes (no live MCP/LLM required)
while exercising real ``create_load_memory_node`` / ``create_save_memory_node``
through ``build_main_graph(long_term_memory=...)``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.agent.graph import build_main_graph
from src.memory.long_term import LongTermMemory


@pytest.mark.asyncio
async def test_long_term_memory_recall_across_two_turns(tmp_path: Path) -> None:
    """Second turn should recall facts memorized during first turn."""
    memory_file = tmp_path / "long_term_memory.jsonl"
    long_term_memory = LongTermMemory(memory_file=str(memory_file))

    graph = build_main_graph(long_term_memory=long_term_memory)

    first_result = await graph.ainvoke(
        {"question": "What is LoRA?"},
        config={"configurable": {"thread_id": "memory-e2e-turn-1"}},
    )
    assert first_result is not None
    assert memory_file.exists(), "First turn should persist memory JSONL"
    memory_lines = [
        line
        for line in memory_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(memory_lines) >= 1, "Expected at least one stored fact after turn 1"

    second_result = await graph.ainvoke(
        {"question": "How is RAG configured in this project?"},
        config={"configurable": {"thread_id": "memory-e2e-turn-2"}},
    )

    accumulated_facts = second_result.get("accumulated_facts", [])
    assert accumulated_facts, "Second turn should recall facts from turn 1"
    assert any(
        "RAG" in fact or "configured" in fact.lower()
        for fact in accumulated_facts
    ), "Recalled facts should include overlap with memory-relevant keywords"
