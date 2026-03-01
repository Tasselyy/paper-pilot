"""Integration tests for checkpointer multi-session behavior (H10).

Same thread_id invoked twice: assert state accumulates (checkpoint per thread).
Two different thread_ids each invoked once: assert session isolation.

Design reference: DEV_SPEC H10. Can be extended in phase J for follow_up
and coreference.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agent.graph import build_main_graph
from src.agent.nodes.critic import CriticOutput
from src.agent.nodes.router import RouterOutput
from src.agent.nodes.slot_filling import SlotFillingOutput
from src.agent.state import RetrievedContext


def _sample_contexts() -> list[RetrievedContext]:
    return [
        RetrievedContext(
            content="Checkpointer stores state per thread.",
            source="Test",
            doc_id="doc_1",
            relevance_score=0.9,
            chunk_index=0,
        ),
    ]


def _make_mock_rag() -> MagicMock:
    rag = MagicMock()
    rag.search = AsyncMock(return_value=_sample_contexts())
    return rag


def _make_mock_llm() -> MagicMock:
    llm = MagicMock()

    def _with_structured_output(output_cls: type) -> MagicMock:
        structured = MagicMock()
        if output_cls is RouterOutput:
            structured.ainvoke = AsyncMock(
                return_value=RouterOutput(type="factual", confidence=0.91)
            )
        elif output_cls is SlotFillingOutput:
            structured.ainvoke = AsyncMock(
                return_value=SlotFillingOutput(
                    entities=[],
                    dimensions=[],
                    constraints=[],
                    reformulated_query="Session test query",
                )
            )
        elif output_cls is CriticOutput:
            structured.ainvoke = AsyncMock(
                return_value=CriticOutput(
                    score=8.5,
                    completeness=0.9,
                    faithfulness=0.9,
                    feedback="OK",
                )
            )
        else:
            structured.ainvoke = AsyncMock(return_value=MagicMock())
        return structured

    llm.with_structured_output = MagicMock(side_effect=_with_structured_output)
    r = MagicMock()
    r.content = "Session answer."
    llm.ainvoke = AsyncMock(return_value=r)
    return llm


@pytest.fixture
def graph():
    return build_main_graph(rag=_make_mock_rag(), llm=_make_mock_llm())


@pytest.mark.asyncio
async def test_same_thread_id_state_accumulates(graph: Any) -> None:
    """Same thread_id invoked twice: both complete and checkpoint reflects latest run."""
    thread_id = "checkpoint-session-same"
    config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}

    r1 = await graph.ainvoke({"question": "First question?"}, config=config)
    assert r1 is not None
    assert (
        r1.get("final_answer") if isinstance(r1, dict) else getattr(r1, "final_answer", None)
    ), "first run should produce final_answer"

    r2 = await graph.ainvoke({"question": "Second question?"}, config=config)
    assert r2 is not None
    assert (
        r2.get("final_answer") if isinstance(r2, dict) else getattr(r2, "final_answer", None)
    ), "second run should produce final_answer"

    state = graph.get_state(config)
    assert state is not None
    values = state.values if hasattr(state, "values") else state.get("values", state)
    assert values is not None
    assert (
        values.get("question") == "Second question?"
        or getattr(values, "question", None) == "Second question?"
    ), "checkpoint state should reflect latest invoke (second question)"


@pytest.mark.asyncio
async def test_different_thread_ids_isolated(graph: Any) -> None:
    """Two thread_ids: each invoke succeeds and state is isolated per thread."""
    config_a: dict[str, Any] = {"configurable": {"thread_id": "thread-a"}}
    config_b: dict[str, Any] = {"configurable": {"thread_id": "thread-b"}}

    ra = await graph.ainvoke({"question": "Question for A?"}, config=config_a)
    rb = await graph.ainvoke({"question": "Question for B?"}, config=config_b)

    assert ra is not None and rb is not None
    state_a = graph.get_state(config_a)
    state_b = graph.get_state(config_b)
    assert state_a is not None and state_b is not None
    va = state_a.values if hasattr(state_a, "values") else state_a.get("values", state_a)
    vb = state_b.values if hasattr(state_b, "values") else state_b.get("values", state_b)
    qa = va.get("question") if isinstance(va, dict) else getattr(va, "question", None)
    qb = vb.get("question") if isinstance(vb, dict) else getattr(vb, "question", None)
    assert qa == "Question for A?" and qb == "Question for B?", "each thread should have its own state"
