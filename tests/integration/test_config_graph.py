"""Integration tests for config-driven graph build (H8).

Loads test settings (or fixture YAML), builds the main graph, invokes once,
and asserts that state contains final_answer or expected keys. Ensures that
config or graph construction changes are caught by automation.

Design reference: DEV_SPEC H8.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agent.graph import build_main_graph
from src.agent.nodes.critic import CriticOutput
from src.agent.nodes.router import RouterOutput
from src.agent.nodes.slot_filling import SlotFillingOutput
from src.agent.state import RetrievedContext
from src.config import load_settings


def _fixture_settings_path() -> Path:
    """Path to the test settings YAML (next to this test file)."""
    return Path(__file__).resolve().parent.parent / "fixtures" / "settings_test.yaml"


def _sample_contexts() -> list[RetrievedContext]:
    """Sample RAG contexts for mock."""
    return [
        RetrievedContext(
            content="LoRA adds low-rank adapters.",
            source="LoRA paper",
            doc_id="doc_001",
            relevance_score=0.95,
            chunk_index=0,
        ),
    ]


def _make_mock_rag() -> MagicMock:
    rag = MagicMock()
    rag.search = AsyncMock(return_value=_sample_contexts())
    return rag


def _make_mock_llm(text: str = "LoRA is a parameter-efficient fine-tuning method.") -> MagicMock:
    """Mock LLM that supports with_structured_output (Router, SlotFilling, Critic) and ainvoke (synthesis)."""
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
                    reformulated_query="What is LoRA?",
                )
            )
        elif output_cls is CriticOutput:
            structured.ainvoke = AsyncMock(
                return_value=CriticOutput(
                    score=8.5,
                    completeness=0.9,
                    faithfulness=0.9,
                    feedback="Answer is complete and faithful.",
                )
            )
        else:
            structured.ainvoke = AsyncMock(return_value=MagicMock())
        return structured

    llm.with_structured_output = MagicMock(side_effect=_with_structured_output)
    r = MagicMock()
    r.content = text
    llm.ainvoke = AsyncMock(return_value=r)
    return llm


class TestConfigDrivenGraph:
    """Config load + build_main_graph + invoke produces expected state keys."""

    @pytest.fixture
    def settings_path(self) -> Path:
        return _fixture_settings_path()

    def test_load_settings_from_fixture_yaml(self, settings_path: Path) -> None:
        """Loading test fixture YAML returns valid Settings."""
        assert settings_path.exists(), "fixture settings_test.yaml must exist"
        settings = load_settings(settings_path)
        assert settings.llm.provider == "openai"
        assert settings.llm.model == "gpt-4o-mini"
        assert "rag_server" in settings.mcp.connections
        assert settings.agent.max_retries == 2

    async def test_build_and_invoke_produces_final_answer_and_keys(
        self,
        settings_path: Path,
    ) -> None:
        """After loading config, building graph with mocks and invoking, state has final_answer and expected keys."""
        settings = load_settings(settings_path)
        mock_rag = _make_mock_rag()
        mock_llm = _make_mock_llm()
        rag_default = getattr(settings.mcp, "rag_default_collection", None)

        graph = build_main_graph(
            rag=mock_rag,
            llm=mock_llm,
            rag_default_collection=rag_default,
        )
        config: dict[str, Any] = {"configurable": {"thread_id": "test-config-graph"}}

        result = await graph.ainvoke(
            {"question": "What is LoRA?"},
            config=config,
        )

        assert result is not None
        final_answer = (
            result.get("final_answer")
            if isinstance(result, dict)
            else getattr(result, "final_answer", None)
        )
        assert final_answer, "state should contain non-empty final_answer"
        assert "question" in result or hasattr(result, "question")
        assert "intent" in result or hasattr(result, "intent")
        assert "retrieved_contexts" in result or hasattr(result, "retrieved_contexts")
