"""Unit tests for the Comparative strategy node (C5).

Verifies that the comparative strategy correctly:
- Uses ``intent.entities`` and ``intent.dimensions`` when populated.
- Falls back to LLM extraction when ``intent.entities`` is empty.
- Performs **parallel** retrieval (one per entity) via ``asyncio.gather``.
- Asserts that the number of parallel retrieval calls == ``len(entities)``.
- Passes per-entity contexts and dimensions to the synthesis LLM.
- Writes ``draft_answer``, ``retrieved_contexts``, ``retrieval_queries``
  into the partial state update.
- Records reasoning trace steps for observability.
- Raises ``ValueError`` when ``state.intent`` is ``None``.
- Factory creates a callable node with correct name and bound parameters.
- Placeholder returns a dict with ``draft_answer``.

All tests use mocked RAG and LLM — no real server connections.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agent.state import AgentState, Intent, ReasoningStep, RetrievedContext
from src.agent.strategies.comparative import (
    DEFAULT_TOP_K,
    CompareEntities,
    create_comparative_strategy_node,
    comparative_strategy_node,
    run_comparative_strategy,
)
from src.prompts.strategies import (
    COMPARATIVE_SYNTHESIS_SYSTEM_PROMPT,
    format_entity_contexts,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_intent(
    *,
    intent_type: str = "comparative",
    reformulated_query: str = "Compare LoRA, QLoRA, and Full Fine-tuning on memory usage and accuracy",
    confidence: float = 0.92,
    entities: list[str] | None = None,
    dimensions: list[str] | None = None,
) -> Intent:
    """Create an Intent with sensible defaults for comparative testing."""
    return Intent(
        type=intent_type,
        confidence=confidence,
        reformulated_query=reformulated_query,
        entities=entities if entities is not None else ["LoRA", "QLoRA", "Full Fine-tuning"],
        dimensions=dimensions if dimensions is not None else ["memory usage", "accuracy"],
    )


def _make_state(
    *,
    question: str = "Compare LoRA, QLoRA, and Full Fine-tuning on memory usage and accuracy",
    intent: Intent | None = None,
) -> AgentState:
    """Create a minimal AgentState for testing."""
    if intent is None:
        intent = _make_intent()
    return AgentState(question=question, intent=intent)


def _sample_contexts(entity: str) -> list[RetrievedContext]:
    """Return sample RetrievedContext objects for a given entity."""
    return [
        RetrievedContext(
            content=f"{entity} achieves high efficiency in fine-tuning.",
            source=f"{entity} Paper",
            doc_id=f"doc_{entity.lower().replace(' ', '_')}_1",
            relevance_score=0.93,
            chunk_index=0,
        ),
        RetrievedContext(
            content=f"{entity} uses specific memory optimization techniques.",
            source=f"{entity} Technical Report",
            doc_id=f"doc_{entity.lower().replace(' ', '_')}_2",
            relevance_score=0.85,
            chunk_index=1,
        ),
    ]


def _make_mock_rag(
    entities: list[str] | None = None,
) -> MagicMock:
    """Create a mock RAGToolWrapper with per-entity context returns.

    The mock returns different contexts for each entity based on call order.
    """
    if entities is None:
        entities = ["LoRA", "QLoRA", "Full Fine-tuning"]
    rag = MagicMock()
    side_effects = [_sample_contexts(entity) for entity in entities]
    rag.search = AsyncMock(side_effect=side_effects)
    return rag


def _make_mock_llm(
    *,
    synthesis_text: str = (
        "| Dimension | LoRA | QLoRA | Full Fine-tuning |\n"
        "|-----------|------|-------|------------------|\n"
        "| Memory    | Low  | Lower | High             |\n"
        "| Accuracy  | Good | Good  | Best             |"
    ),
    fallback_entities: list[str] | None = None,
    fallback_dimensions: list[str] | None = None,
) -> MagicMock:
    """Create a mock LLM supporting both structured output and ainvoke.

    Args:
        synthesis_text: Text returned by direct ``ainvoke`` for synthesis.
        fallback_entities: Entities returned when LLM extraction fallback
            is triggered.
        fallback_dimensions: Dimensions returned by LLM extraction fallback.
    """
    llm = MagicMock()

    if fallback_entities is None:
        fallback_entities = ["LoRA", "QLoRA"]
    if fallback_dimensions is None:
        fallback_dimensions = ["memory", "accuracy"]

    # -- Structured output mock (for CompareEntities fallback) --------
    def _with_structured_output(output_cls):
        structured_mock = MagicMock()

        if output_cls is CompareEntities:

            async def _extract_ainvoke(*args: Any, **kwargs: Any) -> CompareEntities:
                return CompareEntities(
                    entities=fallback_entities,
                    dimensions=fallback_dimensions,
                )

            structured_mock.ainvoke = _extract_ainvoke

        return structured_mock

    llm.with_structured_output = MagicMock(side_effect=_with_structured_output)

    # -- Direct ainvoke (synthesis) ------------------------------------
    synthesis_response = MagicMock()
    synthesis_response.content = synthesis_text
    llm.ainvoke = AsyncMock(return_value=synthesis_response)

    return llm


# ---------------------------------------------------------------------------
# format_entity_contexts (prompt helper)
# ---------------------------------------------------------------------------


class TestFormatEntityContexts:
    """Tests for the per-entity context formatting helper."""

    def test_formats_multiple_entities(self) -> None:
        """Should format contexts grouped by entity with headers."""
        entity_contexts = {
            "LoRA": [
                RetrievedContext(
                    content="LoRA is efficient.", source="LoRA Paper",
                    doc_id="d1", relevance_score=0.9,
                ),
            ],
            "QLoRA": [
                RetrievedContext(
                    content="QLoRA combines quantization.", source="QLoRA Paper",
                    doc_id="d2", relevance_score=0.85,
                ),
            ],
        }
        result = format_entity_contexts(entity_contexts)
        assert "=== LoRA ===" in result
        assert "LoRA is efficient." in result
        assert "=== QLoRA ===" in result
        assert "QLoRA combines quantization." in result

    def test_empty_entity_contexts(self) -> None:
        """Empty mapping should return a placeholder."""
        result = format_entity_contexts({})
        assert "No entity contexts" in result

    def test_entity_with_no_contexts(self) -> None:
        """Entity with empty context list should show no-contexts message."""
        result = format_entity_contexts({"LoRA": []})
        assert "=== LoRA ===" in result
        assert "No relevant contexts" in result


# ---------------------------------------------------------------------------
# run_comparative_strategy — core logic
# ---------------------------------------------------------------------------


class TestRunComparativeStrategy:
    """Tests for the core comparative strategy implementation."""

    async def test_basic_parallel_retrieve_and_synthesize(self) -> None:
        """Happy path: use intent entities, parallel retrieve, synthesize."""
        state = _make_state()
        mock_rag = _make_mock_rag()
        mock_llm = _make_mock_llm()

        result = await run_comparative_strategy(state, mock_rag, mock_llm)

        # Parallel retrieval: one call per entity
        assert mock_rag.search.await_count == 3
        assert result["draft_answer"]
        assert len(result["draft_answer"]) > 0

    async def test_parallel_retrieval_count_equals_entities(self) -> None:
        """Number of parallel retrieval calls must equal len(entities).

        This is the core acceptance criterion for C5.
        """
        entities = ["LoRA", "QLoRA", "Full Fine-tuning"]
        intent = _make_intent(entities=entities)
        state = _make_state(intent=intent)
        mock_rag = _make_mock_rag(entities=entities)
        mock_llm = _make_mock_llm()

        await run_comparative_strategy(state, mock_rag, mock_llm)

        assert mock_rag.search.await_count == len(entities)

    async def test_two_entities_two_retrieval_calls(self) -> None:
        """With 2 entities, exactly 2 RAG calls should be made."""
        entities = ["LoRA", "QLoRA"]
        intent = _make_intent(entities=entities, dimensions=["memory"])
        state = _make_state(intent=intent)
        mock_rag = _make_mock_rag(entities=entities)
        mock_llm = _make_mock_llm()

        await run_comparative_strategy(state, mock_rag, mock_llm)

        assert mock_rag.search.await_count == 2

    async def test_four_entities_four_retrieval_calls(self) -> None:
        """With 4 entities, exactly 4 RAG calls should be made."""
        entities = ["LoRA", "QLoRA", "AdaLoRA", "Full Fine-tuning"]
        intent = _make_intent(entities=entities)
        state = _make_state(intent=intent)
        mock_rag = _make_mock_rag(entities=entities)
        mock_llm = _make_mock_llm()

        await run_comparative_strategy(state, mock_rag, mock_llm)

        assert mock_rag.search.await_count == 4

    async def test_retrieval_queries_contain_entity_and_question(self) -> None:
        """Each retrieval query should combine entity with question."""
        state = _make_state()
        mock_rag = _make_mock_rag()
        mock_llm = _make_mock_llm()

        result = await run_comparative_strategy(state, mock_rag, mock_llm)

        assert len(result["retrieval_queries"]) == 3
        for query in result["retrieval_queries"]:
            assert state.question in query or state.intent.reformulated_query in query

    async def test_rag_called_with_entity_query_format(self) -> None:
        """RAG search should be called with '{entity}: {question}' format."""
        entities = ["LoRA", "QLoRA"]
        intent = _make_intent(entities=entities, dimensions=["memory"])
        state = _make_state(intent=intent)
        mock_rag = _make_mock_rag(entities=entities)
        mock_llm = _make_mock_llm()

        await run_comparative_strategy(state, mock_rag, mock_llm)

        calls = mock_rag.search.call_args_list
        assert len(calls) == 2
        # Verify each call contains the entity
        call_queries = [c[0][0] for c in calls]
        assert any("LoRA" in q for q in call_queries)
        assert any("QLoRA" in q for q in call_queries)

    async def test_uses_intent_entities_and_dimensions(self) -> None:
        """Should use entities and dimensions from intent when populated."""
        entities = ["LoRA", "QLoRA"]
        dimensions = ["memory", "speed"]
        intent = _make_intent(entities=entities, dimensions=dimensions)
        state = _make_state(intent=intent)
        mock_rag = _make_mock_rag(entities=entities)
        mock_llm = _make_mock_llm()

        result = await run_comparative_strategy(state, mock_rag, mock_llm)

        # LLM extraction should NOT be called (intent entities available)
        mock_llm.with_structured_output.assert_not_called()
        assert result["draft_answer"]

    async def test_fallback_to_llm_extraction_when_entities_empty(self) -> None:
        """Should call LLM to extract entities when intent.entities is empty."""
        intent = _make_intent(entities=[], dimensions=[])
        state = _make_state(intent=intent)
        # Fallback entities from mock LLM
        fallback_entities = ["LoRA", "QLoRA"]
        mock_rag = _make_mock_rag(entities=fallback_entities)
        mock_llm = _make_mock_llm(
            fallback_entities=fallback_entities,
            fallback_dimensions=["memory", "accuracy"],
        )

        result = await run_comparative_strategy(state, mock_rag, mock_llm)

        # LLM extraction WAS called
        mock_llm.with_structured_output.assert_called_once_with(CompareEntities)
        # Retrieval count == len(fallback_entities)
        assert mock_rag.search.await_count == len(fallback_entities)
        assert result["draft_answer"]

    async def test_retrieved_contexts_from_all_entities(self) -> None:
        """Contexts from all entities should be collected into a flat list."""
        entities = ["LoRA", "QLoRA", "Full Fine-tuning"]
        state = _make_state()
        mock_rag = _make_mock_rag(entities=entities)
        mock_llm = _make_mock_llm()

        result = await run_comparative_strategy(state, mock_rag, mock_llm)

        # 2 contexts per entity * 3 entities = 6 total
        assert len(result["retrieved_contexts"]) == 6
        assert all(isinstance(ctx, RetrievedContext) for ctx in result["retrieved_contexts"])

    async def test_draft_answer_non_empty(self) -> None:
        """Core acceptance criterion: draft_answer must be non-empty."""
        state = _make_state()
        mock_rag = _make_mock_rag()
        mock_llm = _make_mock_llm()

        result = await run_comparative_strategy(state, mock_rag, mock_llm)

        assert result["draft_answer"]
        assert isinstance(result["draft_answer"], str)
        assert len(result["draft_answer"]) > 0

    async def test_llm_receives_system_and_user_messages(self) -> None:
        """Synthesis LLM should receive system + formatted user message."""
        state = _make_state()
        mock_rag = _make_mock_rag()
        mock_llm = _make_mock_llm()

        await run_comparative_strategy(state, mock_rag, mock_llm)

        call_args = mock_llm.ainvoke.call_args[0][0]
        assert len(call_args) == 2  # system + user

        system_msg = call_args[0]
        user_msg = call_args[1]

        assert system_msg.content == COMPARATIVE_SYNTHESIS_SYSTEM_PROMPT
        # User message should contain entities and dimensions
        assert "LoRA" in user_msg.content
        assert "QLoRA" in user_msg.content

    async def test_synthesis_prompt_contains_dimensions(self) -> None:
        """Synthesis prompt should include the comparison dimensions."""
        dimensions = ["memory usage", "training speed"]
        intent = _make_intent(dimensions=dimensions)
        state = _make_state(intent=intent)
        mock_rag = _make_mock_rag()
        mock_llm = _make_mock_llm()

        await run_comparative_strategy(state, mock_rag, mock_llm)

        user_msg = mock_llm.ainvoke.call_args[0][0][1]
        assert "memory usage" in user_msg.content
        assert "training speed" in user_msg.content

    async def test_custom_top_k(self) -> None:
        """Should respect a custom top_k parameter."""
        entities = ["LoRA", "QLoRA"]
        intent = _make_intent(entities=entities, dimensions=["memory"])
        state = _make_state(intent=intent)
        mock_rag = _make_mock_rag(entities=entities)
        mock_llm = _make_mock_llm()

        await run_comparative_strategy(state, mock_rag, mock_llm, top_k=10)

        for call in mock_rag.search.call_args_list:
            assert call.kwargs.get("top_k") == 10 or call[1].get("top_k") == 10

    async def test_raises_when_intent_is_none(self) -> None:
        """Should raise ValueError when state.intent is None."""
        state = AgentState(question="test")
        mock_rag = _make_mock_rag()
        mock_llm = _make_mock_llm()

        with pytest.raises(ValueError, match="state.intent"):
            await run_comparative_strategy(state, mock_rag, mock_llm)

    async def test_uses_reformulated_query(self) -> None:
        """Should use intent.reformulated_query for retrieval."""
        query = "Compare LoRA and QLoRA on memory efficiency"
        intent = _make_intent(
            reformulated_query=query,
            entities=["LoRA", "QLoRA"],
            dimensions=["memory"],
        )
        state = _make_state(question="lora vs qlora memory", intent=intent)
        mock_rag = _make_mock_rag(entities=["LoRA", "QLoRA"])
        mock_llm = _make_mock_llm()

        await run_comparative_strategy(state, mock_rag, mock_llm)

        # Each RAG call should use the reformulated query
        for call in mock_rag.search.call_args_list:
            assert query in call[0][0]

    async def test_falls_back_to_question_when_reformulated_empty(self) -> None:
        """Should fall back to state.question when reformulated_query is empty."""
        intent = _make_intent(
            reformulated_query="",
            entities=["LoRA", "QLoRA"],
            dimensions=["memory"],
        )
        state = _make_state(
            question="Compare LoRA and QLoRA",
            intent=intent,
        )
        mock_rag = _make_mock_rag(entities=["LoRA", "QLoRA"])
        mock_llm = _make_mock_llm()

        await run_comparative_strategy(state, mock_rag, mock_llm)

        for call in mock_rag.search.call_args_list:
            assert "Compare LoRA and QLoRA" in call[0][0]

    async def test_empty_retrieval_still_calls_llm(self) -> None:
        """LLM synthesis should still be invoked even with empty RAG results."""
        entities = ["LoRA", "QLoRA"]
        intent = _make_intent(entities=entities, dimensions=["memory"])
        state = _make_state(intent=intent)
        rag = MagicMock()
        rag.search = AsyncMock(side_effect=[[], []])
        mock_llm = _make_mock_llm(synthesis_text="Insufficient context for comparison.")

        result = await run_comparative_strategy(state, rag, mock_llm)

        mock_llm.ainvoke.assert_awaited_once()
        assert result["draft_answer"] == "Insufficient context for comparison."
        assert result["retrieved_contexts"] == []

    async def test_reasoning_trace_recorded(self) -> None:
        """Should include reasoning trace steps for observability."""
        state = _make_state()
        mock_rag = _make_mock_rag()
        mock_llm = _make_mock_llm()

        result = await run_comparative_strategy(state, mock_rag, mock_llm)

        trace = result["reasoning_trace"]
        assert len(trace) >= 3
        assert all(isinstance(step, ReasoningStep) for step in trace)

        # First step: comparative plan (thought)
        assert trace[0].step_type == "thought"
        assert "entities" in trace[0].content

        # Second step: parallel RAG search (action)
        assert trace[1].step_type == "action"
        assert "Parallel" in trace[1].content

        # Third step: synthesis (thought)
        assert trace[2].step_type == "thought"
        assert "Synthesized" in trace[2].content

    async def test_trace_metadata_contains_entity_counts(self) -> None:
        """Trace metadata should include entity and dimension counts."""
        state = _make_state()
        mock_rag = _make_mock_rag()
        mock_llm = _make_mock_llm()

        result = await run_comparative_strategy(state, mock_rag, mock_llm)

        plan_step = result["reasoning_trace"][0]
        assert plan_step.metadata["entities"] == ["LoRA", "QLoRA", "Full Fine-tuning"]
        assert plan_step.metadata["dimensions"] == ["memory usage", "accuracy"]

        action_step = result["reasoning_trace"][1]
        assert action_step.metadata["num_entities"] == 3


# ---------------------------------------------------------------------------
# create_comparative_strategy_node (factory)
# ---------------------------------------------------------------------------


class TestCreateComparativeStrategyNode:
    """Tests for the node factory function."""

    async def test_factory_creates_callable_node(self) -> None:
        """Factory should return a callable that accepts AgentState."""
        mock_rag = _make_mock_rag()
        mock_llm = _make_mock_llm()

        node = create_comparative_strategy_node(mock_rag, mock_llm)
        state = _make_state()

        result = await node(state)

        assert "draft_answer" in result
        assert "retrieved_contexts" in result
        assert "retrieval_queries" in result
        assert "reasoning_trace" in result

    async def test_factory_passes_custom_top_k(self) -> None:
        """Factory should respect the top_k parameter."""
        entities = ["LoRA", "QLoRA"]
        intent = _make_intent(entities=entities, dimensions=["memory"])
        state = _make_state(intent=intent)
        mock_rag = _make_mock_rag(entities=entities)
        mock_llm = _make_mock_llm()

        node = create_comparative_strategy_node(mock_rag, mock_llm, top_k=7)

        await node(state)

        for call in mock_rag.search.call_args_list:
            assert call.kwargs.get("top_k") == 7 or call[1].get("top_k") == 7

    async def test_factory_node_has_correct_name(self) -> None:
        """Factory-produced node should have the expected __name__."""
        mock_rag = _make_mock_rag()
        mock_llm = _make_mock_llm()

        node = create_comparative_strategy_node(mock_rag, mock_llm)

        assert node.__name__ == "comparative_strategy_node"


# ---------------------------------------------------------------------------
# comparative_strategy_node (default placeholder)
# ---------------------------------------------------------------------------


class TestComparativeStrategyNodePlaceholder:
    """Tests for the default synchronous placeholder node."""

    def test_placeholder_returns_draft_answer(self) -> None:
        """Placeholder should return a dict with draft_answer."""
        state = _make_state()
        result = comparative_strategy_node(state)

        assert "draft_answer" in result
        assert isinstance(result["draft_answer"], str)
        assert len(result["draft_answer"]) > 0
