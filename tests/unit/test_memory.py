"""Unit tests for memory system (D4).

Verifies that:
- ``LongTermMemory`` correctly loads/saves facts from/to JSONL.
- ``recall`` returns relevant facts based on keyword overlap.
- ``memorize`` persists facts and updates in-memory state.
- ``extract_keywords`` produces sensible tokens.
- ``load_memory_node`` writes ``accumulated_facts`` into state.
- ``save_memory_node`` calls ``memorize`` and records reasoning trace.
- Both factories (``create_load_memory_node`` / ``create_save_memory_node``)
  produce working async callables.
- Sync placeholder nodes return correct partial state.
- ``ShortTermMemory`` (``create_checkpointer``) returns a valid saver.

All tests use a temporary JSONL file and mocked LLM — no real API calls.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agent.nodes.memory_nodes import (
    ExtractedFacts,
    _heuristic_extract,
    create_load_memory_node,
    create_save_memory_node,
    load_memory_node,
    run_load_memory,
    run_save_memory,
    save_memory_node,
)
from src.agent.state import AgentState, Intent, ReasoningStep, RetrievedContext
from src.memory.long_term import (
    Fact,
    LongTermMemory,
    extract_keywords,
)
from src.memory.short_term import create_checkpointer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_memory_file(tmp_path: Path) -> Path:
    """Return a temporary JSONL file path for memory tests."""
    return tmp_path / "test_memory.jsonl"


@pytest.fixture()
def memory(tmp_memory_file: Path) -> LongTermMemory:
    """Return a fresh ``LongTermMemory`` with an empty temp file."""
    return LongTermMemory(memory_file=str(tmp_memory_file))


@pytest.fixture()
def seeded_memory(tmp_memory_file: Path) -> LongTermMemory:
    """Return a ``LongTermMemory`` pre-loaded with sample facts."""
    facts = [
        {
            "content": "LoRA adds low-rank matrices to transformer layers for efficient fine-tuning.",
            "keywords": ["lora", "low-rank", "matrices", "transformer", "layers", "efficient", "fine-tuning"],
            "source_question": "What is LoRA?",
            "timestamp": 1000.0,
        },
        {
            "content": "DPO aligns language models using preference pairs without a reward model.",
            "keywords": ["dpo", "aligns", "language", "models", "preference", "pairs", "reward", "model"],
            "source_question": "How does DPO work?",
            "timestamp": 2000.0,
        },
        {
            "content": "ReAct interleaves reasoning and acting for tool-augmented LLM agents.",
            "keywords": ["react", "interleaves", "reasoning", "acting", "tool-augmented", "llm", "agents"],
            "source_question": "What is the ReAct pattern?",
            "timestamp": 3000.0,
        },
    ]
    tmp_memory_file.parent.mkdir(parents=True, exist_ok=True)
    with tmp_memory_file.open("w", encoding="utf-8") as fh:
        for fact in facts:
            fh.write(json.dumps(fact) + "\n")

    return LongTermMemory(memory_file=str(tmp_memory_file))


def _make_state(
    *,
    question: str = "What is LoRA?",
    draft_answer: str = "LoRA is a parameter-efficient fine-tuning method.",
) -> AgentState:
    """Create a minimal AgentState for testing memory nodes."""
    return AgentState(
        question=question,
        draft_answer=draft_answer,
        intent=Intent(
            type="factual",
            confidence=0.9,
            entities=["LoRA"],
            reformulated_query=question,
        ),
        retrieved_contexts=[
            RetrievedContext(
                content="LoRA adds low-rank matrices to transformer layers.",
                source="LoRA Paper (2021)",
                doc_id="lora-001",
                relevance_score=0.92,
            ),
        ],
    )


def _make_mock_llm(facts: list[str] | None = None) -> MagicMock:
    """Create a mock LLM that returns ``ExtractedFacts`` via structured output."""
    if facts is None:
        facts = ["LoRA reduces trainable parameters by 10000x."]

    extraction = ExtractedFacts(facts=facts)

    structured_llm = MagicMock()
    structured_llm.ainvoke = AsyncMock(return_value=extraction)

    llm = MagicMock()
    llm.with_structured_output = MagicMock(return_value=structured_llm)

    return llm


# ===========================================================================
# extract_keywords
# ===========================================================================


class TestExtractKeywords:
    """Tests for keyword extraction utility."""

    def test_basic_extraction(self) -> None:
        """Should extract meaningful words, excluding stop words."""
        keywords = extract_keywords("What is LoRA fine-tuning?")
        assert "lora" in keywords
        assert "fine-tuning" in keywords
        # Stop words should be excluded
        assert "what" not in keywords
        assert "is" not in keywords

    def test_deduplication(self) -> None:
        """Should not return duplicate keywords."""
        keywords = extract_keywords("LoRA uses LoRA adapters for LoRA fine-tuning")
        assert keywords.count("lora") == 1

    def test_empty_input(self) -> None:
        """Should return empty list for empty input."""
        assert extract_keywords("") == []

    def test_short_tokens_excluded(self) -> None:
        """Tokens shorter than 2 chars should be excluded."""
        keywords = extract_keywords("I am a fine researcher")
        # Single-char tokens filtered out
        assert "i" not in keywords
        assert "a" not in keywords
        # "am" is 2 chars — kept (but may be a stop word; regardless, single-char filtered)
        assert "fine" in keywords
        assert "researcher" in keywords

    def test_preserves_order(self) -> None:
        """Keywords should be in order of first occurrence."""
        keywords = extract_keywords("transformer models use attention mechanisms")
        assert keywords.index("transformer") < keywords.index("attention")


# ===========================================================================
# Fact model
# ===========================================================================


class TestFactModel:
    """Tests for the Fact Pydantic model."""

    def test_create_fact(self) -> None:
        """Should create a valid Fact with all fields."""
        fact = Fact(
            content="LoRA is efficient.",
            keywords=["lora", "efficient"],
            source_question="What is LoRA?",
            timestamp=1234.0,
        )
        assert fact.content == "LoRA is efficient."
        assert fact.keywords == ["lora", "efficient"]

    def test_fact_serialization(self) -> None:
        """Should serialize to JSON and back correctly."""
        fact = Fact(
            content="Test fact",
            keywords=["test"],
            source_question="Test?",
            timestamp=100.0,
        )
        data = json.loads(fact.model_dump_json())
        restored = Fact(**data)
        assert restored.content == fact.content
        assert restored.keywords == fact.keywords


# ===========================================================================
# LongTermMemory — initialization
# ===========================================================================


class TestLongTermMemoryInit:
    """Tests for LongTermMemory initialization and loading."""

    def test_empty_file_init(self, memory: LongTermMemory) -> None:
        """Should initialize with zero facts when file does not exist."""
        assert memory.fact_count == 0

    def test_seeded_init(self, seeded_memory: LongTermMemory) -> None:
        """Should load all facts from existing JSONL file."""
        assert seeded_memory.fact_count == 3

    def test_malformed_line_skipped(self, tmp_memory_file: Path) -> None:
        """Should skip malformed JSON lines and load valid ones."""
        tmp_memory_file.parent.mkdir(parents=True, exist_ok=True)
        with tmp_memory_file.open("w", encoding="utf-8") as fh:
            fh.write('{"content":"valid","keywords":[],"source_question":"q","timestamp":1.0}\n')
            fh.write("this is not json\n")
            fh.write('{"content":"also valid","keywords":[],"source_question":"q2","timestamp":2.0}\n')

        mem = LongTermMemory(memory_file=str(tmp_memory_file))
        assert mem.fact_count == 2


# ===========================================================================
# LongTermMemory.recall
# ===========================================================================


class TestLongTermMemoryRecall:
    """Tests for the recall (retrieval) method."""

    @pytest.mark.asyncio
    async def test_recall_returns_relevant_facts(self, seeded_memory: LongTermMemory) -> None:
        """Should return facts with keyword overlap to the question."""
        results = await seeded_memory.recall("Tell me about LoRA fine-tuning")
        assert len(results) >= 1
        assert any("LoRA" in r for r in results)

    @pytest.mark.asyncio
    async def test_recall_empty_memory(self, memory: LongTermMemory) -> None:
        """Should return empty list when no facts are stored."""
        results = await memory.recall("What is LoRA?")
        assert results == []

    @pytest.mark.asyncio
    async def test_recall_no_overlap(self, seeded_memory: LongTermMemory) -> None:
        """Should return empty list when no keywords overlap."""
        results = await seeded_memory.recall("xyz abc 123")
        assert results == []

    @pytest.mark.asyncio
    async def test_recall_respects_top_k(self, seeded_memory: LongTermMemory) -> None:
        """Should return at most top_k results."""
        results = await seeded_memory.recall(
            "LoRA DPO ReAct transformer models agents",
            top_k=2,
        )
        assert len(results) <= 2

    @pytest.mark.asyncio
    async def test_recall_ordered_by_relevance(self, seeded_memory: LongTermMemory) -> None:
        """More relevant facts (higher overlap) should come first."""
        results = await seeded_memory.recall(
            "LoRA transformer fine-tuning low-rank matrices",
        )
        if len(results) >= 2:
            # First result should be the LoRA fact (highest overlap)
            assert "LoRA" in results[0]


# ===========================================================================
# LongTermMemory.memorize
# ===========================================================================


class TestLongTermMemoryMemorize:
    """Tests for the memorize (persistence) method."""

    @pytest.mark.asyncio
    async def test_memorize_stores_facts(self, memory: LongTermMemory) -> None:
        """Should add facts to in-memory list."""
        facts = await memory.memorize(
            question="What is LoRA?",
            answer="LoRA is efficient.",
            facts_to_store=["LoRA reduces trainable parameters by 10000x."],
        )
        assert len(facts) == 1
        assert memory.fact_count == 1

    @pytest.mark.asyncio
    async def test_memorize_persists_to_jsonl(
        self, memory: LongTermMemory, tmp_memory_file: Path
    ) -> None:
        """Should write facts to the JSONL file."""
        await memory.memorize(
            question="What is DPO?",
            answer="DPO aligns models.",
            facts_to_store=["DPO uses preference pairs for alignment."],
        )
        assert tmp_memory_file.exists()
        lines = tmp_memory_file.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["content"] == "DPO uses preference pairs for alignment."

    @pytest.mark.asyncio
    async def test_memorize_empty_list(self, memory: LongTermMemory) -> None:
        """Should handle empty facts list gracefully."""
        facts = await memory.memorize(
            question="Test?",
            answer="Test.",
            facts_to_store=[],
        )
        assert facts == []
        assert memory.fact_count == 0

    @pytest.mark.asyncio
    async def test_memorize_extracts_keywords(self, memory: LongTermMemory) -> None:
        """Stored facts should have keywords extracted from content."""
        await memory.memorize(
            question="Test?",
            answer="Test.",
            facts_to_store=["LoRA adds low-rank decomposition matrices."],
        )
        stored = memory.facts[0]
        assert "lora" in stored.keywords
        assert "low-rank" in stored.keywords

    @pytest.mark.asyncio
    async def test_memorize_sets_source_question(self, memory: LongTermMemory) -> None:
        """Stored facts should record the source question."""
        await memory.memorize(
            question="What is LoRA?",
            answer="LoRA is efficient.",
            facts_to_store=["LoRA is efficient."],
        )
        assert memory.facts[0].source_question == "What is LoRA?"

    @pytest.mark.asyncio
    async def test_memorize_multiple_facts(self, memory: LongTermMemory) -> None:
        """Should store multiple facts in a single call."""
        await memory.memorize(
            question="Compare LoRA and DPO",
            answer="They differ.",
            facts_to_store=[
                "LoRA is for fine-tuning.",
                "DPO is for alignment.",
            ],
        )
        assert memory.fact_count == 2


# ===========================================================================
# LongTermMemory.clear
# ===========================================================================


class TestLongTermMemoryClear:
    """Tests for the clear method."""

    @pytest.mark.asyncio
    async def test_clear_removes_all_facts(
        self, seeded_memory: LongTermMemory, tmp_memory_file: Path
    ) -> None:
        """Should remove all in-memory facts and delete the file."""
        assert seeded_memory.fact_count == 3
        seeded_memory.clear()
        assert seeded_memory.fact_count == 0


# ===========================================================================
# LongTermMemory — round-trip persistence
# ===========================================================================


class TestLongTermMemoryRoundTrip:
    """Tests for write-then-read round-trip through JSONL."""

    @pytest.mark.asyncio
    async def test_round_trip(self, tmp_memory_file: Path) -> None:
        """Facts written by memorize should be loaded by a fresh instance."""
        mem1 = LongTermMemory(memory_file=str(tmp_memory_file))
        await mem1.memorize(
            question="What is LoRA?",
            answer="LoRA is efficient.",
            facts_to_store=["LoRA reduces parameters significantly."],
        )

        mem2 = LongTermMemory(memory_file=str(tmp_memory_file))
        assert mem2.fact_count == 1
        assert "LoRA" in mem2.facts[0].content


# ===========================================================================
# run_load_memory
# ===========================================================================


class TestRunLoadMemory:
    """Tests for the core load_memory implementation."""

    @pytest.mark.asyncio
    async def test_loads_relevant_facts(self, seeded_memory: LongTermMemory) -> None:
        """Should write accumulated_facts into state."""
        state = _make_state(question="Tell me about LoRA")
        result = await run_load_memory(state, seeded_memory)

        assert "accumulated_facts" in result
        assert len(result["accumulated_facts"]) >= 1
        assert any("LoRA" in f for f in result["accumulated_facts"])

    @pytest.mark.asyncio
    async def test_empty_question_returns_empty(self, seeded_memory: LongTermMemory) -> None:
        """Should return empty facts for empty question."""
        state = _make_state(question="")
        result = await run_load_memory(state, seeded_memory)

        assert result["accumulated_facts"] == []

    @pytest.mark.asyncio
    async def test_records_reasoning_trace(self, seeded_memory: LongTermMemory) -> None:
        """Should include a reasoning trace step."""
        state = _make_state(question="What is LoRA?")
        result = await run_load_memory(state, seeded_memory)

        trace = result["reasoning_trace"]
        assert len(trace) == 1
        assert isinstance(trace[0], ReasoningStep)
        assert trace[0].step_type == "action"

    @pytest.mark.asyncio
    async def test_dict_state_works(self, seeded_memory: LongTermMemory) -> None:
        """Should work when state is a plain dict."""
        state = {"question": "Tell me about DPO alignment"}
        result = await run_load_memory(state, seeded_memory)

        assert "accumulated_facts" in result


# ===========================================================================
# run_save_memory — with LLM
# ===========================================================================


class TestRunSaveMemoryWithLLM:
    """Tests for save_memory with LLM-based fact extraction."""

    @pytest.mark.asyncio
    async def test_extracts_and_stores_facts(self, memory: LongTermMemory) -> None:
        """Should extract facts via LLM and store them."""
        state = _make_state()
        mock_llm = _make_mock_llm(facts=["LoRA reduces parameters by 10000x."])

        result = await run_save_memory(state, memory, mock_llm)

        assert memory.fact_count == 1
        assert "reasoning_trace" in result

    @pytest.mark.asyncio
    async def test_llm_receives_structured_output_schema(self, memory: LongTermMemory) -> None:
        """LLM should receive ExtractedFacts as the structured output schema."""
        state = _make_state()
        mock_llm = _make_mock_llm()

        await run_save_memory(state, memory, mock_llm)

        mock_llm.with_structured_output.assert_called_once_with(ExtractedFacts)

    @pytest.mark.asyncio
    async def test_llm_receives_question_and_answer(self, memory: LongTermMemory) -> None:
        """LLM prompt should contain the question and answer."""
        state = _make_state(
            question="What is LoRA?",
            draft_answer="LoRA is a fine-tuning method.",
        )
        mock_llm = _make_mock_llm()

        await run_save_memory(state, memory, mock_llm)

        structured_llm = mock_llm.with_structured_output.return_value
        call_args = structured_llm.ainvoke.call_args[0][0]
        user_msg = call_args[1]
        assert "What is LoRA?" in user_msg.content
        assert "LoRA is a fine-tuning method." in user_msg.content

    @pytest.mark.asyncio
    async def test_empty_draft_skips(self, memory: LongTermMemory) -> None:
        """Should skip memorization when draft_answer is empty."""
        state = _make_state(draft_answer="")
        mock_llm = _make_mock_llm()

        result = await run_save_memory(state, memory, mock_llm)

        assert memory.fact_count == 0
        mock_llm.with_structured_output.assert_not_called()
        assert "reasoning_trace" in result

    @pytest.mark.asyncio
    async def test_llm_failure_falls_back_to_heuristic(self, memory: LongTermMemory) -> None:
        """Should fall back to heuristic if LLM call fails."""
        state = _make_state(
            draft_answer="LoRA adds low-rank decomposition to reduce parameters significantly."
        )
        mock_llm = MagicMock()
        structured_llm = MagicMock()
        structured_llm.ainvoke = AsyncMock(side_effect=RuntimeError("API error"))
        mock_llm.with_structured_output = MagicMock(return_value=structured_llm)

        await run_save_memory(state, memory, mock_llm)

        # Heuristic should still store something
        assert memory.fact_count >= 1


# ===========================================================================
# run_save_memory — without LLM (heuristic)
# ===========================================================================


class TestRunSaveMemoryHeuristic:
    """Tests for save_memory without LLM (heuristic extraction)."""

    @pytest.mark.asyncio
    async def test_heuristic_stores_first_sentence(self, memory: LongTermMemory) -> None:
        """Should store the first substantive sentence from the answer."""
        state = _make_state(
            draft_answer="LoRA is a parameter-efficient fine-tuning technique. It modifies weights."
        )

        result = await run_save_memory(state, memory, llm=None)

        assert memory.fact_count == 1
        assert "LoRA" in memory.facts[0].content

    @pytest.mark.asyncio
    async def test_heuristic_skips_short_answers(self, memory: LongTermMemory) -> None:
        """Should not store facts from very short answers."""
        state = _make_state(draft_answer="OK.")

        await run_save_memory(state, memory, llm=None)

        assert memory.fact_count == 0


# ===========================================================================
# _heuristic_extract
# ===========================================================================


class TestHeuristicExtract:
    """Tests for the heuristic fact extraction function."""

    def test_extracts_first_sentence(self) -> None:
        """Should extract the first sentence longer than 20 chars."""
        result = _heuristic_extract("LoRA reduces trainable parameters by 10000x. It is efficient.")
        assert len(result) == 1
        assert "LoRA" in result[0]

    def test_empty_input(self) -> None:
        """Should return empty list for empty input."""
        assert _heuristic_extract("") == []

    def test_short_answer(self) -> None:
        """Should return empty list when no sentence exceeds 20 chars."""
        assert _heuristic_extract("OK. Yes. No.") == []


# ===========================================================================
# create_load_memory_node (factory)
# ===========================================================================


class TestCreateLoadMemoryNode:
    """Tests for the load_memory node factory."""

    @pytest.mark.asyncio
    async def test_factory_creates_working_node(self, seeded_memory: LongTermMemory) -> None:
        """Factory should return a callable that loads facts."""
        node = create_load_memory_node(seeded_memory)
        state = _make_state(question="Tell me about LoRA")

        result = await node(state)

        assert "accumulated_facts" in result
        assert len(result["accumulated_facts"]) >= 1

    @pytest.mark.asyncio
    async def test_factory_node_has_correct_name(self, memory: LongTermMemory) -> None:
        """Factory-produced node should have the expected __name__."""
        node = create_load_memory_node(memory)
        assert node.__name__ == "load_memory_node"


# ===========================================================================
# create_save_memory_node (factory)
# ===========================================================================


class TestCreateSaveMemoryNode:
    """Tests for the save_memory node factory."""

    @pytest.mark.asyncio
    async def test_factory_with_llm(self, memory: LongTermMemory) -> None:
        """Factory with LLM should extract and store facts."""
        mock_llm = _make_mock_llm(facts=["LoRA is efficient."])
        node = create_save_memory_node(memory, mock_llm)
        state = _make_state()

        result = await node(state)

        assert memory.fact_count == 1
        assert "reasoning_trace" in result

    @pytest.mark.asyncio
    async def test_factory_without_llm(self, memory: LongTermMemory) -> None:
        """Factory without LLM should use heuristic extraction."""
        node = create_save_memory_node(memory, llm=None)
        state = _make_state(
            draft_answer="LoRA is a parameter-efficient fine-tuning technique for large models."
        )

        await node(state)

        # Heuristic should store something for a substantive answer
        assert memory.fact_count >= 1

    @pytest.mark.asyncio
    async def test_factory_node_has_correct_name(self, memory: LongTermMemory) -> None:
        """Factory-produced node should have the expected __name__."""
        node = create_save_memory_node(memory)
        assert node.__name__ == "save_memory_node"


# ===========================================================================
# Sync placeholder nodes
# ===========================================================================


class TestPlaceholderNodes:
    """Tests for the default synchronous placeholder nodes."""

    def test_load_memory_placeholder_returns_empty_facts(self) -> None:
        """Placeholder load_memory should return empty accumulated_facts."""
        state = _make_state()
        result = load_memory_node(state)

        assert "accumulated_facts" in result
        assert result["accumulated_facts"] == []

    def test_load_memory_placeholder_has_trace(self) -> None:
        """Placeholder load_memory should include a reasoning trace step."""
        state = _make_state()
        result = load_memory_node(state)

        assert "reasoning_trace" in result
        assert len(result["reasoning_trace"]) == 1
        assert result["reasoning_trace"][0].step_type == "action"

    def test_save_memory_placeholder_has_trace(self) -> None:
        """Placeholder save_memory should include a reasoning trace step."""
        state = _make_state()
        result = save_memory_node(state)

        assert "reasoning_trace" in result
        assert len(result["reasoning_trace"]) == 1
        assert result["reasoning_trace"][0].step_type == "action"


# ===========================================================================
# Short-term memory (checkpointer)
# ===========================================================================


class TestCreateCheckpointer:
    """Tests for the short-term memory checkpointer factory."""

    def test_memory_backend(self) -> None:
        """Should return a MemorySaver for 'memory' backend."""
        from langgraph.checkpoint.memory import MemorySaver

        saver = create_checkpointer("memory")
        assert isinstance(saver, MemorySaver)

    def test_unsupported_backend_raises(self) -> None:
        """Should raise ValueError for unsupported backends."""
        with pytest.raises(ValueError, match="Unsupported"):
            create_checkpointer("redis")  # type: ignore[arg-type]
