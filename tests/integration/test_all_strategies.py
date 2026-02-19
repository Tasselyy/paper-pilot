"""Integration tests covering all strategy routes (H1).

These tests execute the compiled main graph end-to-end with mocked RAG and LLM
dependencies. Each test drives one intent type and validates that the
corresponding strategy path is exercised successfully:

- factual -> simple
- comparative -> comparative
- multi_hop -> multi_hop
- exploratory -> exploratory
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from src.agent.graph import build_main_graph
from src.agent.nodes.critic import CriticOutput
from src.agent.nodes.router import RouterOutput
from src.agent.nodes.slot_filling import ConstraintEntry, SlotFillingOutput
from src.agent.state import RetrievedContext
from src.agent.strategies.exploratory import ReactDecision
from src.agent.strategies.multi_hop import ReplanDecision, SubQuestionPlan


def _get(state: Any, key: str, default: Any = None) -> Any:
    """Return a key/attribute from LangGraph state objects."""
    if isinstance(state, dict):
        return state.get(key, default)
    return getattr(state, key, default)


def _mk_contexts(query: str, n: int = 2) -> list[RetrievedContext]:
    """Create deterministic retrieved contexts for a query."""
    safe_query = query.replace("\n", " ").strip() or "empty-query"
    return [
        RetrievedContext(
            content=f"Context {i + 1} for: {safe_query}",
            source=f"source-{safe_query[:24]}-{i + 1}",
            doc_id=f"doc-{abs(hash((safe_query, i))) % 10000}",
            relevance_score=0.9 - (i * 0.05),
            chunk_index=i,
        )
        for i in range(n)
    ]


@dataclass
class _Scenario:
    """Inputs used to drive one strategy path through the graph."""

    name: str
    intent_type: str
    question: str
    reformulated_query: str
    entities: list[str] = field(default_factory=list)
    dimensions: list[str] = field(default_factory=list)
    constraints: dict[str, str] = field(default_factory=dict)
    synthesis_text: str = "Default synthesized answer."
    multi_hop_plan: list[str] = field(default_factory=list)
    multi_hop_decisions: list[ReplanDecision] = field(default_factory=list)
    exploratory_decisions: list[ReactDecision] = field(default_factory=list)


class _ScenarioLLM:
    """Scenario-driven mock LLM for graph integration tests."""

    def __init__(self, scenario: _Scenario) -> None:
        self.scenario = scenario
        self._replan_idx = 0
        self._react_idx = 0
        self.with_structured_output = MagicMock(side_effect=self._with_structured_output)
        self.ainvoke = AsyncMock(side_effect=self._ainvoke_text)

    async def _ainvoke_text(self, messages: Any) -> Any:
        response = MagicMock()
        response.content = self.scenario.synthesis_text
        return response

    def _with_structured_output(self, output_cls: type[Any]) -> MagicMock:
        structured = MagicMock()

        async def _ainvoke(_messages: Any) -> Any:
            if output_cls is RouterOutput:
                return RouterOutput(type=self.scenario.intent_type, confidence=0.91)

            if output_cls is SlotFillingOutput:
                return SlotFillingOutput(
                    entities=self.scenario.entities,
                    dimensions=self.scenario.dimensions,
                    constraints=[
                        ConstraintEntry(key=k, value=v)
                        for k, v in self.scenario.constraints.items()
                    ],
                    reformulated_query=self.scenario.reformulated_query,
                )

            if output_cls is SubQuestionPlan:
                return SubQuestionPlan(questions=self.scenario.multi_hop_plan)

            if output_cls is ReplanDecision:
                if not self.scenario.multi_hop_decisions:
                    return ReplanDecision(action="synthesize", reason="No further steps.")
                idx = min(self._replan_idx, len(self.scenario.multi_hop_decisions) - 1)
                self._replan_idx += 1
                return self.scenario.multi_hop_decisions[idx]

            if output_cls is ReactDecision:
                if not self.scenario.exploratory_decisions:
                    return ReactDecision(
                        action="synthesize",
                        query="",
                        reasoning="Enough information gathered.",
                    )
                idx = min(self._react_idx, len(self.scenario.exploratory_decisions) - 1)
                self._react_idx += 1
                return self.scenario.exploratory_decisions[idx]

            if output_cls is CriticOutput:
                return CriticOutput(
                    score=8.5,
                    completeness=0.88,
                    faithfulness=0.9,
                    feedback="Answer is complete and grounded.",
                )

            raise AssertionError(f"Unexpected structured output class: {output_cls}")

        structured.ainvoke = AsyncMock(side_effect=_ainvoke)
        return structured


def _make_mock_rag() -> MagicMock:
    """Create a mock RAG wrapper with query-dependent contexts."""
    rag = MagicMock()

    async def _search(query: str, *args: Any, **kwargs: Any) -> list[RetrievedContext]:
        return _mk_contexts(query, n=2)

    rag.search = AsyncMock(side_effect=_search)
    return rag


async def _invoke_graph_for_scenario(scenario: _Scenario) -> tuple[Any, MagicMock]:
    """Build and invoke the graph for one strategy scenario."""
    rag = _make_mock_rag()
    llm = _ScenarioLLM(scenario)
    graph = build_main_graph(rag=rag, llm=llm)

    result = await graph.ainvoke(
        {"question": scenario.question},
        config={"configurable": {"thread_id": f"test-all-strategies-{scenario.name}"}},
    )
    return result, rag


class TestAllStrategiesIntegration:
    """Integration coverage for simple/comparative/multi_hop/exploratory paths."""

    async def test_main_graph_factual_route_uses_simple_strategy(self) -> None:
        """factual intent should execute simple path and produce final answer."""
        scenario = _Scenario(
            name="simple",
            intent_type="factual",
            question="What is LoRA?",
            reformulated_query="Explain LoRA in one paragraph.",
            entities=["LoRA"],
            synthesis_text="LoRA is a parameter-efficient fine-tuning method.",
        )
        result, rag = await _invoke_graph_for_scenario(scenario)

        assert _get(result, "final_answer")
        assert "Sources" in _get(result, "final_answer")
        assert _get(result, "intent").type == "factual"
        assert _get(result, "critic_verdict").passed is True
        assert _get(result, "retrieval_queries") == [scenario.reformulated_query]
        assert rag.search.await_count == 1

    async def test_main_graph_comparative_route_uses_parallel_entity_search(self) -> None:
        """comparative intent should execute comparative path with one call per entity."""
        scenario = _Scenario(
            name="comparative",
            intent_type="comparative",
            question="Compare LoRA and QLoRA on memory and quality.",
            reformulated_query="Compare LoRA and QLoRA across memory and quality.",
            entities=["LoRA", "QLoRA"],
            dimensions=["memory", "quality"],
            synthesis_text="QLoRA generally saves more memory while preserving quality.",
        )
        result, rag = await _invoke_graph_for_scenario(scenario)

        queries = _get(result, "retrieval_queries", [])
        assert _get(result, "final_answer")
        assert "Sources" in _get(result, "final_answer")
        assert _get(result, "intent").type == "comparative"
        assert _get(result, "critic_verdict").passed is True
        assert len(queries) == len(scenario.entities)
        assert all(": " in q for q in queries)
        assert rag.search.await_count == len(scenario.entities)

    async def test_main_graph_multi_hop_route_executes_plan_then_synthesize(self) -> None:
        """multi_hop intent should execute planned sub-questions then synthesize."""
        plan = [
            "What is LoRA?",
            "How does QLoRA extend LoRA with quantization?",
        ]
        scenario = _Scenario(
            name="multi-hop",
            intent_type="multi_hop",
            question="How does QLoRA build on LoRA?",
            reformulated_query="Explain how QLoRA builds on LoRA.",
            entities=["LoRA", "QLoRA"],
            multi_hop_plan=plan,
            multi_hop_decisions=[ReplanDecision(action="next_step", reason="Continue plan.")],
            synthesis_text="QLoRA combines LoRA adapters with 4-bit quantization.",
        )
        result, rag = await _invoke_graph_for_scenario(scenario)

        assert _get(result, "final_answer")
        assert "Sources" in _get(result, "final_answer")
        assert _get(result, "intent").type == "multi_hop"
        assert _get(result, "critic_verdict").passed is True
        assert _get(result, "sub_questions") == plan
        assert _get(result, "retrieval_queries") == plan
        assert rag.search.await_count == len(plan)

    async def test_main_graph_exploratory_route_executes_react_loop(self) -> None:
        """exploratory intent should execute ReAct search then synthesize."""
        search_query = "Recent PEFT methods after LoRA"
        scenario = _Scenario(
            name="exploratory",
            intent_type="exploratory",
            question="What are recent trends in PEFT?",
            reformulated_query="Survey recent trends in parameter-efficient fine-tuning.",
            entities=["PEFT"],
            exploratory_decisions=[
                ReactDecision(
                    action="search",
                    query=search_query,
                    reasoning="Need one focused retrieval step.",
                ),
                ReactDecision(
                    action="synthesize",
                    query="",
                    reasoning="Enough observations collected.",
                ),
            ],
            synthesis_text="Recent PEFT trends include QLoRA, DoRA, and adaptive adapters.",
        )
        result, rag = await _invoke_graph_for_scenario(scenario)

        assert _get(result, "final_answer")
        assert "Sources" in _get(result, "final_answer")
        assert _get(result, "intent").type == "exploratory"
        assert _get(result, "critic_verdict").passed is True
        assert _get(result, "retrieval_queries") == [search_query]
        assert rag.search.await_count == 1
