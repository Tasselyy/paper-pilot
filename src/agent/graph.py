"""Main LangGraph graph definition.

Provides ``build_main_graph()`` which assembles the full agent graph
(nodes, edges, conditional routing) and returns a compiled graph.

When ``rag`` and ``llm`` dependencies are supplied, the *simple* strategy
uses the real async implementation (``create_simple_strategy_node``);
otherwise, a synchronous placeholder is used so the graph can still
compile and invoke without external services.

Design reference: PAPER_PILOT_DESIGN.md §5.1, DEV_SPEC B4.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.agent.edges import critic_gate, route_by_intent
from src.agent.nodes.critic import critic_node
from src.agent.nodes.format_output import format_output_node
from src.agent.nodes.memory_nodes import load_memory_node, save_memory_node
from src.agent.nodes.retry_refine import retry_refine_node
from src.agent.nodes.router import create_router_node, router_node
from src.agent.nodes.slot_filling import create_slot_filling_node, slot_filling_node
from src.agent.state import AgentState
from src.agent.strategies.comparative import comparative_strategy_node
from src.agent.strategies.exploratory import exploratory_strategy_node
from src.agent.strategies.multi_hop import (
    create_multi_hop_strategy_node,
    multi_hop_strategy_node,
)
from src.agent.strategies.simple import (
    create_simple_strategy_node,
    simple_strategy_node,
)

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

    from src.tools.tool_wrapper import RAGToolWrapper


def build_main_graph(
    *,
    rag: RAGToolWrapper | None = None,
    llm: BaseChatModel | None = None,
) -> StateGraph:
    """Build and return the compiled main agent graph.

    The graph implements the full PaperPilot workflow::

        START → load_memory → route → slot_fill
              → [route_by_intent] → simple | multi_hop | comparative | exploratory
              → critic → [critic_gate] → save_memory → format_output → END
                                        ↘ retry_refine → route (loop)

    Args:
        rag: Optional ``RAGToolWrapper`` instance.  When provided together
            with *llm*, the **simple** strategy node uses the real async
            implementation (retrieve + synthesize) instead of the sync
            placeholder.
        llm: Optional ``BaseChatModel`` instance for answer synthesis.

    Returns:
        A compiled LangGraph ``StateGraph`` with ``MemorySaver`` checkpointer.
    """
    graph = StateGraph(AgentState)

    # ── Resolve strategy / LLM-dependent nodes ─────────
    if rag is not None and llm is not None:
        _simple_node = create_simple_strategy_node(rag, llm)
        _multi_hop_node = create_multi_hop_strategy_node(rag, llm)
    else:
        _simple_node = simple_strategy_node
        _multi_hop_node = multi_hop_strategy_node

    if llm is not None:
        _router_node = create_router_node(llm)
        _slot_filling_node = create_slot_filling_node(llm)
    else:
        _router_node = router_node
        _slot_filling_node = slot_filling_node

    # ── Node registration ─────────────────────────────
    graph.add_node("load_memory", load_memory_node)
    graph.add_node("route", _router_node)
    graph.add_node("slot_fill", _slot_filling_node)
    graph.add_node("simple", _simple_node)
    graph.add_node("multi_hop", _multi_hop_node)
    graph.add_node("comparative", comparative_strategy_node)
    graph.add_node("exploratory", exploratory_strategy_node)
    graph.add_node("critic", critic_node)
    graph.add_node("retry_refine", retry_refine_node)
    graph.add_node("save_memory", save_memory_node)
    graph.add_node("format_output", format_output_node)

    # ── Edges ─────────────────────────────────────────

    # Entry: load memory → route (intent classification) → slot filling
    graph.add_edge(START, "load_memory")
    graph.add_edge("load_memory", "route")
    graph.add_edge("route", "slot_fill")

    # After slot filling → branch by intent strategy
    graph.add_conditional_edges(
        "slot_fill",
        route_by_intent,
        {
            "simple": "simple",
            "multi_hop": "multi_hop",
            "comparative": "comparative",
            "exploratory": "exploratory",
        },
    )

    # All strategies → critic
    graph.add_edge("simple", "critic")
    graph.add_edge("multi_hop", "critic")
    graph.add_edge("comparative", "critic")
    graph.add_edge("exploratory", "critic")

    # Critic → pass / retry
    graph.add_conditional_edges(
        "critic",
        critic_gate,
        {
            "pass": "save_memory",
            "retry": "retry_refine",
        },
    )

    # Retry → back to route
    graph.add_edge("retry_refine", "route")

    # Pass → save memory → format output → END
    graph.add_edge("save_memory", "format_output")
    graph.add_edge("format_output", END)

    return graph.compile(checkpointer=MemorySaver())
