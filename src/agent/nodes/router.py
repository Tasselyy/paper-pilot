"""Router node: intent classification via LoRA model or Cloud LLM fallback.

Placeholder implementation — sets a default ``factual`` intent so the graph
can run end-to-end.  Real classification is wired in task C1.
"""

from __future__ import annotations

from src.agent.state import Intent


def router_node(state) -> dict:
    """Classify the user question into an intent type.

    Args:
        state: The current AgentState.

    Returns:
        Partial state update with a placeholder ``intent``.
    """
    question = state.question if hasattr(state, "question") else state.get("question", "")
    return {
        "intent": Intent(
            type="factual",
            confidence=1.0,
            reformulated_query=question or "placeholder query",
        ),
    }
