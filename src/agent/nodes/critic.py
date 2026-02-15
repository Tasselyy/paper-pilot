"""Critic node: evaluate answer quality via DPO model or Cloud LLM.

Placeholder implementation — always passes with a good score.
Real evaluation is wired in task D1.
"""

from __future__ import annotations

from src.agent.state import CriticVerdict


def critic_node(state) -> dict:
    """Evaluate the draft answer and produce a CriticVerdict.

    Args:
        state: The current AgentState.

    Returns:
        Partial state update with a placeholder ``critic_verdict``.
    """
    return {
        "critic_verdict": CriticVerdict(
            passed=True,
            score=7.0,
            completeness=0.8,
            faithfulness=0.9,
            feedback="Placeholder — critic not yet implemented.",
        ),
    }
