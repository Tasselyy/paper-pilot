"""AgentState, Intent, and related Pydantic models.

Defines the core state schema for the LangGraph agent, including intent
classification types, sub-structures (RetrievedContext, CriticVerdict,
ReasoningStep), and the main ``AgentState`` used as the graph's state.

Design reference: PAPER_PILOT_DESIGN.md §4.
"""

from __future__ import annotations

import operator
import time
from typing import Annotated, Any, Literal

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from src.types import IntentType

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

StrategyName = Literal["simple", "multi_hop", "comparative", "exploratory"]
"""Graph strategy node names mapped from ``IntentType``."""

# ---------------------------------------------------------------------------
# Sub-structures
# ---------------------------------------------------------------------------


class Intent(BaseModel):
    """Intent understanding output (Router classification + Slot Filling).

    Attributes:
        type: Intent type from LoRA classification.
        confidence: Classification confidence in [0, 1].
        entities: Key entities (paper names, technique names, etc.).
        dimensions: Comparison dimensions (non-empty only for *comparative*).
        constraints: Implicit constraints (e.g. model_scale, time_range).
        reformulated_query: Rewritten query for retrieval and synthesis.
    """

    type: IntentType = Field(description="Intent type (LoRA classification result)")
    confidence: float = Field(ge=0, le=1, description="Classification confidence")

    # Populated by Slot Filling (Cloud LLM)
    entities: list[str] = Field(
        default_factory=list,
        description="Key entities (paper names, technique names)",
    )
    dimensions: list[str] = Field(
        default_factory=list,
        description="Comparison dimensions (comparative only)",
    )
    constraints: dict[str, Any] = Field(
        default_factory=dict,
        description="Implicit constraints (model_scale, time_range, etc.)",
    )
    reformulated_query: str = Field(
        default="",
        description="Rewritten clear query for retrieval",
    )

    def to_strategy(self) -> StrategyName:
        """Map intent type to a LangGraph strategy node name.

        ``follow_up`` is handled by the *simple* strategy with dialogue
        context carried in ``reformulated_query``.

        Returns:
            One of ``"simple"``, ``"multi_hop"``, ``"comparative"``,
            ``"exploratory"``.
        """
        if self.type in ("factual", "follow_up"):
            return "simple"
        # comparative, multi_hop, exploratory map directly
        return self.type  # type: ignore[return-value]


class RetrievedContext(BaseModel):
    """A single retrieval result from the RAG knowledge base."""

    content: str
    source: str = Field(description="Paper title / document name")
    doc_id: str = Field(description="Document identifier")
    relevance_score: float = Field(description="Relevance score from retrieval")
    chunk_index: int | None = Field(default=None, description="Chunk index within doc")


class CriticVerdict(BaseModel):
    """Critic evaluation result (Reflexion pattern)."""

    passed: bool
    score: float = Field(ge=0, le=10, description="Answer quality score 0-10")
    completeness: float = Field(ge=0, le=1, description="Completeness rating")
    faithfulness: float = Field(ge=0, le=1, description="Faithfulness (no hallucination)")
    feedback: str = Field(description="Improvement suggestions (when passed=False)")


class ReasoningStep(BaseModel):
    """A single step in the reasoning trace (for ReAct and observability)."""

    step_type: Literal["thought", "action", "observation", "critique", "route"]
    content: str
    timestamp: float = Field(default_factory=time.time)
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Main state
# ---------------------------------------------------------------------------


class AgentState(BaseModel):
    """PaperPilot Agent state — the core schema for the LangGraph StateGraph.

    Design notes (§4.1):
        * ``messages`` uses ``add_messages`` reducer for append semantics.
        * All sub-structures are Pydantic models for type safety and
          compatibility with LLM structured output.
        * Control-flow variables (``retry_count``, ``current_react_step``)
          let conditional edges make routing decisions.
    """

    # Conversation
    messages: Annotated[list[BaseMessage], add_messages] = Field(default_factory=list)
    question: str = Field(default="", description="Current user question (raw)")

    # Intent understanding (Router + Slot Filling)
    intent: Intent | None = Field(default=None, description="Fully populated intent")

    # Retrieval
    sub_questions: list[str] = Field(
        default_factory=list,
        description="Sub-questions decomposed by multi_hop strategy",
    )
    retrieved_contexts: list[RetrievedContext] = Field(
        default_factory=list,
        description="Retrieved RAG contexts",
    )
    retrieval_queries: list[str] = Field(
        default_factory=list,
        description="Actual queries sent to RAG",
    )

    # Generation
    draft_answer: str = Field(default="", description="Current draft answer")
    final_answer: str = Field(default="", description="Final answer after formatting")

    # Evaluation
    critic_verdict: CriticVerdict | None = Field(
        default=None,
        description="Critic evaluation result",
    )

    # Control flow
    retry_count: int = Field(default=0, description="Number of critic retries so far")
    max_retries: int = Field(default=2, description="Maximum critic retry rounds")
    current_react_step: int = Field(default=0, description="Current ReAct loop step")
    max_react_steps: int = Field(default=5, description="Maximum ReAct loop steps")

    # Reasoning trace (observability) — uses ``operator.add`` reducer so
    # trace entries from each node are *appended* rather than replaced.
    reasoning_trace: Annotated[list[ReasoningStep], operator.add] = Field(
        default_factory=list,
        description="Step-by-step reasoning log",
    )

    # Memory
    accumulated_facts: list[str] = Field(
        default_factory=list,
        description="Relevant facts loaded from long-term memory",
    )
