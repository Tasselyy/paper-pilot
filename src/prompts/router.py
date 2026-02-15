"""Prompt templates for the Router node (intent classification).

The Router classifies a user question into one of five intent types:
``factual``, ``comparative``, ``multi_hop``, ``exploratory``, ``follow_up``.

It outputs a structured ``RouterOutput`` with the intent type and a
confidence score in [0, 1].

Design reference: PAPER_PILOT_DESIGN.md §6.1, DEV_SPEC C1.
"""

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

ROUTER_SYSTEM_PROMPT = """\
You are PaperPilot's intent classification module.  Your job is to analyse \
the user's question about academic papers / machine-learning research and \
classify it into exactly **one** of the five intent types below.

## Intent Types

| Type | Description | Example |
|------|-------------|---------|
| **factual** | A direct factual question that can be answered from a single paper or concept. | "What is LoRA?" |
| **comparative** | Asks to compare two or more papers, methods, or concepts along specific dimensions. | "Compare LoRA and QLoRA in terms of memory efficiency" |
| **multi_hop** | Requires information from multiple sources combined through a chain of reasoning steps. | "How does the attention mechanism in Transformers relate to the improvements proposed in FlashAttention?" |
| **exploratory** | An open-ended exploration of a topic, survey-style, or broad question needing iterative research. | "What are the latest trends in efficient fine-tuning?" |
| **follow_up** | A continuation of the previous conversation turn — implicit references to prior context. | "Can you elaborate on that?" / "What about its limitations?" |

## Rules

1. Return **exactly one** type and a **confidence** score between 0 and 1.
2. If the question is ambiguous, pick the most likely type and lower the \
confidence accordingly.
3. ``follow_up`` should only be chosen when the question clearly references \
prior conversation context (e.g., pronouns like "it", "that", "those" with \
no explicit topic, or explicit phrases like "tell me more").
4. ``multi_hop`` requires reasoning across **multiple** distinct sources or \
concepts; a question about a single concept is ``factual``.
5. ``comparative`` requires explicit or implicit comparison between **at \
least two** named entities.
6. ``exploratory`` is for broad, survey-style, or open-ended research \
questions.
"""

# ---------------------------------------------------------------------------
# User prompt template
# ---------------------------------------------------------------------------

ROUTER_USER_TEMPLATE = """\
Question: {question}

Classify the above question into one of the five intent types \
(factual, comparative, multi_hop, exploratory, follow_up) with a \
confidence score.\
"""
