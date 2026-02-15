"""Prompt templates for the retry_refine node (Reflexion pattern).

When the Critic rejects a draft answer, the retry_refine node uses these
prompts to rewrite the user question so that the next strategy cycle
addresses the feedback.  The LLM produces a refined question that
incorporates the Critic's suggestions, guiding the retrieval and
synthesis steps toward a better answer.

Design reference: PAPER_PILOT_DESIGN.md §6.8, DEV_SPEC D3.
"""

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

RETRY_REFINE_SYSTEM_PROMPT = """\
You are PaperPilot's question-refinement assistant.  A draft answer to \
the user's research question was evaluated by a quality critic and \
found lacking.  Your job is to **rewrite the question** so the next \
retrieval and synthesis cycle produces a better answer.

## Guidelines

1. Read the critic's feedback carefully — it explains what is missing \
or incorrect in the draft answer.
2. Incorporate the feedback into a refined question that explicitly \
asks for the missing information or corrects the focus.
3. Preserve the original question's intent and scope; do not change \
the topic.
4. If the feedback mentions specific aspects (e.g., "missing details \
about rank decomposition"), the refined question should explicitly \
mention those aspects.
5. Keep the refined question concise and clear — suitable for a \
knowledge-base search.
6. Provide a brief summary of what you changed and why.\
"""

# ---------------------------------------------------------------------------
# User prompt template
# ---------------------------------------------------------------------------

RETRY_REFINE_USER_TEMPLATE = """\
Original question: {question}

Strategy used: {strategy}

Draft answer (rejected):
{draft_answer}

Critic feedback:
{feedback}

Rewrite the question to address the critic's feedback. The refined \
question should guide the next retrieval cycle toward a better answer.\
"""
