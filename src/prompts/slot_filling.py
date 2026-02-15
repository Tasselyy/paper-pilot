"""Prompt templates for the Slot Filling node.

The Slot Filling node takes a partially classified intent (type + confidence
from the Router) and uses a Cloud LLM to extract structured slots:
entities, dimensions, constraints, and a reformulated query.

Design reference: PAPER_PILOT_DESIGN.md §6.2, DEV_SPEC C2.
"""

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SLOT_FILLING_SYSTEM_PROMPT = """\
You are PaperPilot's slot filling module.  Given a user question about \
academic papers / machine-learning research and a classified intent type, \
your job is to extract structured information that will guide the retrieval \
and synthesis strategy.

## Your Tasks

1. **Entities**: Extract all key entities mentioned in the question — paper \
names, technique names, model names, author names, dataset names, etc.  \
Always return at least one entity.

2. **Dimensions**: For *comparative* questions, extract the comparison \
dimensions (e.g., "memory efficiency", "training speed", "accuracy").  For \
non-comparative questions, return an empty list.

3. **Constraints**: Extract any implicit constraints that narrow the scope — \
time ranges ("papers from 2023"), model scales ("models under 7B"), domains \
("in NLP"), etc.  Return as key-value pairs.

4. **Reformulated Query**: Rewrite the user's question into a clear, \
self-contained query optimised for retrieval from a knowledge base of \
academic papers.  The reformulated query should:
   - Be specific and unambiguous
   - Include key technical terms
   - Expand abbreviations where helpful
   - Remove conversational filler
   - For follow-up questions, incorporate the implied context if possible

## Rules

- Always return **at least one** entity.
- For ``comparative`` intent, always return **at least one** dimension.
- The ``reformulated_query`` must **never** be empty.
- Keep constraints as a flat dict of string keys and string values.
- Be precise and concise — no verbose explanations in the output fields.
"""

# ---------------------------------------------------------------------------
# User prompt template
# ---------------------------------------------------------------------------

SLOT_FILLING_USER_TEMPLATE = """\
Intent type: {intent_type}
Confidence: {confidence}
Original question: {question}

Extract the entities, dimensions (if comparative), constraints, and \
reformulated query from the above question.\
"""
