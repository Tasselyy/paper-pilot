"""Generate Router training data with LLM-produced questions (diverse phrasings).

Output is Alpaca JSONL (instruction / input / output / intent) consumed by sft_router.py.
For each intent, the LLM generates diverse user questions. Requires OPENAI_API_KEY (or OPENAI_BASE_URL for vLLM).

Usage:
  python training/data/generate_router_data_with_llm.py --samples-per-intent 40 --output training/data/router_train_llm.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

_root = Path(__file__).resolve().parent.parent.parent
if _root not in sys.path:
    sys.path.insert(0, str(_root))

from training.data.llm_client import DEFAULT_MODEL, chat_text, get_client

INTENT_TYPES: tuple[str, ...] = (
    "factual",
    "comparative",
    "multi_hop",
    "exploratory",
    "follow_up",
)

DEFAULT_SAMPLES_PER_INTENT = 40
DEFAULT_SEED = 42

ROUTER_INSTRUCTION = (
    "Classify the user question into one intent type. "
    "Allowed types: factual, comparative, multi_hop, exploratory, follow_up. "
    "Return strict JSON with keys: type, confidence."
)


def _generate_questions_for_intent(
    intent: str,
    count: int,
    model: str,
    client: Any,
) -> list[str]:
    """Ask LLM to generate `count` diverse questions for the given intent."""
    user = (
        f"Generate exactly {count} diverse user questions that a research assistant "
        f"would classify as intent \"{intent}\". "
        "Topics: ML/NLP (e.g. LoRA, RAG, transformers, quantization). "
        "One question per line. No numbering, no bullets. Plain text only."
    )
    system = "You output only the requested list of questions, one per line, nothing else."
    text, err = chat_text(system=system, user=user, model=model, client=client)
    if err or not text:
        raise RuntimeError(f"LLM failed for intent {intent}: {err or 'empty'}")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip() and len(ln.strip()) > 10]
    return lines[:count]


def generate_router_dataset_with_llm(
    *,
    samples_per_intent: int = DEFAULT_SAMPLES_PER_INTENT,
    seed: int = DEFAULT_SEED,
    model: str = DEFAULT_MODEL,
    client: Any | None = None,
) -> list[dict[str, Any]]:
    """Generate Router rows: per-intent questions from LLM, output format same as template script."""
    if samples_per_intent <= 0:
        raise ValueError("samples_per_intent must be > 0")
    if client is None:
        client = get_client()
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for intent in INTENT_TYPES:
        questions = _generate_questions_for_intent(
            intent=intent,
            count=samples_per_intent,
            model=model,
            client=client,
        )
        for idx, question in enumerate(questions):
            confidence = round(0.80 + (idx % 7) * 0.02 + rng.uniform(0, 0.05), 2)
            confidence = max(0.80, min(0.99, confidence))
            output = json.dumps(
                {"type": intent, "confidence": confidence},
                ensure_ascii=False,
                separators=(",", ":"),
            )
            rows.append({
                "instruction": ROUTER_INSTRUCTION,
                "input": question,
                "output": output,
                "intent": intent,
            })
    rng.shuffle(rows)
    return rows


def write_jsonl(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Router training data with LLM-generated diverse questions."
    )
    parser.add_argument(
        "--samples-per-intent",
        type=int,
        default=DEFAULT_SAMPLES_PER_INTENT,
        help="Number of questions to generate per intent (via LLM).",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("training/data/router_train_llm.jsonl"),
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--api-base", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.api_base:
        import os
        os.environ["OPENAI_BASE_URL"] = args.api_base
    client = get_client()
    rows = generate_router_dataset_with_llm(
        samples_per_intent=args.samples_per_intent,
        seed=args.seed,
        model=args.model,
        client=client,
    )
    write_jsonl(rows=rows, output_path=args.output)
    print(f"Generated {len(rows)} Router samples (LLM questions) at: {args.output}")


if __name__ == "__main__":
    main()
