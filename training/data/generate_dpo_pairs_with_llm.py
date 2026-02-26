"""Generate DPO preference pairs with LLM-produced chosen/rejected verdicts (content-grounded).

Output is JSONL (prompt, chosen, rejected) consumed by dpo_critic.py. For each
(question, draft, contexts), the LLM produces one high- and one low-quality verdict
that reference the actual content. Requires OPENAI_API_KEY (or OPENAI_BASE_URL for vLLM).

Usage:
  python training/data/generate_dpo_pairs_with_llm.py --num-pairs 200 --output training/data/dpo_llm.jsonl
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

from src.models.inference import build_critic_evaluation_prompt

from training.data.llm_client import DEFAULT_MODEL, chat_json, get_client

DEFAULT_PAIRS = 200
DEFAULT_SEED = 42

SYSTEM_CHOSEN = (
    "You are an evaluator. Given a question, a draft answer, and retrieved contexts, "
    "output a single JSON object with keys: score, completeness, faithfulness, feedback. "
    "Produce a FAVORABLE evaluation (score between 7 and 10) that is justified by the actual "
    "draft and contexts. completeness and faithfulness in [0, 1]. feedback: short string. "
    "Output only the JSON, no other text."
)

SYSTEM_REJECTED = (
    "You are an evaluator. Given a question, a draft answer, and retrieved contexts, "
    "output a single JSON object with keys: score, completeness, faithfulness, feedback. "
    "Produce an UNFAVORABLE evaluation (score between 0 and 6) that is justified by real "
    "gaps or errors in the draft relative to the contexts. completeness and faithfulness in [0, 1]. "
    "feedback: short string. Output only the JSON, no other text."
)


def _sample_question(rng: random.Random) -> str:
    pool = (
        "What is LoRA and when would you use it?",
        "Compare QLoRA and full fine-tuning for memory usage.",
        "How does RAG improve factual consistency in LLM outputs?",
        "What are the trade-offs of 4-bit quantization?",
        "How does DPO differ from SFT in training objectives?",
    )
    return rng.choice(pool)


def _sample_draft(rng: random.Random) -> str:
    pool = (
        "LoRA is a parameter-efficient method that trains low-rank adapters.",
        "QLoRA uses quantized base models and typically reduces memory.",
        "RAG retrieves relevant documents and conditions the model on them.",
        "4-bit quantization saves memory but can slightly reduce accuracy.",
        "DPO is a direct preference optimization method.",
    )
    return rng.choice(pool)


def _sample_context_block(rng: random.Random) -> str:
    pool = (
        "[1] Source: LoRA Paper\nLoRA injects low-rank adapters into attention layers.\n\n"
        "[2] Source: QLoRA\nQLoRA combines 4-bit quantization with LoRA fine-tuning.",
        "[1] Source: RAG Survey\nRAG improves factual grounding by conditioning on retrieved passages.\n\n"
        "[2] Source: Faithfulness Benchmarks\nGrounded generation reduces unsupported claims.",
        "(no retrieved contexts)",
    )
    return rng.choice(pool)


def _sample_strategy(rng: random.Random) -> str:
    return rng.choice(("simple", "comparative", "multi_hop", "exploratory"))


def _verdict_to_json_string(obj: dict[str, Any]) -> str:
    score = max(0, min(10, round(float(obj.get("score", 0)), 1)))
    completeness = max(0, min(1, round(float(obj.get("completeness", 0)), 2)))
    faithfulness = max(0, min(1, round(float(obj.get("faithfulness", 0)), 2)))
    feedback = str(obj.get("feedback", "")) or "No feedback."
    payload = {
        "score": score,
        "completeness": completeness,
        "faithfulness": faithfulness,
        "feedback": feedback,
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def generate_dpo_with_llm(
    *,
    num_pairs: int = DEFAULT_PAIRS,
    seed: int = DEFAULT_SEED,
    model: str = DEFAULT_MODEL,
    client: Any | None = None,
) -> list[dict[str, Any]]:
    """Generate DPO rows: prompt from (question, draft, contexts), chosen/rejected from LLM."""
    if num_pairs <= 0:
        raise ValueError("num_pairs must be > 0")
    if client is None:
        client = get_client()
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for i in range(num_pairs):
        question = _sample_question(rng)
        draft = _sample_draft(rng)
        contexts = _sample_context_block(rng)
        strategy = _sample_strategy(rng)
        prompt = build_critic_evaluation_prompt(
            question=question,
            draft_answer=draft,
            contexts=contexts,
            strategy=strategy,
        )
        chosen_out, err1 = chat_json(
            system=SYSTEM_CHOSEN, user=prompt, model=model, client=client
        )
        if err1 or not chosen_out:
            raise RuntimeError(f"Pair {i + 1}/{num_pairs} chosen failed: {err1 or 'empty'}")
        rejected_out, err2 = chat_json(
            system=SYSTEM_REJECTED, user=prompt, model=model, client=client
        )
        if err2 or not rejected_out:
            raise RuntimeError(f"Pair {i + 1}/{num_pairs} rejected failed: {err2 or 'empty'}")
        chosen_score = float(chosen_out.get("score", 0))
        rejected_score = float(rejected_out.get("score", 10))
        if chosen_score <= rejected_score:
            chosen_out["score"] = max(7.0, round(chosen_score + 1.0, 1))
            rejected_out["score"] = min(6.0, round(rejected_score - 1.0, 1))
        rows.append({
            "prompt": prompt,
            "chosen": _verdict_to_json_string(chosen_out),
            "rejected": _verdict_to_json_string(rejected_out),
        })
    random.Random(seed + 1).shuffle(rows)
    return rows


def write_jsonl(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate DPO preference pairs with LLM-generated chosen/rejected verdicts."
    )
    parser.add_argument("--num-pairs", type=int, default=DEFAULT_PAIRS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path, default=Path("training/data/dpo_llm.jsonl"))
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--api-base", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.api_base:
        import os
        os.environ["OPENAI_BASE_URL"] = args.api_base
    client = get_client()
    rows = generate_dpo_with_llm(
        num_pairs=args.num_pairs,
        seed=args.seed,
        model=args.model,
        client=client,
    )
    write_jsonl(rows=rows, output_path=args.output)
    print(f"Generated {len(rows)} DPO pairs (LLM verdicts) at: {args.output}")


if __name__ == "__main__":
    main()
