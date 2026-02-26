"""Generate Critic SFT data with LLM-generated verdicts (content-grounded).

Output is Alpaca JSONL (instruction / input / output) consumed by sft_critic.py.
The verdict (score, completeness, faithfulness, feedback) is produced by an LLM
given the actual question, draft answer, and contexts. Requires OPENAI_API_KEY (or OPENAI_BASE_URL for vLLM).

Usage:
  python training/data/generate_critic_sft_with_llm.py --num-samples 200 --output training/data/critic_sft_llm.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

# Project root for imports when run as script
_root = Path(__file__).resolve().parent.parent.parent
if _root not in sys.path:
    sys.path.insert(0, str(_root))

from src.models.inference import build_critic_evaluation_prompt

from training.data.llm_client import DEFAULT_MODEL, chat_json, get_client

DEFAULT_NUM_SAMPLES = 200
DEFAULT_SEED = 42

CRITIC_INSTRUCTION = (
    "Evaluate the answer quality for the given question and retrieved context.\n"
    "Score rubric:\n"
    "- score: overall quality in [0, 10]\n"
    "- completeness: coverage in [0, 1]\n"
    "- faithfulness: grounding in context in [0, 1]\n"
    "Return ONLY valid JSON with keys: score, completeness, faithfulness, feedback."
)

SYSTEM_VERDICT = (
    "You are an evaluator. Given a question, a draft answer, and retrieved contexts, "
    "output a single JSON object with exactly these keys: score, completeness, faithfulness, feedback. "
    "score: number in [0, 10] (one decimal). completeness and faithfulness: numbers in [0, 1] (two decimals). "
    "feedback: short string explaining your assessment based on the actual draft and contexts. "
    "Output only the JSON, no markdown or other text."
)


def _sample_question(rng: random.Random) -> str:
    pool = (
        "What is LoRA and when is it preferable to full fine-tuning?",
        "Compare LoRA and QLoRA in terms of memory usage and quality.",
        "How does retrieval augmentation improve factual consistency?",
        "Explain why attention is important in Transformers.",
        "What are the trade-offs of 4-bit quantization for inference?",
        "How does DPO differ from SFT in training objectives?",
    )
    return rng.choice(pool)


def _sample_draft(rng: random.Random) -> str:
    """Sample a draft answer (mix of bad, mid, good)."""
    pool = (
        "LoRA is just a dataset and usually slower than all methods.",
        "RAG mainly increases hallucinations because of extra context.",
        "LoRA adds trainable low-rank adapters and reduces fine-tuning memory.",
        "RAG retrieves related documents and can reduce unsupported claims.",
        "4-bit quantization lowers memory cost but may slightly affect quality.",
        "LoRA inserts low-rank trainable matrices while freezing base weights.",
        "QLoRA combines quantized base weights with LoRA adapters.",
        "RAG conditions generation on retrieved context, which improves factual grounding.",
    )
    return rng.choice(pool)


def _sample_context_block(rng: random.Random) -> str:
    pool = (
        "[1] Source: LoRA Paper\nLoRA reduces trainable parameters via low-rank updates.\n\n"
        "[2] Source: QLoRA\nQLoRA uses quantized base weights with LoRA adapters.",
        "[1] Source: RAG Survey\nRAG improves grounding by conditioning on retrieved passages.\n\n"
        "[2] Source: Hallucination Study\nGrounded generation reduces unsupported claims.",
        "(no retrieved contexts)",
    )
    return rng.choice(pool)


def _sample_strategy(rng: random.Random) -> str:
    return rng.choice(("simple", "comparative", "multi_hop", "exploratory"))


def _verdict_to_output_string(obj: dict[str, Any]) -> str:
    """Normalize and serialize verdict to compact JSON string."""
    score = float(obj.get("score", 0))
    score = max(0, min(10, round(score, 1)))
    completeness = float(obj.get("completeness", 0))
    completeness = max(0, min(1, round(completeness, 2)))
    faithfulness = float(obj.get("faithfulness", 0))
    faithfulness = max(0, min(1, round(faithfulness, 2)))
    feedback = str(obj.get("feedback", "")) or "No feedback."
    payload = {
        "score": score,
        "completeness": completeness,
        "faithfulness": faithfulness,
        "feedback": feedback,
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def generate_critic_sft_with_llm(
    *,
    num_samples: int = DEFAULT_NUM_SAMPLES,
    seed: int = DEFAULT_SEED,
    model: str = DEFAULT_MODEL,
    client: Any | None = None,
) -> list[dict[str, Any]]:
    """Generate Critic SFT rows: same (question, draft, contexts) sampling as template script, verdict from LLM."""
    if num_samples <= 0:
        raise ValueError("num_samples must be > 0")
    if client is None:
        client = get_client()
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for i in range(num_samples):
        question = _sample_question(rng)
        draft = _sample_draft(rng)
        contexts = _sample_context_block(rng)
        strategy = _sample_strategy(rng)
        input_text = build_critic_evaluation_prompt(
            question=question,
            draft_answer=draft,
            contexts=contexts,
            strategy=strategy,
        )
        out, err = chat_json(system=SYSTEM_VERDICT, user=input_text, model=model, client=client)
        if err or not out:
            raise RuntimeError(f"Sample {i + 1}/{num_samples} LLM call failed: {err or 'empty'}")
        output_text = _verdict_to_output_string(out)
        rows.append(
            {
                "instruction": CRITIC_INSTRUCTION,
                "input": input_text,
                "output": output_text,
            }
        )
    rng.shuffle(rows)
    return rows


def write_jsonl(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Critic SFT data with LLM-generated content-grounded verdicts."
    )
    parser.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path, default=Path("training/data/critic_sft_llm.jsonl"))
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Chat model name.")
    parser.add_argument("--api-base", type=str, default=None, help="Override OPENAI_BASE_URL.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.api_base:
        import os
        os.environ["OPENAI_BASE_URL"] = args.api_base
    client = get_client()
    rows = generate_critic_sft_with_llm(
        num_samples=args.num_samples,
        seed=args.seed,
        model=args.model,
        client=client,
    )
    write_jsonl(rows=rows, output_path=args.output)
    print(f"Generated {len(rows)} Critic SFT samples (LLM verdicts) at: {args.output}")


if __name__ == "__main__":
    main()
