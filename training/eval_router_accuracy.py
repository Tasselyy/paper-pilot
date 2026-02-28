"""Evaluate Router (intent classification) accuracy on a JSONL eval set.

Use after training to measure intent recognition accuracy. Supports local LoRA
or vLLM backend. Run from project root:

  # Local LoRA
  python training/eval_router_accuracy.py --router-model-path /path/to/router-lora

  # vLLM
  python training/eval_router_accuracy.py --vllm --router-model router-lora

  # Custom eval file and holdout split
  python training/eval_router_accuracy.py --router-model-path ./out/router --eval-file training/data/router_train.jsonl --split-ratio 0.2
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

# Project root for imports
if __name__ == "__main__" and __package__ is None:
    _root = Path(__file__).resolve().parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from src.types import IntentType

INTENT_TYPES: tuple[IntentType, ...] = (
    "factual",
    "comparative",
    "multi_hop",
    "exploratory",
    "follow_up",
)


def load_eval_pairs(path: Path) -> list[tuple[str, IntentType]]:
    """Load (question, gold_intent) from JSONL. Expects 'input' and 'intent' keys."""
    pairs: list[tuple[str, IntentType]] = []
    for line in path.read_text(encoding="utf-8").strip().splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        q = obj.get("input") or obj.get("question") or ""
        intent = (obj.get("intent") or "").strip().lower()
        if not q:
            continue
        if intent not in INTENT_TYPES:
            intent = "factual"  # fallback
        pairs.append((q, intent))
    return pairs


def split_holdout(
    pairs: list[tuple[str, IntentType]],
    ratio: float,
    seed: int,
) -> list[tuple[str, IntentType]]:
    """Use `ratio` of data as test set (e.g. 0.2 = 20% test). Returns test pairs."""
    if ratio <= 0 or ratio >= 1:
        return pairs
    rng = random.Random(seed)
    indices = list(range(len(pairs)))
    rng.shuffle(indices)
    n_test = max(1, int(len(pairs) * ratio))
    test_indices = set(indices[-n_test:])
    return [pairs[i] for i in sorted(test_indices)]


def run_eval(
    pairs: list[tuple[str, IntentType]],
    *,
    router_model_path: str | None = None,
    vllm: bool = False,
    router_model: str | None = None,
    vllm_base_url: str = "http://localhost:8000/v1",
) -> list[tuple[str, IntentType, IntentType, float]]:
    """Run classifier on all pairs. Returns list of (question, gold, pred, confidence)."""
    if vllm and router_model:
        from src.models.vllm_client import VLLMInferenceClient

        client = VLLMInferenceClient(
            base_url=vllm_base_url,
            router_model=router_model,
            critic_model=None,
        )
        classify = client.classify_question
    elif router_model_path:
        from src.models.loader import LocalModelManager

        manager = LocalModelManager(router_model_path=router_model_path)
        classify = manager.classify_question
    else:
        raise ValueError(
            "Provide either --router-model-path (local) or --vllm and --router-model (vLLM)."
        )

    results: list[tuple[str, IntentType, IntentType, float]] = []
    for question, gold in pairs:
        pred, conf = classify(question)
        results.append((question, gold, pred, conf))
    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate Router intent classification accuracy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--eval-file",
        type=Path,
        default=Path("training/data/router_train.jsonl"),
        help="JSONL with 'input' and 'intent' per line",
    )
    parser.add_argument(
        "--split-ratio",
        type=float,
        default=0.0,
        metavar="R",
        help="Use R of data as test (e.g. 0.2). 0 = use all.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split")
    # Local LoRA
    parser.add_argument(
        "--router-model-path",
        type=str,
        default=None,
        help="Path to local Router LoRA (use this for local inference)",
    )
    # vLLM
    parser.add_argument(
        "--vllm",
        action="store_true",
        help="Use vLLM OpenAI-compatible endpoint",
    )
    parser.add_argument(
        "--router-model",
        type=str,
        default=None,
        help="vLLM model name (required if --vllm)",
    )
    parser.add_argument(
        "--vllm-base-url",
        type=str,
        default="http://localhost:8000/v1",
        help="vLLM API base URL",
    )
    # Output
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Write accuracy and per-class stats to JSON",
    )
    parser.add_argument(
        "--no-per-class",
        action="store_true",
        help="Do not print per-intent accuracy",
    )
    args = parser.parse_args()

    if not args.eval_file.exists():
        print(f"Error: eval file not found: {args.eval_file}", file=sys.stderr)
        return 1

    if args.vllm and not args.router_model:
        print("Error: --router-model is required when using --vllm", file=sys.stderr)
        return 1
    if not args.vllm and not args.router_model_path:
        print(
            "Error: provide --router-model-path (local) or --vllm and --router-model",
            file=sys.stderr,
        )
        return 1

    pairs = load_eval_pairs(args.eval_file)
    if args.split_ratio > 0:
        pairs = split_holdout(pairs, args.split_ratio, args.seed)
        print(f"Using {len(pairs)} samples (holdout {args.split_ratio:.0%}, seed={args.seed})")
    else:
        print(f"Using full eval file: {len(pairs)} samples")

    if not pairs:
        print("No samples to evaluate.", file=sys.stderr)
        return 1

    print("Running inference...")
    results = run_eval(
        pairs,
        router_model_path=args.router_model_path,
        vllm=args.vllm,
        router_model=args.router_model,
        vllm_base_url=args.vllm_base_url,
    )

    correct = sum(1 for _, g, p, _ in results if g == p)
    total = len(results)
    acc = correct / total if total else 0.0

    print(f"\nAccuracy: {correct}/{total} = {acc:.2%}")

    stats: dict = {"accuracy": acc, "correct": correct, "total": total}

    if not args.no_per_class:
        by_intent: dict[IntentType, list[bool]] = {t: [] for t in INTENT_TYPES}
        for _, gold, pred, _ in results:
            by_intent[gold].append(gold == pred)
        print("\nPer-intent accuracy:")
        for t in INTENT_TYPES:
            lst = by_intent[t]
            if not lst:
                continue
            acc_t = sum(lst) / len(lst)
            print(f"  {t}: {sum(lst)}/{len(lst)} = {acc_t:.2%}")
            stats.setdefault("per_intent", {})[t] = {
                "correct": sum(lst),
                "total": len(lst),
                "accuracy": acc_t,
            }

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        print(f"\nWrote {args.output_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
