"""One-command: generate all training data (Router, Critic SFT, DPO) from LLM.

By default everything is from the LLM: Router questions per intent; for Critic/DPO
the (question, draft, contexts) scenario and the verdicts/chosen-rejected are all
LLM-generated. Use --no-full-scenario to use fixed pools for scenario (faster).

Requires OPENAI_API_KEY (or OPENAI_BASE_URL). Run from repo root:

  python training/data/generate_all_with_llm.py

Options:
  --router-per-intent   Questions per intent (default 40)
  --critic-samples      Critic SFT rows (default 150; ignored if --full-scenario)
  --dpo-pairs           DPO pairs (default 150; ignored if --full-scenario)
  --scenarios           When --full-scenario: number of LLM-generated scenarios (default 100), each yields 1 SFT + 1 DPO row
  --full-scenario       Generate (question, draft, contexts) via LLM for Critic/DPO (default: True, i.e. all from LLM)
  --no-full-scenario    Use fixed pools for question/draft/contexts, only verdicts from LLM (faster, fewer API calls)
  --output-dir          Directory for JSONL files (default training/data)
  --model, --api-base   Pass-through to LLM client
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

_root = Path(__file__).resolve().parent.parent.parent
if _root not in sys.path:
    sys.path.insert(0, str(_root))

from src.models.inference import build_critic_evaluation_prompt

from training.data.generate_router_data_with_llm import (
    generate_router_dataset_with_llm,
    write_jsonl as write_router_jsonl,
)
from training.data.generate_critic_sft_with_llm import (
    CRITIC_INSTRUCTION,
    _verdict_to_output_string,
    generate_critic_sft_with_llm,
)
from training.data.generate_dpo_pairs_with_llm import (
    _verdict_to_json_string,
    generate_dpo_with_llm,
)
from training.data.llm_client import DEFAULT_MODEL, chat_json, get_client

# Default output filenames (same as training config)
ROUTER_FILENAME = "router_train.jsonl"
CRITIC_FILENAME = "critic_sft_train.jsonl"
DPO_FILENAME = "dpo_train.jsonl"

DEFAULT_ROUTER_PER_INTENT = 40
DEFAULT_CRITIC_SAMPLES = 150
DEFAULT_DPO_PAIRS = 150
DEFAULT_SCENARIOS = 100
DEFAULT_SEED = 42

# For full-scenario: one LLM call to get (question, draft_answer, contexts, strategy)
SYSTEM_SCENARIO = (
    "You produce a single JSON object for a research Q&A training example. "
    "Keys: question (string), draft_answer (string, 2-4 sentences), contexts (string: "
    "numbered sources or '(no retrieved contexts)'), strategy (one of: simple, comparative, multi_hop, exploratory). "
    "Topic: ML/NLP (e.g. LoRA, RAG, transformers, quantization). "
    "Draft answer can be good or bad. Output only the JSON, no other text."
)
USER_SCENARIO = "Generate one diverse example. Output only the JSON."

SYSTEM_VERDICT = (
    "You are an evaluator. Given a question, draft answer, and contexts, "
    "output a single JSON with keys: score, completeness, faithfulness, feedback. "
    "score in [0, 10] (one decimal), completeness and faithfulness in [0, 1] (two decimals). "
    "Base your assessment on the actual content. Output only the JSON."
)
SYSTEM_CHOSEN = (
    "You are an evaluator. Output a single JSON with keys: score, completeness, faithfulness, feedback. "
    "Produce a FAVORABLE evaluation (score 7-10) justified by the draft and contexts. "
    "Output only the JSON."
)
SYSTEM_REJECTED = (
    "You are an evaluator. Output a single JSON with keys: score, completeness, faithfulness, feedback. "
    "Produce an UNFAVORABLE evaluation (score 0-6) justified by gaps or errors in the draft. "
    "Output only the JSON."
)


def _generate_one_scenario(client: Any, model: str) -> dict[str, Any] | None:
    out, err = chat_json(system=SYSTEM_SCENARIO, user=USER_SCENARIO, model=model, client=client)
    if err or not out:
        return None
    q = out.get("question") or ""
    d = out.get("draft_answer") or ""
    c = out.get("contexts") or "(no retrieved contexts)"
    s = out.get("strategy") or "simple"
    if s not in ("simple", "comparative", "multi_hop", "exploratory"):
        s = "simple"
    return {"question": q, "draft_answer": d, "contexts": c, "strategy": s}


def _run_full_scenario_mode(
    output_dir: Path,
    num_scenarios: int,
    seed: int,
    model: str,
    client: Any,
) -> None:
    """Generate num_scenarios (question, draft, contexts) via LLM; for each get one SFT row and one DPO row."""
    rng = random.Random(seed)
    sft_rows: list[dict[str, Any]] = []
    dpo_rows: list[dict[str, Any]] = []
    for i in range(num_scenarios):
        scenario = _generate_one_scenario(client, model)
        if not scenario:
            continue
        prompt = build_critic_evaluation_prompt(
            question=scenario["question"],
            draft_answer=scenario["draft_answer"],
            contexts=scenario["contexts"],
            strategy=scenario["strategy"],
        )
        # SFT: one verdict
        verdict_out, verr = chat_json(system=SYSTEM_VERDICT, user=prompt, model=model, client=client)
        if not verr and verdict_out:
            sft_rows.append({
                "instruction": CRITIC_INSTRUCTION,
                "input": prompt,
                "output": _verdict_to_output_string(verdict_out),
            })
        # DPO: chosen + rejected
        chosen_out, cerr = chat_json(system=SYSTEM_CHOSEN, user=prompt, model=model, client=client)
        rejected_out, rerr = chat_json(system=SYSTEM_REJECTED, user=prompt, model=model, client=client)
        if not cerr and not rerr and chosen_out and rejected_out:
            cs = float(chosen_out.get("score", 0))
            rs = float(rejected_out.get("score", 10))
            if cs <= rs:
                chosen_out["score"] = max(7.0, cs + 1.0)
                rejected_out["score"] = min(6.0, rs - 1.0)
            dpo_rows.append({
                "prompt": prompt,
                "chosen": _verdict_to_json_string(chosen_out),
                "rejected": _verdict_to_json_string(rejected_out),
            })
    rng.shuffle(sft_rows)
    rng.shuffle(dpo_rows)
    critic_path = output_dir / CRITIC_FILENAME
    dpo_path = output_dir / DPO_FILENAME
    output_dir.mkdir(parents=True, exist_ok=True)
    with critic_path.open("w", encoding="utf-8") as f:
        for row in sft_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with dpo_path.open("w", encoding="utf-8") as f:
        for row in dpo_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Full-scenario: {len(sft_rows)} Critic SFT -> {critic_path}")
    print(f"Full-scenario: {len(dpo_rows)} DPO pairs -> {dpo_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate all training data (Router, Critic SFT, DPO) from LLM."
    )
    parser.add_argument("--router-per-intent", type=int, default=DEFAULT_ROUTER_PER_INTENT)
    parser.add_argument("--critic-samples", type=int, default=DEFAULT_CRITIC_SAMPLES)
    parser.add_argument("--dpo-pairs", type=int, default=DEFAULT_DPO_PAIRS)
    parser.add_argument(
        "--scenarios",
        type=int,
        default=DEFAULT_SCENARIOS,
        help="When --full-scenario: number of scenarios (each → 1 SFT + 1 DPO row).",
    )
    parser.add_argument(
        "--full-scenario",
        action="store_true",
        default=True,
        help="Generate (question, draft, contexts) via LLM for Critic/DPO (default: True).",
    )
    parser.add_argument(
        "--no-full-scenario",
        action="store_false",
        dest="full_scenario",
        help="Use fixed pools for scenario; only verdicts from LLM.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("training/data"),
        help="Directory for router_train.jsonl, critic_sft_train.jsonl, dpo_train.jsonl.",
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--api-base", type=str, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    if args.api_base:
        os.environ["OPENAI_BASE_URL"] = args.api_base
    client = get_client()
    output_dir = args.output_dir.resolve()
    if not output_dir.is_absolute() and _root:
        output_dir = (_root / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) Router: LLM questions per intent
    router_path = output_dir / ROUTER_FILENAME
    router_rows = generate_router_dataset_with_llm(
        samples_per_intent=args.router_per_intent,
        seed=args.seed,
        model=args.model,
        client=client,
    )
    write_router_jsonl(router_rows, router_path)
    print(f"Router: {len(router_rows)} rows -> {router_path}")

    # 2) Critic SFT + 3) DPO: either pool+LLM verdicts or full-scenario
    if args.full_scenario:
        _run_full_scenario_mode(
            output_dir=output_dir,
            num_scenarios=args.scenarios,
            seed=args.seed,
            model=args.model,
            client=client,
        )
    else:
        critic_path = output_dir / CRITIC_FILENAME
        critic_rows = generate_critic_sft_with_llm(
            num_samples=args.critic_samples,
            seed=args.seed,
            model=args.model,
            client=client,
        )
        with critic_path.open("w", encoding="utf-8") as f:
            for row in critic_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Critic SFT: {len(critic_rows)} rows -> {critic_path}")

        dpo_path = output_dir / DPO_FILENAME
        dpo_rows = generate_dpo_with_llm(
            num_pairs=args.dpo_pairs,
            seed=args.seed,
            model=args.model,
            client=client,
        )
        with dpo_path.open("w", encoding="utf-8") as f:
            for row in dpo_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"DPO: {len(dpo_rows)} pairs -> {dpo_path}")

    print("Done. Use these files with sft_router.py, sft_critic.py, dpo_critic.py (default dataset paths).")


if __name__ == "__main__":
    main()
