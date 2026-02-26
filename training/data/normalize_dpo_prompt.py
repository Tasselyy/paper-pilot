"""Normalize DPO JSONL so prompt matches build_critic_evaluation_prompt() exactly.

Gemini often outputs:
- "context.Score rubric:" instead of "context.\nScore rubric:"
- "score:" / "completeness:" / "faithfulness:" without "- " prefix
- "faithfulness: ... [0, 1]Return" instead of "... [0, 1]\nReturn"
- "feedback.{\"score\"" instead of "feedback.\n{\"score\""
- "Contexts:[1]" instead of "Contexts:\n[1]"

Usage:
  python training/data/normalize_dpo_prompt.py < input.jsonl > normalized.jsonl
  python training/data/normalize_dpo_prompt.py --in-place training/data/dpo_train.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


# Target prefix for the instruction part (from src.models.inference.build_critic_evaluation_prompt)
EXPECTED_HEAD = (
    "Evaluate the answer quality for the given question and retrieved context.\n"
    "Score rubric:\n"
    "- score: overall quality in [0, 10]\n"
    "- completeness: coverage in [0, 1]\n"
    "- faithfulness: grounding in context in [0, 1]\n"
    "Return ONLY valid JSON with keys: score, completeness, faithfulness, feedback.\n"
    '{"score":7.8,"completeness":0.82,"faithfulness":0.86,"feedback":"Add a concrete example."}\n'
)


def normalize_prompt(prompt: str) -> str:
    """Fix common Gemini prompt format so it matches inference-time prompt."""
    s = prompt

    # Ensure newline after "context." before "Score rubric:"
    s = re.sub(r"context\.Score rubric:", "context.\nScore rubric:", s, count=1)

    # Rubric lines: add "- " when missing (only first occurrence of each, rubric is early in string)
    s = re.sub(r"\nscore: overall quality", "\n- score: overall quality", s, count=1)
    s = re.sub(r"\ncompleteness: coverage", "\n- completeness: coverage", s, count=1)
    s = re.sub(r"\nfaithfulness: grounding", "\n- faithfulness: grounding", s, count=1)

    # Newline before "Return ONLY"
    s = re.sub(r"\[0, 1\]Return ONLY", "[0, 1]\nReturn ONLY", s, count=1)

    # Newline before example JSON
    s = re.sub(r"feedback\.\s*\{\s*\"score\"", 'feedback.\n{"score"', s, count=1)

    # Newline after "Contexts:" when followed by [ or (
    s = re.sub(r"Contexts:(\s*)(\[|\()", r"Contexts:\n\1\2", s, count=1)

    return s


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize DPO prompt format in JSONL.")
    parser.add_argument("input", nargs="?", type=Path, default=None, help="Input JSONL file (default: stdin)")
    parser.add_argument("--in-place", action="store_true", help="Overwrite input file")
    parser.add_argument("--output", "-o", type=Path, default=None, help="Output JSONL file (default: stdout)")
    args = parser.parse_args()

    if args.in_place:
        if not args.input or not args.input.exists():
            print("--in-place requires an existing input file.", file=sys.stderr)
            sys.exit(1)
        lines = args.input.read_text(encoding="utf-8").strip().split("\n")
        out_path = args.input
    else:
        if args.input and args.input.exists():
            lines = args.input.read_text(encoding="utf-8").strip().split("\n")
        else:
            lines = sys.stdin.read().strip().split("\n")
        out_path = args.output

    out_lines: list[str] = []
    for raw in lines:
        raw = raw.strip()
        if not raw:
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            out_lines.append(raw)
            continue
        if "prompt" in row:
            row["prompt"] = normalize_prompt(row["prompt"])
        out_lines.append(json.dumps(row, ensure_ascii=False))

    text = "\n".join(out_lines) + ("\n" if out_lines else "")
    if out_path:
        out_path.write_text(text, encoding="utf-8")
        print(f"Wrote {len(out_lines)} lines to {out_path}", file=sys.stderr)
    else:
        print(text)


if __name__ == "__main__":
    main()
