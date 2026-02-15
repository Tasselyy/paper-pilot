#!/usr/bin/env python3
"""
Spec Sync — splits DEV_SPEC.md into chapter files under auto-coder/specs/.

Usage:
    python .cursor/skills/auto-coder/scripts/sync_spec.py [--force]

Features:
    - Detects chapters by '## N. Title' (tolerant of extra whitespace)
    - Hash-based skip: only re-syncs when DEV_SPEC.md changes (or --force)
    - Cleans orphan spec files that no longer correspond to a chapter
    - Prints a summary table of synced chapters
"""

import hashlib
import re
import sys
from pathlib import Path
from typing import List, Tuple, NamedTuple


class Chapter(NamedTuple):
    number: int
    cn_title: str
    filename: str
    start_line: int
    end_line: int
    line_count: int


# Chapter number -> English slug (encoding-independent)
NUMBER_SLUG_MAP = {
    1: "overview",
    2: "features",
    3: "tech-stack",
    4: "testing",
    5: "architecture",
    6: "schedule",
    7: "future",
}


def _slug(chapter_num: int, title: str) -> str:
    """Return English slug for a chapter number, with fallback sanitization."""
    if chapter_num in NUMBER_SLUG_MAP:
        return NUMBER_SLUG_MAP[chapter_num]
    # Fallback: sanitize whatever title text we have
    clean = re.sub(r'[^\w]+', '-', title, flags=re.ASCII).strip('-').lower()
    return clean or f"chapter-{chapter_num}"


def detect_chapters(content: str) -> List[Chapter]:
    """
    Detect chapters matching '## N. Title' (tolerant of extra spaces,
    full-width digits, or trailing whitespace).
    """
    lines = content.split('\n')
    starts: List[Tuple[int, str, int]] = []

    # Pattern: ## <digits>. <title>  (allow multiple spaces, optional trailing spaces)
    pattern = re.compile(r'^##\s+(\d+)\.\s+(.+?)\s*$')

    for i, line in enumerate(lines):
        m = pattern.match(line)
        if m:
            starts.append((int(m.group(1)), m.group(2).strip(), i))

    if not starts:
        # Provide helpful diagnostics
        h2_lines = [(i, line) for i, line in enumerate(lines) if line.startswith('## ')]
        hint = ""
        if h2_lines:
            samples = h2_lines[:3]
            hint = "\n  Found these ## headers (not matching '## N. Title'):\n"
            for ln, txt in samples:
                hint += f"    L{ln + 1}: {txt}\n"
        raise ValueError(
            f"No chapters found in DEV_SPEC.md. Expected '## N. Title' pattern.{hint}"
        )

    # Warn about gaps in numbering
    nums = [n for n, _, _ in starts]
    expected = list(range(nums[0], nums[0] + len(nums)))
    if nums != expected:
        print(f"  WARN: chapter numbers {nums} are not sequential (expected {expected})")

    chapters = []
    for idx, (num, title, start) in enumerate(starts):
        end = starts[idx + 1][2] if idx + 1 < len(starts) else len(lines)
        chapters.append(Chapter(
            num, title,
            f"{num:02d}-{_slug(num, title)}.md",
            start, end, end - start,
        ))
    return chapters


def sync(force: bool = False):
    """Main sync logic: read DEV_SPEC, detect chapters, write spec files."""
    skill_dir = Path(__file__).resolve().parent.parent   # auto-coder/
    repo_root = skill_dir.parent.parent.parent           # project root
    dev_spec  = repo_root / "DEV_SPEC.md"
    specs_dir = skill_dir / "specs"
    hash_file = skill_dir / ".spec_hash"

    if not dev_spec.exists():
        print(f"ERROR: DEV_SPEC.md not found at {dev_spec}")
        print(f"  Hint: run this script from the project root, or check repo_root={repo_root}")
        sys.exit(1)

    # Hash check — skip if unchanged
    current_hash = hashlib.sha256(dev_spec.read_bytes()).hexdigest()
    if not force and hash_file.exists() and hash_file.read_text().strip() == current_hash:
        print("specs up-to-date (hash match, use --force to override)")
        return

    content = dev_spec.read_text(encoding='utf-8')
    chapters = detect_chapters(content)
    lines = content.split('\n')

    specs_dir.mkdir(parents=True, exist_ok=True)

    # Clean orphan files
    old = {f.name for f in specs_dir.glob("*.md")}
    new = {ch.filename for ch in chapters}
    orphans = old - new
    for f in orphans:
        (specs_dir / f).unlink()
        print(f"  removed orphan: {f}")

    # Write chapter files
    for ch in chapters:
        (specs_dir / ch.filename).write_text(
            '\n'.join(lines[ch.start_line:ch.end_line]),
            encoding='utf-8',
        )

    hash_file.write_text(current_hash)

    # Summary
    print(f"synced {len(chapters)} chapters → {specs_dir.relative_to(repo_root)}/")
    print(f"  {'File':<25} {'Title':<30} {'Lines':>5}")
    print(f"  {'─' * 25} {'─' * 30} {'─' * 5}")
    for ch in chapters:
        print(f"  {ch.filename:<25} {ch.cn_title:<30} {ch.line_count:>5}")


if __name__ == "__main__":
    sync(force="--force" in sys.argv)
