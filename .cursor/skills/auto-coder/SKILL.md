---
name: auto-coder
description: Autonomous spec-driven development agent. Reads DEV_SPEC.md, identifies next task, implements code, runs tests, and persists progress — all in one command with minimal user intervention. Use when user says "auto code", "自动开发", "自动写代码", "auto dev", "一键开发", "autopilot", or wants fully automated spec-to-code workflow. Replaces manual dev-workflow pipeline with autonomous execution.
---

# Auto Coder

Autonomous agent: one trigger completes **read spec → find task → code → test → persist progress**.

## Trigger

| User Says | Behavior |
|-----------|----------|
| "auto code" / "自动开发" | Next task, full cycle |
| "auto code B2" | Specific task |
| "auto code --no-commit" | Skip git commit |

---

## Pipeline

```
Sync Spec → Find Task → Implement → Test (≤3 fix rounds) → Persist
```

Only pause at the very end for commit confirmation. Everything else runs autonomously.

> **⚠️ CRITICAL: ALL Python commands MUST run inside the project venv.**
> Before executing ANY `python` or `pytest` command, activate the venv first:
> ```powershell
> .\.venv\Scripts\Activate.ps1
> ```
> Verify by checking `Get-Command python` points to `.venv\Scripts\python.exe`.
> **Never use system Python. Never skip this step.**

---

### 1. Sync Spec

Activate venv first, then sync:
```powershell
.\.venv\Scripts\Activate.ps1
python .cursor/skills/auto-coder/scripts/sync_spec.py
```

Then read the schedule file to get task statuses:
- Read `.cursor/skills/auto-coder/specs/06-schedule.md`

Task markers:

| Marker | Status |
|--------|--------|
| `[ ]` / `⬜` | Not started |
| `[~]` / `🔶` / `(进行中)` | In progress |
| `[x]` / `✅` / `(已完成)` | Completed |

---

### 2. Find Task

Priority: first `IN_PROGRESS`, then first `NOT_STARTED`. If user specified a task ID, use that directly.

Quick-check predecessor artifacts exist (file-level only). On mismatch, log warning and continue — only stop if the target task itself is blocked.

---

### 3. Implement

1. **Read relevant spec** from `.cursor/skills/auto-coder/specs/`:
   - Architecture: `05-architecture.md`
   - Tech details: `03-tech-stack.md`
   - Testing conventions: `04-testing.md`

2. **Extract** from spec: inputs/outputs, design principles (Pluggable? Config-driven? Factory?), file list, acceptance criteria.

3. **Plan** files to create/modify before writing any code.

4. **Code** — mandatory standards:
   - Type hints on all signatures
   - Google-style docstrings on public APIs
   - No hardcoded values (use config)
   - Single responsibility, short functions
   - Error handling for external integrations

5. **Write tests** alongside code:
   - `tests/unit/test_<module>.py` or `tests/integration/` per spec
   - Naming: `test_<func>_<scenario>_<expected>`
   - Mock external deps in unit tests

6. **Self-review** before running tests: all planned files exist, type hints present, no hardcoded values, tests import correctly.

---

### 4. Test & Auto-Fix

```

Round 0..2:
  Run pytest on relevant test file
  If pass → go to step 5
  If fail → analyze error, apply fix, re-run

Round 3 still failing → STOP, show failure report to user
```

---

### 5. Persist

1. **Update `DEV_SPEC.md`** (global file): change task marker `[ ]` → `[x]`
2. **Re-sync**: `python .cursor/skills/auto-coder/scripts/sync_spec.py --force`
3. **Show summary & ask**:

```
✅ [A3] 配置加载与校验 — done
   Files: src/core/settings.py, tests/unit/test_settings.py
   Tests: 8/8 passed
   Commit: feat(config): [A3] implement config loader

   "commit" → git add + commit
   "skip"   → end
   "next"   → commit + start next task
```

On "next", loop back to step 1 for the next task.

---

## Guardrails

- One task per cycle, atomic commits
- Spec is single source of truth
- 3-round test fix limit
- Match existing codebase style
- **MUST activate `.venv` before ANY `python`/`pytest` command** — no exceptions. If unsure whether venv is active, run `.\.venv\Scripts\Activate.ps1` again (idempotent)

---

## Directory Structure

```
auto-coder/
├── SKILL.md              ← this file
├── .spec_hash            ← auto-generated hash
├── scripts/
│   └── sync_spec.py      ← splits DEV_SPEC.md into chapters
└── specs/                ← auto-generated from DEV_SPEC (## N. Title)
    ├── 01-overview.md
    ├── 02-features.md
    ├── 03-tech-stack.md
    ├── 04-testing.md
    ├── 05-architecture.md
    └── 06-schedule.md
```

All paths are self-contained. This skill has no external dependencies on other skills.
