# Agent instructions

Guidance for any AI agent (Claude Code, Codex, …) working in this repo. This is
the single source of truth: `CLAUDE.md` is a symlink to this file.

## What this is

`agentloop` is a generic multi-agent loop that drives CLI coding agents
(`claude`, `codex`) to solve problems and review each other's work. The loop
engine is task-agnostic; "PR review with two models" is just one configuration.

## Environment & commands

- Python 3.13, managed with **uv**. Don't use bare `pip`/`python`.
- `uv sync` — install deps into `.venv`.
- `uv run pytest` — tests (run offline against a fake agent; no network/cost).
- `uv run ruff check .` and `uv run ruff format .` — lint (incl. import sort) + format.
- `uv run ty check` — type check.
- `uv run agentloop --help` — the CLI.

Always run **ruff, ty, and pytest** before committing. All three must be clean.

## Architecture

```
Orchestrator ── select speaker → compose prompt → agent.send → record → check stop
   ├── Agent / CliAgent     subprocess + session resume + JSON parsing
   │     ├── ClaudeAgent     claude -p --output-format json
   │     └── CodexAgent      codex exec --json
   ├── Policy                who speaks + what they see   ← the generic knob
   ├── StopCondition         MaxRounds · Consensus · BudgetUSD
   └── Store (optional)      JSONL journal → durable & resumable
```

- `src/agentloop/domain.py` — `Message`, `Transcript`, `TurnResult`.
- `src/agentloop/agent.py` — `Agent` ABC + `CliAgent` subprocess base.
- `src/agentloop/adapters/` — one module per CLI; parses that CLI's real output.
- `src/agentloop/policy.py` / `stop.py` / `orchestrator.py` — the loop.
- `src/agentloop/store.py` — resumable journal.
- `src/agentloop/report.py` — human-readable run report (plan + implementation
  verdict + transcript/findings) written into the run dir. Distinct from the
  journal: the journal resumes a run, the report is the finished product.
- `src/agentloop/pipeline.py` — adaptive flow with a write-gate: triage →
  (optional) parallel discovery scouts → serial debate → **agreed plan** →
  `[--write]` → fix loop (implement → self-verify → review diff → APPROVED).
  `solve()` is the entry point; `fix()` is the stage-2 loop.
- `src/agentloop/cli.py` — `run` (a single freeform-task command).

### Pipeline & CLI surface

```
triage → discovery → debate ──► AGREED PLAN ──[gate]──► implement → self-verify → review ──► APPROVED
        (read-only, always)      (plan.md)      │                                  ↑ loop until APPROVED / max attempts
                                                ├─ default:           stop, save the plan (read-only)
                                                ├─ --write:           cross now (one-shot)
                                                └─ --resume + --write: cross later (review-first handoff)
```

Read-only is the safe default; writing the repo is **always** an explicit
`--write` opt-in, never inferred from the prompt. The only writer is a single
`Build.implementer()` (codex `workspace-write`) past the gate — one writer, so no
worktree/concurrency hazard; reviewers stay read-only and inspect the diff via
`git diff`. The whole CLI is `task` + `--write` + `--resume <run-dir>` + `--repo`
+ `-v`; everything else (discovery breadth, rounds, budget backstop, journaling,
the report dir) is a default or triage-inferred, not a flag.

Each run writes `.agentloop/<timestamp>-<slug>/`: `report.md`, `plan.md` (the
handoff artifact), `journal.jsonl` + `fix.journal.jsonl` (resume), `meta.json`
(task), `transcript/debate.md` + `transcript/fix.md`, and `findings/`. It's a
dotted, tool-owned dir so it doesn't collide with the inspected project's `out/`.

The CLI is intentionally thin: it hands the agents a prompt and lets them do the
work with their own tools. Task-specific logic (e.g. fetching a diff) belongs in
the prompt, not in Python — don't reintroduce git plumbing in the CLI.

### State model (read before touching sessions or the store)

Three layers; the engine owns only the first two:

1. **Transcript** (in memory) — the turns; the only thing agents share.
2. **Session pointer** — each agent's `session_id`; private per agent.
3. **CLI history** (on disk, owned by the CLIs) — each agent's full context.

Agents never read each other's Layer-3 history. Information crosses between them
only as text, via `Policy.compose()` quoting one agent's turn into another's
prompt. The journal persists Layers 1 & 2 so a resumed run reconnects each agent
to its real Layer-3 session.

## Conventions

- Fully typed; keep `ty check` clean. Prefer `Sequence`/`Protocol` over concrete
  containers in public signatures (list invariance bites otherwise).
- Adapters are built against the CLIs' **actual** output — if a CLI's JSON
  changes, re-probe it (`claude -p --output-format json`, `codex exec --json`)
  rather than guessing the schema. On failure, surface the CLI's *real* error:
  `codex exec` exits non-zero but reports the cause as a stdout event (stderr is
  just stdin noise), so `CodexAgent._failure_detail` digs it out. `solve()`
  preflights both CLIs first so misconfig (bad model, not logged in) fails in
  seconds; per-CLI model overrides come from `AGENTLOOP_CODEX_MODEL` /
  `AGENTLOOP_CLAUDE_MODEL` so the user's global CLI config is untouched.
- Keep the loop engine task-agnostic: new behavior belongs in a `Policy`,
  `StopCondition`, or `Agent` subclass, not in `Orchestrator`.
- Commits: small and logical, imperative subject, Conventional-Commit prefix
  (`feat:`/`fix:`/`test:`/`docs:`/`chore:`).
