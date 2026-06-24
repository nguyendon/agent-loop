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
  rather than guessing the schema.
- Keep the loop engine task-agnostic: new behavior belongs in a `Policy`,
  `StopCondition`, or `Agent` subclass, not in `Orchestrator`.
- Commits: small and logical, imperative subject, Conventional-Commit prefix
  (`feat:`/`fix:`/`test:`/`docs:`/`chore:`).
