# agentloop

A small, generic **agent loop** that drives CLI coding agents (`claude`, `codex`)
to solve problems — and to argue with each other until they converge.

The engine knows nothing about any specific task. Every turn it asks a **policy**
who speaks next and what they see, runs that agent as a subprocess, records the
result in a shared transcript, and checks the **stop conditions**. Reviewing a PR
with two models is just one configuration of those parts.

```
Orchestrator ── select speaker → compose prompt → agent.send → record → check stop
   ├── Agent            the unit that takes a turn
   │     └── CliAgent   subprocess + session resume + JSON parsing
   │           ├── ClaudeAgent   claude -p --output-format json
   │           └── CodexAgent    codex exec --json
   ├── Policy           who speaks + what they see   ← the generic knob
   │     ├── RoundRobinPolicy
   │     └── DebatePolicy        A reviews → B critiques → A revises …
   └── StopCondition    MaxRounds · Consensus("AGREED") · BudgetUSD
```

## Setup

```bash
uv sync
```

Requires the `claude` and `codex` CLIs installed and logged in. Each run does a
quick **preflight** — a trivial ping to both CLIs — and aborts in seconds with the
real error if one is misconfigured, rather than dying minutes into a debate.

If a CLI rejects its default model (e.g. `codex exec` on a ChatGPT account that
can't use `gpt-*-codex`), point agentloop at a supported one without touching the
CLI's global config:

```bash
export AGENTLOOP_CODEX_MODEL=gpt-5.5      # scoped to agentloop's codex agents
export AGENTLOOP_CLAUDE_MODEL=...         # optional; same for claude
```

## Quick start

The loop is task-agnostic: the task is just a prompt, and the agents use their own
tools (`git`, `gh`, file reads) to fetch the PR or read the diff. It is **read-only
by default** — it reviews and converges on a plan, and never touches your tree.
`--write` is the only thing that can change the repo.

### Review a PR or codebase (read-only)

```bash
# A specific PR — the agents fetch the diff themselves with gh
uv run agentloop "review PR #42: fetch it with gh, find correctness, security, and perf issues, and agree on the top ones"

# Your uncommitted / staged work
uv run agentloop "review the uncommitted changes (git diff) and agree on the prioritized issues"

# This branch vs main
uv run agentloop "review the diff between this branch and main; agree on what must change before merge"

# A slice of the codebase (a standing audit, no diff)
uv run agentloop "audit src/agentloop/adapters for correctness and error handling; agree on the top findings"
```

Each run writes `.agentloop/<timestamp>-<slug>/` with `report.md`, `plan.md`, and the
full debate. (A dotted, tool-owned dir — agentloop runs *inside* the repo it's
inspecting, so it stays out of that project's `out/`/build dirs. Add `.agentloop/`
to the target repo's `.gitignore` if you don't want the runs tracked.) Nothing in
your tree is modified.

### Review *and* implement the fix (`--write`)

Same prompts, plus `--write`: it reviews to an agreed plan, then crosses the gate —
a write-capable agent implements the plan and self-verifies (runs the tests/build),
and the two agents review the resulting diff until they converge on `APPROVED`.

```bash
uv run agentloop "fix the failing test in tests/test_orchestrator.py" --write
uv run agentloop "address the issues in PR #42 and verify the suite still passes" --write
```

**Committing is left outside the loop, on purpose.** `--write` edits your working
tree but never commits — the engine stays out of git. Review the diff and commit
it yourself, or hand the finished run (the `report:` path printed on `done`) to a
follow-up agent to turn into clean commits. Keeping it uncommitted is also why the
reviewers can read the live `git diff`.

### Review first, then hand off to the fix loop (staged)

The careful path — you see the plan before anything writes:

```bash
# 1. read-only review → produces the plan
uv run agentloop "review the uncommitted changes and agree on the fixes"
#    → done panel prints e.g.  report: .agentloop/20260624-143005-review-the-uncommitted-changes

# 2. read .agentloop/<run>/plan.md and decide it's right

# 3. hand that exact plan to the fix loop — no re-debate
uv run agentloop --resume .agentloop/20260624-143005-review-the-uncommitted-changes --write
```

The whole surface is `task` + `--write` + `--resume <run-dir>` + `--repo` + `-v`.

#### Expect it to take a while

This is **not** instant — every turn runs a full agent CLI to completion (with
tools), and there are several turns across triage, discovery, debate, and the fix
loop. Rough wall-clock, dominated by how much the agents read and how broad the task is:

| Run | Typical |
| --- | --- |
| read-only review/plan | **~2–10 min** |
| `--write` (plan → implement → review) | **~10–25+ min** |

The CLI prints this estimate when it starts and the actual `elapsed` time when it
finishes. While it runs you'll see a spinner naming the current phase; **long
pauses on one phase are normal, not a hang** — a single tool-using turn can take
minutes. Pass `-v`/`-vv` to replace the spinner with per-turn log lines (timings,
cost, the command run) on stderr. Narrowing the task (`review src/foo.py` rather
than `review this repo`) is the biggest lever on how long it takes.

### How a task is approached

A cheap **triage** turn decides the shape:

- **Simple question** → straight to debate (two agents open in parallel, then
  argue to consensus).
- **Open-ended task** (code review, audit, design) → **discovery** first: triage
  picks independent angles and fans out parallel scouts, pools their findings, then
  seeds the debate with them.

The agents have **read-only** tool access for review (claude in plan mode, codex in
its read-only sandbox), so they ground findings in the real code and history. Only
`--write` lifts that, and only past the agreed plan: a single write-capable agent
(codex `workspace-write`) makes the edits while the reviewers stay read-only. The
final panel reports the shape used (`N scouts → debate`, `… → fix`) and the total
cost across all phases.

## Run it on another repo

Install the CLI once, then point it anywhere — both agents run in the target
directory, so they work on whatever repo you aim them at.

```bash
uv tool install /path/to/pr-review-agent-loop   # puts `agentloop` on your PATH

cd /path/to/other/repo && agentloop "review the changes"   # cd in...
agentloop "review the changes" --repo /path/to/other/repo  # ...or pass --repo
```

Pick up later changes with `uv tool upgrade agentloop`. To run without installing:
`uv run --project /path/to/pr-review-agent-loop agentloop "..." --repo /path/to/other/repo`.

## As a library

```python
from agentloop import ClaudeAgent, CodexAgent, DebatePolicy, Consensus, Orchestrator

loop = Orchestrator(
    agents=[ClaudeAgent("claude"), CodexAgent("codex", sandbox="read-only")],
    policy=DebatePolicy("Find the bug in foo.py and agree on a fix."),
    stop=[Consensus("AGREED")],
    max_rounds=6,
    on_message=lambda m: print(f"{m.author}: {m.content}"),
)
result = loop.run()
print(result.stopped_by, result.transcript.total_cost_usd)
```

## State & durability

Three layers, and the loop only owns the first two:

| Layer | What | Where it lives | Persisted by |
| --- | --- | --- | --- |
| 1. Transcript | the turns every agent contributes | in memory (`Transcript`) | the journal, if set |
| 2. Session pointer | each agent's `session_id` | in memory (`CliAgent`) | the journal, if set |
| 3. Conversation history | each agent's full private context | `~/.claude/...`, `~/.codex/...` | the CLIs themselves |

Agents **don't share memory** — they never read each other's Layer-3 history.
Information crosses between them only as text: one agent's turn is recorded in
the transcript, and `Policy.compose()` quotes it into the next agent's prompt.

Every run is **durable and resumable**. Stage 1 (triage/discovery/debate)
journals to `journal.jsonl`; when the write gate is crossed, stage 2
(implement/review) journals to the sibling `fix.journal.jsonl`. The CLI creates
both automatically under `.agentloop/<run>/`; library callers get the same default
by passing a `JournalStore` to `solve()`, or can wire the journals explicitly.
Every turn is appended as it happens, so a crash loses at most the in-flight
turn. Replaying the run (`--resume .agentloop/<run>`, or reusing the same journal path
in code) rebuilds the transcript *and* restores each agent's `session_id`, so
the CLIs reload their real Layer-3 context and the loop continues exactly where
it stopped — not from a cold start.

```python
from agentloop import JournalStore, Orchestrator

loop = Orchestrator(agents, policy, store=JournalStore("run.jsonl"))
loop.run()  # first call records; a later call with the same path resumes
```

`max_rounds` counts *total* turns across the journal, so a resumed run caps the
combined length rather than restarting the budget.

## Extending it

- **New agent** (Gemini, Aider, a local model): subclass `CliAgent`, implement
  `_build_command` and `_parse`. Session resume and timeouts come for free.
- **New interaction pattern**: subclass `Policy`. `select` picks the speaker;
  `compose` decides what context that speaker receives. This is where
  brainstorming vs. debate vs. judge-and-revise lives.
- **New stop rule**: any `(Context) -> bool` callable — e.g. "stop when a
  verdict line matches a regex" or "stop after N findings".

## Develop

```bash
uv run pytest          # loop logic is tested offline with a fake agent
uv run ruff check .    # lint + import sorting
uv run ruff format .   # format
uv run ty check        # type check
```
