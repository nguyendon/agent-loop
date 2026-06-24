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

Requires the `claude` and `codex` CLIs installed and logged in.

## Use it

The loop is task-agnostic. `run` hands the two agents a freeform prompt and they
do the work with their own tools (git, file reads, `gh`, …). `review` is a thin
preset over `run` with a reviewer prompt.

```bash
# Any task — the agents inspect the repo / fetch the PR / etc. themselves:
uv run agentloop run "review the uncommitted changes and agree on the top issues"
uv run agentloop run "review PR #42"
uv run agentloop run "find and fix the flaky test in tests/"

# Pure reasoning, no repo access (cheaper, faster):
uv run agentloop run "Design a token-bucket rate limiter" --no-tools

# Review preset (defaults to the current working changes):
uv run agentloop review
uv run agentloop review "the last 3 commits" --budget 1.50

# Persist a run, then resume it by reusing the same journal path:
uv run agentloop run "review the changes" --rounds 2 --journal run.jsonl
uv run agentloop run "review the changes" --rounds 6 --journal run.jsonl
```

By default the agents have **read-only** tool access (claude in plan mode, codex
in its read-only sandbox), so they ground findings in the real code and history.
`--no-tools` drops claude to prompt-only for tasks that need no repo access — it's
cheaper and faster but the agents only see what you put in the prompt.

## Run it on another repo

Install the CLI once, then point it anywhere — both agents run in the target
directory, so they review whatever repo you aim them at.

```bash
uv tool install /path/to/pr-review-agent-loop   # puts `agentloop` on your PATH

cd /path/to/other/repo && agentloop review                  # cd in...
agentloop review --repo /path/to/other/repo                 # ...or pass --repo
agentloop run "review PR #42" --repo /path/to/other/repo
```

Pick up later changes with `uv tool upgrade agentloop`. To run without installing:
`uv run --project /path/to/pr-review-agent-loop agentloop review --repo /path/to/other/repo`.

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

Pass a `JournalStore` (or `--journal PATH`) to make a run **durable and
resumable**. Every turn is appended to a JSONL file as it happens, so a crash
loses at most the in-flight turn. Reusing the same path replays the journal:
it rebuilds the transcript *and* restores each agent's `session_id`, so the CLIs
reload their real Layer-3 context and the loop continues exactly where it
stopped — not from a cold start.

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
