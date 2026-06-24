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

```bash
# Two agents debate an open problem until they agree (or hit the round cap):
uv run agentloop debate "Design a token-bucket rate limiter for our API"

# Two agents review the current branch's diff and reconcile findings:
uv run agentloop review --base main --head HEAD --budget 1.50

# Same review, as a library script you can edit:
uv run python examples/pr_review.py --base main --head HEAD

# Persist a run, then resume it by reusing the same journal path:
uv run agentloop debate "Design a cache eviction policy" --rounds 2 --journal run.jsonl
uv run agentloop debate "Design a cache eviction policy" --rounds 6 --journal run.jsonl
```

## Run it on another repo

Install the CLI once, then point it anywhere. `git` and both agents run in the
target directory, so the loop reviews whatever repo you aim it at.

```bash
uv tool install /path/to/pr-review-agent-loop   # puts `agentloop` on your PATH

# either cd into the target repo...
cd /path/to/other/repo && agentloop review --base main --head my-feature

# ...or stay put and pass --repo:
agentloop review --repo /path/to/other/repo --base main --head my-feature
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
