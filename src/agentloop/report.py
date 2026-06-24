"""Human-readable run reports written into a run directory.

Distinct from the journal (``store.py``): the journal is a replay log keyed to
agent sessions, for resuming a run. A report is the *finished product* -- the
agreed plan, what (if anything) was implemented, and the review verdict.

Layout per run::

    out/<timestamp>-<slug>/         (gitignored, created by the CLI up front)
      report.md          plan + implementation verdict + run metadata
      plan.md            the agreed plan -- the handoff artifact for --resume --write
      journal.jsonl      stage-1 resume journal (triage/discovery/debate)
      fix.journal.jsonl  stage-2 resume journal (implement/review; only with --write)
      transcript/
        debate.md        stage-1 debate (omitted on a resume-handoff)
        fix.md           stage-2 implement + review (only when --write ran)
      findings/          one file per discovery scout (omitted if none)
        01-<focus>.md
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

from .domain import USER, Message, Transcript
from .pipeline import FixResult

_SLUG_MAX = 60


def slug(text: str) -> str:
    """A filesystem-safe, lowercase-kebab slug; empty input yields ``run``."""
    cleaned = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return cleaned[:_SLUG_MAX].strip("-") or "run"


def _last_by_author(transcript: Transcript, author: str) -> Message | None:
    return next((m for m in reversed(transcript.messages) if m.author == author), None)


def _render_report(
    *,
    task: str,
    when: str,
    shape: str,
    triage_reason: str,
    stopped_by: str,
    turns: int,
    resumed: bool,
    total_cost: float,
    plan_text: str,
    fix: FixResult | None,
) -> str:
    lines = [
        f"# {slug(triage_reason or task).replace('-', ' ')}",
        "",
        f"- **task:** {task}",
        f"- **when:** {when}",
        f"- **shape:** {shape}",
    ]
    if triage_reason:
        lines.append(f"- **triage:** {triage_reason}")
    lines += [
        f"- **stopped by:** {stopped_by}",
        f"- **turns:** {turns}{' (resumed)' if resumed else ''}",
        f"- **total cost:** ${total_cost:.4f}",
        "",
        "## Plan",
        "",
        plan_text.strip() or "_(no plan produced)_",
        "",
    ]
    if fix is not None:
        verdict = "APPROVED" if fix.approved else "NOT approved"
        lines += [
            "## Implementation",
            "",
            f"**verdict:** {verdict} after {fix.attempts} attempt(s)",
            "",
        ]
        change = _last_by_author(fix.transcript, "implementer")
        if change is not None:
            lines += [change.content.strip(), ""]
    return "\n".join(lines).rstrip() + "\n"


def _render_turns(title: str, transcript: Transcript, *, seed: str | None = None) -> str:
    lines = [f"# {title}", ""]
    if seed is not None:
        lines += ["## task (seed)", "", seed.strip(), ""]
    for message in transcript.messages:
        if message.author == USER:
            continue
        cost = f" (${message.cost_usd:.4f})" if message.cost_usd else ""
        lines += [f"## {message.author}{cost}", "", message.content.strip(), ""]
    return "\n".join(lines).rstrip() + "\n"


def write_run_report(
    run_dir: Path,
    *,
    task: str,
    when: str,
    shape: str,
    triage_reason: str,
    stopped_by: str,
    turns: int,
    resumed: bool,
    total_cost: float,
    plan_text: str,
    transcript: Transcript | None = None,
    scouts: Sequence[Message] = (),
    focuses: Sequence[str] = (),
    fix: FixResult | None = None,
) -> Path:
    """Write a run's report tree into ``run_dir`` (already created) and return it.

    ``transcript`` is the stage-1 debate (absent on a resume-handoff, where stage
    1 ran in a prior process). ``scouts`` are the discovery turns -- they never
    join the debate transcript -- and ``focuses`` labels them positionally.
    """
    (run_dir / "transcript").mkdir(parents=True, exist_ok=True)

    (run_dir / "report.md").write_text(
        _render_report(
            task=task,
            when=when,
            shape=shape,
            triage_reason=triage_reason,
            stopped_by=stopped_by,
            turns=turns,
            resumed=resumed,
            total_cost=total_cost,
            plan_text=plan_text,
            fix=fix,
        ),
        encoding="utf-8",
    )
    (run_dir / "plan.md").write_text(
        f"# Plan\n\n_for: {task}_\n\n{plan_text.strip()}\n", encoding="utf-8"
    )

    if transcript is not None:
        (run_dir / "transcript" / "debate.md").write_text(
            _render_turns("Debate transcript", transcript, seed=task), encoding="utf-8"
        )
    if fix is not None:
        (run_dir / "transcript" / "fix.md").write_text(
            _render_turns("Fix transcript (implement + review)", fix.transcript), encoding="utf-8"
        )

    if scouts:
        findings = run_dir / "findings"
        findings.mkdir(exist_ok=True)
        for index, message in enumerate(scouts):
            focus = focuses[index] if index < len(focuses) else message.author
            name = f"{index + 1:02d}-{slug(focus)}.md"
            body = f"# {focus}\n\n_via {message.author}_\n\n{message.content.strip()}\n"
            (findings / name).write_text(body, encoding="utf-8")

    return run_dir
