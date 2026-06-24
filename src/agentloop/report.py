"""Human-readable run reports written to ``out/<timestamp>-<slug>/``.

Distinct from the journal (``store.py``): the journal is a replay log keyed to
agent sessions, for resuming a run. A report is the *finished product* -- the
converged answer plus the supporting material a person reads afterwards.

Layout per run::

    out/                                  (gitignored)
      20260624-143005-<slug>/
        report.md          final positions + run metadata
        findings/          one file per discovery scout (omitted if none)
          01-<focus>.md
        transcript/
          debate.md        the full turn-by-turn debate

The slug comes from the triage reason when there is one (a one-line summary of
how the task was approached), falling back to the task text.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

from .domain import USER, Message, Transcript

_SLUG_MAX = 60


def slug(text: str) -> str:
    """A filesystem-safe, lowercase-kebab slug; empty input yields ``run``."""
    cleaned = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return cleaned[:_SLUG_MAX].strip("-") or "run"


def _final_positions(transcript: Transcript) -> list[Message]:
    """Each debater's last word, in the order they last spoke.

    On a debate that ends in consensus the very last turn may be a bare
    "AGREED", so the converged answer lives in the closing turn from *each*
    side, not just the final speaker -- show both.
    """
    last_by_author: dict[str, Message] = {}
    for message in transcript.agent_messages:
        last_by_author[message.author] = message
    return list(last_by_author.values())


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
    transcript: Transcript,
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
        "## Final positions",
        "",
    ]
    for message in _final_positions(transcript):
        lines += [f"### {message.author}", "", message.content.strip(), ""]
    return "\n".join(lines).rstrip() + "\n"


def _render_debate(task: str, transcript: Transcript) -> str:
    lines = ["# Debate transcript", "", "## task (seed)", "", task.strip(), ""]
    for message in transcript.messages:
        if message.author == USER:
            continue
        cost = f" (${message.cost_usd:.4f})" if message.cost_usd else ""
        lines += [f"## {message.author}{cost}", "", message.content.strip(), ""]
    return "\n".join(lines).rstrip() + "\n"


def write_run_report(
    *,
    out_root: Path,
    timestamp: str,
    task: str,
    shape: str,
    triage_reason: str,
    stopped_by: str,
    turns: int,
    resumed: bool,
    total_cost: float,
    transcript: Transcript,
    scouts: Sequence[Message] = (),
    focuses: Sequence[str] = (),
) -> Path:
    """Write a run's report tree under ``out_root`` and return the run directory.

    ``scouts`` are the discovery turns (absent from the debate transcript, which
    only holds the seed and the debaters); ``focuses`` labels them positionally.
    """
    run_dir = out_root / f"{timestamp}-{slug(triage_reason or task)}"
    (run_dir / "transcript").mkdir(parents=True, exist_ok=True)

    (run_dir / "report.md").write_text(
        _render_report(
            task=task,
            when=timestamp,
            shape=shape,
            triage_reason=triage_reason,
            stopped_by=stopped_by,
            turns=turns,
            resumed=resumed,
            total_cost=total_cost,
            transcript=transcript,
        ),
        encoding="utf-8",
    )
    (run_dir / "transcript" / "debate.md").write_text(
        _render_debate(task, transcript), encoding="utf-8"
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
