"""Run-report writer tests: layout, contents, and slugging."""

from __future__ import annotations

from pathlib import Path

from agentloop.domain import USER, Message, Transcript
from agentloop.pipeline import FixResult
from agentloop.report import slug, write_run_report


def _debate() -> Transcript:
    transcript = Transcript()
    transcript.add(Message(USER, "review the diff"))
    transcript.add(Message("claude", "claude opening", cost_usd=0.01))
    transcript.add(Message("codex", "codex opening", cost_usd=0.02))
    transcript.add(Message("claude", "AGREED, nothing to add", cost_usd=0.01))
    return transcript


def _run_dir(tmp_path: Path) -> Path:
    run_dir = tmp_path / "20260624-120000-review-the-diff"
    run_dir.mkdir()
    return run_dir


def test_writes_report_tree_with_findings(tmp_path: Path) -> None:
    run_dir = write_run_report(
        _run_dir(tmp_path),
        task="review the diff",
        when="20260624-120000",
        shape="2 scouts → debate",
        triage_reason="open-ended code review",
        stopped_by="Consensus",
        turns=3,
        resumed=False,
        total_cost=0.04,
        plan_text="Fix A, then B; they are the agreed priorities for this change.",
        transcript=_debate(),
        scouts=[Message("scout1", "found a bug"), Message("scout2", "perf nit")],
        focuses=["correctness", "performance"],
    )

    report = (run_dir / "report.md").read_text()
    assert "## Plan" in report
    assert "agreed priorities" in report
    assert "open-ended code review" in report
    assert "$0.0400" in report
    # No write phase, so no implementation section or fix transcript.
    assert "## Implementation" not in report
    assert not (run_dir / "transcript" / "fix.md").exists()

    # The plan is also persisted standalone as the handoff artifact.
    assert "agreed priorities" in (run_dir / "plan.md").read_text()

    debate = (run_dir / "transcript" / "debate.md").read_text()
    assert "review the diff" in debate and "codex opening" in debate

    findings = sorted((run_dir / "findings").glob("*.md"))
    assert [p.name for p in findings] == ["01-correctness.md", "02-performance.md"]
    assert "found a bug" in findings[0].read_text()


def test_write_phase_adds_implementation_and_fix_transcript(tmp_path: Path) -> None:
    fix_transcript = Transcript()
    fix_transcript.add(
        Message("implementer", "edited foo.py; ran pytest, all green", cost_usd=0.05)
    )
    fix_transcript.add(Message("reviewer-claude", "APPROVED", cost_usd=0.01))
    fix = FixResult(transcript=fix_transcript, approved=True, attempts=1, cost_usd=0.06)

    run_dir = write_run_report(
        _run_dir(tmp_path),
        task="fix the bug",
        when="20260624-120000",
        shape="debate only → fix",
        triage_reason="",
        stopped_by="APPROVED",
        turns=1,
        resumed=False,
        total_cost=0.10,
        plan_text="Change foo.py so the off-by-one is corrected, then re-run the suite.",
        transcript=_debate(),
        fix=fix,
    )

    report = (run_dir / "report.md").read_text()
    assert "## Implementation" in report
    assert "**verdict:** APPROVED after 1 attempt(s)" in report
    assert "edited foo.py" in report  # last implementer message surfaces as "what changed"

    fix_md = (run_dir / "transcript" / "fix.md").read_text()
    assert "implementer" in fix_md and "reviewer-claude" in fix_md


def test_summary_drives_report_while_plan_md_stays_raw(tmp_path: Path) -> None:
    run_dir = write_run_report(
        _run_dir(tmp_path),
        task="review the diff",
        when="20260624-120000",
        shape="debate only",
        triage_reason="",
        stopped_by="Consensus",
        turns=2,
        resumed=False,
        total_cost=0.02,
        plan_text="RAW agreed plan: change foo.py line 12, the executable handoff.",
        summary="# Findings\n\nClean human-readable synthesis for the report.",
        transcript=_debate(),
    )
    report = (run_dir / "report.md").read_text()
    plan = (run_dir / "plan.md").read_text()
    # report.md shows the synthesized summary; plan.md keeps the raw handoff plan.
    assert "human-readable synthesis" in report
    assert "RAW agreed plan" not in report
    assert "RAW agreed plan" in plan
    assert "human-readable synthesis" not in plan


def test_no_transcript_on_resume_handoff(tmp_path: Path) -> None:
    run_dir = write_run_report(
        _run_dir(tmp_path),
        task="fix the bug",
        when="20260624-120000",
        shape="resumed plan → fix",
        triage_reason="",
        stopped_by="APPROVED",
        turns=1,
        resumed=True,
        total_cost=0.06,
        plan_text="The agreed plan carried over from the prior read-only run goes here.",
    )
    # Stage 1 ran in a prior process, so there's no debate transcript this time.
    assert not (run_dir / "transcript" / "debate.md").exists()
    assert "carried over" in (run_dir / "plan.md").read_text()
    assert "(resumed)" in (run_dir / "report.md").read_text()


def test_slug_is_kebab_and_bounded() -> None:
    assert slug("Open-ended Code Review!") == "open-ended-code-review"
    assert slug("") == "run"
    assert len(slug("x" * 200)) <= 60
