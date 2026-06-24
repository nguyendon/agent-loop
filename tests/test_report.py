"""Run-report writer tests: layout, contents, and slugging."""

from __future__ import annotations

from pathlib import Path

from agentloop.domain import USER, Message, Transcript
from agentloop.report import slug, write_run_report


def _debate() -> Transcript:
    transcript = Transcript()
    transcript.add(Message(USER, "review the diff"))
    transcript.add(Message("claude", "claude opening", cost_usd=0.01))
    transcript.add(Message("codex", "codex opening", cost_usd=0.02))
    transcript.add(Message("claude", "AGREED, nothing to add", cost_usd=0.01))
    return transcript


def test_writes_report_tree_with_findings(tmp_path: Path) -> None:
    run_dir = write_run_report(
        out_root=tmp_path,
        timestamp="20260624-120000",
        task="review the diff",
        shape="2 scouts → debate",
        triage_reason="open-ended code review",
        stopped_by="Consensus",
        turns=3,
        resumed=False,
        total_cost=0.04,
        transcript=_debate(),
        scouts=[Message("scout1", "found a bug"), Message("scout2", "perf nit")],
        focuses=["correctness", "performance"],
    )

    assert run_dir == tmp_path / "20260624-120000-open-ended-code-review"
    report = (run_dir / "report.md").read_text()
    # Final positions show each debater's last word, not just the final speaker.
    assert "### claude" in report and "### codex" in report
    assert "AGREED, nothing to add" in report
    assert "open-ended code review" in report
    assert "$0.0400" in report

    debate = (run_dir / "transcript" / "debate.md").read_text()
    assert "review the diff" in debate and "codex opening" in debate

    findings = sorted((run_dir / "findings").glob("*.md"))
    assert [p.name for p in findings] == ["01-correctness.md", "02-performance.md"]
    assert "found a bug" in findings[0].read_text()


def test_no_findings_dir_without_scouts(tmp_path: Path) -> None:
    run_dir = write_run_report(
        out_root=tmp_path,
        timestamp="20260624-120000",
        task="what is 2+2",
        shape="debate only",
        triage_reason="",
        stopped_by="Consensus",
        turns=2,
        resumed=False,
        total_cost=0.01,
        transcript=_debate(),
    )
    assert not (run_dir / "findings").exists()
    # Empty triage reason falls back to the task for the slug.
    assert run_dir.name == "20260624-120000-what-is-2-2"


def test_slug_is_kebab_and_bounded() -> None:
    assert slug("Open-ended Code Review!") == "open-ended-code-review"
    assert slug("") == "run"
    assert len(slug("x" * 200)) <= 60
