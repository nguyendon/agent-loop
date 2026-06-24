"""Pipeline tests: the write-gate and the fix loop, offline against fake agents.

``FakeBuild`` overrides every agent factory ``solve``/``fix`` use, so no CLI
subprocess ever runs. Reviewer replies are driven by the attempt counter so we
can exercise both the one-shot-approve and loop-until-approved paths.
"""

from __future__ import annotations

import pytest

from agentloop.adapters.claude import ClaudeAgent
from agentloop.adapters.codex import CodexAgent
from agentloop.agent import Agent, AgentError
from agentloop.domain import Message, TurnResult
from agentloop.pipeline import Build, fix, preflight, solve
from agentloop.store import FixJournal, JournalStore

_TRIAGE_NO_DISCOVERY = '{"discovery": false, "focuses": [], "reason": "simple"}'
_PLAN = "Plan: correct the off-by-one in foo.py and re-run the suite to confirm."


class _Fake(Agent):
    """Replays a fixed list of responses; ``error`` makes ``send`` raise instead."""

    def __init__(self, name: str, replies: list[str], *, error: str | None = None) -> None:
        self.name = name
        self._replies = replies
        self._error = error
        self.prompts: list[str] = []
        self.session_id: str | None = None
        self.turns = 0

    @property
    def is_first_turn(self) -> bool:
        return self.turns == 0

    def send(self, prompt: str) -> TurnResult:
        self.prompts.append(prompt)
        if self._error is not None:
            raise AgentError(self._error)
        reply = self._replies[min(self.turns, len(self._replies) - 1)]
        self.turns += 1
        self.session_id = f"{self.name}-session"
        return TurnResult(text=reply, session_id=self.session_id, cost_usd=0.01)


class FakeBuild(Build):
    """A Build whose agents are scripted fakes; records every agent created."""

    def __init__(self, *, approve_on_attempt: int = 1) -> None:
        super().__init__(repo=None)
        self.approve_on_attempt = approve_on_attempt
        self.attempt = 0
        self.created: list[str] = []
        self.last_implementer: _Fake | None = None

    def _debater(self, name: str) -> _Fake:
        return _Fake(name, [_PLAN, "AGREED"])

    def _reviewer(self, name: str) -> _Fake:
        approved = self.attempt >= self.approve_on_attempt
        return _Fake(
            name, ["APPROVED, correct and complete" if approved else "needs work: edge case"]
        )

    def claude(
        self,
        name: str = "claude",
        system_prompt: str | None = None,
        *,
        fast: bool = False,
        tools: bool = True,
    ) -> Agent:
        self.created.append(name)
        if name == "triage":
            return _Fake(name, [_TRIAGE_NO_DISCOVERY])
        if name == "synthesis":
            return _Fake(name, [f"SYNTHESIZED: {_PLAN}"])
        if name.startswith("reviewer"):
            return self._reviewer(name)
        return self._debater(name)

    def codex(
        self, name: str = "codex", system_prompt: str | None = None, *, fast: bool = False
    ) -> Agent:
        self.created.append(name)
        if name.startswith("reviewer"):
            return self._reviewer(name)
        return self._debater(name)

    def implementer(self, name: str = "implementer") -> Agent:
        self.attempt += 1
        self.created.append(name)
        agent = _Fake(name, [f"implemented attempt {self.attempt}; ran tests, all green"])
        self.last_implementer = agent
        return agent


def test_gate_stops_at_plan_without_write() -> None:
    build = FakeBuild()
    result = solve("fix it", build=build, rounds=8, stop=[], write=False)

    assert result.fix is None  # never crossed the gate
    assert result.loop is not None
    # plan_text is the RAW agreed plan (the executable handoff); the synthesized,
    # human-readable text is kept separate in summary (shown in report.md).
    assert "off-by-one" in result.plan_text
    assert "SYNTHESIZED" not in result.plan_text
    assert "SYNTHESIZED" in result.summary
    assert "synthesis" in build.created
    assert "implementer" not in build.created


class _StrongClaudeBrokenBuild(FakeBuild):
    """Fast-tier claude (triage) works, but the strong tier is misconfigured."""

    def claude(
        self,
        name: str = "claude",
        system_prompt: str | None = None,
        *,
        fast: bool = False,
        tools: bool = True,
    ) -> Agent:
        self.created.append(name)
        if fast:
            return _Fake(name, [_TRIAGE_NO_DISCOVERY if name == "triage" else "ok"])
        return _Fake(name, [], error="claude: exit 1: the 'bad-strong' model is not supported")


def test_triage_path_still_preflights_the_strong_claude_model() -> None:
    # Regression guard: triage runs on the fast tier, so the strong claude (used
    # by the debate) must still be validated up front -- not fail late mid-debate.
    with pytest.raises(AgentError) as excinfo:
        solve("review it", build=_StrongClaudeBrokenBuild(), rounds=8, stop=[], write=False)
    assert "not supported" in str(excinfo.value)


def test_write_crosses_gate_and_approves_first_try() -> None:
    build = FakeBuild(approve_on_attempt=1)
    result = solve("fix it", build=build, rounds=8, stop=[], write=True)

    assert result.fix is not None
    assert result.fix.approved
    assert result.fix.attempts == 1
    assert build.created.count("implementer") == 1
    # The implementer's report is captured in the fix transcript.
    assert any(m.author == "implementer" for m in result.fix.transcript.messages)
    # The handoff implements the RAW agreed plan, never the human synthesis.
    assert build.last_implementer is not None
    implement_prompt = build.last_implementer.prompts[0]
    assert "off-by-one" in implement_prompt
    assert "SYNTHESIZED" not in implement_prompt


def test_fix_loops_until_reviewers_approve() -> None:
    build = FakeBuild(approve_on_attempt=2)
    result = fix("fix it", _PLAN, build=build, max_attempts=3, review_rounds=2)

    assert result.approved
    assert result.attempts == 2  # first attempt rejected, second approved
    assert build.created.count("implementer") == 2


def test_fix_gives_up_after_max_attempts() -> None:
    build = FakeBuild(approve_on_attempt=99)  # never approves
    result = fix("fix it", _PLAN, build=build, max_attempts=2, review_rounds=2)

    assert not result.approved
    assert result.attempts == 2


def test_resumed_plan_skips_stage_one() -> None:
    build = FakeBuild(approve_on_attempt=1)
    result = solve("fix it", build=build, rounds=8, stop=[], write=True, resumed_plan=_PLAN)

    assert result.loop is None  # stage 1 was skipped
    assert "triage" not in build.created  # no triage either
    assert result.fix is not None and result.fix.approved
    assert result.plan_text == _PLAN


class _UnhealthyBuild(FakeBuild):
    """codex preflight fails as it would for an unsupported model."""

    def codex(
        self, name: str = "codex", system_prompt: str | None = None, *, fast: bool = False
    ) -> Agent:
        self.created.append(name)
        return _Fake(name, [], error="codex: exit 1: The 'gpt-5.3-codex' model is not supported")


def test_preflight_passes_for_healthy_agents() -> None:
    preflight(FakeBuild())  # does not raise


def test_preflight_fails_fast_with_real_error_and_hint() -> None:
    with pytest.raises(AgentError) as excinfo:
        preflight(_UnhealthyBuild())
    message = str(excinfo.value)
    assert "not supported" in message  # the true cause, surfaced
    assert "AGENTLOOP_CODEX_MODEL" in message  # and how to fix it


class _ResumeReviewBuild(Build):
    def __init__(self) -> None:
        super().__init__(repo=None)
        self.created: list[str] = []

    def claude(
        self,
        name: str = "claude",
        system_prompt: str | None = None,
        *,
        fast: bool = False,
        tools: bool = True,
    ) -> Agent:
        self.created.append(name)
        return _Fake(name, ["APPROVED, correct and complete"])

    def codex(
        self, name: str = "codex", system_prompt: str | None = None, *, fast: bool = False
    ) -> Agent:
        self.created.append(name)
        return _Fake(name, ["APPROVED, correct and complete"])

    def implementer(self, name: str = "implementer") -> Agent:
        raise AssertionError("resume should continue the in-flight review, not re-run implementer")


def test_build_reads_model_overrides_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENTLOOP_CODEX_MODEL", "gpt-5.5")
    monkeypatch.setenv("AGENTLOOP_CLAUDE_MODEL", "claude-haiku-4-5")
    build = Build()
    assert build.codex_model == "gpt-5.5"
    assert build.claude_model == "claude-haiku-4-5"


def test_fast_tier_uses_fast_model_and_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AGENTLOOP_FAST_CLAUDE_MODEL", raising=False)
    monkeypatch.delenv("AGENTLOOP_FAST_CODEX_MODEL", raising=False)
    monkeypatch.setenv("AGENTLOOP_CODEX_MODEL", "gpt-5.5")
    build = Build()

    # fast claude defaults to haiku; fast codex falls back to the strong codex
    # model so it never reaches for an unsupported default.
    assert build.fast_claude_model == "claude-haiku-4-5"
    triage = build.claude("triage", fast=True, tools=False)
    assert isinstance(triage, ClaudeAgent)
    assert triage.model == "claude-haiku-4-5"
    assert triage.permission_mode is None  # tool-less
    debater = build.claude()
    assert isinstance(debater, ClaudeAgent)
    assert debater.permission_mode == "plan"  # strong tier keeps tools
    fast_codex = build.codex("scout2", fast=True)
    assert isinstance(fast_codex, CodexAgent)
    assert fast_codex.model == "gpt-5.5"


def test_fix_resumes_in_flight_review_without_restarting_attempt(tmp_path) -> None:
    journal = FixJournal(tmp_path / "fix.journal.jsonl")
    journal.record_implement(
        attempt=1,
        session_id="implementer-session",
        turns=1,
        message=_message("implementer", "implemented attempt 1", 0.01),
    )
    journal.record_review_turn(
        attempt=1,
        name="reviewer-claude",
        session_id="reviewer-claude-session",
        turns=1,
        message=_message("reviewer-claude", "APPROVED, looks correct", 0.01),
    )

    build = _ResumeReviewBuild()
    result = fix("fix it", _PLAN, build=build, store=journal, max_attempts=3, review_rounds=2)

    assert result.approved
    assert result.attempts == 1
    assert build.created == ["reviewer-claude", "reviewer-codex"]
    assert [m.author for m in result.transcript.messages] == [
        "implementer",
        "reviewer-claude",
        "reviewer-codex",
    ]


def test_solve_derives_fix_journal_from_main_store_by_default(tmp_path) -> None:
    build = FakeBuild(approve_on_attempt=1)
    store = JournalStore(tmp_path / "journal.jsonl")

    result = solve("fix it", build=build, rounds=8, stop=[], write=True, store=store)

    restored = store.fix_journal().restore()
    assert result.fix is not None and result.fix.approved
    assert restored is not None
    assert restored.completed
    assert restored.approved
    assert restored.attempt == 1
    assert restored.implement_message is not None


def _message(author: str, content: str, cost_usd: float):
    return Message(author, content, cost_usd=cost_usd)
