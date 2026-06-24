"""Pipeline tests: the write-gate and the fix loop, offline against fake agents.

``FakeBuild`` overrides every agent factory ``solve``/``fix`` use, so no CLI
subprocess ever runs. Reviewer replies are driven by the attempt counter so we
can exercise both the one-shot-approve and loop-until-approved paths.
"""

from __future__ import annotations

import pytest

from agentloop.agent import Agent, AgentError
from agentloop.domain import TurnResult
from agentloop.pipeline import Build, fix, preflight, solve

_TRIAGE_NO_DISCOVERY = '{"discovery": false, "focuses": [], "reason": "simple"}'
_PLAN = "Plan: correct the off-by-one in foo.py and re-run the suite to confirm."


class _Fake(Agent):
    """Replays a fixed list of responses; ``error`` makes ``send`` raise instead."""

    def __init__(self, name: str, replies: list[str], *, error: str | None = None) -> None:
        self.name = name
        self._replies = replies
        self._error = error
        self.session_id: str | None = None
        self.turns = 0

    @property
    def is_first_turn(self) -> bool:
        return self.turns == 0

    def send(self, prompt: str) -> TurnResult:
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

    def _debater(self, name: str) -> _Fake:
        return _Fake(name, [_PLAN, "AGREED"])

    def _reviewer(self, name: str) -> _Fake:
        approved = self.attempt >= self.approve_on_attempt
        return _Fake(
            name, ["APPROVED, correct and complete" if approved else "needs work: edge case"]
        )

    def claude(self, name: str = "claude", system_prompt: str | None = None) -> Agent:
        self.created.append(name)
        if name == "triage":
            return _Fake(name, [_TRIAGE_NO_DISCOVERY])
        if name.startswith("reviewer"):
            return self._reviewer(name)
        return self._debater(name)

    def codex(self, name: str = "codex", system_prompt: str | None = None) -> Agent:
        self.created.append(name)
        if name.startswith("reviewer"):
            return self._reviewer(name)
        return self._debater(name)

    def implementer(self, name: str = "implementer") -> Agent:
        self.attempt += 1
        self.created.append(name)
        return _Fake(name, [f"implemented attempt {self.attempt}; ran tests, all green"])


def test_gate_stops_at_plan_without_write() -> None:
    build = FakeBuild()
    result = solve("fix it", build=build, rounds=8, stop=[], write=False)

    assert result.fix is None  # never crossed the gate
    assert result.loop is not None
    assert "off-by-one" in result.plan_text  # the agreed plan was extracted
    assert "implementer" not in build.created


def test_write_crosses_gate_and_approves_first_try() -> None:
    build = FakeBuild(approve_on_attempt=1)
    result = solve("fix it", build=build, rounds=8, stop=[], write=True)

    assert result.fix is not None
    assert result.fix.approved
    assert result.fix.attempts == 1
    assert build.created.count("implementer") == 1
    # The implementer's report is captured in the fix transcript.
    assert any(m.author == "implementer" for m in result.fix.transcript.messages)


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

    def codex(self, name: str = "codex", system_prompt: str | None = None) -> Agent:
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


def test_build_reads_model_overrides_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENTLOOP_CODEX_MODEL", "gpt-5.5")
    monkeypatch.setenv("AGENTLOOP_CLAUDE_MODEL", "claude-haiku-4-5")
    build = Build()
    assert build.codex_model == "gpt-5.5"
    assert build.claude_model == "claude-haiku-4-5"
