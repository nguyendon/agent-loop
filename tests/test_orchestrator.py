"""Loop-engine tests that run offline against a scripted fake agent."""

from __future__ import annotations

from pathlib import Path

from agentloop import (
    Consensus,
    DebatePolicy,
    JournalStore,
    Message,
    Orchestrator,
    RoundRobinPolicy,
    fan_out,
)
from agentloop.agent import Agent
from agentloop.domain import TurnResult


class FakeAgent(Agent):
    """Replays a fixed list of responses; records every prompt it was given."""

    def __init__(self, name: str, replies: list[str]) -> None:
        self.name = name
        self._replies = replies
        self.prompts: list[str] = []
        self.session_id: str | None = None
        self.turns = 0

    @property
    def is_first_turn(self) -> bool:
        return self.turns == 0

    def send(self, prompt: str) -> TurnResult:
        self.prompts.append(prompt)
        reply = self._replies[min(self.turns, len(self._replies) - 1)]
        self.turns += 1
        self.session_id = f"{self.name}-session"
        return TurnResult(text=reply, session_id=self.session_id, cost_usd=0.01)


def test_round_robin_alternates_and_seeds_task() -> None:
    a = FakeAgent("a", ["a1", "a2"])
    b = FakeAgent("b", ["b1", "b2"])
    loop = Orchestrator([a, b], RoundRobinPolicy("do the thing"), max_rounds=4)

    result = loop.run()

    assert [m.author for m in result.transcript.agent_messages] == ["a", "b", "a", "b"]
    # Each agent's opening prompt is the raw task; later prompts quote the other.
    assert a.prompts[0] == "do the thing"
    assert "b said:" in a.prompts[1]


def test_debate_opens_in_parallel_then_goes_serial() -> None:
    a = FakeAgent("a", ["a-open", "a-rebut"])
    b = FakeAgent("b", ["b-open", "b-rebut"])
    loop = Orchestrator([a, b], DebatePolicy("the task"), max_rounds=4)

    result = loop.run()

    authors = [m.author for m in result.transcript.agent_messages]
    assert authors == ["a", "b", "a", "b"]
    # Both openings are independent: each agent's first prompt is the same task
    # framing and neither quotes the other.
    assert a.prompts[0] == b.prompts[0]
    assert "b-open" not in a.prompts[0]
    # The serial rebuttal that follows does quote the other's opening.
    assert "b-open" in a.prompts[1]


def test_fan_out_preserves_agent_order() -> None:
    agents = [FakeAgent("a", ["ra"]), FakeAgent("b", ["rb"]), FakeAgent("c", ["rc"])]
    results = fan_out(agents, ["pa", "pb", "pc"])

    assert [r.text for r in results] == ["ra", "rb", "rc"]
    # Each agent received its matching prompt.
    assert [agent.prompts[0] for agent in agents] == ["pa", "pb", "pc"]


def test_consensus_stops_when_all_agents_agree() -> None:
    a = FakeAgent("a", ["here are findings", "AGREED, nothing to add"])
    b = FakeAgent("b", ["my critique", "AGREED"])
    loop = Orchestrator([a, b], DebatePolicy("review this"), stop=[Consensus()], max_rounds=20)

    result = loop.run()

    # a1, b1, a2(AGREED), b2(AGREED) -> stops at 4, not the 20-round cap.
    assert result.turns == 4
    assert result.stopped_by == "Consensus"


def test_consensus_requires_marker_as_leading_token() -> None:
    # "Not quite AGREED" means the opposite -- a substring check would stop the
    # loop here at turn 2; a leading-token check must not.
    a = FakeAgent("a", ["Not quite AGREED, one issue remains", "AGREED"])
    b = FakeAgent("b", ["AGREED", "AGREED"])
    loop = Orchestrator([a, b], DebatePolicy("x"), stop=[Consensus()], max_rounds=10)

    result = loop.run()

    assert result.turns == 3  # not 2: the negated turn didn't count as agreement
    assert result.stopped_by == "Consensus"


def test_real_consensus_wins_label_over_max_rounds_on_tie() -> None:
    # Consensus is reached exactly as the round cap is hit; the meaningful
    # condition should win the label, not the backstop.
    a = FakeAgent("a", ["AGREED"])
    b = FakeAgent("b", ["AGREED"])
    loop = Orchestrator([a, b], DebatePolicy("x"), stop=[Consensus()], max_rounds=2)

    result = loop.run()

    assert result.turns == 2
    assert result.stopped_by == "Consensus"


def test_max_rounds_is_a_backstop() -> None:
    a = FakeAgent("a", ["never agree"])
    b = FakeAgent("b", ["also never"])
    loop = Orchestrator([a, b], DebatePolicy("x"), stop=[Consensus()], max_rounds=3)

    result = loop.run()

    assert result.turns == 3
    assert result.stopped_by == "MaxRounds"


def test_policy_seed_recorded_as_user_message() -> None:
    a = FakeAgent("a", ["ok"])
    loop = Orchestrator([a], RoundRobinPolicy("task"), max_rounds=1)

    result = loop.run()

    # The seed comes from the policy -- the single source of truth for the task.
    assert result.transcript.messages[0] == Message("user", "task")


def test_journal_round_trips_transcript_and_sessions(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    a, b = FakeAgent("a", ["a1"]), FakeAgent("b", ["b1"])
    Orchestrator([a, b], DebatePolicy("the task"), max_rounds=2, store=JournalStore(path)).run()

    restored = JournalStore(path).restore()

    assert restored is not None
    assert restored.transcript.messages[0] == Message("user", "the task")
    assert [m.author for m in restored.transcript.agent_messages] == ["a", "b"]
    # The agent's session pointer survives, so a resumed agent can pick its
    # real CLI history back up instead of starting cold.
    assert restored.agents["a"].session_id == "a-session"


def test_resume_continues_without_reseeding_or_repeating(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    # First run stops at the 2-turn cap, mid-conversation.
    a, b = FakeAgent("a", ["a1"]), FakeAgent("b", ["b1"])
    first = Orchestrator([a, b], DebatePolicy("task"), max_rounds=2, store=JournalStore(path)).run()
    assert first.turns == 2 and not first.resumed

    # Fresh agent objects + same journal -> resume and carry on.
    a2, b2 = FakeAgent("a", ["a2"]), FakeAgent("b", ["b2"])
    second = Orchestrator(
        [a2, b2], DebatePolicy("task"), max_rounds=4, store=JournalStore(path)
    ).run()

    assert second.resumed is True
    assert second.turns == 4  # 2 restored + 2 new, not restarted from zero
    assert [m.author for m in second.transcript.agent_messages] == ["a", "b", "a", "b"]
    # Seed recorded exactly once across both runs.
    assert sum(1 for m in second.transcript.messages if m.author == "user") == 1
    # Restored agents resumed their sessions rather than treating turn 1 as fresh.
    assert a2.turns == 2  # 1 restored + 1 taken this run


def test_parse_plan_extracts_focuses_and_caps() -> None:
    from agentloop.pipeline import parse_plan

    raw = '```json\n{"discovery": true, "focuses": ["a","b","c","d","e"], "reason": "big"}\n```'
    plan = parse_plan(raw, max_agents=3)
    assert plan.discovery is True
    assert plan.focuses == ["a", "b", "c"]  # capped at max_agents


def test_parse_plan_simple_question_no_discovery() -> None:
    from agentloop.pipeline import parse_plan

    plan = parse_plan('{"discovery": false, "focuses": [], "reason": "simple"}', max_agents=4)
    assert plan.discovery is False
    assert plan.focuses == []


def test_parse_plan_falls_back_on_garbage() -> None:
    from agentloop.pipeline import parse_plan

    plan = parse_plan("the model rambled with no json", max_agents=4)
    assert plan.discovery is False  # safe default
