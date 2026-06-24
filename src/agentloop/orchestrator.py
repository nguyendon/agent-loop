"""The generic loop engine.

It knows nothing about reviews, debates, or any particular task. Every turn it
asks the policy who speaks and what they see, runs that agent, records the
result, and checks the stop conditions. That's the whole agent loop.

If given a ``store``, it also persists each turn and can resume a prior run: on
startup it replays the journal into the transcript and restores every agent's
session pointer, so the loop picks up exactly where it left off.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from .agent import Agent
from .domain import USER, Message, Transcript
from .policy import Context, Policy
from .stop import MaxRounds, StopCondition
from .store import RestoreState, Store


@dataclass(slots=True)
class LoopResult:
    transcript: Transcript
    turns: int
    stopped_by: str
    resumed: bool = False


class Orchestrator:
    def __init__(
        self,
        agents: list[Agent],
        policy: Policy,
        *,
        stop: Sequence[StopCondition] | None = None,
        max_rounds: int = 12,
        on_message: Callable[[Message], None] | None = None,
        store: Store | None = None,
    ) -> None:
        if not agents:
            raise ValueError("need at least one agent")
        self.agents = agents
        self.policy = policy
        # User stops are checked first so a meaningful exit (e.g. Consensus) wins
        # the label when it and the backstop fire on the same turn. MaxRounds is
        # the last-resort cap; it counts *total* transcript turns, so on resume
        # it bounds the combined length of the original and continued runs.
        self.stop: list[StopCondition] = [*(stop or []), MaxRounds(max_rounds)]
        self.on_message = on_message
        self.store = store

    def run(self) -> LoopResult:
        transcript, resumed = self._start()

        turn = len(transcript.agent_messages)
        stopped_by = "no_speaker"
        while True:
            ctx = Context(transcript, self.agents, turn)

            # Checked at the top so a resumed-but-already-finished run exits
            # without taking a needless extra turn.
            fired = next((s for s in self.stop if s(ctx)), None)
            if fired is not None:
                stopped_by = type(fired).__name__
                break

            speaker = self.policy.select(ctx)
            if speaker is None:
                stopped_by = "policy"
                break

            prompt = self.policy.compose(speaker, ctx)
            result = speaker.send(prompt)
            message = transcript.add(
                Message(
                    author=speaker.name,
                    content=result.text,
                    usage=result.usage,
                    cost_usd=result.cost_usd,
                )
            )
            if self.store is not None:
                self.store.record_turn(
                    name=speaker.name,
                    session_id=speaker.session_id,
                    turns=speaker.turns,
                    message=message,
                )
            if self.on_message is not None:
                self.on_message(message)

            turn += 1

        return LoopResult(transcript=transcript, turns=turn, stopped_by=stopped_by, resumed=resumed)

    # --- startup: fresh seed or restore from the journal ---------------------

    def _start(self) -> tuple[Transcript, bool]:
        restored: RestoreState | None = self.store.restore() if self.store else None
        if restored is not None:
            self._apply_agent_state(restored)
            return restored.transcript, True

        transcript = Transcript()
        seed = self.policy.seed()
        if seed:
            message = transcript.add(Message(USER, seed))
            if self.store is not None:
                self.store.record_seed(message)
        return transcript, False

    def _apply_agent_state(self, restored: RestoreState) -> None:
        by_name = {agent.name: agent for agent in self.agents}
        for name, state in restored.agents.items():
            agent = by_name.get(name)
            if agent is not None:
                agent.session_id = state.session_id
                agent.turns = state.turns
