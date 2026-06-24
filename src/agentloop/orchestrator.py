"""The generic loop engine.

It knows nothing about reviews, debates, or any particular task. Every turn it
asks the policy who speaks and what they see, runs that agent, records the
result, and checks the stop conditions. That's the whole agent loop.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from .agent import Agent
from .domain import USER, Message, Transcript
from .policy import Context, Policy
from .stop import MaxRounds, StopCondition


@dataclass(slots=True)
class LoopResult:
    transcript: Transcript
    turns: int
    stopped_by: str


class Orchestrator:
    def __init__(
        self,
        agents: list[Agent],
        policy: Policy,
        *,
        stop: Sequence[StopCondition] | None = None,
        max_rounds: int = 12,
        on_message: Callable[[Message], None] | None = None,
    ) -> None:
        if not agents:
            raise ValueError("need at least one agent")
        self.agents = agents
        self.policy = policy
        # A hard turn cap is always present as a backstop against runaway loops.
        self.stop: list[StopCondition] = [MaxRounds(max_rounds), *(stop or [])]
        self.on_message = on_message

    def run(self) -> LoopResult:
        transcript = Transcript()
        seed = self.policy.seed()
        if seed:
            transcript.add(Message(USER, seed))

        turn = 0
        stopped_by = "no_speaker"
        while True:
            ctx = Context(transcript, self.agents, turn)
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
            if self.on_message:
                self.on_message(message)

            turn += 1
            ctx = Context(transcript, self.agents, turn)
            fired = next((s for s in self.stop if s(ctx)), None)
            if fired is not None:
                stopped_by = type(fired).__name__
                break

        return LoopResult(transcript=transcript, turns=turn, stopped_by=stopped_by)
