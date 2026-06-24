"""The generic loop engine.

It knows nothing about reviews, debates, or any particular task. Every turn it
asks the policy who speaks and what they see, runs that agent, records the
result, and checks the stop conditions. That's the whole agent loop.

If given a ``store``, it also persists each turn and can resume a prior run: on
startup it replays the journal into the transcript and restores every agent's
session pointer, so the loop picks up exactly where it left off.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from .agent import Agent
from .domain import USER, Message, Transcript, TurnResult
from .policy import Context, Policy
from .stop import MaxRounds, StopCondition
from .store import RestoreState, Store

log = logging.getLogger("agentloop.orchestrator")


def fan_out(agents: Sequence[Agent], prompts: Sequence[str]) -> list[TurnResult]:
    """Run each agent's turn concurrently and return results in agent order.

    ``send()`` is blocking subprocess I/O, so the GIL is released while each CLI
    runs -- a thread pool gives real wall-clock parallelism here. Exceptions
    propagate from ``.result()``.
    """
    with ThreadPoolExecutor(max_workers=max(1, len(agents))) as pool:
        futures = [
            pool.submit(agent.send, prompt) for agent, prompt in zip(agents, prompts, strict=True)
        ]
        return [future.result() for future in futures]


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
        on_turn_start: Callable[[Agent, int], None] | None = None,
        on_parallel_start: Callable[[Sequence[Agent]], None] | None = None,
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
        # Fired right before an agent's (blocking) turn -- lets a UI show that
        # work is happening during the long silent wait for the subprocess.
        self.on_turn_start = on_turn_start
        # Fired before the parallel opening round (all agents at once).
        self.on_parallel_start = on_parallel_start
        self.store = store

    def run(self) -> LoopResult:
        transcript, resumed = self._start()

        turn = len(transcript.agent_messages)
        log.info(
            "%s loop: %s, max %d turns",
            "resuming" if resumed else "starting",
            " vs ".join(a.name for a in self.agents),
            self.stop[-1].n if isinstance(self.stop[-1], MaxRounds) else turn,
        )
        # Parallel opening: on a fresh run, if the policy says every agent's first
        # turn is independent, run them all at once -- nobody waits on the other
        # for context, and the serial debate begins as soon as both are ready.
        if turn == 0:
            openings = self.policy.parallel_opening(Context(transcript, self.agents, 0))
            if openings is not None:
                log.info("parallel opening: %s", ", ".join(a.name for a in self.agents))
                if self.on_parallel_start is not None:
                    self.on_parallel_start(self.agents)
                results = fan_out(self.agents, openings)
                for agent, result in zip(self.agents, results, strict=True):
                    self._record(agent, result, transcript)
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
            log.info("turn %d → %s", turn + 1, speaker.name)
            if self.on_turn_start is not None:
                self.on_turn_start(speaker, turn)
            self._record(speaker, speaker.send(prompt), transcript)
            turn += 1

        log.info(
            "stopped by %s after %d turns ($%.4f)", stopped_by, turn, transcript.total_cost_usd
        )
        return LoopResult(transcript=transcript, turns=turn, stopped_by=stopped_by, resumed=resumed)

    def _record(self, speaker: Agent, result: TurnResult, transcript: Transcript) -> None:
        """Append a turn to the transcript, journal it, and notify -- shared by
        the parallel opening and the serial loop."""
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
