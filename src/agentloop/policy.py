"""Policies decide *who speaks next* and *what they see*.

This is the knob that makes the loop generic: the orchestrator just asks the
policy to pick a speaker and compose its prompt. Swap the policy and the same
engine goes from round-robin brainstorming to an adversarial two-agent debate.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from .agent import Agent
from .domain import Transcript


@dataclass(slots=True)
class Context:
    """Everything a policy needs to make its decision."""

    transcript: Transcript
    agents: list[Agent]
    turn: int  # 0-based count of agent turns taken so far


class Policy(ABC):
    @abstractmethod
    def select(self, ctx: Context) -> Agent | None:
        """Return the next speaker, or ``None`` to end the loop."""

    @abstractmethod
    def compose(self, speaker: Agent, ctx: Context) -> str:
        """Build the prompt string handed to ``speaker`` this turn."""

    def seed(self) -> str | None:
        """Opening message recorded as the user turn before the loop starts.

        The task lives here, in the policy, as the single source of truth -- the
        orchestrator records it but never owns a second copy."""
        return None

    def parallel_opening(self, ctx: Context) -> list[str] | None:
        """If every agent's first turn is independent of the others, return one
        opening prompt per agent (in ``ctx.agents`` order) so the orchestrator can
        run them concurrently. Return ``None`` to keep the loop fully serial."""
        return None


class RoundRobinPolicy(Policy):
    """Cycle through agents in order. First time each agent speaks it gets the
    full task; afterwards it just sees the latest message from someone else."""

    def __init__(self, task: str) -> None:
        self.task = task

    def seed(self) -> str | None:
        return self.task

    def select(self, ctx: Context) -> Agent | None:
        return ctx.agents[ctx.turn % len(ctx.agents)]

    def compose(self, speaker: Agent, ctx: Context) -> str:
        last = ctx.transcript.last_from_others(speaker.name)
        if _is_first_turn(speaker) or last is None:
            return self.task
        return f"{last.author} said:\n\n{last.content}\n\nYour turn — respond."


class DebatePolicy(Policy):
    """Two agents alternate. Each opens with its own review of the task, then
    rebuts the other until they converge (let a ``Consensus`` stop end it)."""

    def __init__(
        self,
        task: str,
        *,
        opening_instructions: str = (
            "Independently complete the task below. Be concrete and specific."
        ),
        rebuttal_instructions: str = (
            "Critique the other agent's response: what did they miss, get wrong, "
            "or over-claim? Revise your own position. If you fully agree and have "
            "nothing to add, begin your reply with the single word AGREED."
        ),
    ) -> None:
        self.task = task
        self.opening_instructions = opening_instructions
        self.rebuttal_instructions = rebuttal_instructions

    def seed(self) -> str | None:
        return self.task

    def select(self, ctx: Context) -> Agent | None:
        if len(ctx.agents) != 2:
            raise ValueError("DebatePolicy requires exactly two agents")
        return ctx.agents[ctx.turn % 2]

    def _opening(self) -> str:
        return f"{self.opening_instructions}\n\n--- TASK ---\n{self.task}"

    def parallel_opening(self, ctx: Context) -> list[str] | None:
        # Each agent's opening depends only on the task, never on the others, so
        # the whole first round can run at once instead of A-waits-for-B.
        return [self._opening() for _ in ctx.agents]

    def compose(self, speaker: Agent, ctx: Context) -> str:
        last = ctx.transcript.last_from_others(speaker.name)
        if _is_first_turn(speaker) or last is None or last.author == "user":
            return self._opening()
        return (
            f"{self.rebuttal_instructions}\n\n"
            f"--- {last.author}'s latest response ---\n{last.content}"
        )


def _is_first_turn(speaker: Agent) -> bool:
    # CliAgent tracks this; non-CLI agents are treated as always-fresh.
    return getattr(speaker, "is_first_turn", True)
