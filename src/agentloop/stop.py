"""Stop conditions: cross-cutting checks evaluated after every turn.

Each is a callable ``(Context) -> bool``. The loop halts as soon as one fires.
Keeping them separate from the policy means you can mix and match: cap the
rounds *and* the spend *and* watch for consensus, independently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .policy import Context


class StopCondition(Protocol):
    def __call__(self, ctx: Context) -> bool: ...


@dataclass(slots=True)
class MaxRounds:
    """Stop once this many agent turns have been taken."""

    n: int

    def __call__(self, ctx: Context) -> bool:
        return ctx.turn >= self.n


@dataclass(slots=True)
class Consensus:
    """Stop when the last turn from every agent contains ``marker``.

    Useful for debates: the loop ends only once *all* participants have signalled
    agreement on their most recent turn, not just the latest speaker.
    """

    marker: str = "AGREED"

    def __call__(self, ctx: Context) -> bool:
        marker = self.marker.lower()
        for agent in ctx.agents:
            last = next((m for m in reversed(ctx.transcript.by(agent.name))), None)
            if last is None or marker not in last.content.lower():
                return False
        return True


@dataclass(slots=True)
class BudgetUSD:
    """Stop once the transcript's total reported cost crosses ``limit``."""

    limit: float

    def __call__(self, ctx: Context) -> bool:
        return ctx.transcript.total_cost_usd >= self.limit
