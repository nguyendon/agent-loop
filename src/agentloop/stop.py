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
    """Stop when every agent's last turn *begins with* ``marker``.

    Useful for debates: the loop ends only once *all* participants have signalled
    agreement on their most recent turn, not just the latest speaker.

    The marker must be a leading token, matching the policy instruction to
    "begin your reply with AGREED". A substring check would be fooled by a
    negation like "Not quite AGREED..." -- which means the opposite.
    """

    marker: str = "AGREED"

    def __call__(self, ctx: Context) -> bool:
        marker = self.marker.lower()
        for agent in ctx.agents:
            last = next((m for m in reversed(ctx.transcript.by(agent.name))), None)
            if last is None or not last.content.strip().lower().startswith(marker):
                return False
        return True


@dataclass(slots=True)
class BudgetUSD:
    """Stop once the transcript's total reported cost crosses ``limit``."""

    limit: float

    def __call__(self, ctx: Context) -> bool:
        return ctx.transcript.total_cost_usd >= self.limit
