"""Core data types shared across the loop.

These are deliberately tiny and serializable so a transcript can be logged,
persisted, or replayed without dragging in any agent/CLI machinery.
"""

from __future__ import annotations

from dataclasses import dataclass, field

USER = "user"

# Free-form usage payload as emitted by the underlying CLI (token counts, etc.).
type Usage = dict[str, object]


@dataclass(frozen=True, slots=True)
class TurnResult:
    """What a single agent turn produced."""

    text: str
    session_id: str | None = None
    usage: Usage | None = None
    cost_usd: float | None = None
    raw: object = None  # the untouched CLI payload, for debugging


@dataclass(frozen=True, slots=True)
class Message:
    """One entry in the shared transcript."""

    author: str  # agent name, or USER for the seed task
    content: str
    usage: Usage | None = None
    cost_usd: float | None = None


@dataclass(slots=True)
class Transcript:
    """The shared, append-only conversation every agent contributes to."""

    messages: list[Message] = field(default_factory=list)

    def add(self, message: Message) -> Message:
        self.messages.append(message)
        return message

    def last_from_others(self, author: str) -> Message | None:
        """The most recent message not written by ``author`` (incl. the user seed)."""
        for message in reversed(self.messages):
            if message.author != author:
                return message
        return None

    def by(self, author: str) -> list[Message]:
        return [m for m in self.messages if m.author == author]

    @property
    def agent_messages(self) -> list[Message]:
        return [m for m in self.messages if m.author != USER]

    @property
    def total_cost_usd(self) -> float:
        return sum(m.cost_usd or 0.0 for m in self.messages)
