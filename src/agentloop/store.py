"""Durable, resumable persistence for a loop's transcript and agent sessions.

A run is one JSONL file. Every turn is appended as a self-contained line *as it
happens*, so a crash loses at most the in-flight turn. Replaying the file
rebuilds two things: the shared transcript (Layer 1) AND each agent's session
pointer (Layer 2). Because the CLIs keep their own history on disk (Layer 3), a
restored ``session_id`` is enough for an agent to resume its real context -- so a
resumed loop continues with the same transcript and the same live sessions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from .domain import USER, Message, Transcript


@dataclass(slots=True)
class AgentState:
    """The bit of an agent worth persisting: where its CLI session lives."""

    session_id: str | None
    turns: int


@dataclass(slots=True)
class RestoreState:
    transcript: Transcript
    agents: dict[str, AgentState] = field(default_factory=dict)


class Store(Protocol):
    """What the orchestrator needs from a persistence backend."""

    def record_seed(self, message: Message) -> None: ...
    def record_turn(
        self, *, name: str, session_id: str | None, turns: int, message: Message
    ) -> None: ...
    def restore(self) -> RestoreState | None: ...


class JournalStore:
    """Append-only JSONL journal. One file == one resumable run."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def exists(self) -> bool:
        return self.path.exists() and self.path.stat().st_size > 0

    def _append(self, record: dict[str, object]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def record_seed(self, message: Message) -> None:
        self._append({"kind": "seed", "content": message.content})

    def record_turn(
        self, *, name: str, session_id: str | None, turns: int, message: Message
    ) -> None:
        self._append(
            {
                "kind": "turn",
                "name": name,
                "session_id": session_id,
                "turns": turns,
                "content": message.content,
                "usage": message.usage,
                "cost_usd": message.cost_usd,
            }
        )

    def restore(self) -> RestoreState | None:
        """Replay the journal into a transcript + per-agent session state."""
        if not self.exists():
            return None
        transcript = Transcript()
        agents: dict[str, AgentState] = {}
        with self.path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                match record.get("kind"):
                    case "seed":
                        transcript.add(Message(USER, record["content"]))
                    case "turn":
                        transcript.add(
                            Message(
                                author=record["name"],
                                content=record["content"],
                                usage=record.get("usage"),
                                cost_usd=record.get("cost_usd"),
                            )
                        )
                        # Last writer wins: the newest line for an agent holds
                        # its current session id and turn count.
                        agents[record["name"]] = AgentState(
                            session_id=record.get("session_id"),
                            turns=record.get("turns", 0),
                        )
        return RestoreState(transcript=transcript, agents=agents)
