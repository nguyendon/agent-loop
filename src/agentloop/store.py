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
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from .domain import USER, Message, Transcript

log = logging.getLogger("agentloop.store")


@dataclass(slots=True)
class AgentState:
    """The bit of an agent worth persisting: where its CLI session lives."""

    session_id: str | None
    turns: int


@dataclass(slots=True)
class RestoreState:
    transcript: Transcript
    agents: dict[str, AgentState] = field(default_factory=dict)


@dataclass(slots=True)
class FixRestoreState:
    transcript: Transcript
    review_transcript: Transcript
    implement_message: Message | None
    feedback: str
    attempt: int
    approved: bool
    completed: bool
    reviewer_agents: dict[str, AgentState] = field(default_factory=dict)


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

    def fix_journal(self) -> FixJournal:
        """The sibling stage-2 journal used for the write path."""
        return FixJournal(self.path.with_name("fix.journal.jsonl"))

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
        log.info("restored %d turns from %s", len(transcript.agent_messages), self.path)
        return RestoreState(transcript=transcript, agents=agents)


class FixJournal:
    """Append-only JSONL journal for the stage-2 implement/review loop."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def exists(self) -> bool:
        return self.path.exists() and self.path.stat().st_size > 0

    def _append(self, record: dict[str, object]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def record_implement(
        self, *, attempt: int, session_id: str | None, turns: int, message: Message
    ) -> None:
        self._append(
            {
                "kind": "implement",
                "attempt": attempt,
                "session_id": session_id,
                "turns": turns,
                "content": message.content,
                "usage": message.usage,
                "cost_usd": message.cost_usd,
            }
        )

    def record_review_turn(
        self, *, attempt: int, name: str, session_id: str | None, turns: int, message: Message
    ) -> None:
        self._append(
            {
                "kind": "review_turn",
                "attempt": attempt,
                "name": name,
                "session_id": session_id,
                "turns": turns,
                "content": message.content,
                "usage": message.usage,
                "cost_usd": message.cost_usd,
            }
        )

    def record_feedback(self, *, attempt: int, feedback: str) -> None:
        self._append({"kind": "feedback", "attempt": attempt, "feedback": feedback})

    def record_outcome(self, *, attempt: int, approved: bool) -> None:
        self._append({"kind": "outcome", "attempt": attempt, "approved": approved})

    def restore(self) -> FixRestoreState | None:
        if not self.exists():
            return None

        transcript = Transcript()
        review_by_attempt: dict[int, Transcript] = {}
        reviewer_agents_by_attempt: dict[int, dict[str, AgentState]] = {}
        implement_by_attempt: dict[int, Message] = {}
        feedback_by_attempt: dict[int, str] = {}
        attempt = 0
        outcome_attempt = 0
        approved = False
        completed = False

        with self.path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                kind = record.get("kind")
                if kind == "implement":
                    record_attempt = int(record.get("attempt", 0))
                    attempt = max(attempt, record_attempt)
                    message = Message(
                        author="implementer",
                        content=record["content"],
                        usage=record.get("usage"),
                        cost_usd=record.get("cost_usd"),
                    )
                    transcript.add(message)
                    implement_by_attempt[record_attempt] = message
                elif kind == "review_turn":
                    record_attempt = int(record.get("attempt", 0))
                    attempt = max(attempt, record_attempt)
                    message = Message(
                        author=record["name"],
                        content=record["content"],
                        usage=record.get("usage"),
                        cost_usd=record.get("cost_usd"),
                    )
                    transcript.add(message)
                    review_by_attempt.setdefault(record_attempt, Transcript()).add(message)
                    reviewer_agents_by_attempt.setdefault(record_attempt, {})[record["name"]] = (
                        AgentState(
                            session_id=record.get("session_id"),
                            turns=record.get("turns", 0),
                        )
                    )
                elif kind == "feedback":
                    record_attempt = int(record.get("attempt", 0))
                    attempt = max(attempt, record_attempt)
                    feedback_by_attempt[record_attempt] = str(record.get("feedback", ""))
                elif kind == "outcome":
                    outcome_attempt = int(record.get("attempt", 0))
                    attempt = max(attempt, outcome_attempt)
                    approved = bool(record.get("approved"))
                    completed = True

        current_attempt = outcome_attempt or attempt
        review_transcript = review_by_attempt.get(current_attempt, Transcript())
        reviewer_agents = reviewer_agents_by_attempt.get(current_attempt, {})
        implement_message = implement_by_attempt.get(current_attempt)
        feedback = feedback_by_attempt.get(current_attempt, "")
        log.info("restored %d stage-2 turns from %s", len(transcript.messages), self.path)
        return FixRestoreState(
            transcript=transcript,
            review_transcript=review_transcript,
            implement_message=implement_message,
            feedback=feedback,
            attempt=current_attempt,
            approved=approved,
            completed=completed,
            reviewer_agents=reviewer_agents,
        )
