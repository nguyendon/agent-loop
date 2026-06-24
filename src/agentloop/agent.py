"""The ``Agent`` abstraction and a subprocess-backed base for CLI agents.

An agent is the generic unit the loop drives: hand it a prompt, get a
``TurnResult`` back. ``CliAgent`` adds the plumbing every CLI tool needs --
spawn a process, enforce a timeout, parse output, and remember the session id so
the next turn resumes the same conversation instead of starting cold.
"""

from __future__ import annotations

import logging
import subprocess
import time
from abc import ABC, abstractmethod
from pathlib import Path

from .domain import TurnResult

log = logging.getLogger("agentloop.agent")


class AgentError(RuntimeError):
    """Raised when an agent's underlying process fails or returns nothing usable."""


class Agent(ABC):
    """Anything the orchestrator can ask to take a turn.

    ``session_id`` and ``turns`` are the resumable-session contract: the
    orchestrator reads them and the store persists them, so a restored agent can
    pick its real history back up. Stateless agents just leave them at default.
    """

    name: str
    session_id: str | None = None
    turns: int = 0

    @abstractmethod
    def send(self, prompt: str) -> TurnResult:
        """Run one turn against ``prompt`` and return the result."""

    def reset(self) -> None:  # noqa: B027 -- optional hook; stateless agents need no-op
        """Forget any conversation state so the next turn starts fresh."""


class CliAgent(Agent):
    """Base class for agents backed by a non-interactive CLI subprocess."""

    def __init__(
        self,
        name: str,
        *,
        model: str | None = None,
        system_prompt: str | None = None,
        cwd: str | Path | None = None,
        timeout: float = 600.0,
        extra_args: list[str] | None = None,
    ) -> None:
        self.name = name
        self.model = model
        self.system_prompt = system_prompt
        self.cwd = Path(cwd) if cwd else None
        self.timeout = timeout
        self.extra_args = extra_args or []
        self.session_id: str | None = None
        self.turns = 0

    @property
    def is_first_turn(self) -> bool:
        return self.session_id is None

    def reset(self) -> None:
        self.session_id = None
        self.turns = 0

    # --- subclass contract ---------------------------------------------------

    @abstractmethod
    def _build_command(self, prompt: str, *, first_turn: bool) -> list[str]:
        """Return argv for this turn. ``first_turn`` lets adapters decide whether
        to start a session or resume one, and whether to inject the system prompt."""

    @abstractmethod
    def _parse(self, stdout: str) -> TurnResult:
        """Turn the process's stdout into a ``TurnResult``."""

    # --- driver --------------------------------------------------------------

    def send(self, prompt: str) -> TurnResult:
        first_turn = self.is_first_turn
        command = self._build_command(prompt, first_turn=first_turn)
        # The prompt is the last arg and can be huge; log the rest plus its size.
        log.debug("%s: exec %s (prompt %d chars)", self.name, command[:-1], len(prompt))

        start = time.monotonic()
        try:
            proc = subprocess.run(
                command,
                capture_output=True,
                text=True,
                cwd=self.cwd,
                stdin=subprocess.DEVNULL,  # keep the CLI from blocking on stdin
                timeout=self.timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise AgentError(f"{self.name}: timed out after {self.timeout}s") from exc
        elapsed = time.monotonic() - start

        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "").strip()
            raise AgentError(f"{self.name}: exit {proc.returncode}: {detail[:500]}")

        result = self._parse(proc.stdout)
        if result.session_id:
            self.session_id = result.session_id
        self.turns += 1
        cost = f", ${result.cost_usd:.4f}" if result.cost_usd else ""
        log.info("%s: turn done in %.1fs (%d chars%s)", self.name, elapsed, len(result.text), cost)
        return result
