"""Adapter for the ``codex`` CLI (OpenAI Codex) in non-interactive exec mode."""

from __future__ import annotations

import json
from pathlib import Path

from ..agent import CliAgent
from ..domain import TurnResult


class CodexAgent(CliAgent):
    """Drives ``codex exec --json`` and resumes via ``codex exec resume <id>``.

    ``--json`` emits a stream of newline-delimited events::

        {"type": "thread.started", "thread_id": "..."}
        {"type": "item.completed", "item": {"type": "agent_message", "text": "..."}}
        {"type": "turn.completed", "usage": {...}}

    Codex has no system-prompt flag in exec mode, so on the first turn we prepend
    the role to the prompt; resumed turns already carry it in session history.
    """

    binary = "codex"

    def __init__(
        self,
        name: str = "codex",
        *,
        model: str | None = None,
        system_prompt: str | None = None,
        cwd: str | Path | None = None,
        timeout: float = 600.0,
        extra_args: list[str] | None = None,
        sandbox: str = "read-only",
    ) -> None:
        super().__init__(
            name,
            model=model,
            system_prompt=system_prompt,
            cwd=cwd,
            timeout=timeout,
            extra_args=extra_args,
        )
        self.sandbox = sandbox

    def _build_command(self, prompt: str, *, first_turn: bool) -> list[str]:
        if not first_turn and self.session_id:
            cmd = [
                self.binary,
                "exec",
                "resume",
                self.session_id,
                "--json",
                "--skip-git-repo-check",
            ]
            if self.model:
                cmd += ["-m", self.model]
        else:
            cmd = [self.binary, "exec", "--json", "--skip-git-repo-check", "-s", self.sandbox]
            if self.model:
                cmd += ["-m", self.model]
            if self.system_prompt:
                prompt = f"{self.system_prompt}\n\n---\n\n{prompt}"
        cmd += self.extra_args
        cmd += [prompt]
        return cmd

    def _parse(self, stdout: str) -> TurnResult:
        session_id: str | None = None
        usage: dict[str, object] | None = None
        parts: list[str] = []

        for line in stdout.splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            match event.get("type"):
                case "thread.started":
                    session_id = event.get("thread_id")
                case "item.completed":
                    item = event.get("item", {})
                    if item.get("type") == "agent_message" and item.get("text"):
                        parts.append(str(item["text"]))
                case "turn.completed":
                    usage = event.get("usage")

        return TurnResult(
            text="\n".join(parts).strip(),
            session_id=session_id,
            usage=usage,
            raw=stdout,
        )
