"""Adapter for the ``claude`` CLI (Claude Code) in non-interactive print mode."""

from __future__ import annotations

import json
from pathlib import Path

from ..agent import AgentError, CliAgent
from ..domain import TurnResult


class ClaudeAgent(CliAgent):
    """Drives ``claude -p --output-format json`` and resumes via ``--resume``.

    The JSON result looks like::

        {"type": "result", "result": "...", "session_id": "...",
         "is_error": false, "usage": {...}, "total_cost_usd": 0.04}
    """

    binary = "claude"

    def __init__(
        self,
        name: str = "claude",
        *,
        model: str | None = None,
        system_prompt: str | None = None,
        cwd: str | Path | None = None,
        timeout: float = 600.0,
        extra_args: list[str] | None = None,
        permission_mode: str | None = None,
    ) -> None:
        super().__init__(
            name,
            model=model,
            system_prompt=system_prompt,
            cwd=cwd,
            timeout=timeout,
            extra_args=extra_args,
        )
        self.permission_mode = permission_mode

    def _build_command(self, prompt: str, *, first_turn: bool) -> list[str]:
        cmd = [self.binary, "-p", "--output-format", "json"]
        if self.model:
            cmd += ["--model", self.model]
        if self.permission_mode:
            cmd += ["--permission-mode", self.permission_mode]
        # Only inject the system prompt once; resumed turns already carry it.
        if first_turn and self.system_prompt:
            cmd += ["--append-system-prompt", self.system_prompt]
        if not first_turn and self.session_id:
            cmd += ["--resume", self.session_id]
        cmd += self.extra_args
        cmd += [prompt]
        return cmd

    def _parse(self, stdout: str) -> TurnResult:
        try:
            data = json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise AgentError(f"{self.name}: could not parse claude JSON: {stdout[:300]}") from exc

        if data.get("is_error"):
            raise AgentError(f"{self.name}: claude reported an error: {data.get('result')}")

        return TurnResult(
            text=str(data.get("result", "")).strip(),
            session_id=data.get("session_id"),
            usage=data.get("usage"),
            cost_usd=data.get("total_cost_usd"),
            raw=data,
        )
