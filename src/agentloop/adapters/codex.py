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

    def _failure_detail(self, stdout: str, stderr: str) -> str:
        # codex exec exits non-zero on a failed turn but still emits the real
        # cause as an {"type": "error"|"turn.failed"} event on stdout; stderr is
        # just "Reading additional input from stdin...". Prefer the event.
        for line in stdout.splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") in ("error", "turn.failed"):
                return _error_message(event)
        return (stderr or stdout or "").strip()

    def _parse(self, stdout: str) -> TurnResult:
        session_id: str | None = None
        usage: dict[str, object] | None = None
        cost_usd: float | None = None
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
                    cost_usd = _cost_usd(event)

        return TurnResult(
            text="\n".join(parts).strip(),
            session_id=session_id,
            usage=usage,
            cost_usd=cost_usd,
            raw=stdout,
        )


def _error_message(event: dict[str, object]) -> str:
    """The human-readable message from a codex error/turn.failed event.

    The ``message`` field is often itself a JSON string like
    ``{"error": {"message": "The 'gpt-5.3-codex' model is not supported..."}}``;
    unwrap one level to surface just the sentence.
    """
    raw = event.get("message")
    error = event.get("error")
    if isinstance(error, dict):
        nested = error.get("message")
        if isinstance(nested, str):
            raw = nested
    if not isinstance(raw, str):
        return str(event)
    try:
        inner = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw
    if isinstance(inner, dict):
        nested_error = inner.get("error")
        if isinstance(nested_error, dict):
            message = nested_error.get("message")
            if isinstance(message, str):
                return message
        message = inner.get("message")
        if isinstance(message, str):
            return message
    return raw


def _cost_usd(event: dict[str, object]) -> float | None:
    for candidate in (event.get("total_cost_usd"), event.get("cost_usd")):
        if isinstance(candidate, int | float):
            return float(candidate)
    usage = event.get("usage")
    if isinstance(usage, dict):
        for key in ("total_cost_usd", "cost_usd"):
            candidate = usage.get(key)
            if isinstance(candidate, int | float):
                return float(candidate)
    return None
