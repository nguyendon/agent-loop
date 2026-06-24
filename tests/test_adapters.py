"""Adapter tests against the CLIs' real output shapes (no subprocess)."""

from __future__ import annotations

import json

from agentloop.adapters.codex import CodexAgent


def test_codex_failure_detail_surfaces_event_not_stdin_noise() -> None:
    # codex exits non-zero but reports the real cause as an error event on
    # stdout; stderr is just the "Reading additional input from stdin..." line.
    inner = json.dumps(
        {
            "type": "error",
            "status": 400,
            "error": {
                "message": "The 'gpt-5.3-codex' model is not supported with a ChatGPT account."
            },
        }
    )
    stdout = "\n".join(
        [
            '{"type":"thread.started","thread_id":"x"}',
            '{"type":"turn.started"}',
            json.dumps({"type": "error", "message": inner}),
        ]
    )

    detail = CodexAgent("codex")._failure_detail(stdout, "Reading additional input from stdin...")

    assert "not supported" in detail
    assert "stdin" not in detail


def test_codex_failure_detail_falls_back_to_stderr() -> None:
    # No structured error event → fall back to stderr rather than swallow it.
    detail = CodexAgent("codex")._failure_detail("not json at all", "boom: command not found")
    assert detail == "boom: command not found"


def test_codex_parse_captures_cost_from_turn_completed() -> None:
    stdout = "\n".join(
        [
            '{"type":"thread.started","thread_id":"thread-123"}',
            '{"type":"item.completed","item":{"type":"agent_message","text":"done"}}',
            '{"type":"turn.completed","usage":{"input_tokens":12},"total_cost_usd":0.125}',
        ]
    )

    result = CodexAgent("codex")._parse(stdout)

    assert result.session_id == "thread-123"
    assert result.text == "done"
    assert result.cost_usd == 0.125
