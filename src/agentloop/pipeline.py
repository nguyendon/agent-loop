"""Adaptive pipeline: triage → (optional) parallel discovery → serial debate.

A cheap triage turn decides how to approach a task. Simple questions skip
discovery and go straight to the debate; open-ended work (code review, audits,
design) fans out up to ``num_agents`` independent scouts, pools their findings,
and seeds the debate with them. The triage agent picks the angles; the engine
runs the phases.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from .adapters.claude import ClaudeAgent
from .adapters.codex import CodexAgent
from .agent import Agent
from .domain import Message
from .orchestrator import LoopResult, Orchestrator, fan_out
from .policy import DebatePolicy
from .stop import StopCondition
from .store import Store

log = logging.getLogger("agentloop.pipeline")

_TRIAGE_PROMPT = (
    "You are triaging a task to decide how a two-agent debate system should approach it.\n"
    "Guidance:\n"
    "- Simple factual or opinion questions need NO discovery -> discovery=false.\n"
    "- Open-ended or investigative tasks (code review, audits, design, debugging) benefit\n"
    "  from parallel discovery first -> discovery=true, with 2-{max} INDEPENDENT angles\n"
    "  (e.g. for a code review: correctness, security, performance, tests).\n"
    "Reply with ONLY a JSON object, no prose or code fences:\n"
    '{{"discovery": true|false, "focuses": ["angle 1", "angle 2"], "reason": "one sentence"}}\n'
    "focuses is [] when discovery is false; otherwise one short phrase per parallel agent, "
    "at most {max}.\n\nTASK:\n{task}"
)


@dataclass(slots=True)
class Plan:
    discovery: bool
    focuses: list[str]
    reason: str


@dataclass(slots=True)
class SolveResult:
    loop: LoopResult
    plan: Plan
    extra_cost_usd: float  # triage + discovery (the debate cost is in loop.transcript)


@dataclass(slots=True)
class Build:
    """Constructs agents with consistent cwd / tool settings for every phase."""

    repo: str | None = None
    tools: bool = True

    def claude(self, name: str = "claude", system_prompt: str | None = None) -> ClaudeAgent:
        return ClaudeAgent(
            name,
            cwd=self.repo,
            permission_mode="plan" if self.tools else None,
            system_prompt=system_prompt,
        )

    def codex(self, name: str = "codex", system_prompt: str | None = None) -> CodexAgent:
        return CodexAgent(name, cwd=self.repo, sandbox="read-only", system_prompt=system_prompt)

    def scout(self, index: int, focus: str) -> Agent:
        # Alternate model families for diversity; the lens comes from the focus.
        system_prompt = f"You are an investigator. Focus specifically on: {focus}."
        name = f"scout{index + 1}"
        return self.codex(name, system_prompt) if index % 2 else self.claude(name, system_prompt)


def parse_plan(raw: str, *, max_agents: int) -> Plan:
    """Parse the triage agent's JSON reply, defensively (any failure → no discovery)."""
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match is None:
        log.warning("triage: no JSON in reply, defaulting to no discovery")
        return Plan(discovery=False, focuses=[], reason="triage parse failed")
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        log.warning("triage: invalid JSON, defaulting to no discovery")
        return Plan(discovery=False, focuses=[], reason="triage parse failed")

    focuses = [str(f).strip() for f in data.get("focuses", []) if str(f).strip()][:max_agents]
    discovery = bool(data.get("discovery")) and len(focuses) > 0
    return Plan(discovery=discovery, focuses=focuses, reason=str(data.get("reason", "")))


def _noop(*_: object) -> None:
    pass


def solve(
    task: str,
    *,
    build: Build,
    rounds: int,
    stop: list[StopCondition],
    num_agents: int = 4,
    triage: bool = True,
    store: Store | None = None,
    on_status: Callable[[str], None] = _noop,
    on_message: Callable[[Message], None] = _noop,
    on_turn_start: Callable[[Agent, int], None] = _noop,
    on_parallel_start: Callable[[Sequence[Agent]], None] = _noop,
) -> SolveResult:
    plan = Plan(discovery=False, focuses=[], reason="triage disabled")
    extra_cost = 0.0

    if triage:
        on_status("triaging task…")
        result = build.claude("triage").send(_TRIAGE_PROMPT.format(max=num_agents, task=task))
        extra_cost += result.cost_usd or 0.0
        plan = parse_plan(result.text, max_agents=num_agents)
        log.info(
            "triage: discovery=%s, %d scouts (%s)", plan.discovery, len(plan.focuses), plan.reason
        )

    seed = task
    if plan.discovery:
        scouts = [build.scout(i, focus) for i, focus in enumerate(plan.focuses)]
        prompts = [
            f"{task}\n\nInvestigate independently. Focus on: {focus}. List concrete, specific "
            f"findings; cite file:line where relevant."
            for focus in plan.focuses
        ]
        on_parallel_start(scouts)
        pooled: list[str] = []
        for scout, focus, result in zip(
            scouts, plan.focuses, fan_out(scouts, prompts), strict=True
        ):
            extra_cost += result.cost_usd or 0.0
            on_message(Message(scout.name, result.text, cost_usd=result.cost_usd))
            pooled.append(f"## {focus} (via {scout.name})\n{result.text}")
        seed = (
            f"{task}\n\nA discovery panel investigated this and surfaced the following candidate "
            f"findings (some may be wrong or redundant):\n\n" + "\n\n".join(pooled) + "\n\n"
            "Debate these, discard false positives, and converge on an agreed, prioritized set."
        )

    debaters: list[Agent] = [build.claude(), build.codex()]
    loop = Orchestrator(
        debaters,
        DebatePolicy(seed),
        stop=stop,
        max_rounds=rounds,
        on_message=on_message,
        on_turn_start=on_turn_start,
        on_parallel_start=on_parallel_start,
        store=store,
    )
    return SolveResult(loop=loop.run(), plan=plan, extra_cost_usd=extra_cost)
