"""Adaptive pipeline: triage → (optional) discovery → debate, with a write-gate.

A cheap triage turn decides how to approach a task. Simple questions skip
discovery and go straight to the debate; open-ended work (code review, audits,
design) fans out up to ``num_agents`` independent scouts, pools their findings,
and seeds the debate with them. The debate converges on an agreed plan.

That agreed plan is a gate. Read-only by default: the run stops there with the
plan saved. With ``write=True`` the pipeline crosses the gate into the fix loop
-- a single write-capable agent implements the plan and self-verifies, then the
two agents review the resulting diff and converge on APPROVED. ``plan_text``
lets a later run skip straight to the fix loop from a previously agreed plan
(the review-first handoff).
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from .adapters.claude import ClaudeAgent
from .adapters.codex import CodexAgent
from .agent import Agent, AgentError
from .domain import Message, Transcript
from .orchestrator import LoopResult, Orchestrator, fan_out
from .policy import DebatePolicy
from .stop import Consensus, StopCondition
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

# Stage-2 review instructions: same machinery as a debate, but the consensus
# marker is APPROVED -- reviewers must inspect the real diff, not just agree.
_REVIEW_OPENING = (
    "An implementer just applied changes to this repository to carry out the agreed plan "
    "below. Inspect the ACTUAL changes -- run `git diff` (and read the touched files) -- "
    "and judge whether they correctly and completely implement the plan with no bugs or "
    "regressions. If they do, begin your reply with the single word APPROVED. Otherwise do "
    "NOT write APPROVED; list the specific changes still required."
)
_REVIEW_REBUTTAL = (
    "Consider the other reviewer's assessment and re-examine the diff with `git diff`. If "
    "you now agree the changes are correct and complete, begin your reply with APPROVED. "
    "Otherwise list the remaining required changes."
)

_MAX_FIX_ATTEMPTS = 3  # implement→review rounds before giving up
_REVIEW_ROUNDS = 4  # max reviewer turns per attempt


@dataclass(slots=True)
class Plan:
    discovery: bool
    focuses: list[str]
    reason: str


@dataclass(slots=True)
class FixResult:
    """The stage-2 outcome: every implement + review turn, and the verdict."""

    transcript: Transcript
    approved: bool
    attempts: int
    cost_usd: float


@dataclass(slots=True)
class SolveResult:
    plan: Plan
    plan_text: str  # the agreed plan (stage-1 output, or a resumed plan)
    extra_cost_usd: float  # triage + discovery + fix (the debate cost is in loop.transcript)
    loop: LoopResult | None = None  # None on a resume-handoff (stage 1 already ran)
    fix: FixResult | None = None  # None unless write=True


@dataclass(slots=True)
class Build:
    """Constructs agents with a consistent cwd for every phase.

    Debate and review agents are read-only (claude plan mode, codex read-only);
    only ``implementer`` may write, and it's used as a single writer past the
    gate so there's no concurrent-edit hazard.
    """

    repo: str | None = None
    # Per-CLI model overrides, scoped to agentloop -- set via env so the user's
    # global codex/claude config is untouched. codex exec's default model isn't
    # always allowed on every account (e.g. ChatGPT-auth rejects gpt-*-codex),
    # so this is how you point it at a supported one (e.g. gpt-5.5).
    claude_model: str | None = field(default_factory=lambda: os.getenv("AGENTLOOP_CLAUDE_MODEL"))
    codex_model: str | None = field(default_factory=lambda: os.getenv("AGENTLOOP_CODEX_MODEL"))

    # Factories return the Agent abstraction (callers never need the concrete
    # CLI type); this also lets tests substitute a Build with fake agents.
    def claude(self, name: str = "claude", system_prompt: str | None = None) -> Agent:
        return ClaudeAgent(
            name,
            model=self.claude_model,
            cwd=self.repo,
            permission_mode="plan",
            system_prompt=system_prompt,
        )

    def codex(self, name: str = "codex", system_prompt: str | None = None) -> Agent:
        return CodexAgent(
            name,
            model=self.codex_model,
            cwd=self.repo,
            sandbox="read-only",
            system_prompt=system_prompt,
        )

    def implementer(self, name: str = "implementer") -> Agent:
        # codex workspace-write can edit files AND run tests inside its sandbox
        # (no network/escape) -- more contained than claude headless write.
        return CodexAgent(name, model=self.codex_model, cwd=self.repo, sandbox="workspace-write")

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


def agreed_plan_text(transcript: Transcript) -> str:
    """The substantive converged plan from a debate.

    The final turn is often a bare "AGREED", so walk back to the last turn whose
    content (minus a leading AGREED marker) is non-trivial; fall back to the last
    turn if none qualifies.
    """
    messages = transcript.agent_messages
    for message in reversed(messages):
        stripped = re.sub(r"^AGREED[\s:,.\-]*", "", message.content.strip(), flags=re.IGNORECASE)
        if len(stripped.strip()) >= 40:
            return stripped.strip()
    return messages[-1].content.strip() if messages else ""


def _implement_prompt(task: str, plan_text: str, feedback: str) -> str:
    prompt = (
        "You are implementing an agreed plan in this repository. Make the changes directly "
        "and keep them precise and minimal. Then verify your own work: run the project's "
        "tests or build (use whatever the repo provides) and confirm it passes.\n\n"
        f"--- TASK ---\n{task}\n\n--- AGREED PLAN ---\n{plan_text}\n"
    )
    if feedback:
        prompt += f"\n--- REVIEWERS REQUESTED THESE CHANGES (address them) ---\n{feedback}\n"
    prompt += (
        "\nReport what you changed (files touched + a short summary) and the exact "
        "verification commands you ran and their outcome."
    )
    return prompt


def _review_seed(task: str, plan_text: str, change_report: str) -> str:
    return (
        f"{task}\n\n--- AGREED PLAN ---\n{plan_text}\n\n"
        f"--- IMPLEMENTER'S REPORT ---\n{change_report}"
    )


def _requested_changes(transcript: Transcript) -> str:
    """The reviewers' latest feedback, fed back into the next implement pass."""
    by_author: dict[str, Message] = {}
    for message in transcript.agent_messages:
        by_author[message.author] = message
    return "\n\n".join(f"{m.author}: {m.content.strip()}" for m in by_author.values())


def _noop(*_: object) -> None:
    pass


_PREFLIGHT_PROMPT = "Reply with the single word: ok"
_MODEL_HINT = (
    "\nIf an agent rejects its model, set AGENTLOOP_CODEX_MODEL / "
    "AGENTLOOP_CLAUDE_MODEL to one your account supports (e.g. gpt-5.5)."
)


def preflight(build: Build, *, on_status: Callable[[str], None] = _noop) -> None:
    """Cheap health check before the real work: ping each CLI in parallel and
    fail fast with the actual error (bad model, not logged in, missing binary)
    instead of dying minutes later mid-debate."""
    on_status("checking agents…")
    # The AgentError already carries the agent's name, so a flat list is enough.
    agents = [build.claude("preflight"), build.codex("preflight")]
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=len(agents)) as pool:
        for future in [pool.submit(a.send, _PREFLIGHT_PROMPT) for a in agents]:
            try:
                future.result()
            except AgentError as exc:
                errors.append(str(exc))
    if errors:
        raise AgentError("preflight failed:\n  " + "\n  ".join(errors) + _MODEL_HINT)


def fix(
    task: str,
    plan_text: str,
    *,
    build: Build,
    max_attempts: int = _MAX_FIX_ATTEMPTS,
    review_rounds: int = _REVIEW_ROUNDS,
    on_message: Callable[[Message], None] = _noop,
    on_turn_start: Callable[[Agent, int], None] = _noop,
    on_parallel_start: Callable[[Sequence[Agent]], None] = _noop,
) -> FixResult:
    """Implement the plan with a single write agent, then review the diff; loop
    until the reviewers converge on APPROVED or ``max_attempts`` is hit."""
    transcript = Transcript()
    approved = False
    cost = 0.0
    feedback = ""
    attempt = 0

    while attempt < max_attempts:
        attempt += 1
        implementer = build.implementer()
        on_turn_start(implementer, 0)
        result = implementer.send(_implement_prompt(task, plan_text, feedback))
        cost += result.cost_usd or 0.0
        change = transcript.add(Message(implementer.name, result.text, cost_usd=result.cost_usd))
        on_message(change)

        # A fresh read-only debate over the actual diff; consensus on APPROVED ends it.
        reviewers = [build.claude("reviewer-claude"), build.codex("reviewer-codex")]
        review = Orchestrator(
            reviewers,
            DebatePolicy(
                _review_seed(task, plan_text, result.text),
                opening_instructions=_REVIEW_OPENING,
                rebuttal_instructions=_REVIEW_REBUTTAL,
            ),
            stop=[Consensus("APPROVED")],
            max_rounds=review_rounds,
            on_message=on_message,
            on_turn_start=on_turn_start,
            on_parallel_start=on_parallel_start,
        ).run()
        for message in review.transcript.agent_messages:
            transcript.add(message)
        cost += review.transcript.total_cost_usd

        if review.stopped_by == "Consensus":
            approved = True
            break
        feedback = _requested_changes(review.transcript)
        log.info("fix attempt %d not approved; looping with reviewer feedback", attempt)

    return FixResult(transcript=transcript, approved=approved, attempts=attempt, cost_usd=cost)


def solve(
    task: str,
    *,
    build: Build,
    rounds: int,
    stop: list[StopCondition],
    num_agents: int = 4,
    triage: bool = True,
    write: bool = False,
    resumed_plan: str | None = None,
    store: Store | None = None,
    on_status: Callable[[str], None] = _noop,
    on_message: Callable[[Message], None] = _noop,
    on_turn_start: Callable[[Agent, int], None] = _noop,
    on_parallel_start: Callable[[Sequence[Agent]], None] = _noop,
) -> SolveResult:
    plan = Plan(discovery=False, focuses=[], reason="resumed plan" if resumed_plan else "")
    extra_cost = 0.0
    loop_result: LoopResult | None = None
    plan_text = resumed_plan or ""

    preflight(build, on_status=on_status)

    # ---- stage 1: produce the plan (skipped on a resume-handoff) ----
    if resumed_plan is None:
        if triage:
            on_status("triaging task…")
            result = build.claude("triage").send(_TRIAGE_PROMPT.format(max=num_agents, task=task))
            extra_cost += result.cost_usd or 0.0
            plan = parse_plan(result.text, max_agents=num_agents)
            log.info(
                "triage: discovery=%s, %d scouts (%s)",
                plan.discovery,
                len(plan.focuses),
                plan.reason,
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
                f"{task}\n\nA discovery panel investigated this and surfaced the following "
                f"candidate findings (some may be wrong or redundant):\n\n"
                + "\n\n".join(pooled)
                + "\n\nDebate these, discard false positives, and converge on an agreed, "
                "prioritized set."
            )

        debaters: list[Agent] = [build.claude(), build.codex()]
        loop_result = Orchestrator(
            debaters,
            DebatePolicy(seed),
            stop=stop,
            max_rounds=rounds,
            on_message=on_message,
            on_turn_start=on_turn_start,
            on_parallel_start=on_parallel_start,
            store=store,
        ).run()
        plan_text = agreed_plan_text(loop_result.transcript)

    # ---- gate: cross into the fix loop only when asked ----
    fix_result: FixResult | None = None
    if write and plan_text:
        on_status("implementing the agreed plan…")
        fix_result = fix(
            task,
            plan_text,
            build=build,
            on_message=on_message,
            on_turn_start=on_turn_start,
            on_parallel_start=on_parallel_start,
        )
        extra_cost += fix_result.cost_usd

    return SolveResult(
        plan=plan,
        plan_text=plan_text,
        extra_cost_usd=extra_cost,
        loop=loop_result,
        fix=fix_result,
    )
