"""Command-line entry point.

The loop is task-agnostic: it hands the agents a freeform prompt and lets them
do the work with their own tools (git, file reads, gh, …). A single command, so
Typer promotes it -- invoke it as `agentloop "<task>"`, no subcommand.

Read-only by default; writing to the repo is always an explicit opt-in:

    uv run agentloop "review the uncommitted changes and agree on the top issues"
    uv run agentloop "fix the flaky test in test_orchestrator.py" --write
    uv run agentloop --resume out/20260624-120000-fix-the-flaky-test --write
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.rule import Rule
from rich.status import Status

from .agent import Agent, AgentError
from .domain import Message
from .pipeline import Build, solve
from .report import slug, write_run_report
from .stop import BudgetUSD, Consensus, StopCondition
from .store import FixJournal, JournalStore

app = typer.Typer(add_completion=False, help="Run a multi-agent loop over the claude & codex CLIs.")
console = Console()

_STYLES = {
    "claude": "bold magenta",
    "codex": "bold cyan",
    "reviewer-claude": "bold magenta",
    "reviewer-codex": "bold cyan",
    "implementer": "bold green",
    "user": "dim",
}

_ROUNDS = 8  # max debate turns
_NUM_AGENTS = 4  # max discovery scouts (triage picks 2..N)
_BUDGET_USD = 10.0  # runaway-cost backstop (no flag; tunable here)


def _setup_logging(verbosity: int) -> None:
    """-v → INFO, -vv → DEBUG; default stays quiet (warnings only)."""
    level = logging.WARNING if verbosity <= 0 else logging.INFO if verbosity == 1 else logging.DEBUG
    logger = logging.getLogger("agentloop")
    logger.setLevel(level)
    logger.handlers.clear()
    logger.addHandler(
        RichHandler(console=Console(stderr=True), show_path=False, show_time=False, markup=True)
    )


def _fmt_elapsed(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes}m {secs:02d}s" if minutes else f"{secs}s"


def _print_message(message: Message) -> None:
    style = _STYLES.get(message.author, "white")
    cost = f"  [dim](${message.cost_usd:.4f})[/dim]" if message.cost_usd else ""
    console.print(Rule(f"[{style}]{message.author}[/{style}]{cost}"))
    console.print(message.content)
    console.print()


def _write_meta(run_dir: Path, task: str, when: str) -> None:
    (run_dir / "meta.json").write_text(json.dumps({"task": task, "when": when}), encoding="utf-8")


def _read_meta(run_dir: Path) -> dict[str, str]:
    path = run_dir / "meta.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _read_plan(run_dir: Path) -> str | None:
    """The agreed plan body from a prior run's plan.md (header stripped)."""
    path = run_dir / "plan.md"
    if not path.exists():
        return None
    lines = path.read_text(encoding="utf-8").splitlines()
    start = next((i + 1 for i, ln in enumerate(lines) if ln.startswith("_for:")), 0)
    return "\n".join(lines[start:]).strip() or None


def _run_loop(
    task: str | None, *, write: bool, resume: str | None, repo: str | None, verbose: int
) -> None:
    # --- resolve the run directory, task, and any prior agreed plan ----------
    resumed_plan: str | None = None
    if resume is not None:
        run_dir = Path(resume)
        if not run_dir.is_dir():
            console.print(f"[red]--resume path is not a directory: {resume}[/red]")
            raise typer.Exit(2)
        meta = _read_meta(run_dir)
        task = task or meta.get("task")
        when = meta.get("when", "")
        # A finished stage 1 leaves plan.md; with --write that's the handoff.
        # Otherwise we fall through and resume the stage-1 debate from the journal.
        resumed_plan = _read_plan(run_dir) if write else None
        console.print(f"[dim]resuming {run_dir}[/dim]\n")
    else:
        when = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = Path("out") / f"{when}-{slug(task or '')}"
        run_dir.mkdir(parents=True, exist_ok=True)

    if not task:
        console.print("[red]no task: pass one, or --resume a run dir that has meta.json[/red]")
        raise typer.Exit(2)
    if resume is None:
        _write_meta(run_dir, task, when)

    store = JournalStore(run_dir / "journal.jsonl")
    fix_store = FixJournal(run_dir / "fix.journal.jsonl")
    stop: list[StopCondition] = [Consensus("AGREED"), BudgetUSD(_BUDGET_USD)]
    scouts: list[Message] = []

    # A turn blocks while the agent's subprocess runs (often minutes), so without
    # feedback it looks frozen. Show a spinner -- except when logging is on, where
    # the log lines are the play-by-play instead.
    spinner_on = verbose <= 0
    spinner: Status | None = None

    def _set_spinner(text: str | None) -> None:
        nonlocal spinner
        if spinner is not None:
            spinner.stop()
            spinner = None
        if text is not None and spinner_on:
            spinner = console.status(text, spinner="dots")
            spinner.start()

    def _on_status(text: str) -> None:
        _set_spinner(f"[dim]{text}[/dim]")

    def _on_turn_start(agent: Agent, turn: int) -> None:
        style = _STYLES.get(agent.name, "white")
        _set_spinner(f"[{style}]{agent.name}[/{style}] thinking… (turn {turn + 1})")

    def _on_parallel_start(agents: Sequence[Agent]) -> None:
        names = ", ".join(a.name for a in agents)
        _set_spinner(f"[bold]{names}[/bold] thinking in parallel…")

    def _on_message(message: Message) -> None:
        _set_spinner(None)
        if message.author.startswith("scout"):
            scouts.append(message)
        _print_message(message)

    mode = "write" if write else "read-only"
    # Set expectations up front: each turn is a whole agent subprocess, so the
    # spinner can sit on one phase for minutes -- that's normal, not a hang.
    eta = "expect ~10-25 min" if write else "expect ~2-10 min"
    console.print(
        f"[dim]claude ⇄ codex · {mode} · up to {_ROUNDS} turns · {eta} · ctrl-c to stop[/dim]"
    )
    console.print(
        "[dim]each turn runs a full agent with tools; long pauses are normal, not a hang.[/dim]\n"
    )

    start = time.monotonic()
    try:
        outcome = solve(
            task,
            build=Build(repo=repo),
            rounds=_ROUNDS,
            stop=stop,
            num_agents=_NUM_AGENTS,
            write=write,
            resumed_plan=resumed_plan,
            store=store,
            fix_store=fix_store,
            on_status=_on_status,
            on_message=_on_message,
            on_turn_start=_on_turn_start,
            on_parallel_start=_on_parallel_start,
        )
    except KeyboardInterrupt:
        _set_spinner(None)
        console.print("\n[yellow]interrupted[/yellow]")
        raise typer.Exit(130) from None
    except AgentError as exc:
        _set_spinner(None)
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from None
    finally:
        _set_spinner(None)

    plan = outcome.plan
    loop = outcome.loop
    fix = outcome.fix
    total = (loop.transcript.total_cost_usd if loop else 0.0) + outcome.extra_cost_usd

    if resumed_plan is not None:
        base_shape = "resumed plan"
    elif plan.discovery:
        base_shape = f"{len(plan.focuses)} scouts → debate"
    else:
        base_shape = "debate only"
    shape = base_shape + (" → fix" if fix is not None else "")

    if fix is not None:
        stopped_by = "APPROVED" if fix.approved else "fix incomplete"
        turns = fix.attempts
    elif loop is not None:
        stopped_by, turns = loop.stopped_by, loop.turns
    else:
        stopped_by, turns = "n/a", 0

    write_run_report(
        run_dir,
        task=task,
        when=when or run_dir.name,
        shape=shape,
        triage_reason=plan.reason,
        stopped_by=stopped_by,
        turns=turns,
        resumed=resume is not None,
        total_cost=total,
        plan_text=outcome.plan_text,
        transcript=loop.transcript if loop else None,
        scouts=scouts,
        focuses=plan.focuses,
        fix=fix,
    )

    verdict = ""
    if fix is not None:
        verdict = (
            f"\nverdict: [bold]{'APPROVED' if fix.approved else 'changes still needed'}[/bold]"
        )
    elapsed = time.monotonic() - start
    console.print(
        Panel(
            f"shape: [bold]{shape}[/bold]\n"
            f"stopped by: [bold]{stopped_by}[/bold]\n"
            f"turns: {turns}{verdict}\n"
            f"elapsed: {_fmt_elapsed(elapsed)}\n"
            f"total cost: ${total:.4f}\n"
            f"report: {run_dir}",
            title="done",
        )
    )

    # The fix loop edits the tree but deliberately never commits -- committing is
    # left outside the loop, to a human or to a follow-up agent handed this run.
    if fix is not None:
        console.print(
            f"[dim]changes are left uncommitted by design. review and commit them yourself, "
            f"or hand this run to an agent: “make clean commits for the changes from "
            f"{run_dir}”.[/dim]"
        )


@app.command()
def run(
    task: str | None = typer.Argument(None, help="What you want the agents to do."),
    write: bool = typer.Option(
        False, "--write", help="Allow the agents to modify the repo: plan → implement → review."
    ),
    resume: str | None = typer.Option(
        None,
        "--resume",
        help="Continue a prior run directory (e.g. out/<run>); add --write to fix.",
    ),
    repo: str | None = typer.Option(
        None, "--repo", help="Directory to run the agents in (defaults to the current directory)."
    ),
    verbose: int = typer.Option(
        0, "--verbose", "-v", count=True, help="Log progress to stderr (-v info, -vv debug)."
    ),
) -> None:
    """Triage → discovery → debate to an agreed plan; --write crosses the gate into the fix loop."""
    if repo is not None and not Path(repo).is_dir():
        console.print(f"[red]--repo path is not a directory: {repo}[/red]")
        raise typer.Exit(2)
    _setup_logging(verbose)
    _run_loop(task, write=write, resume=resume, repo=repo, verbose=verbose)


if __name__ == "__main__":
    app()
