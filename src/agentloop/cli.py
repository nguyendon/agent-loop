"""Command-line entry point.

The loop is task-agnostic: it hands the two agents a freeform prompt and lets
them do the work with their own tools (git, file reads, gh, …). A single command,
so Typer promotes it -- invoke it as `agentloop "<task>"`, no subcommand.

    uv run agentloop "review the uncommitted changes and agree on the top issues"
    uv run agentloop "review PR #42"
    uv run agentloop "design a token-bucket rate limiter" --no-tools
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.rule import Rule
from rich.status import Status

from .agent import Agent
from .domain import Message
from .pipeline import Build, solve
from .report import write_run_report
from .stop import BudgetUSD, Consensus, StopCondition
from .store import JournalStore

app = typer.Typer(add_completion=False, help="Run a multi-agent loop over the claude & codex CLIs.")
console = Console()

_STYLES = {"claude": "bold magenta", "codex": "bold cyan", "user": "dim"}


def _setup_logging(verbosity: int) -> None:
    """-v → INFO, -vv → DEBUG; default stays quiet (warnings only)."""
    level = logging.WARNING if verbosity <= 0 else logging.INFO if verbosity == 1 else logging.DEBUG
    logger = logging.getLogger("agentloop")
    logger.setLevel(level)
    logger.handlers.clear()
    logger.addHandler(
        RichHandler(console=Console(stderr=True), show_path=False, show_time=False, markup=True)
    )


def _print_message(message: Message) -> None:
    style = _STYLES.get(message.author, "white")
    cost = f"  [dim](${message.cost_usd:.4f})[/dim]" if message.cost_usd else ""
    console.print(Rule(f"[{style}]{message.author}[/{style}]{cost}"))
    console.print(message.content)
    console.print()


def _check_repo(repo: str | None) -> None:
    if repo is not None and not Path(repo).is_dir():
        console.print(f"[red]--repo path is not a directory: {repo}[/red]")
        raise typer.Exit(2)


def _run_loop(
    task: str,
    *,
    rounds: int,
    budget: float | None,
    journal: str | None,
    repo: str | None,
    tools: bool,
    num_agents: int,
    triage: bool,
    out_dir: str | None,
    verbose: int,
) -> None:
    stop: list[StopCondition] = [Consensus("AGREED")]
    if budget:
        stop.append(BudgetUSD(budget))

    # Scouts surface only through on_message (they never join the debate
    # transcript), so capture them here for the report's findings/ subdir.
    scouts: list[Message] = []

    store = JournalStore(journal) if journal else None
    if store and store.exists():
        console.print(f"[dim]resuming from {journal}[/dim]\n")

    # A turn blocks while the agent's subprocess runs (often minutes with tools),
    # so without feedback it looks frozen. Show a spinner -- except when logging is
    # on, where the log lines are the play-by-play instead.
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

    console.print(
        f"[dim]claude ⇄ codex · up to {rounds} turns · "
        f"{'tools' if tools else 'no tools'} · ctrl-c to stop[/dim]\n"
    )

    try:
        outcome = solve(
            task,
            build=Build(repo=repo, tools=tools),
            rounds=rounds,
            stop=stop,
            num_agents=num_agents,
            triage=triage,
            store=store,
            on_status=_on_status,
            on_message=_on_message,
            on_turn_start=_on_turn_start,
            on_parallel_start=_on_parallel_start,
        )
    except KeyboardInterrupt:
        _set_spinner(None)
        console.print("\n[yellow]interrupted[/yellow]")
        raise typer.Exit(130) from None
    finally:
        _set_spinner(None)

    result = outcome.loop
    total = result.transcript.total_cost_usd + outcome.extra_cost_usd
    plan = outcome.plan
    shape = f"{len(plan.focuses)} scouts → debate" if plan.discovery else "debate only"

    report_dir: Path | None = None
    if out_dir is not None:
        report_dir = write_run_report(
            out_root=Path(out_dir),
            timestamp=datetime.now().strftime("%Y%m%d-%H%M%S"),
            task=task,
            shape=shape,
            triage_reason=plan.reason,
            stopped_by=result.stopped_by,
            turns=result.turns,
            resumed=result.resumed,
            total_cost=total,
            transcript=result.transcript,
            scouts=scouts,
            focuses=plan.focuses,
        )

    console.print(
        Panel(
            f"shape: [bold]{shape}[/bold]\n"
            f"stopped by: [bold]{result.stopped_by}[/bold]\n"
            f"turns: {result.turns}{' (resumed)' if result.resumed else ''}\n"
            f"total cost: ${total:.4f}"
            + (f"\njournal: {journal}" if journal else "")
            + (f"\nreport: {report_dir}" if report_dir else ""),
            title="done",
        )
    )


@app.command()
def run(
    task: str = typer.Argument(..., help="What you want the two agents to do."),
    rounds: int = typer.Option(8, help="Max agent turns before stopping."),
    budget: float | None = typer.Option(None, help="Stop once total cost (USD) exceeds this."),
    repo: str | None = typer.Option(
        None, help="Directory to run the agents in (defaults to the current directory)."
    ),
    journal: str | None = typer.Option(
        None, help="JSONL file to persist/resume the run. Reuse the same path to resume."
    ),
    no_tools: bool = typer.Option(
        False,
        "--no-tools",
        help="Drop claude's plan-mode tools (pure reasoning). codex keeps its read-only sandbox.",
    ),
    num_agents: int = typer.Option(
        4, "--num-agents", help="Max parallel discovery scouts when triage calls for it."
    ),
    no_triage: bool = typer.Option(
        False, "--no-triage", help="Skip triage and go straight to debate (no discovery)."
    ),
    out_dir: str = typer.Option(
        "out",
        "--out-dir",
        help="Root for timestamped run reports (report.md + transcript/findings).",
    ),
    no_report: bool = typer.Option(
        False, "--no-report", help="Don't write a run report to --out-dir."
    ),
    verbose: int = typer.Option(
        0, "--verbose", "-v", count=True, help="Log progress to stderr (-v info, -vv debug)."
    ),
) -> None:
    """Run the two-agent loop on any task; triage picks discovery depth, then debate."""
    _check_repo(repo)
    _setup_logging(verbose)
    _run_loop(
        task,
        rounds=rounds,
        budget=budget,
        journal=journal,
        repo=repo,
        tools=not no_tools,
        num_agents=num_agents,
        triage=not no_triage,
        out_dir=None if no_report else out_dir,
        verbose=verbose,
    )


if __name__ == "__main__":
    app()
