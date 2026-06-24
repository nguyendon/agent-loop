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
from pathlib import Path

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.rule import Rule
from rich.status import Status

from .adapters.claude import ClaudeAgent
from .adapters.codex import CodexAgent
from .agent import Agent
from .domain import Message
from .orchestrator import Orchestrator
from .policy import DebatePolicy
from .stop import BudgetUSD, Consensus
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
    verbose: int,
) -> None:
    # cwd makes the tool location-independent: git and both agents run in the
    # target repo. With tools, claude gets plan mode (read-only tools, no edits);
    # without, claude runs in default mode (no plan-mode tools). codex always uses
    # its read-only sandbox -- exec has no prompt-only mode -- so `tools` is a
    # claude-only lever, not a hard sandbox guarantee for the pair.
    claude = ClaudeAgent("claude", cwd=repo, permission_mode="plan" if tools else None)
    codex = CodexAgent("codex", cwd=repo, sandbox="read-only")

    stop: list[Consensus | BudgetUSD] = [Consensus("AGREED")]
    if budget:
        stop.append(BudgetUSD(budget))

    store = JournalStore(journal) if journal else None
    if store and store.exists():
        console.print(f"[dim]resuming from {journal}[/dim]\n")

    # A turn blocks while the agent's subprocess runs (often minutes with tools),
    # so without feedback it looks frozen. Show a spinner per turn -- except when
    # logging is on, where the log lines are the play-by-play instead.
    spinner_on = verbose <= 0
    spinner: Status | None = None

    def _stop_spinner() -> None:
        nonlocal spinner
        if spinner is not None:
            spinner.stop()
            spinner = None

    def _on_turn_start(agent: Agent, turn: int) -> None:
        nonlocal spinner
        if spinner_on:
            style = _STYLES.get(agent.name, "white")
            spinner = console.status(
                f"[{style}]{agent.name}[/{style}] thinking… (turn {turn + 1})", spinner="dots"
            )
            spinner.start()

    def _on_parallel_start(agents: Sequence[Agent]) -> None:
        nonlocal spinner
        if spinner_on:
            names = ", ".join(a.name for a in agents)
            spinner = console.status(f"[bold]{names}[/bold] thinking in parallel…", spinner="dots")
            spinner.start()

    def _on_message(message: Message) -> None:
        _stop_spinner()
        _print_message(message)

    console.print(
        f"[dim]claude ⇄ codex · up to {rounds} turns · "
        f"{'tools' if tools else 'no tools'} · ctrl-c to stop[/dim]\n"
    )

    loop = Orchestrator(
        [claude, codex],
        DebatePolicy(task),
        stop=stop,
        max_rounds=rounds,
        on_message=_on_message,
        on_turn_start=_on_turn_start,
        on_parallel_start=_on_parallel_start,
        store=store,
    )
    try:
        result = loop.run()
    except KeyboardInterrupt:
        _stop_spinner()
        console.print("\n[yellow]interrupted[/yellow]")
        raise typer.Exit(130) from None
    finally:
        _stop_spinner()

    console.print(
        Panel(
            f"stopped by: [bold]{result.stopped_by}[/bold]\n"
            f"turns: {result.turns}{' (resumed)' if result.resumed else ''}\n"
            f"total cost: ${result.transcript.total_cost_usd:.4f}"
            + (f"\njournal: {journal}" if journal else ""),
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
    verbose: int = typer.Option(
        0, "--verbose", "-v", count=True, help="Log progress to stderr (-v info, -vv debug)."
    ),
) -> None:
    """Run the two-agent loop on any task; the agents do the work themselves."""
    _check_repo(repo)
    _setup_logging(verbose)
    _run_loop(
        task,
        rounds=rounds,
        budget=budget,
        journal=journal,
        repo=repo,
        tools=not no_tools,
        verbose=verbose,
    )


if __name__ == "__main__":
    app()
