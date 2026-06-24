"""Command-line entry point.

The loop is task-agnostic: it hands the two agents a freeform prompt and lets
them do the work with their own tools (git, file reads, gh, …). A single command,
so Typer promotes it -- invoke it as `agentloop "<task>"`, no subcommand.

    uv run agentloop "review the uncommitted changes and agree on the top issues"
    uv run agentloop "review PR #42"
    uv run agentloop "design a token-bucket rate limiter" --no-tools
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

from .adapters.claude import ClaudeAgent
from .adapters.codex import CodexAgent
from .domain import Message
from .orchestrator import Orchestrator
from .policy import DebatePolicy
from .stop import BudgetUSD, Consensus
from .store import JournalStore

app = typer.Typer(add_completion=False, help="Run a multi-agent loop over the claude & codex CLIs.")
console = Console()

_STYLES = {"claude": "bold magenta", "codex": "bold cyan", "user": "dim"}


def _printer() -> Callable[[Message], None]:
    def show(message: Message) -> None:
        style = _STYLES.get(message.author, "white")
        cost = f"  [dim](${message.cost_usd:.4f})[/dim]" if message.cost_usd else ""
        console.print(Rule(f"[{style}]{message.author}[/{style}]{cost}"))
        console.print(message.content)
        console.print()

    return show


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

    loop = Orchestrator(
        [claude, codex],
        DebatePolicy(task),
        stop=stop,
        max_rounds=rounds,
        on_message=_printer(),
        store=store,
    )
    result = loop.run()

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
) -> None:
    """Run the two-agent loop on any task; the agents do the work themselves."""
    _check_repo(repo)
    _run_loop(task, rounds=rounds, budget=budget, journal=journal, repo=repo, tools=not no_tools)


if __name__ == "__main__":
    app()
