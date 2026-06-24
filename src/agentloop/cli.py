"""Command-line entry point: run the loop without writing any Python.

uv run agentloop debate "Design a rate limiter for our API"
uv run agentloop review --base main --head HEAD
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

from . import git
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


def _run(
    task: str,
    *,
    rounds: int,
    budget: float | None,
    marker: str,
    journal: str | None,
    repo: str | None,
) -> None:
    # cwd is what makes the tool location-independent: git and both agent
    # subprocesses run in the target repo instead of wherever you launched from.
    claude = ClaudeAgent("claude", cwd=repo)
    codex = CodexAgent("codex", sandbox="read-only", cwd=repo)

    stop: list[Consensus | BudgetUSD] = [Consensus(marker)]
    if budget:
        stop.append(BudgetUSD(budget))

    store = JournalStore(journal) if journal else None
    if store and store.exists():
        console.print(f"[dim]resuming from {journal}[/dim]\n")

    policy = DebatePolicy(task)
    loop = Orchestrator(
        [claude, codex],
        policy,
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
def debate(
    task: str = typer.Argument(..., help="The problem the two agents should hash out."),
    rounds: int = typer.Option(8, help="Max agent turns before stopping."),
    budget: float | None = typer.Option(None, help="Stop once total cost (USD) exceeds this."),
    marker: str = typer.Option("AGREED", help="Word that signals consensus."),
    journal: str | None = typer.Option(
        None, help="JSONL file to persist/resume the run. Reuse the same path to resume."
    ),
    repo: str | None = typer.Option(
        None, help="Directory to run the agents in (defaults to the current directory)."
    ),
) -> None:
    """Have claude and codex debate an open-ended problem until they converge."""
    _check_repo(repo)
    _run(task, rounds=rounds, budget=budget, marker=marker, journal=journal, repo=repo)


@app.command()
def review(
    base: str = typer.Option("main", help="Base branch to diff against."),
    head: str = typer.Option("HEAD", help="Branch/ref under review."),
    rounds: int = typer.Option(6, help="Max agent turns before stopping."),
    budget: float | None = typer.Option(None, help="Stop once total cost (USD) exceeds this."),
    journal: str | None = typer.Option(
        None, help="JSONL file to persist/resume the run. Reuse the same path to resume."
    ),
    repo: str | None = typer.Option(
        None, help="Git repo to review (defaults to the current directory)."
    ),
) -> None:
    """Two agents review a branch's diff and reconcile their findings."""
    _check_repo(repo)
    patch = git.diff(base, head, cwd=repo)
    if not patch.strip():
        console.print(f"[yellow]No diff between {base} and {head}.[/yellow]")
        raise typer.Exit(1)

    task = (
        "You are reviewing a pull request. Find correctness bugs, security issues, "
        "and risky changes. Ignore style nits. Output a concise, prioritized list of "
        "findings; cite file and line. Here is the diff:\n\n"
        f"```diff\n{patch}\n```"
    )
    _run(task, rounds=rounds, budget=budget, marker="AGREED", journal=journal, repo=repo)


if __name__ == "__main__":
    app()
