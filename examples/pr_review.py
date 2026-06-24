"""Two agents review the current repo and critique each other until they agree.

Prompt-driven: the agents inspect the changes themselves with their own tools
(git, file reads) -- nothing here fetches a diff. The only review-specific bits
are the prompt and the reviewers' system prompts; the loop is generic.

    uv run python examples/pr_review.py --repo /path/to/repo
"""

from __future__ import annotations

import argparse

from agentloop import ClaudeAgent, CodexAgent, Consensus, DebatePolicy, Orchestrator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=None, help="Repo to review (defaults to cwd).")
    parser.add_argument("--rounds", type=int, default=6)
    args = parser.parse_args()

    task = (
        "Review the current changes in this git repository. Inspect them with your "
        "tools (`git status`, `git diff HEAD`, `git show HEAD`) and read related "
        "files to ground your findings. Give a prioritized list of correctness and "
        "security findings citing file and line, then converge on an agreed set."
    )

    # Read-only tool access so each agent inspects the repo itself, plus divergent
    # lenses so the debate is productive.
    claude = ClaudeAgent(
        "claude",
        cwd=args.repo,
        permission_mode="plan",  # read-only tools, no edits
        system_prompt="You are a meticulous correctness reviewer. You care about edge "
        "cases, error handling, and logic bugs.",
    )
    codex = CodexAgent(
        "codex",
        cwd=args.repo,
        sandbox="read-only",
        system_prompt="You are a security-focused reviewer. You care about injection, "
        "auth, secrets, and unsafe input handling.",
    )

    loop = Orchestrator(
        [claude, codex],
        DebatePolicy(task),
        stop=[Consensus("AGREED")],
        max_rounds=args.rounds,
        on_message=lambda m: print(f"\n===== {m.author} =====\n{m.content}"),
    )
    result = loop.run()
    print(
        f"\n[stopped by {result.stopped_by} after {result.turns} turns; "
        f"${result.transcript.total_cost_usd:.4f}]"
    )


if __name__ == "__main__":
    main()
