"""Two agents review a branch and critique each other until they agree.

This is the whole "PR review with claude + codex" use case, built from the
generic pieces -- nothing here is review-specific except the prompt and the
choice of policy/stop conditions.

    uv run python examples/pr_review.py --base main --head HEAD
"""

from __future__ import annotations

import argparse

from agentloop import ClaudeAgent, CodexAgent, Consensus, DebatePolicy, Orchestrator, git


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="main")
    parser.add_argument("--head", default="HEAD")
    parser.add_argument("--rounds", type=int, default=6)
    args = parser.parse_args()

    patch = git.diff(args.base, args.head)
    if not patch.strip():
        raise SystemExit(f"No diff between {args.base} and {args.head}.")

    task = (
        "Review this pull request for correctness bugs and security issues. "
        "Give a prioritized list of findings citing file and line.\n\n"
        f"```diff\n{patch}\n```"
    )

    # Two reviewers with explicit, divergent lenses so the debate is productive.
    claude = ClaudeAgent(
        "claude",
        system_prompt="You are a meticulous correctness reviewer. You care about edge "
        "cases, error handling, and logic bugs.",
    )
    codex = CodexAgent(
        "codex",
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
