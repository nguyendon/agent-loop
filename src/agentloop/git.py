"""Small git helpers used to seed review tasks."""

from __future__ import annotations

import subprocess
from pathlib import Path


def diff(base: str = "main", head: str = "HEAD", *, cwd: str | Path | None = None) -> str:
    """Return the diff of ``head`` against the merge-base with ``base``.

    Uses the three-dot form so you see exactly what ``head`` adds relative to
    where it branched off ``base`` -- the same diff a PR would show.
    """
    out = subprocess.run(
        ["git", "diff", f"{base}...{head}"],
        capture_output=True,
        text=True,
        cwd=cwd,
        check=True,
    )
    return out.stdout


def changed_files(
    base: str = "main", head: str = "HEAD", *, cwd: str | Path | None = None
) -> list[str]:
    out = subprocess.run(
        ["git", "diff", "--name-only", f"{base}...{head}"],
        capture_output=True,
        text=True,
        cwd=cwd,
        check=True,
    )
    return [line for line in out.stdout.splitlines() if line]
