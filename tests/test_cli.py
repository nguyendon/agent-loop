"""CLI helper tests (offline; uses a real local git repo, no network)."""

from __future__ import annotations

import subprocess
from pathlib import Path

from agentloop.cli import _tree_is_dirty


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True)


def _init_repo(repo: Path) -> None:
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")
    (repo / "a.txt").write_text("one\n")
    _git(repo, "add", "a.txt")
    _git(repo, "commit", "-qm", "init")


def test_dirty_guard_clean_tree(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    assert _tree_is_dirty(str(tmp_path)) is False


def test_dirty_guard_modified_tracked_file(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    (tmp_path / "a.txt").write_text("changed\n")
    assert _tree_is_dirty(str(tmp_path)) is True


def test_dirty_guard_ignores_untracked(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    (tmp_path / "scratch.txt").write_text("untracked\n")  # not in git diff, shouldn't block
    assert _tree_is_dirty(str(tmp_path)) is False


def test_dirty_guard_non_git_dir_returns_none(tmp_path: Path) -> None:
    # Not a git repo → can't tell → don't block (None).
    assert _tree_is_dirty(str(tmp_path)) is None
