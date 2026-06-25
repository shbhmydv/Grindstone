"""``add_worktree`` crash-safety: force-create over a stale leftover wip branch.

A dead run can leave a ``grind-wip/*`` branch behind. A FRESH run never razes (only
resume does), so ``git worktree add -b`` would HARD-FAIL on the name collision and
brick the new run. ``-B`` resets the stale branch to the requested base, which is
safe because a wip branch is only (re)built when its task is being (re)done. ``-B``
still refuses a branch checked out in a LIVE worktree (a real conflict), which must
keep failing loudly.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from grindstone import worktree as wt
from grindstone.worktree import GitError
from tests.grindstone.conftest import git, init_git_repo


def _branch_of(path: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=str(path), capture_output=True, text=True, check=True,
    ).stdout.strip()


def test_add_worktree_force_creates_over_stale_branch(tmp_path: Path) -> None:
    repo = init_git_repo(tmp_path / "repo")
    base = wt.resolve_commit(repo, "main")
    # A dead run's leftover wip branch (advanced past base) the fresh run must reset.
    git(repo, "branch", "grind-wip/E1-T1/attempt-1", "main")

    path = tmp_path / "wt" / "E1-T1" / "attempt-1"
    wt.add_worktree(repo, path, branch="grind-wip/E1-T1/attempt-1", base="main")

    assert _branch_of(path) == "grind-wip/E1-T1/attempt-1"
    assert wt.resolve_commit(path, "HEAD") == base


def test_add_worktree_refuses_branch_checked_out_in_live_worktree(
    tmp_path: Path,
) -> None:
    repo = init_git_repo(tmp_path / "repo")
    live = tmp_path / "wt" / "live"
    wt.add_worktree(repo, live, branch="feature", base="main")
    # The same branch is now live in `live`; -B must refuse to reset it elsewhere.
    other = tmp_path / "wt" / "other"
    with pytest.raises(GitError):
        wt.add_worktree(repo, other, branch="feature", base="main")
