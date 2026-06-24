"""Shared fixtures for the loop/worker tests: a throwaway git repo, an external
worktree base (the escape-proofing lesson), and a run dir under that repo.

All git ops target the caller-supplied tmp repo, never the Grindstone checkout.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from grindstone.rundir import RunDir, create_run_dir


def git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=str(repo), capture_output=True, text=True, check=True
    )


def init_git_repo(path: Path) -> Path:
    """A throwaway repo with one base commit; ignores the run dir + bytecode."""

    path.mkdir(parents=True, exist_ok=True)
    git(path, "init", "-q", "-b", "main")
    git(path, "config", "user.email", "toy@grindstone.local")
    git(path, "config", "user.name", "toy")
    (path / ".gitignore").write_text(".grindstone/\n__pycache__/\n", encoding="utf-8")
    (path / "README.md").write_text("base\n", encoding="utf-8")
    git(path, "add", "-A")
    git(path, "commit", "-q", "-m", "base")
    return path


@pytest.fixture(autouse=True)
def _external_worktree_base(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Redirect throwaway worktrees to an EXTERNAL tmp base (never under the repo)."""

    monkeypatch.setenv("GRINDSTONE_WORKTREE_BASE", str(tmp_path / "wt-base"))


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    return init_git_repo(tmp_path / "repo")


@pytest.fixture
def run_dir(git_repo: Path) -> RunDir:
    return create_run_dir(git_repo, "run-1")
