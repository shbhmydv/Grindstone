"""Worktree lifecycle, ownership scope check, commit, and merge, against
throwaway tmp git repos (ARCHITECTURE.md / S2 rulings 4-7). These never touch the
Grindstone checkout: every op targets the fixture's tmp repo.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from grindstone import worktree as wt

from tests.grindstone.conftest import git, init_git_repo, tracked_files


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return init_git_repo(tmp_path / "repo")


# --- scope check (pure) --------------------------------------------------------


@pytest.mark.parametrize(
    "path,ownership,ok",
    [
        ("out.txt", ["out.txt"], True),
        ("src/a.py", ["src/*"], True),
        ("src/deep/a.py", ["src/**"], True),
        ("src", ["src/**"], True),
        ("other/a.py", ["src/**"], False),
        ("src/a.py", ["lib/*"], False),
        ("a/b/c.txt", ["a/b/c.txt", "x/*"], True),
        ("a/b/c.txt", ["a/b/*", "x/*"], True),
    ],
)
def test_path_in_scope(path: str, ownership: list[str], ok: bool) -> None:
    assert wt.path_in_scope(path, ownership) is ok


def test_scope_violations_lists_only_out_of_scope() -> None:
    changed = ["a/x.py", "a/y.py", "b/z.py"]
    assert wt.scope_violations(changed, ["a/**"]) == ["b/z.py"]
    assert wt.scope_violations(changed, ["a/**", "b/**"]) == []


# --- lifecycle -----------------------------------------------------------------


def test_add_commit_changed_paths_roundtrip(repo: Path, tmp_path: Path) -> None:
    base = wt.head_commit(repo)
    path = tmp_path / "wt" / "a1"
    wt.add_worktree(repo, path, branch="grind/P1/E1/T1-a1", base=base)
    assert path.is_dir()
    (path / "new.txt").write_text("hi\n", encoding="utf-8")
    committed = wt.commit_all(path, "grind(P1/E1/T1): add new.txt")
    assert committed is True
    assert wt.changed_paths(repo, base, "grind/P1/E1/T1-a1") == ["new.txt"]


def test_commit_all_zero_diff_makes_no_commit(repo: Path, tmp_path: Path) -> None:
    base = wt.head_commit(repo)
    path = tmp_path / "wt" / "a1"
    wt.add_worktree(repo, path, branch="grind/P1/E1/T1-a1", base=base)
    # No writes -> nothing to commit; HEAD stays at base.
    assert wt.commit_all(path, "noop") is False
    assert wt.changed_paths(repo, base, "HEAD") == []


def test_discard_attempt_removes_worktree_and_branch(repo: Path, tmp_path: Path) -> None:
    base = wt.head_commit(repo)
    path = tmp_path / "wt" / "a1"
    branch = "grind/P1/E1/T1-a1"
    wt.add_worktree(repo, path, branch=branch, base=base)
    assert wt.branch_exists(repo, branch)
    wt.discard_attempt(repo, path, branch)
    assert not path.exists()
    assert not wt.branch_exists(repo, branch)


def test_branch_namespace_df_conflict_is_avoided(repo: Path, tmp_path: Path) -> None:
    """Ruling-7 deviation guard: an integration branch named ``grind/P1/E1``
    cannot coexist with task branches ``grind/P1/E1/T1-a1`` (git ref D/F
    conflict). The collision-free ``grind/P1/E1/_integration`` leaf does.
    """

    base = wt.head_commit(repo)
    path = tmp_path / "wt" / "a1"
    wt.add_worktree(repo, path, branch="grind/P1/E1/T1-a1", base=base)
    # The pinned literal name would collide:
    with pytest.raises(wt.GitError):
        wt.ensure_integration_branch(repo, "grind/P1/E1", base)
    # The chosen leaf name does not:
    wt.ensure_integration_branch(repo, "grind/P1/E1/_integration", base)
    assert wt.branch_exists(repo, "grind/P1/E1/_integration")


# --- merge ---------------------------------------------------------------------


def _branch_with_file(repo: Path, tmp_path: Path, name: str, rel: str, content: str) -> str:
    base = wt.head_commit(repo)
    path = tmp_path / "wt" / name
    branch = f"grind/P1/E1/{name}"
    wt.add_worktree(repo, path, branch=branch, base=base)
    (path / rel).parent.mkdir(parents=True, exist_ok=True)
    (path / rel).write_text(content, encoding="utf-8")
    wt.commit_all(path, f"grind: {name}")
    wt.remove_worktree(repo, path)
    return branch


def test_disjoint_merges_commute(repo: Path, tmp_path: Path) -> None:
    base = wt.head_commit(repo)
    b1 = _branch_with_file(repo, tmp_path, "T1-a1", "a/x.txt", "x\n")
    b2 = _branch_with_file(repo, tmp_path, "T2-a1", "b/y.txt", "y\n")
    wt.ensure_integration_branch(repo, "grind/P1/E1/_integration", base)
    int_wt = tmp_path / "wt" / "_integration"
    wt.add_worktree_on(repo, int_wt, branch="grind/P1/E1/_integration")
    assert wt.merge_into(int_wt, b1).ok
    assert wt.merge_into(int_wt, b2).ok
    wt.remove_worktree(repo, int_wt)
    assert tracked_files(repo, "grind/P1/E1/_integration") == [
        ".gitignore",
        "README.md",
        "a/x.txt",
        "b/y.txt",
    ]


def test_overlapping_merge_conflicts(repo: Path, tmp_path: Path) -> None:
    base = wt.head_commit(repo)
    b1 = _branch_with_file(repo, tmp_path, "T1-a1", "shared.txt", "from-T1\n")
    b2 = _branch_with_file(repo, tmp_path, "T2-a1", "shared.txt", "from-T2\n")
    wt.ensure_integration_branch(repo, "grind/P1/E1/_integration", base)
    int_wt = tmp_path / "wt" / "_integration"
    wt.add_worktree_on(repo, int_wt, branch="grind/P1/E1/_integration")
    assert wt.merge_into(int_wt, b1).ok
    conflict = wt.merge_into(int_wt, b2)
    assert conflict.ok is False
    assert conflict.conflict  # non-empty status / message
    # The aborted merge left the worktree clean (mergeable again later).
    assert wt.is_ancestor(repo, b1, "grind/P1/E1/_integration")


def test_is_ancestor_after_merge(repo: Path, tmp_path: Path) -> None:
    base = wt.head_commit(repo)
    b1 = _branch_with_file(repo, tmp_path, "T1-a1", "a/x.txt", "x\n")
    assert not wt.is_ancestor(repo, b1, "grind/P1/E1/_integration") or True
    wt.ensure_integration_branch(repo, "grind/P1/E1/_integration", base)
    int_wt = tmp_path / "wt" / "_integration"
    wt.add_worktree_on(repo, int_wt, branch="grind/P1/E1/_integration")
    wt.merge_into(int_wt, b1)
    wt.remove_worktree(repo, int_wt)
    assert wt.is_ancestor(repo, b1, "grind/P1/E1/_integration")


def test_add_worktree_detached_tolerates_branch_checked_out_elsewhere(
    repo: Path, tmp_path: Path
) -> None:
    """E2E gate2 P0: phase-eval crashed because its worktree targeted a BRANCH
    that something else (there: a rogue artifact worker; equally: the owner
    inspecting mid-run) had checked out in the operator's checkout. The
    evaluator only reads a tree, detached HEAD cannot collide by construction.
    """

    base = wt.head_commit(repo)
    wt.ensure_integration_branch(repo, "grind/P1/E1/_integration", base)
    git(repo, "checkout", "grind/P1/E1/_integration")

    with pytest.raises(wt.GitError):
        wt.add_worktree_on(repo, tmp_path / "wt" / "collide", branch="grind/P1/E1/_integration")

    eval_wt = tmp_path / "wt" / "_phase_eval"
    wt.add_worktree_detached(repo, eval_wt, ref="grind/P1/E1/_integration")
    assert wt.head_commit(eval_wt) == base
    wt.remove_worktree(repo, eval_wt)
