"""Run-dir layout: traversal containment, atomic write crash semantics, and
creation-collision behavior.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from grindstone.rundir import atomic_write_json, create_run_dir


def test_create_run_dir_paths_and_collision(tmp_path: Path) -> None:
    run = create_run_dir(tmp_path, "run-1")
    assert run.root == tmp_path / ".grindstone" / "runs" / "run-1"
    assert run.root.is_dir()
    assert run.state_path == run.root / "state.json"
    assert run.events_path == run.root / "events.ndjson"
    with pytest.raises(FileExistsError):
        create_run_dir(tmp_path, "run-1")


def test_worktrees_root_is_external_and_collision_safe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The throwaway git worktrees must live OUTSIDE the target repo: a nested
    # worktree lets a worker strip its CWD back to the repo root and write there
    # (the run 124321Z RCA). So the base is external, keyed by repo-id + run-id.
    base = tmp_path / "wt-base"
    monkeypatch.setenv("GRINDSTONE_WORKTREE_BASE", str(base))
    run = create_run_dir(tmp_path / "repo", "run-1")
    wt_root = run.worktrees_root
    # Not under the run dir, and not under the repo at all.
    assert run.root.resolve() not in wt_root.resolve().parents
    repo_root = (tmp_path / "repo").resolve()
    assert repo_root not in wt_root.resolve().parents
    # Sits under the configured base, ends in the run-id + worktrees leaf.
    assert base.resolve() in wt_root.resolve().parents
    assert wt_root.name == "worktrees"
    assert wt_root.parent.name == "run-1"

    # A second repo with the SAME run-id never collides (repo-id segment differs).
    other = create_run_dir(tmp_path / "repo2", "run-1")
    assert other.worktrees_root != wt_root


def test_worktrees_root_default_base_when_env_unset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("GRINDSTONE_WORKTREE_BASE", raising=False)
    run = create_run_dir(tmp_path / "repo", "run-1")
    assert str(run.worktrees_root).startswith("/tmp/cache/grindstone/")
    assert run.worktrees_root.name == "worktrees"


def test_resolve_accepts_valid_log_key(tmp_path: Path) -> None:
    run = create_run_dir(tmp_path, "run-1")
    resolved = run.resolve("p2/e3/t1/handoff.json")
    assert resolved == (run.root / "p2/e3/t1/handoff.json").resolve()
    assert run.root.resolve() in resolved.parents


@pytest.mark.parametrize(
    "bad",
    [
        "../escape",          # leading .. rejected by the log-key grammar
        "/etc/passwd",        # absolute path rejected by the grammar
        "a/../../etc/passwd", # grammar-legal but escapes via .. (containment guard)
    ],
)
def test_resolve_rejects_traversal(tmp_path: Path, bad: str) -> None:
    run = create_run_dir(tmp_path, "run-1")
    with pytest.raises(ValueError):
        run.resolve(bad)


def test_resolve_rejects_symlink_escape(tmp_path: Path) -> None:
    run = create_run_dir(tmp_path, "run-1")
    outside = tmp_path / "outside"
    outside.mkdir()
    (run.root / "link").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError):
        run.resolve("link/secret")


def test_artifacts_dir_created_under_root(tmp_path: Path) -> None:
    run = create_run_dir(tmp_path, "run-1")
    d = run.artifacts_dir("P1/E1/T1")
    assert d.is_dir()
    assert d == (run.root / "artifacts" / "P1/E1/T1").resolve()
    assert run.root.resolve() in d.parents


def test_find_artifact_exact_key(tmp_path: Path) -> None:
    run = create_run_dir(tmp_path, "run-1")
    target = run.resolve("P2/E3/T1/findings.md")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("found", encoding="utf-8")
    assert run.find_artifact("P2/E3/T1/findings.md") == target


def test_find_artifact_bare_filename_matches_unique_artifact(tmp_path: Path) -> None:
    # Gate-6 RCA: a phase exit criterion is written at SKELETON time, when the
    # P*/E*/T*/ placement an epoch will choose is unknowable, a bare filename
    # must match the one logged artifact carrying that name.
    run = create_run_dir(tmp_path, "run-1")
    target = run.resolve("P2/E3/T1/findings.md")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("found", encoding="utf-8")
    assert run.find_artifact("findings.md") == target


def test_find_artifact_ambiguous_or_missing_is_none(tmp_path: Path) -> None:
    run = create_run_dir(tmp_path, "run-1")
    for key in ("P1/E1/T1/notes.md", "P2/E1/T1/notes.md"):
        path = run.resolve(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x", encoding="utf-8")
    assert run.find_artifact("notes.md") is None  # two matches: ambiguous
    assert run.find_artifact("ghost.md") is None  # no match
    assert run.find_artifact("P9/E9/T9/ghost.md") is None  # exact key, no file


def test_atomic_write_json_roundtrip(tmp_path: Path) -> None:
    target = tmp_path / "state.json"
    atomic_write_json(target, {"b": 2, "a": 1})
    assert json.loads(target.read_text()) == {"a": 1, "b": 2}
    # Overwrite is atomic and complete.
    atomic_write_json(target, {"a": 9})
    assert json.loads(target.read_text()) == {"a": 9}


def test_atomic_write_json_leaves_no_temp_on_failure(tmp_path: Path) -> None:
    target = tmp_path / "state.json"
    with pytest.raises(TypeError):
        atomic_write_json(target, {"bad": object()})
    assert not target.exists()
    assert list(tmp_path.glob("*.tmp")) == []
