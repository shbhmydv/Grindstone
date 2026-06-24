"""Run-dir layout: the keyed-log index, the log-key traversal guard, the external
worktrees_root, and the atomic JSON write."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from grindstone.rundir import RunDir, atomic_write_json, create_run_dir


def _run_dir(tmp_path: Path) -> RunDir:
    repo = tmp_path / "repo"
    repo.mkdir()
    return create_run_dir(repo, "20260624T000000Z")


def test_create_run_dir_layout(tmp_path: Path) -> None:
    rd = _run_dir(tmp_path)
    assert rd.root.is_dir()
    assert rd.events_path == rd.root / "events.ndjson"
    assert rd.state_path == rd.root / "state.json"


def test_create_run_dir_rejects_existing(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    create_run_dir(repo, "dup")
    with pytest.raises(FileExistsError):
        create_run_dir(repo, "dup")


def test_log_index_keys_phase_dirs_only(tmp_path: Path) -> None:
    rd = _run_dir(tmp_path)
    (rd.root / "P1" / "E1" / "T1").mkdir(parents=True)
    (rd.root / "P1" / "E1" / "T1" / "handoff.json").write_text("{}", encoding="utf-8")
    (rd.root / "P1" / "report.md").write_text("hi", encoding="utf-8")
    # Non-phase run state must NOT appear in the keyed log.
    rd.events_path.write_text("", encoding="utf-8")
    rd.state_path.write_text("{}", encoding="utf-8")
    index = rd.log_index()
    assert "P1/report.md" in index
    assert "P1/E1/T1/handoff.json" in index
    assert "events.ndjson" not in index
    assert "state.json" not in index


def test_resolve_rejects_bad_grammar(tmp_path: Path) -> None:
    rd = _run_dir(tmp_path)
    with pytest.raises(ValueError):
        rd.resolve("../escape")
    with pytest.raises(ValueError):
        rd.resolve("/abs/path")


def test_resolve_rejects_embedded_traversal(tmp_path: Path) -> None:
    rd = _run_dir(tmp_path)
    with pytest.raises(ValueError):
        rd.resolve("P1/../../etc/passwd")


def test_resolve_accepts_valid_key(tmp_path: Path) -> None:
    rd = _run_dir(tmp_path)
    resolved = rd.resolve("P1/E1/T1/handoff.json")
    assert resolved == (rd.root / "P1/E1/T1/handoff.json").resolve()


def test_worktrees_root_is_external_and_honors_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = tmp_path / "wt-base"
    monkeypatch.setenv("GRINDSTONE_WORKTREE_BASE", str(base))
    rd = _run_dir(tmp_path)
    wroot = rd.worktrees_root
    # External: NOT under the run dir / repo (the worktree-escape lesson).
    assert not str(wroot).startswith(str(rd.root))
    assert str(wroot).startswith(str(base))
    assert wroot.name == "worktrees"


def test_atomic_write_json_roundtrips(tmp_path: Path) -> None:
    target = tmp_path / "state.json"
    atomic_write_json(target, {"b": 2, "a": 1})
    assert json.loads(target.read_text()) == {"a": 1, "b": 2}
    # No stray temp files left beside it.
    assert [p.name for p in tmp_path.iterdir()] == ["state.json"]
