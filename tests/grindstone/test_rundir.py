"""Run-dir layout: the keyed-log index, the log-key traversal guard, and the
external worktrees_root."""

from __future__ import annotations

from pathlib import Path

import pytest

from grindstone.rundir import RunDir, create_run_dir


def _run_dir(tmp_path: Path) -> RunDir:
    repo = tmp_path / "repo"
    repo.mkdir()
    return create_run_dir(repo, "20260624T000000Z")


def test_create_run_dir_layout(tmp_path: Path) -> None:
    rd = _run_dir(tmp_path)
    assert rd.root.is_dir()
    assert rd.events_path == rd.root / "events.ndjson"
    assert rd.journal_path == rd.root / "journal.md"


def test_create_run_dir_rejects_existing(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    create_run_dir(repo, "dup")
    with pytest.raises(FileExistsError):
        create_run_dir(repo, "dup")


def test_log_index_keys_epoch_dirs_only(tmp_path: Path) -> None:
    rd = _run_dir(tmp_path)
    (rd.root / "E1" / "T1").mkdir(parents=True)
    (rd.root / "E1" / "T1" / "handoff.json").write_text("{}", encoding="utf-8")
    (rd.root / "E1" / "report.md").write_text("hi", encoding="utf-8")
    # Non-epoch run state must NOT appear in the keyed log.
    rd.events_path.write_text("", encoding="utf-8")
    rd.journal_path.write_text("# journal\n", encoding="utf-8")
    index = rd.log_index()
    assert "E1/report.md" in index
    assert "E1/T1/handoff.json" in index
    assert "events.ndjson" not in index
    assert "journal.md" not in index


def test_baton_path_and_read_baton(tmp_path: Path) -> None:
    rd = _run_dir(tmp_path)
    # Absent baton (and the epoch-0 sentinel for the first epoch) reads as "".
    assert rd.read_baton(1) == ""
    assert rd.read_baton(0) == ""
    # baton_path is the E<n>/baton.md keyed-log path; once written, read_baton returns it.
    p = rd.baton_path(2)
    assert p == (rd.root / "E2/baton.md").resolve()
    p.parent.mkdir(parents=True)
    p.write_text("## Project summary\nso far so good\n", encoding="utf-8")
    assert rd.read_baton(2) == "## Project summary\nso far so good\n"


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
    resolved = rd.resolve("E1/T1/handoff.json")
    assert resolved == (rd.root / "E1/T1/handoff.json").resolve()


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
