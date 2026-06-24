"""CLI spine: --help works, watch renders a journal, run/resume are clear stubs."""

from __future__ import annotations

from pathlib import Path

import pytest

from grindstone import cli
from grindstone.events import EpochStarted, JournalWriter, RunStarted, TaskRef
from grindstone.rundir import create_run_dir


def test_help_exits_zero() -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["--help"])
    assert exc.value.code == 0


def test_run_is_a_clear_stub(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = cli.main(["run", str(tmp_path / "job.md"), "--repo", str(tmp_path)])
    assert code == 2
    assert "later part" in capsys.readouterr().out


def test_resume_is_a_clear_stub(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = cli.main(["resume", "r1", "--repo", str(tmp_path)])
    assert code == 2
    assert "later part" in capsys.readouterr().out


def test_watch_renders_tree(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    rd = create_run_dir(repo, "run1")
    with JournalWriter(rd.events_path) as jw:
        jw.emit(lambda s: RunStarted(seq=s, ts="t0", run_id="run1", job_path="job.md"))
        jw.emit(lambda s: EpochStarted(seq=s, ts="t1", epoch_id="P1/E1", title="build",
                                       tasks=[TaskRef(id="T1", mode="implement")]))
    code = cli.main(["watch", "run1", "--repo", str(repo)])
    out = capsys.readouterr().out
    assert code == 0
    assert "run run1" in out
    assert "epoch P1/E1" in out
    assert "T1 (implement)" in out
