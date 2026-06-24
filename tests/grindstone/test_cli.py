"""CLI spine + the run/resume wiring: --help works; run/resume build the planner +
backends from config and reach a terminal exit code (driven by Part 1-4 mocks, never
a real rig); a missing config / run surfaces as exit 2; watch renders the journal.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from grindstone import cli, loop
from grindstone.contracts.models import EndDecision, Epoch, EpochDecision, Task
from grindstone.events import EpochStarted, JournalWriter, RunStarted, TaskRef
from tests.grindstone.mock_planner import MockDecisionPlanner
from tests.grindstone.mock_worker import LoopWorker
from grindstone.rundir import create_run_dir
from grindstone.worker import Backends


# --- builders ------------------------------------------------------------------


def _epoch(*owned: str) -> EpochDecision:
    tasks = [
        Task(id=f"T{i + 1}", mode="implement", goal=f"build {f}", file_ownership=[f])
        for i, f in enumerate(owned)
    ]
    return EpochDecision(kind="epoch", epoch=Epoch(title="build", tasks=tasks))


def _end(summary: str = "done") -> EndDecision:
    return EndDecision(kind="end", summary=summary)


def _write_config(repo: Path, *, done_when: str | None = None) -> None:
    cfg = repo / ".grindstone" / "config.yaml"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    body = (
        "roles:\n"
        "  planner:\n    rig: claude\n    slots: 1\n    timeout_s: 60\n"
        "  worker:\n    rig: claude\n    slots: 1\n    timeout_s: 60\n"
    )
    if done_when is not None:
        body += f"done_when: {done_when!r}\n"
    cfg.write_text(body, encoding="utf-8")


@pytest.fixture
def job(tmp_path: Path) -> Path:
    p = tmp_path / "job.md"
    p.write_text("# job\nbuild it\n", encoding="utf-8")
    return p


def _inject(monkeypatch: pytest.MonkeyPatch, planner: MockDecisionPlanner, worker: LoopWorker) -> None:
    """Swap the real (subprocess) planner + backends for the Part 1-4 mocks, so the
    wiring is exercised end to end without ever launching a rig."""

    monkeypatch.setattr(loop, "_build_planner", lambda cfg, repo: planner)
    monkeypatch.setattr(
        loop, "build_backends", lambda cfg, *, log_root: Backends.single(worker)
    )


# --- --help --------------------------------------------------------------------


def test_help_exits_zero() -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["--help"])
    assert exc.value.code == 0


# --- run wiring ----------------------------------------------------------------


def test_loop_run_drives_to_completed(
    git_repo: Path, job: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_config(git_repo)
    _inject(monkeypatch, MockDecisionPlanner([_epoch("a.py"), _end("shipped")]), LoopWorker())
    code = loop.run(job, git_repo, run_id="r1")
    assert code == 0
    # The journal was rendered at the terminal.
    journal = (git_repo / ".grindstone" / "runs" / "r1" / "journal.md").read_text()
    assert "completed" in journal


def test_cli_run_returns_two_without_config(
    git_repo: Path, job: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = cli.main(["run", str(job), "--repo", str(git_repo)])
    assert code == 2
    assert "config.yaml" in capsys.readouterr().out


def test_loop_run_generates_timestamp_run_id(
    git_repo: Path, job: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_config(git_repo)
    _inject(monkeypatch, MockDecisionPlanner([_end("nothing to do")]), LoopWorker())
    code = loop.run(job, git_repo)  # no run_id -> a UTC slug is minted
    assert code == 0
    runs = list((git_repo / ".grindstone" / "runs").iterdir())
    assert len(runs) == 1 and runs[0].name.endswith("Z")


# --- the final-acceptance invariant (#2) ---------------------------------------


def test_acceptance_passing_done_when_completes(
    git_repo: Path, job: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The epoch builds a.py; done_when checks it on the integration tip -> completed.
    _write_config(git_repo, done_when="test -f a.py")
    _inject(monkeypatch, MockDecisionPlanner([_epoch("a.py"), _end()]), LoopWorker())
    assert loop.run(job, git_repo, run_id="ok") == 0


def test_acceptance_failing_done_when_ends(
    git_repo: Path, job: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A done_when that cannot pass turns the planner's END into a clean partial-end.
    _write_config(git_repo, done_when="false")
    _inject(monkeypatch, MockDecisionPlanner([_end("partial handoff")]), LoopWorker())
    assert loop.run(job, git_repo, run_id="bad") == 1


def test_no_done_when_trusts_planner(
    git_repo: Path, job: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_config(git_repo)  # no done_when
    _inject(monkeypatch, MockDecisionPlanner([_end("trusted")]), LoopWorker())
    assert loop.run(job, git_repo, run_id="trust") == 0


# --- resume wiring -------------------------------------------------------------


def test_loop_resume_completed_run_is_idempotent(
    git_repo: Path, job: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_config(git_repo)
    _inject(monkeypatch, MockDecisionPlanner([_end("first")]), LoopWorker())
    assert loop.run(job, git_repo, run_id="r2") == 0
    # Resuming an already-completed run short-circuits to completed (no planner call).
    monkeypatch.setattr(loop, "_build_planner", lambda cfg, repo: MockDecisionPlanner([]))
    assert loop.resume("r2", git_repo) == 0


def test_cli_resume_returns_two_without_run(
    git_repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_config(git_repo)
    code = cli.main(["resume", "nope", "--repo", str(git_repo)])
    assert code == 2
    assert "no run" in capsys.readouterr().out


# --- watch renders the journal -------------------------------------------------


def test_watch_renders_journal(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    rd = create_run_dir(repo, "run1")
    with JournalWriter(rd.events_path) as jw:
        jw.emit(lambda s: RunStarted(seq=s, ts="2026-06-24T00:00:00+00:00",
                                     run_id="run1", job_path="job.md"))
        jw.emit(lambda s: EpochStarted(seq=s, ts="2026-06-24T00:00:01+00:00",
                                       epoch_id="P1/E1", title="build",
                                       tasks=[TaskRef(id="T1", mode="implement")]))
    code = cli.main(["watch", "run1", "--repo", str(repo)])
    out = capsys.readouterr().out
    assert code == 0
    assert "# Run run1" in out
    assert "## P1/E1 - build" in out
    assert "T1 (implement)" in out


def test_watch_missing_run_returns_two(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = cli.main(["watch", str(tmp_path / "nope")])
    assert code == 2
    assert "not a run dir" in capsys.readouterr().out
