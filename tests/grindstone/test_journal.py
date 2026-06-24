"""The journal render: a scripted events.ndjson -> a markdown post-mortem timeline
with epoch/task sections, durations, the critic verdict notes, rate-limit + resume
markers, and the terminal. Plus sibling-journal reaping and the never-raise guard.
"""

from __future__ import annotations

from pathlib import Path

from grindstone.events import (
    EpochCompleted,
    EpochStarted,
    WorkGateRejected,
    JournalWriter,
    RateLimited,
    RunCompleted,
    RunEnded,
    RunResumed,
    RunStarted,
    TaskDispatched,
    TaskDone,
    TaskRef,
    Verdict,
    read_events,
    replay,
)
from grindstone.journal import (
    reap_sibling_journals,
    render_journal,
    render_run_dir,
    write_journal,
)
from grindstone.rundir import create_run_dir


def _ts(sec: int) -> str:
    return f"2026-06-24T00:00:{sec:02d}+00:00"


def _scripted(events_path: Path) -> None:
    """A complete run: one epoch with a passing implement task, then completed."""

    with JournalWriter(events_path) as jw:
        jw.emit(lambda s: RunStarted(seq=s, ts=_ts(0), run_id="run-x",
                                     job_path="/jobs/build.md", max_epochs=5))
        jw.emit(lambda s: EpochStarted(seq=s, ts=_ts(1), epoch_id="E1", title="scaffold",
                                       tasks=[TaskRef(id="T1", mode="implement"),
                                              TaskRef(id="T2", mode="review")]))
        jw.emit(lambda s: TaskDispatched(seq=s, ts=_ts(2), epoch_id="E1", task_id="T1"))
        jw.emit(lambda s: Verdict(seq=s, ts=_ts(8), epoch_id="E1", task_id="T1",
                                  outcome="PASS", reason="good enough to build on"))
        jw.emit(lambda s: TaskDone(seq=s, ts=_ts(9), epoch_id="E1", task_id="T1"))
        jw.emit(lambda s: TaskDispatched(seq=s, ts=_ts(2), epoch_id="E1", task_id="T2"))
        jw.emit(lambda s: WorkGateRejected(seq=s, ts=_ts(6), epoch_id="E1",
                                          task_id="T2", reason="missing citation"))
        jw.emit(lambda s: EpochCompleted(seq=s, ts=_ts(30), epoch_id="E1"))
        jw.emit(lambda s: RunCompleted(seq=s, ts=_ts(31)))


def test_render_timeline_sections_and_durations(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    rd = create_run_dir(repo, "run-x")
    _scripted(rd.events_path)
    md = render_journal(replay(read_events(rd.events_path)))

    assert "# Run run-x - completed" in md
    assert "- Job: `/jobs/build.md`" in md
    assert "Epochs: 1/5" in md
    assert "## E1 - scaffold  [completed]" in md
    # The whole run spanned 0s..31s.
    assert "Duration: 31s" in md
    # Task T1 ran 2s..9s; reaching done clears any stale verdict note.
    assert "T1 (implement) [done]  (7s)" in md
    assert "good enough to build on" not in md
    # A rejected task surfaces its reason as the gate note.
    assert "T2 (review) [gate_rejected]" in md
    assert "missing citation" in md


def test_render_marks_rate_limit_and_ended(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    rd = create_run_dir(repo, "run-y")
    with JournalWriter(rd.events_path) as jw:
        jw.emit(lambda s: RunStarted(seq=s, ts=_ts(0), run_id="run-y", job_path="j.md"))
        jw.emit(lambda s: RateLimited(seq=s, ts=_ts(1), role="planner", detail="429"))
        jw.emit(lambda s: RunEnded(seq=s, ts=_ts(2), summary="partial: parser only"))
    md = render_journal(replay(read_events(rd.events_path)))
    assert "# Run run-y - ended" in md
    assert "Rate-limited: planner: 429" in md
    assert "Ended: partial: parser only" in md


def test_render_run_dir_renders_resume_marker(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    rd = create_run_dir(repo, "run-z")
    with JournalWriter(rd.events_path) as jw:
        jw.emit(lambda s: RunStarted(seq=s, ts=_ts(0), run_id="run-z", job_path="j.md"))
        jw.emit(lambda s: EpochStarted(seq=s, ts=_ts(1), epoch_id="E1", title="x",
                                       tasks=[TaskRef(id="T1", mode="implement")]))
        jw.emit(lambda s: RunResumed(seq=s, ts=_ts(2), run_id="run-z", razed_epoch="E1"))
    md = render_run_dir(rd)
    assert md is not None and "# Run run-z - running" in md


def test_render_run_dir_none_when_empty(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    rd = create_run_dir(repo, "empty")
    assert render_run_dir(rd) is None  # no events file yet
    rd.events_path.write_text("", encoding="utf-8")
    assert render_run_dir(rd) is None  # empty stream


def test_write_journal_writes_markdown(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    rd = create_run_dir(repo, "run-x")
    _scripted(rd.events_path)
    write_journal(rd)
    assert rd.journal_path.is_file()
    assert "# Run run-x - completed" in rd.journal_path.read_text(encoding="utf-8")


def test_reap_sibling_journals(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    keep = create_run_dir(repo, "keep")
    other = create_run_dir(repo, "other")
    keep.journal_path.write_text("# keep\n", encoding="utf-8")
    other.journal_path.write_text("# other\n", encoding="utf-8")
    reap_sibling_journals(keep)
    assert keep.journal_path.is_file()  # the current run keeps its journal
    assert not other.journal_path.exists()  # siblings are reaped
