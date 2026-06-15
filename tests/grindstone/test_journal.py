"""The per-run journal: a markdown render of events.ndjson (the human-facing
post-mortem), written at terminal, and reaped for every run BUT the latest when
a new run starts. Derived from events, so reaping a journal loses nothing the
retained events.ndjson can't re-render.
"""

from __future__ import annotations

from pathlib import Path

from grindstone.events import (
    EpochCompleted,
    EpochStarted,
    Event,
    JournalWriter,
    PhasePassed,
    PhaseRef,
    PhaseStarted,
    PlannerCallStarted,
    RunCompleted,
    RunFailed,
    RunStarted,
    SkeletonProposed,
    TaskDispatched,
    TaskDone,
    TaskFailed,
    TaskRef,
    replay,
)
from grindstone.journal import reap_sibling_journals, render_journal, write_journal
from grindstone.rundir import RunDir, create_run_dir

TS = "2026-06-10T00:00:00+00:00"
TS_END = "2026-06-10T00:05:00+00:00"


def _journal() -> list[Event]:
    seq = iter(range(100))
    return [
        RunStarted(seq=next(seq), ts=TS, run_id="r1", job_path="job.md", max_planner_calls=96),
        PlannerCallStarted(seq=next(seq), ts=TS),
        SkeletonProposed(
            seq=next(seq), ts=TS,
            phases=[PhaseRef(id="P1", title="build"), PhaseRef(id="P2", title="verify")],
        ),
        PhaseStarted(seq=next(seq), ts=TS, phase_id="P1"),
        EpochStarted(
            seq=next(seq), ts=TS, phase_id="P1", epoch_id="E1", title="make files",
            tasks=[TaskRef(id="T1", mode="implement"), TaskRef(id="T2", mode="implement")],
        ),
        TaskDispatched(seq=next(seq), ts=TS, epoch_id="E1", task_id="T1"),
        TaskDispatched(seq=next(seq), ts=TS, epoch_id="E1", task_id="T2"),
        TaskDone(seq=next(seq), ts="2026-06-10T00:00:30+00:00", epoch_id="E1", task_id="T1"),
        TaskFailed(seq=next(seq), ts="2026-06-10T00:00:12+00:00", epoch_id="E1", task_id="T2"),
        EpochCompleted(seq=next(seq), ts=TS, epoch_id="E1"),
        PhasePassed(seq=next(seq), ts=TS, phase_id="P1"),
        PhaseStarted(seq=next(seq), ts=TS, phase_id="P2"),
        RunCompleted(seq=next(seq), ts=TS_END),
    ]


def test_render_journal_covers_tree_as_markdown() -> None:
    out = render_journal(replay(_journal()))
    # Run header: id, status, job, planner-call counter (count/cap).
    assert "# Run r1" in out
    assert "completed" in out
    assert "job.md" in out
    assert "1/96" in out  # planner calls / cap
    # Phases are markdown headings with status.
    assert "## P1 · build" in out and "[passed]" in out
    assert "## P2 · verify" in out and "[started]" in out
    # Epoch + task leaves with statuses + glyphs.
    assert "E1 · make files" in out and "[completed]" in out
    assert "T1 (implement)" in out and "[done]" in out and "✓" in out
    assert "T2 (implement)" in out and "[failed]" in out and "✗" in out
    # A duration surfaced somewhere (run span 5m, task spans).
    assert "5m00s" in out


def test_render_journal_shows_terminal_reason() -> None:
    events = _journal()[:-1] + [
        RunFailed(seq=99, ts=TS_END, reason="safety valve: 96 planner calls reached")
    ]
    out = render_journal(replay(events))
    assert "failed" in out
    assert "safety valve" in out


def test_write_journal_renders_from_events(tmp_path: Path) -> None:
    rd = create_run_dir(tmp_path, "r1")
    with JournalWriter(rd.events_path) as j:
        for ev in _journal():
            j.append(ev)
    assert not rd.journal_path.exists()
    write_journal(rd)
    assert rd.journal_path.exists()
    body = rd.journal_path.read_text(encoding="utf-8")
    assert "# Run r1" in body and "## P1 · build" in body


def test_write_journal_noop_without_run_started(tmp_path: Path) -> None:
    rd = create_run_dir(tmp_path, "empty")
    rd.events_path.write_text("", encoding="utf-8")  # journal present but no events
    write_journal(rd)  # must not raise
    assert not rd.journal_path.exists()  # nothing to render -> no file


def test_reap_sibling_journals_keeps_latest_and_all_events(tmp_path: Path) -> None:
    runs = tmp_path / ".grindstone" / "runs"
    old = RunDir(root=runs / "old")
    new = RunDir(root=runs / "new")
    for rd in (old, new):
        rd.root.mkdir(parents=True)
        rd.journal_path.write_text("# stale\n", encoding="utf-8")
        rd.events_path.write_text('{"seq":0}\n', encoding="utf-8")

    reap_sibling_journals(new)

    # The previous run's journal.md is gone; its events.ndjson survives.
    assert not old.journal_path.exists()
    assert old.events_path.exists()
    # The current run keeps its journal.
    assert new.journal_path.exists()


def test_write_journal_never_raises_on_inconsistent_stream(tmp_path: Path) -> None:
    """write_journal renders AFTER the run is durably terminal, so it must be
    best-effort: a KeyError from an inconsistent event stream (or an OSError on
    write) must never propagate and throw away the already-durable RunOutcome."""

    rd = create_run_dir(tmp_path, "bad")
    with JournalWriter(rd.events_path) as j:
        j.append(RunStarted(seq=0, ts=TS, run_id="r", job_path="j"))
        # a TaskDone for an epoch/task never declared -> replay raises KeyError
        j.append(TaskDone(seq=1, ts=TS, epoch_id="E9", task_id="T9"))
    write_journal(rd)  # must NOT raise
    assert not rd.journal_path.exists()  # render failed -> no stub file, no crash
