"""TUI render path (S4 ruling 4): the pure events -> RunTree -> rich render is
snapshot-tested via plain-text export. No live terminal, no wall clock, the
poll loop's timing is deliberately excluded; only the renderer is asserted.
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
    PlannerCallFailed,
    PlannerCallStarted,
    RunCompleted,
    RunFailed,
    RunStarted,
    SkeletonProposed,
    TaskDispatched,
    TaskDone,
    TaskEscalated,
    TaskFailed,
    TaskRef,
    replay,
)
from grindstone.rundir import RunDir, create_run_dir
from grindstone.tui import read_tree, render_plain, render_waiting, watch

TS = "2026-06-10T00:00:00+00:00"


def _journal() -> list[Event]:
    seq = iter(range(100))
    return [
        RunStarted(seq=next(seq), ts=TS, run_id="r1", job_path="job.md"),
        PlannerCallStarted(seq=next(seq), ts=TS),
        SkeletonProposed(
            seq=next(seq), ts=TS,
            phases=[PhaseRef(id="P1", title="build"), PhaseRef(id="P2", title="verify")],
        ),
        PhaseStarted(seq=next(seq), ts=TS, phase_id="P1"),
        PlannerCallStarted(seq=next(seq), ts=TS),
        EpochStarted(
            seq=next(seq), ts=TS, phase_id="P1", epoch_id="E1", title="make files",
            tasks=[TaskRef(id="T1", mode="implement"), TaskRef(id="T2", mode="implement")],
        ),
        TaskDispatched(seq=next(seq), ts=TS, epoch_id="E1", task_id="T1"),
        TaskDispatched(seq=next(seq), ts=TS, epoch_id="E1", task_id="T2"),
        TaskDone(seq=next(seq), ts=TS, epoch_id="E1", task_id="T1"),
        TaskFailed(seq=next(seq), ts=TS, epoch_id="E1", task_id="T2"),
        EpochCompleted(seq=next(seq), ts=TS, epoch_id="E1"),
        PhasePassed(seq=next(seq), ts=TS, phase_id="P1"),
        PhaseStarted(seq=next(seq), ts=TS, phase_id="P2"),
        RunCompleted(seq=next(seq), ts=TS),
    ]


def test_render_snapshot_covers_full_tree() -> None:
    out = render_plain(replay(_journal()))
    # Run header: status + planner-call counter (2 PlannerCallStarted events).
    assert "run r1" in out
    assert "[completed]" in out
    assert "planner calls: 2" in out
    # Phase nodes with their resolved statuses.
    assert "P1 · build" in out and "[passed]" in out
    assert "P2 · verify" in out and "[started]" in out
    # Epoch + task leaves with statuses.
    assert "E1 · make files" in out and "[completed]" in out
    assert "T1 (implement)" in out and "[done]" in out
    assert "T2 (implement)" in out and "[failed]" in out
    # Status glyphs survive the plain-text export (colour does not).
    assert "✓" in out and "✗" in out


def test_render_reflects_run_status() -> None:
    # Truncating before RunCompleted leaves the run "running".
    running = render_plain(replay(_journal()[:-1]))
    assert "[running]" in running


def test_render_shows_durations_reasons_and_planner_state() -> None:
    events = [
        RunStarted(seq=0, ts="2026-06-10T00:00:00+00:00", run_id="r1", job_path="job.md"),
        SkeletonProposed(seq=1, ts=TS, phases=[PhaseRef(id="P1", title="build")]),
        PhaseStarted(seq=2, ts="2026-06-10T00:00:00+00:00", phase_id="P1"),
        EpochStarted(
            seq=3, ts="2026-06-10T00:00:00+00:00", phase_id="P1", epoch_id="E1",
            title="make", tasks=[TaskRef(id="T1", mode="implement")],
        ),
        TaskDispatched(seq=4, ts="2026-06-10T00:00:00+00:00", epoch_id="E1", task_id="T1"),
        TaskEscalated(seq=5, ts="2026-06-10T00:01:23+00:00", epoch_id="E1", task_id="T1", tier="senior"),
        PlannerCallStarted(seq=6, ts=TS),
        PlannerCallFailed(seq=7, ts=TS, classification="rate_limit"),
    ]
    out = render_plain(replay(events), now="2026-06-10T00:02:00+00:00")
    assert "1m23s" in out          # T1 escalated 1m23s after dispatch (ended - started)
    assert "→ senior" in out       # escalation tier surfaced on the task
    assert "rate_limit" in out     # planner failure classification shown in header
    assert "[running]" in out      # run is still live


def test_header_shows_phase_progress_and_planner_cap() -> None:
    events = [
        RunStarted(seq=0, ts=TS, run_id="r1", job_path="job.md", max_planner_calls=96),
        PlannerCallStarted(seq=1, ts=TS),
        SkeletonProposed(
            seq=2, ts=TS,
            phases=[PhaseRef(id="P1", title="a"), PhaseRef(id="P2", title="b")],
        ),
        PhaseStarted(seq=3, ts=TS, phase_id="P1"),
        PhasePassed(seq=4, ts=TS, phase_id="P1"),
    ]
    out = render_plain(replay(events))
    assert "1/2 phases" in out      # 1 of 2 phases passed
    assert "planner calls: 1/96" in out  # count / cap


def test_waiting_placeholder_before_run_starts() -> None:
    assert "waiting" in render_waiting().plain.lower()


def test_render_shows_failed_run_with_reason() -> None:
    events = _journal()[:-1] + [
        RunFailed(seq=99, ts=TS, reason="safety valve: 96 planner calls reached")
    ]
    out = render_plain(replay(events))
    assert "[failed]" in out
    assert "safety valve" in out


def test_read_tree_handles_empty_and_missing(tmp_path: Path) -> None:
    run_dir = RunDir(root=tmp_path / "nope")
    assert read_tree(run_dir) is None  # missing journal -> None, never raises
    rd = create_run_dir(tmp_path, "empty")
    rd.events_path.write_text("", encoding="utf-8")
    assert read_tree(rd) is None  # empty journal -> None


def test_watch_returns_on_terminal_run_without_sleeping(tmp_path: Path) -> None:
    rd = create_run_dir(tmp_path, "done")
    with JournalWriter(rd.events_path) as j:
        for ev in _journal():
            j.append(ev)

    def _boom(_delay: float) -> None:  # the loop must NOT sleep on a terminal run
        raise AssertionError("watch slept on an already-terminal run")

    tree = watch(rd, sleep=_boom)
    assert tree is not None and tree.status == "completed"


def test_watch_exits_on_failed_run(tmp_path: Path) -> None:
    rd = create_run_dir(tmp_path, "failed")
    events = _journal()[:-1] + [RunFailed(seq=99, ts=TS, reason="safety valve")]
    with JournalWriter(rd.events_path) as j:
        for ev in events:
            j.append(ev)

    def _boom(_delay: float) -> None:  # a capped/failed run is terminal -> no sleep
        raise AssertionError("watch slept on a failed (terminal) run")

    tree = watch(rd, sleep=_boom)
    assert tree is not None and tree.status == "failed"


def test_watch_heartbeats_while_running(tmp_path: Path) -> None:
    # Even with NO new journal events, the loop must repaint each tick so the
    # elapsed clock advances (the "looks frozen" fix). We prove it by counting
    # now_fn calls: one per render, several across ticks despite a static file.
    rd = create_run_dir(tmp_path, "live")
    with JournalWriter(rd.events_path) as j:
        for ev in _journal()[:-1]:  # truncate RunCompleted -> still running
            j.append(ev)

    ticks = {"n": 0}

    def _sleep(_delay: float) -> None:
        ticks["n"] += 1
        if ticks["n"] >= 3:
            raise KeyboardInterrupt

    nows: list[str] = []

    def _now() -> str:
        t = f"2026-06-10T00:00:{len(nows):02d}+00:00"
        nows.append(t)
        return t

    tree = watch(rd, sleep=_sleep, now_fn=_now)
    assert tree is not None and tree.status == "running"
    assert len(nows) >= 3  # repainted on each tick, not just on file change
