"""Journal vocabulary proof: a full-run event stream must replay into the exact
run -> phase -> epoch -> task tree, the writer must enforce seq monotonicity,
and the reader must tolerate a crash-truncated final line.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from grindstone.events import (
    EpochCompleted,
    EpochStarted,
    HandoffRejected,
    JournalWriter,
    PhaseEscalated,
    PhasePassed,
    PhaseRef,
    PhaseStarted,
    PhasesRevised,
    PlannerCallFailed,
    PlannerCallStarted,
    PlannerCallSucceeded,
    RunCompleted,
    RunEscalated,
    RunFailed,
    RunResumed,
    RunStarted,
    SkeletonProposed,
    TaskDispatched,
    TaskDone,
    TaskEscalated,
    TaskFailed,
    TaskRef,
    TaskRetried,
    read_events,
    replay,
)

TS = "2026-06-10T00:00:00+00:00"


def _full_run() -> list:
    s = iter(range(1000))
    return [
        RunStarted(seq=next(s), ts=TS, run_id="r1", job_path="job.md"),
        PlannerCallStarted(seq=next(s), ts=TS),
        PlannerCallSucceeded(seq=next(s), ts=TS, tool="propose_skeleton"),
        SkeletonProposed(
            seq=next(s),
            ts=TS,
            phases=[PhaseRef(id="P1", title="build"), PhaseRef(id="P2", title="review")],
        ),
        PhaseStarted(seq=next(s), ts=TS, phase_id="P1"),
        PlannerCallStarted(seq=next(s), ts=TS),
        PlannerCallSucceeded(seq=next(s), ts=TS, tool="implement"),
        EpochStarted(
            seq=next(s),
            ts=TS,
            phase_id="P1",
            epoch_id="E1",
            title="build it",
            tasks=[TaskRef(id="T1", mode="implement"), TaskRef(id="T2", mode="implement")],
        ),
        TaskDispatched(seq=next(s), ts=TS, epoch_id="E1", task_id="T1"),
        TaskDispatched(seq=next(s), ts=TS, epoch_id="E1", task_id="T2"),
        TaskRetried(seq=next(s), ts=TS, epoch_id="E1", task_id="T1", attempt=1),
        HandoffRejected(seq=next(s), ts=TS, epoch_id="E1", task_id="T2", reason="bad json"),
        TaskDone(seq=next(s), ts=TS, epoch_id="E1", task_id="T1"),
        TaskEscalated(seq=next(s), ts=TS, epoch_id="E1", task_id="T2", tier="cloud"),
        TaskFailed(seq=next(s), ts=TS, epoch_id="E1", task_id="T2"),
        EpochCompleted(seq=next(s), ts=TS, epoch_id="E1"),
        PhasePassed(seq=next(s), ts=TS, phase_id="P1"),
        PhaseStarted(seq=next(s), ts=TS, phase_id="P2"),
        PlannerCallStarted(seq=next(s), ts=TS),
        PlannerCallFailed(seq=next(s), ts=TS, classification="transient"),
        PlannerCallStarted(seq=next(s), ts=TS),
        PlannerCallSucceeded(seq=next(s), ts=TS, tool="review"),
        EpochStarted(
            seq=next(s),
            ts=TS,
            phase_id="P2",
            epoch_id="E2",
            title="review it",
            tasks=[TaskRef(id="T1", mode="review")],
        ),
        TaskDispatched(seq=next(s), ts=TS, epoch_id="E2", task_id="T1"),
        TaskDone(seq=next(s), ts=TS, epoch_id="E2", task_id="T1"),
        EpochCompleted(seq=next(s), ts=TS, epoch_id="E2"),
        PhasePassed(seq=next(s), ts=TS, phase_id="P2"),
        PlannerCallStarted(seq=next(s), ts=TS),
        PlannerCallSucceeded(seq=next(s), ts=TS, tool="complete_run"),
        RunCompleted(seq=next(s), ts=TS),
    ]


def test_replay_renders_full_tree(tmp_path: Path) -> None:
    events = _full_run()
    path = tmp_path / "events.ndjson"
    with JournalWriter(path) as writer:
        for ev in events:
            writer.append(ev)

    tree = replay(read_events(path))

    assert tree.run_id == "r1"
    assert tree.job_path == "job.md"
    assert tree.status == "completed"
    assert tree.planner_calls == 5

    assert [(p.id, p.status) for p in tree.phases] == [("P1", "passed"), ("P2", "passed")]

    p1 = tree.phases[0]
    assert [(e.id, e.status) for e in p1.epochs] == [("E1", "completed")]
    e1 = p1.epochs[0]
    assert [(t.id, t.status, t.attempt) for t in e1.tasks] == [
        ("T1", "done", 1),
        ("T2", "failed", 0),
    ]

    p2 = tree.phases[1]
    assert [(t.id, t.status) for t in p2.epochs[0].tasks] == [("T1", "done")]


def test_phases_revised_replaces_unentered_tail() -> None:
    events = [
        RunStarted(seq=0, ts=TS, run_id="r", job_path="j"),
        SkeletonProposed(
            seq=1,
            ts=TS,
            phases=[
                PhaseRef(id="P1", title="a"),
                PhaseRef(id="P2", title="b"),
                PhaseRef(id="P3", title="c"),
            ],
        ),
        PhaseStarted(seq=2, ts=TS, phase_id="P1"),
        PhasesRevised(
            seq=3,
            ts=TS,
            reason="re-scope tail",
            phases=[PhaseRef(id="P2", title="b2"), PhaseRef(id="P4", title="d")],
        ),
    ]
    tree = replay(events)
    assert [(p.id, p.title, p.status) for p in tree.phases] == [
        ("P1", "a", "started"),
        ("P2", "b2", "pending"),
        ("P4", "d", "pending"),
    ]


def test_run_resumed_and_escalated_status() -> None:
    resumed = replay(
        [
            RunStarted(seq=0, ts=TS, run_id="r", job_path="j"),
            RunResumed(seq=1, ts=TS, run_id="r"),
        ]
    )
    assert resumed.status == "running"
    escalated = replay(
        [
            RunStarted(seq=0, ts=TS, run_id="r", job_path="j"),
            RunEscalated(seq=1, ts=TS, reason="ambiguous spec"),
        ]
    )
    assert escalated.status == "escalated"


def test_run_started_carries_planner_cap() -> None:
    capped = replay([RunStarted(seq=0, ts=TS, run_id="r", job_path="j", max_planner_calls=96)])
    assert capped.planner_cap == 96
    # default None when unset (old journals / valve-off test seam) -> no "/N" shown
    uncapped = replay([RunStarted(seq=0, ts=TS, run_id="r", job_path="j")])
    assert uncapped.planner_cap is None


def test_run_failed_is_terminal_with_reason() -> None:
    # The production planner-call cap trips _fail_valve, which now emits a
    # vocabulary event so the journal is self-describing (the TUI can exit).
    tree = replay(
        [
            RunStarted(seq=0, ts=TS, run_id="r", job_path="j"),
            RunFailed(seq=1, ts="2026-06-10T00:00:09+00:00", reason="safety valve: 96 calls"),
        ]
    )
    assert tree.status == "failed"
    assert tree.escalation_reason == "safety valve: 96 calls"
    assert tree.ended_ts == "2026-06-10T00:00:09+00:00"


def test_run_failed_round_trips(tmp_path: Path) -> None:
    events = [
        RunStarted(seq=0, ts=TS, run_id="r", job_path="j"),
        RunFailed(seq=1, ts=TS, reason="safety valve"),
    ]
    path = tmp_path / "events.ndjson"
    with JournalWriter(path) as writer:
        for ev in events:
            writer.append(ev)
    assert read_events(path) == events


def test_replay_threads_timing_and_planner_state() -> None:
    events = [
        RunStarted(seq=0, ts="2026-06-10T00:00:00+00:00", run_id="r", job_path="j"),
        SkeletonProposed(seq=1, ts=TS, phases=[PhaseRef(id="P1", title="build")]),
        PhaseStarted(seq=2, ts="2026-06-10T00:00:05+00:00", phase_id="P1"),
        EpochStarted(
            seq=3, ts="2026-06-10T00:00:06+00:00", phase_id="P1", epoch_id="E1",
            title="t", tasks=[TaskRef(id="T1", mode="implement"),
                              TaskRef(id="T2", mode="implement")],
        ),
        TaskDispatched(seq=4, ts="2026-06-10T00:00:07+00:00", epoch_id="E1", task_id="T1"),
        TaskDone(seq=5, ts="2026-06-10T00:00:30+00:00", epoch_id="E1", task_id="T1"),
        TaskDispatched(seq=6, ts="2026-06-10T00:00:07+00:00", epoch_id="E1", task_id="T2"),
        TaskEscalated(seq=7, ts="2026-06-10T00:00:40+00:00", epoch_id="E1", task_id="T2", tier="senior"),
        PlannerCallStarted(seq=8, ts=TS),
        PlannerCallFailed(seq=9, ts=TS, classification="rate_limit"),
        PlannerCallStarted(seq=10, ts=TS),
        PlannerCallSucceeded(seq=11, ts="2026-06-10T00:01:00+00:00", tool="complete_run"),
    ]
    tree = replay(events)
    assert tree.started_ts == "2026-06-10T00:00:00+00:00"
    assert tree.last_ts == "2026-06-10T00:01:00+00:00"
    # last planner outcome was success -> tool known, prior failure cleared
    assert tree.last_planner_tool == "complete_run"
    assert tree.last_planner_failure is None
    assert tree.planner_waiting is False

    p1 = tree.phases[0]
    assert p1.started_ts == "2026-06-10T00:00:05+00:00"
    e1 = p1.epochs[0]
    assert e1.started_ts == "2026-06-10T00:00:06+00:00"
    t1, t2 = e1.tasks
    assert (t1.started_ts, t1.ended_ts) == ("2026-06-10T00:00:07+00:00", "2026-06-10T00:00:30+00:00")
    assert t2.ended_ts == "2026-06-10T00:00:40+00:00"
    assert t2.note == "→ senior"  # escalation tier surfaced on the task


def test_replay_tracks_planner_waiting_and_failure() -> None:
    waiting = replay(
        [
            RunStarted(seq=0, ts=TS, run_id="r", job_path="j"),
            PlannerCallStarted(seq=1, ts=TS),
        ]
    )
    assert waiting.planner_waiting is True  # started, no outcome yet
    failed = replay(
        [
            RunStarted(seq=0, ts=TS, run_id="r", job_path="j"),
            PlannerCallStarted(seq=1, ts=TS),
            PlannerCallFailed(seq=2, ts=TS, classification="rate_limit"),
        ]
    )
    assert failed.planner_waiting is False
    assert failed.last_planner_failure == "rate_limit"


def test_handoff_rejection_reason_surfaces_until_resolved() -> None:
    base = [
        RunStarted(seq=0, ts=TS, run_id="r", job_path="j"),
        SkeletonProposed(seq=1, ts=TS, phases=[PhaseRef(id="P1", title="b")]),
        PhaseStarted(seq=2, ts=TS, phase_id="P1"),
        EpochStarted(
            seq=3, ts=TS, phase_id="P1", epoch_id="E1", title="t",
            tasks=[TaskRef(id="T1", mode="implement")],
        ),
        TaskDispatched(seq=4, ts=TS, epoch_id="E1", task_id="T1"),
        HandoffRejected(seq=5, ts=TS, epoch_id="E1", task_id="T1", reason="missing handoff.json"),
    ]
    rejected = replay(base)
    assert rejected.phases[0].epochs[0].tasks[0].note == "missing handoff.json"
    # a subsequent success clears the (now-stale) rejection note
    resolved = replay(base + [TaskDone(seq=6, ts=TS, epoch_id="E1", task_id="T1")])
    assert resolved.phases[0].epochs[0].tasks[0].note is None


def test_round_trip_every_event_type(tmp_path: Path) -> None:
    events = _full_run() + [
        RunResumed(seq=500, ts=TS, run_id="r1"),
        RunEscalated(seq=501, ts=TS, reason="x"),
        PhaseEscalated(seq=502, ts=TS, phase_id="P2"),
        PhasesRevised(seq=503, ts=TS, reason="r", phases=[PhaseRef(id="P3", title="t")]),
    ]
    path = tmp_path / "events.ndjson"
    with JournalWriter(path) as writer:
        for ev in events:
            writer.append(ev)
    assert read_events(path) == events


def test_writer_enforces_monotonic_seq(tmp_path: Path) -> None:
    path = tmp_path / "events.ndjson"
    with JournalWriter(path) as writer:
        writer.append(RunStarted(seq=0, ts=TS, run_id="r", job_path="j"))
        with pytest.raises(ValueError):
            writer.append(PlannerCallStarted(seq=0, ts=TS))
        with pytest.raises(ValueError):
            writer.append(PlannerCallStarted(seq=-1, ts=TS))


def test_writer_resumes_seq_floor_across_reopen(tmp_path: Path) -> None:
    path = tmp_path / "events.ndjson"
    with JournalWriter(path) as writer:
        writer.append(RunStarted(seq=0, ts=TS, run_id="r", job_path="j"))
        writer.append(PlannerCallStarted(seq=1, ts=TS))
    with JournalWriter(path) as writer:
        with pytest.raises(ValueError):
            writer.append(PlannerCallStarted(seq=1, ts=TS))
        writer.append(PlannerCallStarted(seq=2, ts=TS))
    assert [e.seq for e in read_events(path)] == [0, 1, 2]


def test_reader_tolerates_truncated_final_line(tmp_path: Path) -> None:
    path = tmp_path / "events.ndjson"
    with JournalWriter(path) as writer:
        writer.append(RunStarted(seq=0, ts=TS, run_id="r", job_path="j"))
        writer.append(PlannerCallStarted(seq=1, ts=TS))
    # Simulate a crash mid-write: a partial JSON line with no trailing newline.
    with open(path, "a", encoding="utf-8") as fh:
        fh.write('{"seq": 2, "ts": "2026", "ev')
    events = read_events(path)
    assert [e.seq for e in events] == [0, 1]


def test_emit_is_thread_safe_under_concurrency(tmp_path: Path) -> None:
    """S2 ruling 2: concurrent ``emit`` assigns the next seq + writes atomically,
    so seq stays strictly monotonic and no event is lost or duplicated even when
    many tasks journal at once.
    """

    path = tmp_path / "events.ndjson"
    n_threads, per_thread = 8, 200
    total = n_threads * per_thread
    with JournalWriter(path) as writer:
        writer.append(RunStarted(seq=0, ts=TS, run_id="r", job_path="j"))

        def hammer() -> None:
            for _ in range(per_thread):
                writer.emit(lambda s: PlannerCallStarted(seq=s, ts=TS))

        with ThreadPoolExecutor(max_workers=n_threads) as pool:
            for fut in [pool.submit(hammer) for _ in range(n_threads)]:
                fut.result()

    seqs = [e.seq for e in read_events(path)]
    assert len(seqs) == total + 1  # +1 for run_started
    assert seqs == list(range(total + 1))  # contiguous, strictly monotonic, unique


def test_reader_raises_on_corrupt_interior_line(tmp_path: Path) -> None:
    path = tmp_path / "events.ndjson"
    with JournalWriter(path) as writer:
        writer.append(RunStarted(seq=0, ts=TS, run_id="r", job_path="j"))
    with open(path, "a", encoding="utf-8") as fh:
        fh.write("not json at all\n")
        fh.write('{"seq": 2, "ts": "2026", "event": "run_completed"}\n')
    with pytest.raises(ValueError):
        read_events(path)


def test_journal_writer_heals_crash_torn_final_line(tmp_path: Path) -> None:
    """A crash mid-write leaves a torn final line (no terminating newline). The
    writer must TRUNCATE it before appending, otherwise the next event fuses onto
    the partial bytes into a corrupt non-final line that every later read rejects
    (a recoverable partial write would become permanent loss of resume/replay)."""

    path = tmp_path / "events.ndjson"
    good = RunStarted(seq=0, ts=TS, run_id="r1", job_path="job.md").model_dump_json()
    # one complete line + a torn partial final line (crash mid-write, no newline):
    path.write_text(good + "\n" + '{"seq":1,"ts":"2026-', encoding="utf-8")
    with JournalWriter(path) as j:
        j.append(PlannerCallStarted(seq=1, ts=TS))  # append AFTER the torn tail
    events = read_events(path)
    assert [e.seq for e in events] == [0, 1]  # healed: both parse, no corruption
