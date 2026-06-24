"""Event journal: append/read round-trip, monotonic seq, torn-tail heal, and the
phase-free replay fold over the bones taxonomy."""

from __future__ import annotations

from pathlib import Path

import pytest

from grindstone.events import (
    EpochCompleted,
    EpochStarted,
    Event,
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


def _seed(path: Path) -> list[Event]:
    with JournalWriter(path) as jw:
        jw.emit(lambda s: RunStarted(seq=s, ts="2026-06-24T00:00:00", run_id="r1",
                                     job_path="job.md", max_epochs=10))
        jw.emit(lambda s: EpochStarted(
            seq=s, ts="2026-06-24T00:00:01", epoch_id="E1", title="build",
            tasks=[TaskRef(id="T1", mode="implement"), TaskRef(id="T2", mode="research")],
        ))
        jw.emit(lambda s: TaskDispatched(seq=s, ts="2026-06-24T00:00:02",
                                         epoch_id="E1", task_id="T1"))
        jw.emit(lambda s: WorkGateRejected(seq=s, ts="2026-06-24T00:00:03",
                                          epoch_id="E1", task_id="T2", reason="no citation"))
        jw.emit(lambda s: Verdict(seq=s, ts="2026-06-24T00:00:04", epoch_id="E1",
                                  task_id="T1", outcome="PASS", reason="ok"))
        jw.emit(lambda s: TaskDone(seq=s, ts="2026-06-24T00:00:05",
                                   epoch_id="E1", task_id="T1"))
    return read_events(path)


def test_append_read_roundtrip(tmp_path: Path) -> None:
    events = _seed(tmp_path / "events.ndjson")
    assert [e.event for e in events][:2] == ["run_started", "epoch_started"]
    assert [e.seq for e in events] == list(range(6))


def test_monotonic_seq_enforced(tmp_path: Path) -> None:
    path = tmp_path / "events.ndjson"
    with JournalWriter(path) as jw:
        jw.append(RunStarted(seq=5, ts="t", run_id="r", job_path="j"))
        with pytest.raises(ValueError):
            jw.append(RunStarted(seq=5, ts="t", run_id="r", job_path="j"))


def test_torn_tail_is_healed(tmp_path: Path) -> None:
    path = tmp_path / "events.ndjson"
    _seed(path)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write('{"event": "run_completed", "seq": 99')  # torn, no newline
    # read tolerates it ...
    assert len(read_events(path)) == 6
    # ... and a re-opened writer heals it so the next append stays parseable.
    with JournalWriter(path) as jw:
        jw.emit(lambda s: RunCompleted(seq=s, ts="2026-06-24T00:00:06"))
    events = read_events(path)
    assert events[-1].event == "run_completed"


def test_replay_builds_phase_free_tree(tmp_path: Path) -> None:
    events = _seed(tmp_path / "events.ndjson")
    tree = replay(events)
    assert tree.run_id == "r1"
    assert tree.status == "running"
    assert tree.max_epochs == 10
    assert len(tree.epochs) == 1
    epoch = tree.epochs[0]
    assert epoch.id == "E1"
    t1 = next(t for t in epoch.tasks if t.id == "T1")
    t2 = next(t for t in epoch.tasks if t.id == "T2")
    assert t1.status == "done"  # done clears the earlier verdict note
    assert t1.note is None
    assert t2.status == "gate_rejected"
    assert t2.note == "no citation"


def test_replay_run_ended_and_rate_limit(tmp_path: Path) -> None:
    path = tmp_path / "events.ndjson"
    with JournalWriter(path) as jw:
        jw.emit(lambda s: RunStarted(seq=s, ts="t0", run_id="r2", job_path="j"))
        jw.emit(lambda s: RateLimited(seq=s, ts="t1", role="planner", detail="429"))
        jw.emit(lambda s: RunEnded(seq=s, ts="t2", summary="C still pending"))
    tree = replay(read_events(path))
    assert tree.status == "ended"
    assert tree.end_summary == "C still pending"
    assert tree.last_rate_limit == "planner: 429"


def test_replay_resume_marker(tmp_path: Path) -> None:
    path = tmp_path / "events.ndjson"
    with JournalWriter(path) as jw:
        jw.emit(lambda s: RunStarted(seq=s, ts="t0", run_id="r3", job_path="j"))
        jw.emit(lambda s: EpochStarted(seq=s, ts="t1", epoch_id="E1", title="x",
                                       tasks=[TaskRef(id="T1", mode="implement")]))
        jw.emit(lambda s: EpochCompleted(seq=s, ts="t2", epoch_id="E1"))
        jw.emit(lambda s: RunResumed(seq=s, ts="t3", run_id="r3", razed_epoch="E2"))
    events = read_events(path)
    assert any(isinstance(e, RunResumed) and e.razed_epoch == "E2" for e in events)
    tree = replay(events)
    assert tree.status == "running"
    assert tree.epochs[0].status == "completed"


def test_replay_rejects_event_before_run_started(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        replay([EpochCompleted(seq=0, ts="t", epoch_id="E1")])
