"""Kill-mid-epoch resume (ARCHITECTURE.md / ruling 9): in-flight tasks are burned,
terminal tasks stay terminal (DONE tasks are never re-dispatched), integration
finishes idempotently, and the journal replays coherently.

A fast crafted-state unit test pins the burn-and-continue semantics; the rung's
signature test drives a real subprocess to SIGKILL after >=1 task DONE and >=1
in flight.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from grindstone import worktree as wt
from grindstone.epoch_loop import (
    EpochState,
    IntegrationState,
)
from grindstone.events import (
    EpochStarted,
    JournalWriter,
    PhaseRef,
    PhaseStarted,
    RunStarted,
    SkeletonProposed,
    TaskDispatched,
    TaskRef,
    read_events,
    replay,
)
from grindstone.rundir import RunDir
from grindstone.task_loop import TaskCursorState, TaskIdentity

from tests.grindstone.conftest import (
    OUT_FILE,
    implement_epoch,
    make_ok_worker,
    make_toy_task,
    reap_kill_target,
    resume_one_epoch,
    tracked_files,
)

_KILL_TARGET = Path(__file__).resolve().parent / "_kill_target.py"
TS = "2026-06-10T00:00:00+00:00"


def _craft_inflight_single(repo: Path, run_dir: RunDir) -> None:
    """Write the journal + state.json a kill leaves with T1 in flight (attempt 1)."""

    ident = TaskIdentity(run_dir.root.name, "P1", "E1", "T1")
    with JournalWriter(run_dir.events_path) as journal:
        journal.append(RunStarted(seq=0, ts=TS, run_id="r", job_path="j"))
        journal.append(SkeletonProposed(seq=1, ts=TS, phases=[PhaseRef(id="P1", title="t")]))
        journal.append(PhaseStarted(seq=2, ts=TS, phase_id="P1"))
        journal.append(
            EpochStarted(
                seq=3,
                ts=TS,
                phase_id="P1",
                epoch_id="E1",
                title="t",
                tasks=[TaskRef(id="T1", mode="implement")],
            )
        )
        journal.append(TaskDispatched(seq=4, ts=TS, epoch_id="E1", task_id="T1"))
    cursor = TaskCursorState(
        fq_task_id="P1/E1/T1",
        task_id="T1",
        mode="implement",
        status="running",
        tier_index=0,
        tier_name="local",
        tier_attempt=1,
        attempt=1,
        scratch=str(run_dir.root / "worktrees" / "T1" / "attempt-1"),
        branch=ident.attempt_branch(1),
        failure_context=[],
        reason=None,
    )
    state = EpochState(
        phase_id="P1",
        epoch_id="E1",
        title="t",
        mode="implement",
        is_implement=True,
        base=wt.head_commit(repo),
        integration=IntegrationState(
            branch=f"grind/{run_dir.root.name}/P1/E1/_integration",
            status="pending", merged=[], conflict=None,
        ),
        tasks={"T1": cursor},
    )
    run_dir.state_path.write_text(state.model_dump_json(), encoding="utf-8")


def test_resume_burns_inflight_and_completes(git_repo: Path, run_dir: RunDir) -> None:
    _craft_inflight_single(git_repo, run_dir)
    outcome = resume_one_epoch(
        run_dir,
        args=implement_epoch(make_toy_task()),
        mode="implement",
        ladder=[("local", make_ok_worker())],
        repo=git_repo,
    )
    assert outcome.status == "completed"
    assert outcome.tasks[0].status == "done"
    # Attempt 1 was burned; success lands on attempt 2.
    assert outcome.tasks[0].attempts == 2
    kinds = [e.event for e in read_events(run_dir.events_path)]
    assert kinds.count("run_started") == 1
    assert kinds.count("run_resumed") == 1
    assert kinds.count("task_done") == 1
    assert kinds.count("handoff_rejected") == 1  # the burned attempt
    branch = outcome.integration.branch
    assert branch is not None
    assert OUT_FILE in tracked_files(git_repo, branch)
    tree = replay(read_events(run_dir.events_path))
    assert tree.status == "completed"
    assert tree.phases[0].epochs[0].tasks[0].status == "done"


def test_resume_monotonic_seq(git_repo: Path, run_dir: RunDir) -> None:
    _craft_inflight_single(git_repo, run_dir)
    resume_one_epoch(
        run_dir,
        args=implement_epoch(make_toy_task()),
        mode="implement",
        ladder=[("local", make_ok_worker())],
        repo=git_repo,
    )
    seqs = [e.seq for e in read_events(run_dir.events_path)]
    assert seqs == sorted(seqs)
    assert len(seqs) == len(set(seqs))


# --- signature test: SIGKILL mid-epoch, then resume ----------------------------


def _busy_wait(predicate: object, *, deadline_s: float) -> bool:
    end = time.monotonic() + deadline_s
    while time.monotonic() < end:
        if predicate():  # type: ignore[operator]
            return True
        os.sched_yield()
    return False


def test_kill_mid_epoch_then_resume(tmp_path: Path) -> None:
    from tests.grindstone.conftest import init_git_repo

    repo = init_git_repo(tmp_path / "repo")
    run_id = "killed-epoch"
    ready = tmp_path / "ready"
    release = tmp_path / "release"  # never created
    run_dir = RunDir(root=repo / ".grindstone" / "runs" / run_id)

    proc = subprocess.Popen(
        [sys.executable, str(_KILL_TARGET), str(repo), run_id, str(ready), str(release)]
    )

    def t1_done_and_t2_inflight() -> bool:
        if not (ready.exists() and run_dir.state_path.exists()):
            return False
        try:
            state = EpochState.model_validate_json(run_dir.state_path.read_text())
        except (ValueError, OSError):
            return False
        return state.tasks.get("T1") is not None and state.tasks["T1"].status == "done"

    try:
        assert _busy_wait(t1_done_and_t2_inflight, deadline_s=60.0), "never reached the kill point"
        os.kill(proc.pid, signal.SIGKILL)
    finally:
        reap_kill_target(proc)
    assert proc.returncode == -signal.SIGKILL

    pre = EpochState.model_validate_json(run_dir.state_path.read_text())
    assert pre.tasks["T1"].status == "done"
    assert pre.tasks["T2"].status == "running"
    t1_dispatches_pre = sum(
        1
        for e in read_events(run_dir.events_path)
        if e.event == "task_dispatched" and getattr(e, "task_id", None) == "T1"
    )
    t1_attempts_pre = pre.tasks["T1"].attempt

    t1 = make_toy_task(task_id="T1", out_file="f1.txt", owned=["f1.txt"])
    t2 = make_toy_task(task_id="T2", out_file="f2.txt", owned=["f2.txt"])
    outcome = resume_one_epoch(
        run_dir,
        args=implement_epoch(t1, t2),
        mode="implement",
        ladder=[("local", make_ok_worker(out_file="f2.txt"))],
        repo=repo,
    )
    assert outcome.status == "completed"
    by_id = {t.task_id: t for t in outcome.tasks}
    assert by_id["T1"].status == "done"
    assert by_id["T2"].status == "done"
    # The DONE task was NOT re-dispatched: its attempt count is unchanged and no
    # new task_dispatched for T1 appears post-resume.
    assert by_id["T1"].attempts == t1_attempts_pre
    t1_dispatches_total = sum(
        1
        for e in read_events(run_dir.events_path)
        if e.event == "task_dispatched" and getattr(e, "task_id", None) == "T1"
    )
    assert t1_dispatches_total == t1_dispatches_pre == 1
    # Coherent terminal replay; both disjoint files integrated.
    tree = replay(read_events(run_dir.events_path))
    assert tree.status == "completed"
    assert {t.status for t in tree.phases[0].epochs[0].tasks} == {"done"}
    branch = outcome.integration.branch
    assert branch is not None
    assert {"f1.txt", "f2.txt"} <= set(tracked_files(repo, branch))
