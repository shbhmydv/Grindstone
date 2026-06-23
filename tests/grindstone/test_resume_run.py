"""Run-level resume (S3 ruling 6): kill-mid-planner-call re-issues (no burn),
crafted ``awaiting_planner`` re-enters the loop, and a real SIGKILL while a
planner call blocks at a file sentinel resumes to a clean completion.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from grindstone import worktree as wt
from grindstone.contracts.models import Phase
from grindstone.epoch_loop import EpochState, IntegrationState
from grindstone.events import (
    EpochStarted,
    JournalWriter,
    PhaseRef,
    PhaseStarted,
    PlannerCallStarted,
    PlannerCallSucceeded,
    RunStarted,
    SkeletonProposed,
    TaskDispatched,
    TaskRef,
    read_events,
    replay,
)
from grindstone.mock_planner import MockPlanner
from grindstone.rundir import RunDir
from grindstone.run_loop import RunState, resume_grind
from grindstone.task_loop import TaskCursorState, TaskIdentity
from grindstone.verify import WorkerTaskVerifier

from tests.grindstone.conftest import (
    OwnershipWorker,
    check_cmd,
    complete_decision,
    handle_failed_epoch_halt,
    impl_task,
    implement_decision,
    phase_complete_decision,
    phase_dict,
    reap_kill_target,
    tracked_files,
)
from tests.grindstone.test_epoch_verification import _VerifierWorker

_KILL_TARGET = Path(__file__).resolve().parent / "_kill_planner_target.py"
_KILL_PHASE_TARGET = Path(__file__).resolve().parent / "_kill_phase_target.py"
TS = "2026-06-10T00:00:00+00:00"


def _ladder() -> list[tuple[str, OwnershipWorker]]:
    return [("worker", OwnershipWorker())]


def _craft_awaiting_planner(run_dir: RunDir) -> None:
    """A run killed right after the skeleton was accepted (awaiting the next call)."""

    with JournalWriter(run_dir.events_path) as j:
        j.append(RunStarted(seq=0, ts=TS, run_id="r", job_path="job.md"))
        j.append(
            SkeletonProposed(
                seq=1, ts=TS,
                phases=[PhaseRef(id="P1", title="build"), PhaseRef(id="P2", title="verify")],
            )
        )
        j.append(PhaseStarted(seq=2, ts=TS, phase_id="P1"))
        j.append(PlannerCallStarted(seq=3, ts=TS))
        j.append(PlannerCallSucceeded(seq=4, ts=TS, tool="propose_skeleton"))
    state = RunState(
        run_id="r", job_path="job.md", job_text="toy job", status="awaiting_planner",
        skeleton=[Phase.model_validate(phase_dict("P1")), Phase.model_validate(phase_dict("P2"))],
        current_phase_id="P1", epoch_counter=0, planner_call_count=1, rate_limit_waits=0,
        last_integration_branch=None, pending_decision=None, terminal_reason=None,
    )
    run_dir.run_state_path.write_text(state.model_dump_json(), encoding="utf-8")


def test_resume_awaiting_planner_reissues_and_completes(git_repo: Path, run_dir: RunDir) -> None:
    _craft_awaiting_planner(run_dir)
    planner = MockPlanner(
        script=[
            implement_decision(impl_task("T1", "f1.txt")),
            complete_decision(check_cmd("test -f f1.txt")),
        ]
    )
    outcome = resume_grind(run_dir, planner=planner, ladder=_ladder(), repo=git_repo)
    assert outcome.status == "completed"
    # The pre-kill skeleton call (1) plus the two re-issued calls = 3.
    assert outcome.planner_calls == 3
    events = read_events(run_dir.events_path)
    kinds = [e.event for e in events]
    assert kinds.count("run_resumed") == 1
    assert kinds.count("skeleton_proposed") == 1  # NOT re-proposed on resume
    assert replay(events).status == "completed"


def test_resume_of_terminal_run_is_idempotent(git_repo: Path, run_dir: RunDir) -> None:
    _craft_awaiting_planner(run_dir)
    planner = MockPlanner(
        script=[
            implement_decision(impl_task("T1", "f1.txt")),
            complete_decision(check_cmd("test -f f1.txt")),
        ]
    )
    resume_grind(run_dir, planner=planner, ladder=_ladder(), repo=git_repo)
    # A second resume sees a completed run and returns without touching anything.
    again = resume_grind(
        run_dir, planner=MockPlanner(script=[]), ladder=_ladder(), repo=git_repo
    )
    assert again.status == "completed"


# --- running_epoch resume re-verifies the finished in-flight epoch (hole #2) ---


def _craft_running_epoch(repo: Path, run_dir: RunDir) -> dict[str, object]:
    """A run killed with status=running_epoch: the epoch-level state has T1 in flight
    and the run-level state holds the pending implement decision (carrying criteria).
    Returns the decision dict so the test can assert it drives the resume re-verify."""

    decision = implement_decision(
        {"id": "T1", "goal": "create f1.txt",
         "done_when": [check_cmd("test -f f1.txt")],
         "criteria": ["f1.txt is correct"], "file_ownership": ["f1.txt"]}
    )
    ident = TaskIdentity(run_dir.root.name, "P1", "E1", "T1")
    with JournalWriter(run_dir.events_path) as journal:
        journal.append(RunStarted(seq=0, ts=TS, run_id="r", job_path="job.md"))
        journal.append(SkeletonProposed(seq=1, ts=TS, phases=[PhaseRef(id="P1", title="build")]))
        journal.append(PhaseStarted(seq=2, ts=TS, phase_id="P1"))
        journal.append(
            EpochStarted(seq=3, ts=TS, phase_id="P1", epoch_id="E1", title="impl",
                         tasks=[TaskRef(id="T1", mode="implement")])
        )
        journal.append(TaskDispatched(seq=4, ts=TS, epoch_id="E1", task_id="T1"))
    cursor = TaskCursorState(
        fq_task_id="P1/E1/T1", task_id="T1", mode="implement", status="running",
        tier_index=0, tier_name="worker", tier_attempt=1, attempt=1,
        scratch=str(run_dir.root / "worktrees" / "T1" / "attempt-1"),
        branch=ident.attempt_branch(1), failure_context=[], reason=None,
    )
    epoch_state = EpochState(
        phase_id="P1", epoch_id="E1", title="impl", mode="implement", is_implement=True,
        base=wt.head_commit(repo),
        integration=IntegrationState(
            branch=f"grind-wip/{run_dir.root.name}/P1/E1/_staging",
            status="pending", merged=[], conflict=None,
        ),
        tasks={"T1": cursor},
    )
    run_dir.state_path.write_text(epoch_state.model_dump_json(), encoding="utf-8")
    run_state = RunState(
        run_id="r", job_path="job.md", job_text="toy", status="running_epoch",
        skeleton=[Phase.model_validate(phase_dict("P1"))], current_phase_id="P1",
        epoch_counter=0, planner_call_count=2, rate_limit_waits=0,
        last_integration_branch=None, pending_decision=decision, terminal_reason=None,
    )
    run_dir.run_state_path.write_text(run_state.model_dump_json(), encoding="utf-8")
    return decision


def test_resume_running_epoch_verifies_inflight_epoch(git_repo: Path, run_dir: RunDir) -> None:
    """A running_epoch resume finishes the in-flight epoch through resume_epoch; the
    resumed task must still be agentically VERIFIED at its tier. Here the verifier always
    fails, so the resumed task's verification fails, its retry ladder exhausts, the task
    FAILS, the epoch fails, and the gap routes through handle_failed_epoch (per-task
    verification, so the in-flight task's criteria are never silently skipped)."""

    _craft_running_epoch(git_repo, run_dir)
    gaps = ["f1.txt is wrong"]
    planner = MockPlanner(script=[handle_failed_epoch_halt("unverified gap")])
    outcome = resume_grind(
        run_dir, planner=planner, ladder=[("worker", OwnershipWorker())], repo=git_repo,
        verifiers={"worker": WorkerTaskVerifier(_VerifierWorker(passed=False, gaps=gaps))},
    )
    assert outcome.status == "escalated"
    kinds = [e.event for e in read_events(run_dir.events_path)]
    assert "task_verification_started" in kinds  # the in-flight task WAS verified
    assert "task_verification_failed" in kinds
    assert "epoch_failed" in kinds  # the unsatisfiable criterion failed the task -> epoch


# --- signature test: SIGKILL mid-planner-call, then resume ---------------------


def _busy_wait(predicate: object, *, deadline_s: float) -> bool:
    end = time.monotonic() + deadline_s
    while time.monotonic() < end:
        if predicate():  # type: ignore[operator]
            return True
        os.sched_yield()
    return False


def test_kill_mid_planner_call_then_resume(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    run_id = "killed-planner"
    ready = tmp_path / "ready"
    release = tmp_path / "release"  # never created
    run_dir = RunDir(root=repo / ".grindstone" / "runs" / run_id)

    proc = subprocess.Popen(
        [sys.executable, str(_KILL_TARGET), str(repo), run_id, str(ready), str(release)]
    )

    def blocked_in_first_call() -> bool:
        if not (ready.exists() and run_dir.run_state_path.exists()):
            return False
        try:
            state = RunState.model_validate_json(run_dir.run_state_path.read_text())
        except (ValueError, OSError):
            return False
        return state.status == "awaiting_planner" and state.planner_call_count >= 1

    try:
        assert _busy_wait(blocked_in_first_call, deadline_s=60.0), "never reached the planner call"
        os.kill(proc.pid, signal.SIGKILL)
    finally:
        reap_kill_target(proc)
    assert proc.returncode == -signal.SIGKILL

    pre = RunState.model_validate_json(run_dir.run_state_path.read_text())
    assert pre.status == "awaiting_planner" and pre.skeleton is None  # blocked on the FIRST call
    pre_events = read_events(run_dir.events_path)
    assert sum(1 for e in pre_events if e.event == "planner_call_started") == 1
    # The in-flight call left NO outcome on disk (no succeeded/failed for it).
    assert sum(1 for e in pre_events if e.event == "planner_call_succeeded") == 0
    assert sum(1 for e in pre_events if e.event == "planner_call_failed") == 0

    # Resume with a real planner: the killed call is re-issued, not burned.
    planner = MockPlanner(
        script=[
            {"schema_version": "1", "tool": "propose_skeleton",
             "args": {"phases": [phase_dict("P1"), phase_dict("P2")]}},
            implement_decision(impl_task("T1", "f1.txt")),
            complete_decision(check_cmd("test -f f1.txt")),
        ]
    )
    outcome = resume_grind(run_dir, planner=planner, ladder=_ladder(), repo=repo)
    assert outcome.status == "completed"
    # 1 burned-but-counted pre-kill start + 3 re-issued = 4 planner_call_started.
    started = sum(1 for e in read_events(run_dir.events_path) if e.event == "planner_call_started")
    assert started == 4
    assert outcome.final_branch is not None
    assert "f1.txt" in tracked_files(repo, outcome.final_branch)
    assert replay(read_events(run_dir.events_path)).status == "completed"


# --- signature test: SIGKILL mid-phase-transition, then resume (S4 ruling 6) ----


def _phase_passed_count(events: list[object], phase_id: str) -> int:
    return sum(
        1 for e in events
        if getattr(e, "event", "") == "phase_passed" and getattr(e, "phase_id", "") == phase_id
    )


def test_kill_mid_phase_transition_then_resume(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    run_id = "killed-phase"
    ready = tmp_path / "ready"
    release = tmp_path / "release"  # never created
    run_dir = RunDir(root=repo / ".grindstone" / "runs" / run_id)

    proc = subprocess.Popen(
        [sys.executable, str(_KILL_PHASE_TARGET), str(repo), run_id, str(ready), str(release)]
    )

    def transitioned() -> bool:
        if not (ready.exists() and run_dir.run_state_path.exists()):
            return False
        try:
            state = RunState.model_validate_json(run_dir.run_state_path.read_text())
        except (ValueError, OSError):
            return False
        # P1 passed + advanced to P2, blocked awaiting the next (P2) planner call.
        return state.current_phase_id == "P2" and state.status == "awaiting_planner"

    try:
        assert _busy_wait(transitioned, deadline_s=60.0), "never reached the phase transition"
        os.kill(proc.pid, signal.SIGKILL)
    finally:
        reap_kill_target(proc)
    assert proc.returncode == -signal.SIGKILL

    pre = RunState.model_validate_json(run_dir.run_state_path.read_text())
    assert pre.current_phase_id == "P2" and pre.passed_phase_ids == ["P1"]
    pre_events = read_events(run_dir.events_path)
    assert _phase_passed_count(pre_events, "P1") == 1  # already on disk pre-kill

    # Resume: the re-issued P2 call lands, P2's epoch builds f2.txt, the planner
    # phase_complete's P2, then completes the run.
    planner = MockPlanner(
        script=[
            implement_decision(impl_task("T1", "f2.txt")),
            phase_complete_decision("f2.txt"),
            complete_decision(check_cmd("test -f f1.txt"), check_cmd("test -f f2.txt")),
        ]
    )
    outcome = resume_grind(run_dir, planner=planner, ladder=_ladder(), repo=repo)
    assert outcome.status == "completed"
    final = read_events(run_dir.events_path)
    # Idempotent re-evaluation: NO duplicate phase_passed for the already-passed P1.
    assert _phase_passed_count(final, "P1") == 1
    assert _phase_passed_count(final, "P2") == 1
    assert outcome.final_branch is not None
    files = tracked_files(repo, outcome.final_branch)
    assert "f1.txt" in files and "f2.txt" in files
    assert replay(final).status == "completed"
