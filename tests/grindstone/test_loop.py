"""The epoch driver end-to-end, driven by a scripted mock planner + a
concurrency-safe loop worker (no real model). Covers the BONES state machine: a
1-epoch run reaching ``completed``; a multi-epoch run; an EndDecision phase-handoff
(``ended``); the planner-declared setup seam; the disjoint-ownership merge of two
passing implement tasks; an ownership OVERLAP rejected (carried, no fast-forward);
a worker RateLimited parking then the epoch restarting clean; a non-write task
reading the integration tip a prior epoch built; and RESUME from a mid-epoch crash
(raze the in-flight epoch, preserve the completed keyed log, re-plan).
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pytest

from grindstone import worktree as wt
from grindstone.contracts.models import EndDecision, Epoch, EpochDecision, Task
from grindstone.events import (
    EpochCompleted,
    EpochStarted,
    JournalWriter,
    RunResumed,
    RunStarted,
    TaskRef,
    read_events,
    replay,
)
from grindstone.loop import PlannerContext, resume_run, start_run
from grindstone.mock_planner import MockDecisionPlanner
from grindstone.mock_worker import LoopWorker
from grindstone.rundir import RunDir
from grindstone.worker import Backends


# --- builders ------------------------------------------------------------------


def _impl(tid: str, owned: list[str]) -> Task:
    return Task(id=tid, mode="implement", goal=f"build {owned}", file_ownership=owned)


def _review(tid: str, out: str) -> Task:
    return Task(id=tid, mode="review", goal="re-derive and reconcile", artifact_out=out)


def _epoch(*tasks: Task, title: str = "build", setup: list[str] | None = None) -> EpochDecision:
    return EpochDecision(
        kind="epoch",
        epoch=Epoch(title=title, tasks=list(tasks), setup=setup or []),
    )


def _end(summary: str = "all done") -> EndDecision:
    return EndDecision(kind="end", summary=summary)


@pytest.fixture
def job_path(tmp_path: Path) -> Path:
    p = tmp_path / "job.md"
    p.write_text("# job\nbuild the thing\n", encoding="utf-8")
    return p


def _backends(worker: LoopWorker, *, slots: int = 2) -> Backends:
    return Backends.single(worker, slots=slots)


def _no_sleep() -> tuple[list[float], Callable[[float], None]]:
    calls: list[float] = []

    def fake(seconds: float) -> None:
        calls.append(seconds)

    return calls, fake


# --- 1-epoch -> completed ------------------------------------------------------


def test_single_epoch_reaches_completed(
    git_repo: Path, run_dir: RunDir, job_path: Path
) -> None:
    planner = MockDecisionPlanner([_epoch(_impl("T1", ["a.py"])), _end("shipped")])
    result = start_run(
        job_path=job_path, run_dir=run_dir, repo=git_repo, planner=planner,
        backends=_backends(LoopWorker()), max_epochs=5,
    )
    assert result.status == "completed"
    assert result.summary == "shipped"
    assert result.epochs == 1
    # The durable run branch carries the work; the operator checkout is untouched.
    assert "a.py" in wt.list_tree(git_repo, "grind/run-1")
    assert not (git_repo / "a.py").exists()
    tree = replay(read_events(run_dir.events_path))
    assert tree.status == "completed"
    assert [e.status for e in tree.epochs] == ["completed"]


def test_multi_epoch_run(git_repo: Path, run_dir: RunDir, job_path: Path) -> None:
    planner = MockDecisionPlanner(
        [_epoch(_impl("T1", ["a.py"])), _epoch(_impl("T1", ["b.py"])), _end()]
    )
    result = start_run(
        job_path=job_path, run_dir=run_dir, repo=git_repo, planner=planner,
        backends=_backends(LoopWorker()), max_epochs=5,
    )
    assert result.status == "completed" and result.epochs == 2
    tip = wt.list_tree(git_repo, "grind/run-1")
    # E2 was based on E1's integrated tip, so both files survive on the run branch.
    assert "a.py" in tip and "b.py" in tip


# --- EndDecision phase-handoff (ended) -----------------------------------------


def test_end_decision_phase_handoff(
    git_repo: Path, run_dir: RunDir, job_path: Path
) -> None:
    summary = "partial: built the parser, the evaluator still needs a decision"
    planner = MockDecisionPlanner([_end(summary)])
    result = start_run(
        job_path=job_path, run_dir=run_dir, repo=git_repo, planner=planner,
        backends=_backends(LoopWorker()),
        # invariant #2 failing => the planner's end is a clean partial-end, not done.
        acceptance=lambda ctx: False,
    )
    assert result.status == "ended"
    assert result.summary == summary  # persisted as the resume seed
    tree = replay(read_events(run_dir.events_path))
    assert tree.status == "ended" and tree.end_summary == summary


# --- planner-declared setup (the trusted host-mutation seam) -------------------


def test_planner_setup_runs_before_tasks(
    git_repo: Path, run_dir: RunDir, job_path: Path
) -> None:
    planner = MockDecisionPlanner(
        [_epoch(_impl("T1", ["a.py"]), setup=["touch setup-ran"]), _end()]
    )
    result = start_run(
        job_path=job_path, run_dir=run_dir, repo=git_repo, planner=planner,
        backends=_backends(LoopWorker()),
    )
    assert result.status == "completed"
    # The orchestrator ran the planner-declared setup on the host (the repo root).
    assert (git_repo / "setup-ran").is_file()


# --- disjoint-ownership merge of two passing implement tasks -------------------


def test_disjoint_merge_two_tasks(
    git_repo: Path, run_dir: RunDir, job_path: Path
) -> None:
    planner = MockDecisionPlanner(
        [_epoch(_impl("T1", ["a.py"]), _impl("T2", ["b.py"])), _end()]
    )
    result = start_run(
        job_path=job_path, run_dir=run_dir, repo=git_repo, planner=planner,
        backends=_backends(LoopWorker()),
    )
    assert result.status == "completed"
    tip = wt.list_tree(git_repo, "grind/run-1")
    assert "a.py" in tip and "b.py" in tip  # both disjoint tasks merged cleanly


# --- ownership OVERLAP rejected (carried, no fast-forward) ----------------------


def test_ownership_overlap_rejected(
    git_repo: Path, run_dir: RunDir, job_path: Path
) -> None:
    planner = MockDecisionPlanner(
        [_epoch(_impl("T1", ["shared.py"]), _impl("T2", ["shared.py"])), _end()]
    )
    result = start_run(
        job_path=job_path, run_dir=run_dir, repo=git_repo, planner=planner,
        backends=_backends(LoopWorker()),
    )
    assert result.status == "completed"
    # No fast-forward: the colliding work never reached the run branch.
    assert not wt.branch_exists(git_repo, "grind/run-1")
    # The overlap is surfaced to the NEXT planner boundary as carried context.
    second_ctx = planner.contexts[1]
    assert any("overlap" in c for c in second_ctx.carried)


# --- rate limit parks then the epoch restarts clean ----------------------------


def test_rate_limit_parks_then_restarts(
    git_repo: Path, run_dir: RunDir, job_path: Path
) -> None:
    calls, fake = _no_sleep()
    planner = MockDecisionPlanner([_epoch(_impl("T1", ["a.py"])), _end()])
    result = start_run(
        job_path=job_path, run_dir=run_dir, repo=git_repo, planner=planner,
        backends=_backends(LoopWorker(rate_limit_once=True)),
        sleep_fn=fake, backoff_s=7.0,
    )
    assert result.status == "completed"
    assert calls == [7.0]  # parked exactly once at the backoff
    # The restart produced a clean integrated tip (partial work was razed).
    assert "a.py" in wt.list_tree(git_repo, "grind/run-1")
    tree = replay(read_events(run_dir.events_path))
    assert tree.epochs[0].status == "completed"
    assert any(
        e.event == "rate_limited" for e in read_events(run_dir.events_path)
    )


# --- non-write task reads the integration tip ----------------------------------


def test_non_write_reads_integration_tip(
    git_repo: Path, run_dir: RunDir, job_path: Path
) -> None:
    planner = MockDecisionPlanner(
        [
            _epoch(_impl("T1", ["feature.py"]), title="build"),
            _epoch(_review("T1", "P1/E2/T1/review.md"), title="review"),
            _end(),
        ]
    )
    result = start_run(
        job_path=job_path, run_dir=run_dir, repo=git_repo, planner=planner,
        backends=_backends(LoopWorker(read_cite="feature.py")),
    )
    assert result.status == "completed"
    review = run_dir.resolve("P1/E2/T1/review.md").read_text(encoding="utf-8")
    # The review worker read feature.py AT THE INTEGRATION TIP that E1 built; the
    # tip-keyed content proves it saw the in-run state, not the stale base checkout.
    assert "value = 'P1/E1/T1'" in review


# --- resume from a mid-epoch crash ---------------------------------------------


def _fabricate_crashed_run(
    git_repo: Path, run_dir: RunDir, job_path: Path
) -> None:
    """E1 completed + integrated (run branch at a feature.py commit, keyed log
    written), E2 STARTED but never completed (throwaway worktree + wip branch +
    partial keyed log + raw logs), with no run terminal event: a faithful kill."""

    # E1 integrated onto the durable run branch (off base, operator checkout untouched).
    wt.ensure_integration_branch(git_repo, "grind/run-1", "main")
    e1_wt = run_dir.worktrees_root / "_e1build"
    wt.add_worktree_on(git_repo, e1_wt, branch="grind/run-1")
    (e1_wt / "feature.py").write_text("value = 'P1/E1/T1'\n", encoding="utf-8")
    wt.commit_all(e1_wt, "e1: feature")
    wt.remove_worktree(git_repo, e1_wt)

    # E1's durable keyed log (the done-list the re-planned epoch must keep).
    h = run_dir.resolve("P1/E1/T1/handoff.json")
    h.parent.mkdir(parents=True, exist_ok=True)
    h.write_text('{"task_id": "P1/E1/T1", "status": "DONE"}', encoding="utf-8")

    # The journal: E1 done, E2 in flight (no EpochCompleted, no run terminal).
    with JournalWriter(run_dir.events_path) as jw:
        jw.emit(lambda s: RunStarted(seq=s, ts="t0", run_id="run-1",
                                     job_path=str(job_path), max_epochs=9))
        jw.emit(lambda s: EpochStarted(seq=s, ts="t1", epoch_id="P1/E1", title="build",
                                       tasks=[TaskRef(id="T1", mode="implement")]))
        jw.emit(lambda s: EpochCompleted(seq=s, ts="t2", epoch_id="P1/E1"))
        jw.emit(lambda s: EpochStarted(seq=s, ts="t3", epoch_id="P1/E2", title="more",
                                       tasks=[TaskRef(id="T1", mode="implement")]))

    # In-flight E2 transient debris: a registered worktree + wip branch, a partial
    # keyed-log subdir, and a raw-log dir.
    wt.add_worktree(
        git_repo, run_dir.worktrees_root / "P1-E2-T1" / "attempt-1",
        branch="grind-wip/run-1/P1-E2-T1/attempt-1", base="grind/run-1",
    )
    partial = run_dir.resolve("P1/E2/T1")
    partial.mkdir(parents=True, exist_ok=True)
    (partial / "scratch.txt").write_text("half-written\n", encoding="utf-8")
    logs = run_dir.root / "logs" / "P1-E2-T1-worker"
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "stdout.log").write_text("noise\n", encoding="utf-8")


def test_resume_razes_inflight_and_replans(
    git_repo: Path, run_dir: RunDir, job_path: Path
) -> None:
    _fabricate_crashed_run(git_repo, run_dir, job_path)
    planner = MockDecisionPlanner([_end("resumed and finished")])

    result = resume_run(
        run_dir=run_dir, repo=git_repo, planner=planner,
        backends=_backends(LoopWorker()),
    )
    assert result.status == "completed"

    # The in-flight epoch was razed: worktree, wip branch, partial keyed log, raw logs.
    assert not (run_dir.worktrees_root / "P1-E2-T1").exists()
    assert not wt.branch_exists(git_repo, "grind-wip/run-1/P1-E2-T1/attempt-1")
    assert not run_dir.resolve("P1/E2").exists()
    assert not (run_dir.root / "logs" / "P1-E2-T1-worker").exists()

    # The completed-epoch keyed log + the run-branch boundary are PRESERVED.
    assert run_dir.resolve("P1/E1/T1/handoff.json").is_file()
    assert "feature.py" in wt.list_tree(git_repo, "grind/run-1")

    # The journal was APPENDED, never truncated: the razed-epoch marker is permanent.
    events = read_events(run_dir.events_path)
    resumed = [e for e in events if isinstance(e, RunResumed)]
    assert len(resumed) == 1 and resumed[0].razed_epoch == "P1/E2"
    # The planner re-entered at the last clean boundary and saw the completed E1.
    assert planner.contexts[0].epoch_index == 2
    assert "P1/E1/T1/handoff.json" in planner.contexts[0].log_index


# --- the planner context is rebuilt from disk each boundary --------------------


def test_planner_context_carries_tip_each_boundary(
    git_repo: Path, run_dir: RunDir, job_path: Path
) -> None:
    planner = MockDecisionPlanner(
        [_epoch(_impl("T1", ["a.py"])), _epoch(_impl("T2", ["b.py"])), _end()]
    )
    start_run(
        job_path=job_path, run_dir=run_dir, repo=git_repo, planner=planner,
        backends=_backends(LoopWorker()),
    )
    # Boundary 1 saw the base (no a.py yet); boundary 2 saw E1's integrated tip.
    assert "a.py" not in planner.contexts[0].tip_files
    assert "a.py" in planner.contexts[1].tip_files
    assert isinstance(planner.contexts[0], PlannerContext)
