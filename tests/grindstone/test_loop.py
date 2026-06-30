"""The epoch driver end-to-end, driven by a scripted mock planner + a
concurrency-safe loop worker (no real model). Covers the BONES state machine: a
1-epoch run reaching ``completed``; a multi-epoch run; an EndDecision phase-handoff
(``ended``); the planner-declared setup seam; the disjoint-ownership merge of two
passing implement tasks; an ownership OVERLAP recorded in the close-out baton (no
fast-forward); a worker RateLimited parking then the epoch restarting clean; a
partial-fail epoch finalizing only the passers; a setup failure skipping the grind
but still closing out and advancing; a close-out rate limit razing + restarting the
epoch; a non-write task reading the integration tip a prior epoch built; and RESUME
from a mid-epoch crash (raze the in-flight epoch, re-plan reading the prior baton).
"""

from __future__ import annotations

import json
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pytest

from grindstone import worktree as wt
from grindstone.contracts.models import (
    Decision,
    EndDecision,
    Epoch,
    EpochDecision,
    Task,
)
from grindstone.events import (
    EpochCompleted,
    EpochStarted,
    JournalWriter,
    RunResumed,
    RunStarted,
    TaskDone,
    TaskRef,
    WorkGateRejected,
    read_events,
    replay,
)
from grindstone.loop import (
    CloseoutContext,
    PlannerContext,
    make_acceptance,
    resume_run,
    start_run,
)
from tests.grindstone.mock_planner import MockDecisionPlanner
from tests.grindstone.mock_worker import LoopWorker
from grindstone.rundir import RunDir
from grindstone.worker import (
    Backends,
    CRITIC_VERDICT_FILENAME,
    HANDOFF_FILENAME,
    WorkerRequest,
    WorkerTransport,
)


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


def _backends(worker: WorkerTransport, *, slots: int = 2) -> Backends:
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


def test_completed_run_reclaims_planner_tip(
    git_repo: Path, run_dir: RunDir, job_path: Path
) -> None:
    # A NON-resumable terminal (done_when passed) reclaims the planner's in-repo
    # _planner_tip worktree (its ~GB of installed deps / build output). The MockDecision
    # planner's discard_tip mirrors ScriptPlanner's removal; the pre-seeded tip lets the
    # test observe the loop calling it at the completed terminal.
    tip = run_dir.root / "_planner_tip"
    (tip / "node_modules").mkdir(parents=True)
    planner = MockDecisionPlanner([_epoch(_impl("T1", ["a.py"])), _end("shipped")])
    result = start_run(
        job_path=job_path, run_dir=run_dir, repo=git_repo, planner=planner,
        backends=_backends(LoopWorker()), max_epochs=5,
    )
    assert result.status == "completed"
    assert not tip.exists()  # reclaimed at the decided terminal


def test_clean_partial_end_reclaims_planner_tip(
    git_repo: Path, run_dir: RunDir, job_path: Path
) -> None:
    # A clean partial-end (the planner declared END but invariant #2 failed) is also a
    # NON-resumable terminal, so it reclaims the tip too.
    tip = run_dir.root / "_planner_tip"
    tip.mkdir(parents=True)
    planner = MockDecisionPlanner([_end("partial")])
    result = start_run(
        job_path=job_path, run_dir=run_dir, repo=git_repo, planner=planner,
        backends=_backends(LoopWorker()), acceptance=lambda ctx: False,
    )
    assert result.status == "ended"
    assert not tip.exists()


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


def test_planner_setup_runs_off_the_operator_checkout(
    git_repo: Path, run_dir: RunDir, job_path: Path, tmp_path: Path
) -> None:
    # FIX 3: setup is the TRUSTED host-global seam (system packages, shared dirs,
    # global tooling), NOT project-local dep installs. It runs in a throwaway
    # checkout, so a host-global side effect (an absolute-path touch) lands while the
    # operator checkout stays clean.
    marker = tmp_path / "host-global-ran"
    planner = MockDecisionPlanner(
        [_epoch(_impl("T1", ["a.py"]), setup=[f"touch {marker}"]), _end()]
    )
    result = start_run(
        job_path=job_path, run_dir=run_dir, repo=git_repo, planner=planner,
        backends=_backends(LoopWorker()),
    )
    assert result.status == "completed"
    # The host-global setup ran (outside the repo)...
    assert marker.is_file()
    # ...and the operator checkout was NOT dirtied by it.
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=str(git_repo),
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert status == ""


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


# --- ownership OVERLAP recorded in the baton (no fast-forward) ------------------


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
    # The overlap is recorded in E1's close-out baton, which the NEXT boundary reads.
    assert "overlap" in run_dir.read_baton(1)
    assert "overlap" in planner.contexts[1].baton
    # The close-out saw the integration conflict deterministically.
    assert planner.closeout_contexts[0].integration_conflict is not None


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


# --- a partial-fail epoch finalizes only the passers ---------------------------


@dataclass
class _SelectiveWorker:
    """A concurrency-safe worker that PASSES every task except those in ``fail_ids``
    (matched on the bare task id), which produce no work and exhaust to an escalate.
    Lets one epoch fan out a passer + a failer to prove partial finalize."""

    fail_ids: set[str]
    critic_outcome: str = "PASS"
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def run(self, request: WorkerRequest) -> str:
        if request.critic is not None:
            (request.scratch / CRITIC_VERDICT_FILENAME).write_text(
                json.dumps({"outcome": self.critic_outcome, "reason": "sel"}),
                encoding="utf-8",
            )
            return ""
        if request.task.id in self.fail_ids:
            return ""  # no work -> zero diff -> gate rejects -> retries exhaust -> escalate
        for rel in request.task.file_ownership:
            path = request.scratch / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"# {request.task_id}\n", encoding="utf-8")
        (request.scratch / HANDOFF_FILENAME).write_text(
            f"# handoff {request.task_id}\nDONE\n", encoding="utf-8"
        )
        return ""


def test_partial_fail_finalizes_only_passers(
    git_repo: Path, run_dir: RunDir, job_path: Path
) -> None:
    # T1 passes, T2 escalates: the epoch STILL completes, finalizing only T1's work
    # onto the run branch, and the baton records T2's escalation for the next boundary.
    planner = MockDecisionPlanner(
        [_epoch(_impl("T1", ["a.py"]), _impl("T2", ["b.py"])), _end()]
    )
    result = start_run(
        job_path=job_path, run_dir=run_dir, repo=git_repo, planner=planner,
        backends=_backends(_SelectiveWorker(fail_ids={"T2"})),
    )
    assert result.status == "completed"
    tree_files = wt.list_tree(git_repo, "grind/run-1")
    assert "a.py" in tree_files and "b.py" not in tree_files  # only the passer merged
    # The close-out saw one passer + one escalation; the baton carries the escalation.
    outcomes = {o.task_id: o.outcome for o in planner.closeout_contexts[0].task_outcomes}
    assert outcomes == {"E1/T1": "passed", "E1/T2": "escalated"}
    assert "escalated" in run_dir.read_baton(1)


# --- a setup failure skips the grind but still closes out and advances ----------


def test_setup_failure_skips_grind_but_advances(
    git_repo: Path, run_dir: RunDir, job_path: Path
) -> None:
    # A bad setup command will not pass on re-run, so it is NOT an abort: the grind is
    # skipped, the close-out baton records the setup error, the epoch finalizes (no
    # fast-forward, nothing was built) and ADVANCES so the planner re-plans next.
    planner = MockDecisionPlanner(
        [_epoch(_impl("T1", ["a.py"]), setup=["false"]), _end()]
    )
    result = start_run(
        job_path=job_path, run_dir=run_dir, repo=git_repo, planner=planner,
        backends=_backends(LoopWorker()),
    )
    assert result.status == "completed"
    # The grind was skipped: T1 never dispatched and nothing reached the run branch.
    events = read_events(run_dir.events_path)
    assert not any(e.event == "task_dispatched" for e in events)
    assert not wt.branch_exists(git_repo, "grind/run-1")
    # The epoch still advanced (E1 completed) with a baton that records the setup error.
    assert any(isinstance(e, EpochCompleted) and e.epoch_id == "E1" for e in events)
    assert planner.closeout_contexts[0].setup_error is not None
    assert "setup_error" in run_dir.read_baton(1)
    # The next boundary's PLAN read that baton (it is the planner's only memory).
    assert "setup_error" in planner.contexts[1].baton


# --- a close-out rate limit razes + restarts the SAME epoch (node #1) -----------


def test_closeout_rate_limit_razes_and_restarts(
    git_repo: Path, run_dir: RunDir, job_path: Path
) -> None:
    calls, fake = _no_sleep()
    # The epoch is planned TWICE: the first close-out hits a rate limit (raze + park +
    # restart the whole epoch), the restart re-plans and closes out clean.
    planner = MockDecisionPlanner(
        [_epoch(_impl("T1", ["a.py"])), _epoch(_impl("T1", ["a.py"])), _end()],
        closeout_rate_limit_once=True,
    )
    result = start_run(
        job_path=job_path, run_dir=run_dir, repo=git_repo, planner=planner,
        backends=_backends(LoopWorker()), sleep_fn=fake, backoff_s=11.0,
    )
    assert result.status == "completed"
    assert calls == [11.0]  # parked exactly once at the close-out backoff
    assert "a.py" in wt.list_tree(git_repo, "grind/run-1")
    # E1 was started twice (razed + restarted) and completed exactly once.
    events = read_events(run_dir.events_path)
    starts = [e for e in events if isinstance(e, EpochStarted) and e.epoch_id == "E1"]
    completes = [e for e in events if isinstance(e, EpochCompleted) and e.epoch_id == "E1"]
    assert len(starts) == 2 and len(completes) == 1
    # The close-out planner was called for both attempts; a rate-limit event was logged.
    assert len(planner.closeout_contexts) == 2
    assert any(e.event == "rate_limited" for e in events)


# --- non-write task reads the integration tip ----------------------------------


def test_non_write_reads_integration_tip(
    git_repo: Path, run_dir: RunDir, job_path: Path
) -> None:
    planner = MockDecisionPlanner(
        [
            _epoch(_impl("T1", ["feature.py"]), title="build"),
            _epoch(_review("T1", "E2/T1/review.md"), title="review"),
            _end(),
        ]
    )
    result = start_run(
        job_path=job_path, run_dir=run_dir, repo=git_repo, planner=planner,
        backends=_backends(LoopWorker(read_cite="feature.py")),
    )
    assert result.status == "completed"
    review = run_dir.resolve("E2/T1/review.md").read_text(encoding="utf-8")
    # The review worker read feature.py AT THE INTEGRATION TIP that E1 built; the
    # tip-keyed content proves it saw the in-run state, not the stale base checkout.
    assert "value = 'E1/T1'" in review


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
    (e1_wt / "feature.py").write_text("value = 'E1/T1'\n", encoding="utf-8")
    wt.commit_all(e1_wt, "e1: feature")
    wt.remove_worktree(git_repo, e1_wt)

    # E1's durable keyed log (the done-list the re-planned epoch must keep).
    h = run_dir.resolve("E1/T1/handoff.json")
    h.parent.mkdir(parents=True, exist_ok=True)
    h.write_text('{"task_id": "E1/T1", "status": "DONE"}', encoding="utf-8")

    # The journal: E1 done, E2 in flight (no EpochCompleted, no run terminal).
    with JournalWriter(run_dir.events_path) as jw:
        jw.emit(lambda s: RunStarted(seq=s, ts="t0", run_id="run-1",
                                     job_path=str(job_path), max_epochs=9))
        jw.emit(lambda s: EpochStarted(seq=s, ts="t1", epoch_id="E1", title="build",
                                       tasks=[TaskRef(id="T1", mode="implement")]))
        jw.emit(lambda s: EpochCompleted(seq=s, ts="t2", epoch_id="E1"))
        jw.emit(lambda s: EpochStarted(seq=s, ts="t3", epoch_id="E2", title="more",
                                       tasks=[TaskRef(id="T1", mode="implement")]))

    # In-flight E2 transient debris: a registered worktree + wip branch, a partial
    # keyed-log subdir, and a raw-log dir.
    wt.add_worktree(
        git_repo, run_dir.worktrees_root / "E2-T1" / "attempt-1",
        branch="grind-wip/run-1/E2-T1/attempt-1", base="grind/run-1",
    )
    partial = run_dir.resolve("E2/T1")
    partial.mkdir(parents=True, exist_ok=True)
    (partial / "scratch.txt").write_text("half-written\n", encoding="utf-8")
    logs = run_dir.root / "logs" / "E2-T1-worker"
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
    assert not (run_dir.worktrees_root / "E2-T1").exists()
    assert not wt.branch_exists(git_repo, "grind-wip/run-1/E2-T1/attempt-1")
    assert not run_dir.resolve("E2").exists()
    assert not (run_dir.root / "logs" / "E2-T1-worker").exists()

    # The completed-epoch keyed log + the run-branch boundary are PRESERVED.
    assert run_dir.resolve("E1/T1/handoff.json").is_file()
    assert "feature.py" in wt.list_tree(git_repo, "grind/run-1")

    # The journal was APPENDED, never truncated: the razed-epoch marker is permanent.
    events = read_events(run_dir.events_path)
    resumed = [e for e in events if isinstance(e, RunResumed)]
    assert len(resumed) == 1 and resumed[0].razed_epoch == "E2"
    # The planner re-entered at the last clean boundary and saw the completed E1.
    assert planner.contexts[0].epoch_index == 2
    assert "E1/T1/handoff.json" in planner.contexts[0].log_index


# --- an unexpected GitError/OSError mid-epoch razes + restarts the SAME epoch ----


def test_unexpected_git_error_mid_epoch_razes_and_restarts(
    git_repo: Path, run_dir: RunDir, job_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # An orchestrator-side fault during finalize (a GitError from the fast-forward,
    # OUTSIDE the worker's attempt try) must NOT crash an unattended run: BONES razes
    # the in-flight epoch and RESTARTS it (an aborted epoch has no baton, so it is
    # never completed). The planner re-plans the SAME epoch on restart.
    real_ff = wt.fast_forward_branch
    calls = {"n": 0}

    def flaky_ff(repo: Path, branch: str, commit: str) -> None:
        calls["n"] += 1
        if calls["n"] == 1:
            raise wt.GitError("injected integration fault")
        real_ff(repo, branch, commit)

    monkeypatch.setattr("grindstone.loop.wt.fast_forward_branch", flaky_ff)
    # Both script entries plan a.py: the first aborts at finalize, the restart re-plans
    # and lands it (the work was razed, so the planner must re-propose it).
    planner = MockDecisionPlanner(
        [_epoch(_impl("T1", ["a.py"])), _epoch(_impl("T1", ["a.py"])), _end()]
    )
    result = start_run(
        job_path=job_path, run_dir=run_dir, repo=git_repo, planner=planner,
        backends=_backends(LoopWorker()), max_epochs=5,
    )
    assert result.status == "completed"  # the run survived the fault by restarting
    assert "a.py" in wt.list_tree(git_repo, "grind/run-1")
    # E1 was STARTED twice (the abort razed it with no baton, then it restarted) and
    # completed exactly once (the durable boundary only the successful finalize emits).
    events = read_events(run_dir.events_path)
    starts = [e for e in events if isinstance(e, EpochStarted) and e.epoch_id == "E1"]
    completes = [
        e for e in events if isinstance(e, EpochCompleted) and e.epoch_id == "E1"
    ]
    assert len(starts) == 2 and len(completes) == 1
    # No baton was ever written for the aborted attempt; only the finalized epoch has one.
    assert run_dir.read_baton(1) != ""


def test_k_consecutive_aborts_end_cleanly(
    git_repo: Path, run_dir: RunDir, job_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A PERSISTENT infra fault must not infinite-loop: after K consecutive aborts the
    # bounded backstop clean-ends the run (node #2), resumable as the next run.
    def always_raise(repo: Path, branch: str, commit: str) -> None:
        raise OSError("persistent integration fault")

    monkeypatch.setattr("grindstone.loop.wt.fast_forward_branch", always_raise)
    planner = MockDecisionPlanner([_epoch(_impl("T1", ["a.py"]))] * 5 + [_end()])
    result = start_run(
        job_path=job_path, run_dir=run_dir, repo=git_repo, planner=planner,
        backends=_backends(LoopWorker()), max_epochs=20,
    )
    assert result.status == "ended"
    assert "consecutive" in result.summary
    # It ended at the backstop, well before max_epochs.
    assert result.epochs <= 3


# --- unified planner-failure retry (no halt on a transient) --------------------


def test_planner_timeout_retries_immediately_then_backoff(
    git_repo: Path, run_dir: RunDir, job_path: Path
) -> None:
    # A flaky planner timeout must NOT halt an unattended run. The FIRST timeout retries
    # IMMEDIATELY (no park); a CONSECUTIVE timeout parks once, then it lands the epoch and
    # the run completes. No premature RunEnded.
    calls, fake = _no_sleep()
    planner = MockDecisionPlanner(
        ["timeout", "timeout", _epoch(_impl("T1", ["a.py"])), _end("shipped")]
    )
    result = start_run(
        job_path=job_path, run_dir=run_dir, repo=git_repo, planner=planner,
        backends=_backends(LoopWorker()), sleep_fn=fake, backoff_s=9.0, max_epochs=5,
    )
    assert result.status == "completed" and result.summary == "shipped"
    assert calls == [9.0]  # the first timeout was immediate; only the repeat parked
    tree = replay(read_events(run_dir.events_path))
    assert tree.status == "completed"  # never a partial-end mid-run


def test_planner_decide_rate_limit_parks_then_retries(
    git_repo: Path, run_dir: RunDir, job_path: Path
) -> None:
    # A PLAN-call rate limit parks ~1/hr then re-issues the boundary (node #1), exactly
    # like the close-out / worker rate limit, and the run completes.
    calls, fake = _no_sleep()
    planner = MockDecisionPlanner(
        ["rate_limit", _epoch(_impl("T1", ["a.py"])), _end()]
    )
    result = start_run(
        job_path=job_path, run_dir=run_dir, repo=git_repo, planner=planner,
        backends=_backends(LoopWorker()), sleep_fn=fake, backoff_s=13.0, max_epochs=5,
    )
    assert result.status == "completed"
    assert calls == [13.0]  # parked once at the backoff
    assert any(e.event == "rate_limited" for e in read_events(run_dir.events_path))


def test_planner_failure_cap_ends_cleanly(
    git_repo: Path, run_dir: RunDir, job_path: Path
) -> None:
    # The ONE backstop: a permanently-broken planner (every PLAN call errors) cannot
    # spin forever. After MAX_CONSECUTIVE_ABORTS consecutive failures the run clean-ends
    # (node #2), resumable by a human - it never reaches a single completed epoch.
    from grindstone.loop import MAX_CONSECUTIVE_ABORTS

    calls, fake = _no_sleep()
    planner = MockDecisionPlanner(["error"] * MAX_CONSECUTIVE_ABORTS)
    result = start_run(
        job_path=job_path, run_dir=run_dir, repo=git_repo, planner=planner,
        backends=_backends(LoopWorker()), sleep_fn=fake, backoff_s=5.0, max_epochs=20,
    )
    assert result.status == "ended" and result.epochs == 0
    assert "consecutively" in result.summary
    tree = replay(read_events(run_dir.events_path))
    assert tree.status == "ended" and not tree.epochs


# --- resume re-plans reading the prior epoch's persisted baton -----------------


def test_resume_reads_prior_baton_from_disk(
    git_repo: Path, run_dir: RunDir, job_path: Path
) -> None:
    from tests.grindstone.mock_worker import CrashingWorker

    # E1 completes with its one task ESCALATED (critic ESCALATE): the close-out baton
    # records the escalation and is persisted at E1/baton.md. The host is then killed
    # during E2 (a SimulatedKill escapes start_run) BEFORE E2 can finalize a baton.
    planner = MockDecisionPlanner(
        [_epoch(_impl("T1", ["a.py"])), _epoch(_impl("T1", ["b.py"]))]
    )
    worker = CrashingWorker(inner=LoopWorker(critic_outcome="ESCALATE"), crash_on=2)
    with pytest.raises(BaseException):
        start_run(
            job_path=job_path, run_dir=run_dir, repo=git_repo, planner=planner,
            backends=_backends(worker), max_epochs=5,
        )
    # E1's baton is on disk (every completed epoch has one) and records the escalation;
    # the razed E2 left none.
    assert "escalated" in run_dir.read_baton(1)
    assert run_dir.read_baton(2) == ""

    # Resume re-enters at the next boundary having READ the prior baton from disk (it is
    # the planner's whole memory; nothing is reconstructed from the journal).
    resume_planner = MockDecisionPlanner([_end("resumed")])
    result = resume_run(
        run_dir=run_dir, repo=git_repo, planner=resume_planner,
        backends=_backends(LoopWorker()),
    )
    assert result.status == "completed"
    assert "escalated" in resume_planner.contexts[0].baton


# --- _build_context surfaces the prior epoch's evidence bundle -----------------


def test_build_context_points_at_prior_baton_artifacts(run_dir: RunDir) -> None:
    from grindstone.loop import _build_context

    # E2 persisted an evidence bundle; planning E3 must point at E2/baton-artifacts/.
    (run_dir.root / "E2" / "baton-artifacts").mkdir(parents=True)
    (run_dir.root / "E2" / "baton-artifacts" / "render.png").write_text("p", encoding="utf-8")
    (run_dir.root / "E2" / "T1").mkdir(parents=True)
    (run_dir.root / "E2" / "T1" / "handoff.md").write_text("h", encoding="utf-8")
    ctx = _build_context(
        job="j", repo=None, run_dir=run_dir, run_branch=None, tip_ref=None,
        epoch_index=3, max_epochs=5,
    )
    assert ctx.baton_artifacts == ("E2/baton-artifacts/render.png",)
    # The handoff (not under baton-artifacts) is NOT a bundle pointer, but stays in the log.
    assert "E2/T1/handoff.md" in ctx.log_index


def test_build_context_no_baton_artifacts_on_first_epoch_or_when_none(
    run_dir: RunDir,
) -> None:
    from grindstone.loop import _build_context

    # Epoch 1: there is no prior epoch, so no bundle pointer.
    first = _build_context(
        job="j", repo=None, run_dir=run_dir, run_branch=None, tip_ref=None,
        epoch_index=1, max_epochs=5,
    )
    assert first.baton_artifacts == ()
    # E2 produced ONLY a baton (functional run, no evidence): planning E3 -> still none.
    (run_dir.root / "E2").mkdir(parents=True)
    (run_dir.root / "E2" / "baton.md").write_text("## done\n", encoding="utf-8")
    none = _build_context(
        job="j", repo=None, run_dir=run_dir, run_branch=None, tip_ref=None,
        epoch_index=3, max_epochs=5,
    )
    assert none.baton_artifacts == ()


# --- the final-acceptance invariant runs done_when against the tip -------------


def _ctx(git_repo: Path, run_dir: RunDir, tip: str | None) -> PlannerContext:
    return PlannerContext(
        job="j", repo=git_repo, run_dir=run_dir, run_branch="grind/run-1",
        tip_ref=tip, log_index=(), baton="", epoch_index=1,
        max_epochs=5,
    )


def test_acceptance_checks_out_the_tip(
    git_repo: Path, run_dir: RunDir, job_path: Path
) -> None:
    # Build a run-branch tip carrying marker.py (off base, operator checkout untouched).
    wt.ensure_integration_branch(git_repo, "grind/run-1", "main")
    build = run_dir.worktrees_root / "_b"
    wt.add_worktree_on(git_repo, build, branch="grind/run-1")
    (build / "marker.py").write_text("x = 1\n", encoding="utf-8")
    wt.commit_all(build, "add marker")
    wt.remove_worktree(git_repo, build)
    tip = wt.resolve_commit(git_repo, "grind/run-1")

    # done_when is run in a throwaway checkout of that tip, so it SEES marker.py.
    passing = make_acceptance("test -f marker.py")
    assert passing(_ctx(git_repo, run_dir, tip)) is True
    # The same command against the base (no marker.py) fails -> a clean partial-end.
    base = wt.head_commit(git_repo)
    assert passing(_ctx(git_repo, run_dir, base)) is False
    # A non-zero command never passes.
    assert make_acceptance("false")(_ctx(git_repo, run_dir, tip)) is False


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
    # The integration tip moved forward between boundaries (E1 fast-forwarded it);
    # the planner greps the tip itself, so the loop need only carry the ref.
    c0, c1 = planner.contexts[0], planner.contexts[1]
    assert isinstance(c0, PlannerContext)
    assert c0.tip_ref != c1.tip_ref
    assert "a.py" not in wt.list_tree(git_repo, c0.tip_ref) if c0.tip_ref else True
    assert c1.tip_ref is not None and "a.py" in wt.list_tree(git_repo, c1.tip_ref)
    # Boundary 1 had no baton (first epoch); boundary 2 read E1's close-out baton.
    assert c0.baton == ""
    assert c1.baton == run_dir.read_baton(1) != ""


# --- a control file NEVER leaks into the integration tree (>=3-epoch E2E) -------


#: The grindstone-internal control files a faithful agent leaves in its scratch CWD
#: but must NEVER let into the integrated tree: the worker's free-form report, the
#: critic's verdict, and the local rig's per-CWD settings.
_CONTROL_NAMES = {HANDOFF_FILENAME, CRITIC_VERDICT_FILENAME}


def _control_leaks(tree: list[str]) -> list[str]:
    """The control-file paths present in a tracked-tree listing (empty == clean): a
    root-or-nested ``handoff.md`` / ``verdict.json``, or anything under ``.pi/``."""

    return [
        p for p in tree
        if Path(p).name in _CONTROL_NAMES or p == ".pi/settings.json" or p.startswith(".pi/")
    ]


@dataclass
class _FaithfulWorker:
    """A worker that FAITHFULLY models the real agent's disk contract: on an implement
    dispatch it writes the claimed files + a free-form ``handoff.md``, then does its
    OWN ``git add -A && git commit`` IN-WORKTREE (via the production ``commit_all``),
    exactly like the real pi agent. That self-commit is the leak vector: any stray
    control file left in scratch is swept into a commit the way the real run swept in
    ``.pi`` / ``verdict.json``, so the orchestrator's relocation (move-not-copy) is the
    only thing keeping the report + verdict out of the integrated tree.

    On a critic dispatch it writes ``verdict.json`` (PASS), or RETRY-then-PASS for a
    task id in ``retry_then_pass`` so the incremental-retry CARRY path (the path the
    verdict leak rides into a later attempt's scope check) is exercised inside a real
    run. Concurrency-safe: the only shared state is the per-task critic counter, guarded
    by a lock (this E2E fans out one task per epoch, so the counter never races)."""

    retry_then_pass: frozenset[str] = frozenset()
    _critic_calls: dict[str, int] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def run(self, request: WorkerRequest) -> str:
        if request.critic is not None:
            self._critic(request)
            return ""
        for rel in request.task.file_ownership:
            path = request.scratch / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                f"# {request.task_id}\nvalue = {request.task_id!r}\n", encoding="utf-8"
            )
        (request.scratch / HANDOFF_FILENAME).write_text(
            f"# handoff {request.task_id}\nDONE\n", encoding="utf-8"
        )
        # The agent commits its own work in the worktree (the disk contract): this is
        # the exact step that historically swept stray control files into the base.
        wt.commit_all(request.scratch, f"agent self-commit {request.task_id}")
        return ""

    def _critic(self, request: WorkerRequest) -> None:
        with self._lock:
            n = self._critic_calls.get(request.task_id, 0)
            self._critic_calls[request.task_id] = n + 1
        retry = request.task_id in self.retry_then_pass and n == 0
        outcome = "RETRY" if retry else "PASS"
        (request.scratch / CRITIC_VERDICT_FILENAME).write_text(
            json.dumps({"outcome": outcome, "reason": f"faithful {outcome}"}),
            encoding="utf-8",
        )


def test_three_epoch_run_never_leaks_a_control_file(
    git_repo: Path, run_dir: RunDir, job_path: Path
) -> None:
    # The control-file worktree-leak class: a faithful agent commits its scratch CWD
    # (handoff.md + the critic's verdict.json live there), so if the orchestrator does
    # not MOVE them out before/around the commit, one rides into the integration base
    # and detonates a LATER epoch (inherited, then deleted -> a spurious out-of-scope
    # rejection). Invisible to <=2-epoch runs, so this drives THREE real implement
    # epochs; E2 takes a critic RETRY-then-PASS so the verdict's incremental-retry CARRY
    # path is exercised (the leak point the critic move-not-copy fix governs).
    planner = MockDecisionPlanner(
        [
            _epoch(_impl("T1", ["a.py"])),
            _epoch(_impl("T1", ["b.py"])),
            _epoch(_impl("T1", ["c.py"])),
            _end("shipped three"),
        ]
    )
    worker = _FaithfulWorker(retry_then_pass=frozenset({"E2/T1"}))
    result = start_run(
        job_path=job_path, run_dir=run_dir, repo=git_repo, planner=planner,
        backends=_backends(worker), max_epochs=6,
    )
    assert result.status == "completed" and result.epochs == 3

    events = read_events(run_dir.events_path)
    # INVARIANT (no spurious rejection): a leaked control file inherited by a later
    # epoch and then deleted would surface as an out-of-scope gate rejection. None must
    # fire across the whole run, so every epoch's task reaches DONE.
    rejections = [e for e in events if isinstance(e, WorkGateRejected)]
    assert rejections == [], (
        "spurious out-of-scope rejection(s): "
        f"{[(e.task_id, e.reason) for e in rejections]}"
    )
    done = [e for e in events if isinstance(e, TaskDone)]
    assert len(done) == 3  # all three epochs' owned tasks passed and merged

    # INVARIANT (clean tip after EVERY epoch): the planner context's ``tip_ref`` IS the
    # run-branch tip the loop rebuilt at each boundary, so checking each one asserts the
    # integrated tree carried ONLY task-owned files, never a control file, after E1/E2/E3.
    for ctx in planner.contexts:
        if ctx.tip_ref is None:
            continue
        leaks = _control_leaks(wt.list_tree(git_repo, ctx.tip_ref))
        assert leaks == [], f"control file leaked into tip {ctx.tip_ref}: {leaks}"

    # The work itself landed: all three owned files are on the final run branch, and the
    # final tree is still free of every control file.
    tip = wt.list_tree(git_repo, "grind/run-1")
    assert {"a.py", "b.py", "c.py"} <= set(tip)
    assert _control_leaks(tip) == []


# --- the cross-epoch work backlog (decision.pending -> baton ## Pending) --------


def _epoch_with_pending(
    *tasks: Task, pending: list[str], title: str = "scaffold"
) -> EpochDecision:
    return EpochDecision(
        kind="epoch",
        epoch=Epoch(title=title, tasks=list(tasks)),
        pending=pending,
    )


def test_decision_pending_is_wired_into_closeout_context(
    git_repo: Path, run_dir: RunDir, job_path: Path
) -> None:
    # The loop must surface the plan's decision.pending additions to the SOLE baton
    # writer (close-out), so the backlog can be reconciled there. The plan stage NEVER
    # writes the baton (the atomic-finalize invariant: EpochCompleted implies baton).
    additions = ["refine a.py to taste later (senior)", "add integration tests later"]
    planner = MockDecisionPlanner(
        [_epoch_with_pending(_impl("T1", ["a.py"]), pending=additions), _end("done")]
    )
    result = start_run(
        job_path=job_path, run_dir=run_dir, repo=git_repo, planner=planner,
        backends=_backends(LoopWorker()), max_epochs=5,
    )
    assert result.status == "completed"
    assert len(planner.closeout_contexts) == 1
    assert list(planner.closeout_contexts[0].pending_additions) == additions


def _pending_items(baton: str) -> list[str]:
    """The bullets under a baton's ``## Pending`` section (the persisted backlog)."""

    items: list[str] = []
    in_section = False
    for line in baton.splitlines():
        if line.startswith("## "):
            in_section = line.strip() == "## Pending"
            continue
        if in_section and line.strip().startswith("- "):
            items.append(line.strip()[2:].strip())
    return items


def _item_tag(item: str) -> str:
    """The leading ``T<k>:`` tag a test backlog item carries (the short id of the task
    scheduled to drain it), or ``""`` for a fresh, not-yet-scheduled addition."""

    head = item.split(":", 1)[0].strip()
    return head if head.startswith("T") and head[1:].isdigit() else ""


@dataclass
class _ReconcilingPlanner:
    """A loop planner whose close-out performs the DETERMINISTIC backlog reconcile from
    the context the loop hands it: the new ``## Pending`` is (prior ``## Pending``) +
    (this decision's pending additions) MINUS (prior items whose scheduled task PASSED).

    "Scheduled-and-passed" is the real model's handoff-read; here it is the test
    convention that a prior backlog item is tagged with the short id of the task meant to
    drain it (``"T2: ..."``), and close-out drops it IFF that task PASSED (read from the
    DETERMINISTIC per-task outcomes, never a guess), so a FAILED scheduled item auto-carries.
    """

    script: list[Decision]
    closeout_contexts: list[CloseoutContext] = field(default_factory=list)
    _calls: int = 0

    def decide(self, context: PlannerContext) -> Decision:
        entry = self.script[self._calls]
        self._calls += 1
        return entry

    def close_out(self, context: CloseoutContext) -> str:
        self.closeout_contexts.append(context)
        passed = {
            o.task_id.rsplit("/", 1)[-1]
            for o in context.task_outcomes
            if o.outcome == "passed"
        }
        kept = [
            it for it in _pending_items(context.prior_baton)
            if _item_tag(it) not in passed
        ]
        kept += list(context.pending_additions)
        body = "\n".join(f"- {it}" for it in kept) or "- (none)"
        return (
            "## Project summary\nreconcile demo\n## Tasks done\n- work merged\n"
            f"## Pending\n{body}\n## Current status\nepoch closed\n"
        )

    def discard_tip(self, repo: Path | None, run_dir: RunDir) -> None:
        return None  # no real tip: Protocol parity only


@dataclass
class _PerTaskCriticWorker:
    """A LoopWorker variant whose critic ESCALATEs the (fully-qualified) task ids in
    ``escalate`` and PASSes the rest, so a scheduled backlog item can be driven to a
    deterministic FAILED outcome (the auto-carry path). Every implement dispatch writes
    its claimed files (keyed on the task id, so re-scaffolding an existing file is a real
    diff) plus a free-form handoff."""

    escalate: frozenset[str] = frozenset()
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def run(self, request: WorkerRequest) -> str:
        if request.critic is not None:
            outcome = "ESCALATE" if request.task_id in self.escalate else "PASS"
            (request.scratch / CRITIC_VERDICT_FILENAME).write_text(
                json.dumps({"outcome": outcome, "reason": f"per-task {outcome}"}),
                encoding="utf-8",
            )
            return ""
        for rel in request.task.file_ownership:
            path = request.scratch / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                f"# {request.task_id}\nvalue = {request.task_id!r}\n", encoding="utf-8"
            )
        (request.scratch / HANDOFF_FILENAME).write_text(
            f"# handoff {request.task_id}\nDONE\n", encoding="utf-8"
        )
        return ""


def test_backlog_reconcile_union_minus_passed_and_auto_carry(
    git_repo: Path, run_dir: RunDir, job_path: Path
) -> None:
    # E1 scaffolds a.py + b.py (both pass) and records TWO refine items as pending
    # additions (tagged with the task that will drain each). E2 schedules both refines:
    # T1 (a.py) PASSES, T2 (b.py) ESCALATES. The close-out reconcile must drop the
    # passed-and-scheduled item (T1) and CARRY the failed one (T2), based ONLY on the
    # deterministic per-task outcomes the loop hands it.
    e1 = _epoch_with_pending(
        _impl("T1", ["a.py"]), _impl("T2", ["b.py"]),
        pending=[
            "T1: refine a.py to taste (senior)",
            "T2: refine b.py to taste (senior)",
        ],
        title="scaffold a + b",
    )
    e2 = _epoch_with_pending(
        _impl("T1", ["a.py"]), _impl("T2", ["b.py"]),
        pending=[], title="refine a + b",
    )
    planner = _ReconcilingPlanner([e1, e2, _end("shipped")])
    worker = _PerTaskCriticWorker(escalate=frozenset({"E2/T2"}))
    result = start_run(
        job_path=job_path, run_dir=run_dir, repo=git_repo, planner=planner,
        backends=_backends(worker), max_epochs=6,
    )
    assert result.status == "completed" and result.epochs == 2

    # E1's baton: the backlog is the UNION of the (empty) prior + the two additions.
    e1_pending = _pending_items(run_dir.read_baton(1))
    assert e1_pending == [
        "T1: refine a.py to taste (senior)",
        "T2: refine b.py to taste (senior)",
    ]
    # E2's baton: T1 was scheduled AND passed -> dropped; T2 was scheduled but ESCALATED
    # -> auto-carried. The reconcile read "done" from the gate outcomes, not a guess.
    e2_pending = _pending_items(run_dir.read_baton(2))
    assert e2_pending == ["T2: refine b.py to taste (senior)"]
    # The close-out saw the deterministic per-task outcomes it reconciled on.
    e2_outcomes = {
        o.task_id: o.outcome for o in planner.closeout_contexts[1].task_outcomes
    }
    assert e2_outcomes == {"E2/T1": "passed", "E2/T2": "escalated"}


# --- run-scoped SIGTERM/SIGINT reaping is a RESUMABLE stop, mutating no git -------


def test_install_reaper_signals_restores_prior_handlers() -> None:
    """``_install_reaper_signals`` must NEVER leave a global handler installed: tests
    and library callers depend on the original disposition being restored."""

    import signal as _signal

    from grindstone import loop as _loop

    orig_term = _signal.getsignal(_signal.SIGTERM)
    orig_int = _signal.getsignal(_signal.SIGINT)
    prior = _loop._install_reaper_signals()
    try:
        # Installed: the live handlers differ from the originals.
        assert _signal.getsignal(_signal.SIGTERM) is not orig_term
        assert _signal.getsignal(_signal.SIGINT) is not orig_int
    finally:
        _loop._restore_reaper_signals(prior)
    # Restored byte-for-byte.
    assert _signal.getsignal(_signal.SIGTERM) is orig_term
    assert _signal.getsignal(_signal.SIGINT) is orig_int


def test_reaper_signal_handler_reaps_then_raises_interrupted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The installed handler REAPS the live subprocess groups (processes only) then
    raises ``_Interrupted`` (a BaseException, so the epoch body's broad excepts cannot
    swallow it). It touches no git/disk."""

    import signal as _signal

    from grindstone import loop as _loop

    reaped: list[bool] = []
    monkeypatch.setattr(_loop.reaper, "reap_all", lambda: reaped.append(True))

    prior = _loop._install_reaper_signals()
    try:
        handler = _signal.getsignal(_signal.SIGTERM)
        assert callable(handler)
        with pytest.raises(_loop._Interrupted):
            handler(_signal.SIGTERM, None)
        assert reaped == [True]
        assert not isinstance(_loop._Interrupted(0), Exception)  # BaseException only
    finally:
        _loop._restore_reaper_signals(prior)


def test_sigterm_midrun_is_resumable_stop_with_no_git_mutation(
    git_repo: Path, run_dir: RunDir, job_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A SIGTERM landing during a PLAN call (simulated by a planner that raises the
    same ``_Interrupted`` the reaper handler would) ends the run as RESUMABLE
    (``ended``) and performs NO git mutation and NO ``_raze_epoch``: the kill path
    only reaps processes, resume owns scratch cleanup."""

    from grindstone import loop as _loop

    razed: list[object] = []
    monkeypatch.setattr(_loop, "_raze_epoch", lambda *a, **k: razed.append(a))

    def _git_state() -> str:
        return subprocess.run(
            ["git", "for-each-ref", "--format=%(refname) %(objectname)"],
            cwd=str(git_repo), capture_output=True, text=True, check=True,
        ).stdout

    before = _git_state()

    class _Interrupting:
        """Stands in for a SIGTERM arriving on the very first boundary."""

        def decide(self, context: PlannerContext) -> Decision:
            raise _loop._Interrupted(15)

        def close_out(self, context: CloseoutContext) -> str:
            return "unused"

        def discard_tip(self, repo: Path | None, run_dir: RunDir) -> None:
            raise AssertionError("a resumable interrupt must NOT reclaim the tip")

    result = start_run(
        job_path=job_path, run_dir=run_dir, repo=git_repo, planner=_Interrupting(),
        backends=_backends(LoopWorker()), max_epochs=5,
    )

    assert result.status == "ended"
    assert razed == []  # the kill path NEVER razes (resume does that)
    assert _git_state() == before  # no branch/ref mutation
    assert not wt.branch_exists(git_repo, "grind/run-1")  # run branch never created
    # A clean RESUMABLE terminal in the journal (RunEnded, not RunCompleted).
    events = read_events(run_dir.events_path)
    from grindstone.events import RunCompleted, RunEnded
    assert any(isinstance(e, RunEnded) for e in events)
    assert not any(isinstance(e, RunCompleted) for e in events)


# --- the strike ladder (per-task repair-escalation) ----------------------------


@dataclass
class _FileEscalatingWorker:
    """A concurrency-safe loop worker for the strike-ladder tests. An implement task
    whose ownership intersects ``escalate_files`` writes its files (so the
    deterministic gate passes) but its critic routes ESCALATE, so that file's lineage
    fails one WHOLE epoch (one strike); every other task PASSES. Records each worker
    grind's full task id in ``dispatched``, so a two-endpoint test (a distinct local +
    senior instance) can prove BOTH tiers ran within one epoch's in-epoch ladder."""

    escalate_files: frozenset[str]
    dispatched: list[str] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def _escalates(self, task: Task) -> bool:
        return task.mode == "implement" and any(
            f in self.escalate_files for f in task.file_ownership
        )

    def run(self, request: WorkerRequest) -> str:
        task = request.task
        if request.critic is not None:
            outcome = "ESCALATE" if self._escalates(task) else "PASS"
            (request.scratch / CRITIC_VERDICT_FILENAME).write_text(
                json.dumps({"outcome": outcome, "reason": f"file {outcome}"}),
                encoding="utf-8",
            )
            return ""
        with self._lock:
            self.dispatched.append(request.task_id)
        if request.mode == "implement":
            for f in task.file_ownership:
                path = request.scratch / f
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    f"# {request.task_id}\nvalue = {request.task_id!r}\n",
                    encoding="utf-8",
                )
        (request.scratch / HANDOFF_FILENAME).write_text(
            f"# handoff {request.task_id}\nattempted\n", encoding="utf-8"
        )
        return ""


def _two_tier_backends(
    local: WorkerTransport, senior: WorkerTransport, *, slots: int = 2
) -> Backends:
    """A Backends with DISTINCT local + senior endpoints, so ``run_task``'s in-epoch
    tier ladder actually escalates (``has_distinct_tier('senior')`` is True). Each tier
    dispatches its own transport instance, so a test can see which tier ran."""

    from grindstone.worker import _Endpoint

    endpoints = {
        "local": _Endpoint(local, threading.Semaphore(slots)),
        "senior": _Endpoint(senior, threading.Semaphore(slots)),
    }
    return Backends(endpoints, {"local": "local", "senior": "senior"})


def _strike_events(run_dir: RunDir) -> list[object]:
    from grindstone.events import StrikeLedger, TaskParked, TierEscalated
    return [
        e for e in read_events(run_dir.events_path)
        if isinstance(e, (StrikeLedger, TaskParked, TierEscalated))
    ]


def test_strike_ladder_blocks_after_two_full_ladder_failures(
    git_repo: Path, run_dir: RunDir, job_path: Path
) -> None:
    # Senior is now reached IN-EPOCH every epoch: E1 grinds x.py at local, the critic
    # ESCALATEs, the ladder re-dispatches it at SENIOR on the carried wip (still in E1),
    # senior ESCALATEs too -> the WHOLE in-epoch ladder failed = strike 1. The planner
    # re-issues at E2 (its one reframe chance) -> strike 2. At E3 the strike-2 lineage
    # is BLOCKED (parked, dropped from dispatch). No cross-epoch force-senior rung.
    from grindstone import strikes
    from grindstone.events import TaskParked, TierEscalated

    script = [_epoch(_impl("T1", ["x.py"])) for _ in range(3)] + [_end("stopping")]
    planner = MockDecisionPlanner(script)
    local_w = _FileEscalatingWorker(escalate_files=frozenset({"x.py"}))
    senior_w = _FileEscalatingWorker(escalate_files=frozenset({"x.py"}))
    result = start_run(
        job_path=job_path, run_dir=run_dir, repo=git_repo, planner=planner,
        backends=_two_tier_backends(local_w, senior_w), max_epochs=8,
        acceptance=lambda ctx: False,
    )
    assert result.status == "ended"

    events = read_events(run_dir.events_path)
    # E1's in-epoch ladder dispatched x.py at BOTH tiers (local first, then senior on the
    # carried wip) within the ONE epoch, signalled by a TierEscalated event.
    assert "E1/T1" in local_w.dispatched
    assert "E1/T1" in senior_w.dispatched
    e1_esc = [e for e in events if isinstance(e, TierEscalated) and e.epoch_id == "E1"]
    assert len(e1_esc) == 1 and e1_esc[0].to_tier == "senior"

    # Strike 2 BLOCKED the lineage at E3: a structured park event + removed from dispatch.
    parked = [e for e in events if isinstance(e, TaskParked)]
    assert len(parked) == 1
    assert parked[0].epoch_id == "E3" and parked[0].strikes == 2
    assert parked[0].descriptor == "x.py"
    e3_dispatched = [
        e for e in events
        if e.event == "task_dispatched" and e.epoch_id == "E3"
    ]
    assert e3_dispatched == []  # blocked => never dispatched

    # The reconstructed ledger persisted the lineage at 2 strikes (resume-safe), and
    # the blocked lineage surfaced in the run summary so the operator sees it.
    ledger = strikes.reconstruct_entries(events)
    assert {e.ownership: e.strikes for e in ledger} == {("x.py",): 2}
    assert "PARKED" in result.summary and "x.py" in result.summary
    # x.py never merged: the blocked, never-passing work is not on the run branch.
    assert "x.py" not in wt.list_tree(git_repo, "grind/run-1")


def test_redecomposed_child_inherits_parent_strikes(
    git_repo: Path, run_dir: RunDir, job_path: Path
) -> None:
    # E1 issues ONE task owning a.py + b.py; it escalates (strike 1 for that lineage).
    # E2 RE-DECOMPOSES it into two tasks (a.py alone, b.py alone): a.py lands, b.py
    # escalates. The b.py child must INHERIT the parent's strike and climb to 2 -
    # relabelling cannot reset the ladder.
    from grindstone import strikes

    planner = MockDecisionPlanner(
        [
            _epoch(_impl("T1", ["a.py", "b.py"])),
            _epoch(_impl("T1", ["a.py"]), _impl("T2", ["b.py"])),
            _end("done"),
        ]
    )
    worker = _FileEscalatingWorker(escalate_files=frozenset({"b.py"}))
    result = start_run(
        job_path=job_path, run_dir=run_dir, repo=git_repo, planner=planner,
        backends=_backends(worker), max_epochs=6, acceptance=lambda ctx: False,
    )
    assert result.status == "ended"

    ledger = strikes.reconstruct_entries(read_events(run_dir.events_path))
    # ONLY the b.py lineage remains, at strike 2 (parent's 1 + this epoch): the a.py
    # half resolved, the parent [a,b] lineage was superseded by the inheriting child.
    assert {e.ownership: e.strikes for e in ledger} == {("b.py",): 2}
    # a.py merged (it passed in E2); b.py never did.
    tip = wt.list_tree(git_repo, "grind/run-1")
    assert "a.py" in tip and "b.py" not in tip


def test_no_carry_run_has_no_strike_events(
    git_repo: Path, run_dir: RunDir, job_path: Path
) -> None:
    # Backward-compat: a run where every task lands never strikes a lineage, so it emits
    # ZERO strike-ladder events (the journal is byte-identical to before the feature).
    planner = MockDecisionPlanner(
        [_epoch(_impl("T1", ["a.py"])), _epoch(_impl("T1", ["b.py"])), _end()]
    )
    result = start_run(
        job_path=job_path, run_dir=run_dir, repo=git_repo, planner=planner,
        backends=_backends(LoopWorker()), max_epochs=5,
    )
    assert result.status == "completed"
    assert _strike_events(run_dir) == []
    assert "PARKED" not in result.summary


def test_resume_reconstructs_strike_state_across_a_fresh_run(
    git_repo: Path, run_dir: RunDir, job_path: Path
) -> None:
    # Two epochs fail the whole in-epoch ladder for x.py (the ledger reaches 2 strikes,
    # persisted on disk), then the planner ENDS -> a resumable partial-end. A FRESH
    # resume_run (new planner, new worker, no in-memory state) re-proposes x.py at E3;
    # the strike count must be rebuilt FROM THE JOURNAL so the deterministic BLOCK (park
    # at strike 2) still fires.
    from grindstone.events import TaskParked

    first = MockDecisionPlanner(
        [_epoch(_impl("T1", ["x.py"])) for _ in range(2)] + [_end("paused")]
    )
    w1 = _FileEscalatingWorker(escalate_files=frozenset({"x.py"}))
    r1 = start_run(
        job_path=job_path, run_dir=run_dir, repo=git_repo, planner=first,
        backends=_backends(w1), max_epochs=8, acceptance=lambda ctx: False,
    )
    assert r1.status == "ended" and r1.epochs == 2
    # No park yet in the first run (it only reached strike 2 AT the E3 boundary, where
    # the planner ended instead of grinding).
    assert not any(isinstance(e, TaskParked) for e in read_events(run_dir.events_path))

    resumed = MockDecisionPlanner([_epoch(_impl("T1", ["x.py"])), _end("done2")])
    w2 = _FileEscalatingWorker(escalate_files=frozenset({"x.py"}))
    r2 = resume_run(
        run_dir=run_dir, repo=git_repo, planner=resumed,
        backends=_backends(w2), max_epochs=8, acceptance=lambda ctx: False,
    )
    assert r2.status == "ended"
    parked = [e for e in read_events(run_dir.events_path) if isinstance(e, TaskParked)]
    assert len(parked) == 1 and parked[0].epoch_id == "E3"  # reconstructed from disk
    assert parked[0].strikes == 2
    assert w2.dispatched == []  # blocked => the fresh worker never ground x.py
