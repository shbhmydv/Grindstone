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
from grindstone.loop import (
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

    def run(self, request: WorkerRequest) -> None:
        if request.critic is not None:
            (request.scratch / CRITIC_VERDICT_FILENAME).write_text(
                json.dumps({"outcome": self.critic_outcome, "reason": "sel"}),
                encoding="utf-8",
            )
            return
        if request.task.id in self.fail_ids:
            return  # no work -> zero diff -> gate rejects -> retries exhaust -> escalate
        for rel in request.task.file_ownership:
            path = request.scratch / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"# {request.task_id}\n", encoding="utf-8")
        (request.scratch / HANDOFF_FILENAME).write_text(
            f"# handoff {request.task_id}\nDONE\n", encoding="utf-8"
        )


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
