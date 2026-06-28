"""The epoch driver: the BONES state machine, the two failure nodes, and resume.

A synchronous, dumb Python loop; ALL sequencing judgment is delegated to the
planner. Python characterizes NOTHING: it owns the state machine, the two
deterministic invariants (the disjoint-ownership merge of the git diff, the one final
acceptance), the durable log, crash-only resume, and bounded fresh-from-disk context
assembly. Each boundary rebuilds the planner's context fresh FROM DISK (the
integration tip + the durable keyed log + the job spec + the prior epoch's BATON),
asks the planner for ONE decision, and disposes of it:

    EpochDecision -> run the planner-declared setup (the trusted host-mutation
        seam), fan the epoch's disjoint tasks out under ONE shared ``Backends``
        (the per-endpoint semaphores are the real concurrency bound), integrate the
        PASSing implement tasks onto a STAGING branch by the disjoint-ownership merge
        invariant, ask the planner to CLOSE OUT the epoch (read the staging tree + the
        per-task verdicts/handoffs, write the updated living baton), then atomically
        finalize (fast-forward the durable run branch + persist the baton +
        ``EpochCompleted``). Close-out runs BEFORE the fast-forward, so the one durable
        commit point already includes the baton and there is no "integrated-but-not-
        summarized" limbo.
    EndDecision  -> run the one final acceptance (invariant #2, an injected seam):
        pass -> ``completed``; otherwise persist the summary as the resume seed and
        end cleanly (``ended``, failure node #2).

Failure model (exactly two nodes):

  #1 RATE LIMIT (planner OR worker / critic): PARK, back off ~1/hr (injectable),
     then re-enter. A planner DECIDE rate-limit re-issues the boundary call; a
     mid-epoch worker OR close-out rate-limit RAZES the in-flight epoch's throwaway
     worktrees + staging and RESTARTS the epoch whole (partial state is never
     trusted).
  #2 CANNOT CONTINUE: the planner ends cleanly (the baton carries the why). An
     UNEXPECTED exception in the epoch body (a GitError/OSError escaping the
     worktree/integration machinery) RAZES + RESTARTS the SAME epoch (an aborted epoch
     has no baton, so it is never completed); ``MAX_CONSECUTIVE_ABORTS`` consecutive
     aborts on the same epoch clean-end the run. The ``max_epochs`` backstop is the
     INVOLUNTARY trigger of the same clean end.

RESUME is the universal crash-only recovery primitive: because the run branch only
fast-forwards on epoch finalize, the git tip needs no rewind. ``resume_run`` razes the
incomplete epoch's worktrees + wip branches + partial keyed log (including a never-
finalized ``E<n>/baton.md``), PRESERVES the completed-epoch keyed log + the
append-only journal (it APPENDS a razed-epoch marker, never truncates), and re-enters
the loop from the planner call; the next PLAN re-reads the prior epoch's baton from
disk, so no in-memory failure context need survive the crash.

This module exposes the testable CORE callables (``start_run`` / ``resume_run``)
with the planner + backends injected; the CLI-facing ``run`` / ``resume`` (which
build the real planner + backends from config) are wired in a later part.
"""

from __future__ import annotations

import shutil
import signal
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import FrameType
from typing import Any, Callable, Literal, Protocol, Sequence

from grindstone import reaper
from grindstone import strikes
from grindstone import worktree as wt
from grindstone.config import (
    GrindstoneConfig,
    load_config,
    models_script,
    resolve_role_script,
    validate_script_paths,
)
from grindstone.contracts.models import (
    Decision,
    EndDecision,
    Epoch,
    Task,
    VerdictOutcome,
)
from grindstone.events import (
    Event,
    EpochCompleted,
    EpochStarted,
    WorkGatePassed,
    WorkGateRejected,
    JournalWriter,
    RateLimited as RateLimitedEvent,
    RunCompleted,
    RunEnded,
    RunResumed,
    RunStarted,
    StrikeLedger,
    StrikeLedgerEntry,
    TaskDispatched,
    TaskDone,
    TaskParked,
    TaskRef,
    TierEscalated,
    read_events,
)
from grindstone.events import Verdict as VerdictEvent
from grindstone.journal import reap_sibling_journals, write_journal
from grindstone.planner import PlannerError, ScriptPlanner
from grindstone.planner import RateLimited as PlannerRateLimited
from grindstone.rundir import RunDir, create_run_dir
from grindstone.script_planner import ScriptPlannerTransport
from grindstone.script_worker import build_backends
from grindstone.worker import (
    Backends,
    CRITIC_VERDICT_FILENAME,
    HANDOFF_FILENAME,
    RateLimited as WorkerRateLimited,
    TaskResult,
    run_task,
)

#: The built-in epoch backstop when the config sets no ``max_epochs`` (BONES: the
#: cap is the involuntary trigger of the clean partial-end, never unbounded).
DEFAULT_MAX_EPOCHS = 40

#: BONES failure node #2 backstop: K CONSECUTIVE aborts on the SAME epoch from an
#: UNEXPECTED error (a GitError/OSError escaping the worktree/integration machinery,
#: not a planned task failure) clean-end the run, so a persistent infra fault cannot
#: infinite-loop an unattended run that keeps razing + restarting the same epoch. A
#: single transient fault just razes + restarts the epoch once.
MAX_CONSECUTIVE_ABORTS = 3

#: Node-#1 backoff (~1/hr). Injected as ``sleep_fn``/``backoff_s`` so tests park
#: without a real wall clock.
DEFAULT_BACKOFF_S = 3600.0

#: Wall-clock cap on each planner-declared setup command (the trusted-tier
#: host-mutation seam runs them via the shell before the tasks).
SETUP_TIMEOUT_S = 1800.0

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


SleepFn = Callable[[float], None]
NowFn = Callable[[], str]


# --- the planner seam (the real planner is a later part) -----------------------


@dataclass(frozen=True)
class PlannerContext:
    """The bounded window the loop reconstructs FROM DISK each boundary and hands the
    stateless PLAN call. The mock ignores it; the real planner renders its prompt from
    it. Externalized context: re-derived, never accumulated. The planner GREPS its own
    workdir for the tree (the file-name dump is gone); its memory across the run is the
    ``baton`` it wrote at the prior close-out."""

    job: str
    repo: Path | None
    run_dir: RunDir
    run_branch: str | None
    #: The integration-tip commit (the durable run-branch tip, or repo HEAD before
    #: the first epoch completes). ``None`` when there is no repo.
    tip_ref: str | None
    #: Durable keyed-log references a task may name as ``inputs``.
    log_index: tuple[str, ...]
    #: The prior completed epoch's ``baton.md`` text (the planner's living plan), or
    #: ``""`` for the first epoch.
    baton: str
    #: The 1-based epoch number about to be planned, and the backstop.
    epoch_index: int
    max_epochs: int
    #: The strike-ladder NUDGE (soft): the lineages carried unfinished across prior
    #: epochs, each flagged with the deterministic action the state machine will take
    #: (force senior at 3, park at 4). Reconstructed from the journal, so it survives a
    #: resume. Empty (the default) for a run that never carried a task -> byte-identical
    #: planner prompt.
    carried: tuple[strikes.CarriedItem, ...] = ()


@dataclass(frozen=True)
class TaskOutcome:
    """A pure REFERENCE to one finished task's outcome (Python characterizes NOTHING).
    The close-out planner reads the pointed-to handoff + verdict and writes the
    partial-vs-none-vs-regression nuance itself; this struct only carries the
    deterministic terminal, the keyed-log pointers, and the model's verbatim reason."""

    task_id: str            # "E3/T2"
    mode: str
    outcome: str            # "passed" | "escalated" (the deterministic terminal)
    handoff_key: str | None  # keyed-log path to the free-form handoff, if any
    verdict_key: str | None  # keyed-log path to the critic verdict, if any
    reason: str             # the escalation / critic reason verbatim (model text)


@dataclass(frozen=True)
class CloseoutContext:
    """The bounded window the loop hands the CLOSE-OUT call at epoch END. Python
    populates ``task_outcomes`` from the in-memory ``TaskResult``s as pure references;
    the planner reads the staging tree + the pointed-to handoffs/verdicts and WRITES
    the updated baton (the markdown returned by ``close_out``)."""

    job: str
    repo: Path | None
    run_dir: RunDir
    #: The tree close-out reads: the staging tip if work merged, else the epoch base.
    staging_ref: str | None
    #: The previous epoch's ``baton.md`` text, ``""`` if this is the first epoch.
    prior_baton: str
    epoch_index: int
    epoch_id: str           # "E3"
    title: str              # the epoch title the planner gave
    task_outcomes: tuple[TaskOutcome, ...]
    setup_error: str | None
    integration_conflict: str | None
    #: This epoch's planned ADDITIONS to the cross-epoch work backlog (the plan's
    #: ``decision.pending``). The close-out (the SOLE baton writer) reconciles these +
    #: the prior baton's ``## Pending`` against the deterministic per-task outcomes into
    #: the new ``## Pending``. Empty by default (an EndDecision never reaches close-out).
    pending_additions: tuple[str, ...] = ()
    #: The lineage descriptors the strike ladder PARKED this epoch (strike 4: removed
    #: from the active set). The close-out notes them in the baton as unclosed. Empty by
    #: default (a run that parked nothing is byte-identical to today).
    parked: tuple[str, ...] = ()


class Planner(Protocol):
    """The stateless one-shot planner the loop calls at each boundary. ``decide`` (PLAN,
    forward) returns ONE typed ``Decision`` (epoch or end); ``close_out`` (back) reads
    the just-finished epoch's outcomes + staging tree and returns the updated baton
    markdown. Either raises ``RateLimited`` (node #1) / ``PlannerError`` (node #2);
    ``close_out`` NEVER hard-fails on content (it reads whatever prose the rig wrote,
    like the free-form handoff)."""

    def decide(self, context: PlannerContext) -> Decision: ...
    def close_out(self, context: CloseoutContext) -> str: ...


#: Invariant #2: the one final acceptance, run ONCE when the planner says done.
#: An injected seam: ``make_acceptance`` runs the job's own ``done_when`` in a
#: throwaway checkout of the integration tip; ``None`` trusts the planner's word
#: (the run completes, the default when no ``done_when`` is configured).
AcceptanceCheck = Callable[[PlannerContext], bool]


def make_acceptance(
    done_when: str, *, timeout_s: float = SETUP_TIMEOUT_S
) -> AcceptanceCheck:
    """The single final gate (invariant #2): run the job's OWN ``done_when`` ONCE.

    When the planner emits END, check out the integration tip in a throwaway
    detached worktree and run ``done_when`` there exactly once: exit 0 -> the run is
    ``completed``; any non-zero exit (or a failure to run) -> the planner's END is a
    clean partial-end (``ended``) and its summary seeds the next appendable run. This
    is deliberately the ONLY deterministic build gate (BONES: no per-epoch build
    gates); it exists so "done" still means something when every per-epoch check is
    agentic. With no repo / no tip there is nothing to check out, so the command runs
    in the run dir (a degenerate but honest fallback)."""

    def _check(context: PlannerContext) -> bool:
        repo, tip = context.repo, context.tip_ref
        if repo is None or tip is None:
            return _run_acceptance(done_when, context.run_dir.root, timeout_s)
        path = context.run_dir.worktrees_root / "_acceptance"
        wt.add_worktree_detached(repo, path, ref=tip)
        try:
            return _run_acceptance(done_when, path, timeout_s)
        finally:
            wt.remove_worktree(repo, path)

    return _check


def _run_acceptance(command: str, cwd: Path, timeout_s: float) -> bool:
    """Run the acceptance command once in ``cwd``; True iff it exits 0."""

    try:
        proc = subprocess.run(
            command, shell=True, cwd=str(cwd), capture_output=True, text=True,
            timeout=timeout_s,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0


@dataclass(frozen=True)
class RunResult:
    """The run's terminal: ``completed`` (planner ended + acceptance passed) or
    ``ended`` (a clean partial-end, failure node #2). ``summary`` is the resume seed;
    ``epochs`` is the number of epochs that ran to completion."""

    status: Literal["completed", "ended"]
    summary: str
    epochs: int


# --- the live per-attempt event sink (threaded into run_task) ------------------


def _short_id(task_id: str) -> str:
    """The per-epoch short task id (``T1``) the journal groups under ``epoch_id``;
    ``run_task`` works in the fully-qualified ``E*/T*`` for the keyed log."""

    return task_id.rsplit("/", 1)[-1]


class _JournalAttemptEvents:
    """Adapts ``run_task``'s ``AttemptEvents`` hooks onto the journal so the gate +
    triage land live (the per-task events carry the SHORT id, grouped under the
    epoch). Thread-safe: ``JournalWriter.emit`` serializes under its lock, so
    concurrent fan-out tasks interleave cleanly."""

    def __init__(self, journal: JournalWriter, epoch_id: str, now_fn: NowFn) -> None:
        self._journal = journal
        self._epoch_id = epoch_id
        self._now = now_fn

    def work_gate_passed(self, task_id: str) -> None:
        tid = _short_id(task_id)
        self._journal.emit(
            lambda s: WorkGatePassed(
                seq=s, ts=self._now(), epoch_id=self._epoch_id, task_id=tid
            )
        )

    def work_gate_rejected(self, task_id: str, reason: str) -> None:
        tid = _short_id(task_id)
        self._journal.emit(
            lambda s: WorkGateRejected(
                seq=s, ts=self._now(), epoch_id=self._epoch_id, task_id=tid,
                reason=reason,
            )
        )

    def verdict(self, task_id: str, outcome: VerdictOutcome, reason: str) -> None:
        tid = _short_id(task_id)
        self._journal.emit(
            lambda s: VerdictEvent(
                seq=s, ts=self._now(), epoch_id=self._epoch_id, task_id=tid,
                outcome=outcome, reason=reason,
            )
        )

    def tier_escalated(self, task_id: str, to_tier: str, attempt: int) -> None:
        tid = _short_id(task_id)
        self._journal.emit(
            lambda s: TierEscalated(
                seq=s, ts=self._now(), epoch_id=self._epoch_id, task_id=tid,
                to_tier=to_tier, attempt=attempt,
            )
        )


# --- tip + context -------------------------------------------------------------


def _run_branch(run_id: str) -> str:
    """The ONE durable run branch ``grind/<run-id>``: the only ref that survives
    between epochs, fast-forwarded only on epoch completion."""

    return f"grind/{run_id}"


def _resolve_tip(repo: Path | None, run_branch: str | None) -> str | None:
    """The integration tip: the run-branch commit if it exists (a completed epoch
    boundary), else repo HEAD (before the first epoch), else ``None`` (no repo)."""

    if repo is None or run_branch is None:
        return None
    if wt.branch_exists(repo, run_branch):
        return wt.resolve_commit(repo, run_branch)
    return wt.head_commit(repo)


def _build_context(
    *,
    job: str,
    repo: Path | None,
    run_dir: RunDir,
    run_branch: str | None,
    tip_ref: str | None,
    epoch_index: int,
    max_epochs: int,
    carried: tuple[strikes.CarriedItem, ...] = (),
) -> PlannerContext:
    """Reconstruct the PLAN window FROM DISK: the keyed-log index + the prior epoch's
    baton (``read_baton(epoch_index - 1)``, ``""`` for epoch 1) + the strike-ladder
    nudge (``carried``, reconstructed from the journal by the caller). No file-name dump
    (the planner greps its own workdir for the tree)."""

    return PlannerContext(
        job=job,
        repo=repo,
        run_dir=run_dir,
        run_branch=run_branch,
        tip_ref=tip_ref,
        log_index=tuple(run_dir.log_index()),
        baton=run_dir.read_baton(epoch_index - 1),
        epoch_index=epoch_index,
        max_epochs=max_epochs,
        carried=carried,
    )


# --- setup (the trusted host-mutation seam) ------------------------------------


def _run_setup(
    commands: list[str], *, repo: Path | None, run_dir: RunDir, base: str | None
) -> str | None:
    """Run the planner-declared setup commands in order; return the first failure's
    message or ``None`` on success.

    BONES safety boundary: these are TRUSTED-tier (planner-authored) HOST-GLOBAL
    mutations (system packages, global tooling, shared directories), so the shell is
    intentional (the untrusted worker never reaches this seam). They run in a
    THROWAWAY detached worktree of the epoch base, torn down after, so setup can
    NEVER dirty the operator checkout. Project-LOCAL dependency installs (the
    project's own package manager) do NOT belong here: this throwaway checkout is not
    the task worktrees, so an install run here would not reach them; an implement task
    installs the project deps it needs inside its OWN worktree instead. With no repo
    there is nothing to check out, so the commands run in the run dir (a degenerate
    but honest fallback)."""

    if not commands:
        return None
    if repo is None or base is None:
        return _run_setup_in(commands, run_dir.root)
    cwd = run_dir.worktrees_root / "_setup"
    wt.add_worktree_detached(repo, cwd, ref=base)
    try:
        return _run_setup_in(commands, cwd)
    finally:
        wt.remove_worktree(repo, cwd)


def _run_setup_in(commands: list[str], cwd: Path) -> str | None:
    """Run the setup commands in ``cwd`` in order; first failure's message or None."""

    for cmd in commands:
        try:
            proc = subprocess.run(
                cmd, shell=True, cwd=str(cwd), capture_output=True, text=True,
                timeout=SETUP_TIMEOUT_S,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return f"{cmd!r}: {type(exc).__name__}: {exc}"
        if proc.returncode != 0:
            tail = (proc.stderr.strip() or proc.stdout.strip())[:200]
            return f"{cmd!r}: exit {proc.returncode}: {tail}"
    return None


# --- fan-out + integration -----------------------------------------------------


def _fan_out(
    epoch: Epoch,
    epoch_id: str,
    base: str | None,
    *,
    repo: Path | None,
    run_dir: RunDir,
    backends: Backends,
    journal: JournalWriter,
    read_root: Path | None,
    now_fn: NowFn,
) -> list[TaskResult]:
    """Dispatch the epoch's tasks concurrently (bounded by the backend semaphores,
    NOT the pool) and return their results in task order. Lets ``RateLimited`` from
    any task escape (the caller parks + restarts)."""

    tasks = list(epoch.tasks)
    sink = _JournalAttemptEvents(journal, epoch_id, now_fn)
    results: dict[str, TaskResult] = {}

    def _one(task: Task) -> TaskResult:
        short = task.id
        journal.emit(
            lambda s: TaskDispatched(
                seq=s, ts=now_fn(), epoch_id=epoch_id, task_id=short
            )
        )
        return run_task(
            task, f"{epoch_id}/{task.id}", run_dir=run_dir, repo=repo, base=base,
            backends=backends, read_root=read_root, events=sink,
        )

    with ThreadPoolExecutor(max_workers=min(8, len(tasks))) as pool:
        futures = {pool.submit(_one, t): t.id for t in tasks}
        for fut in as_completed(futures):
            results[futures[fut]] = fut.result()

    ordered = [results[t.id] for t in tasks]
    for task in tasks:
        if results[task.id].outcome == "passed":
            short = task.id
            journal.emit(
                lambda s: TaskDone(
                    seq=s, ts=now_fn(), epoch_id=epoch_id, task_id=short
                )
            )
    return ordered


def _grind_epoch(
    epoch: Epoch,
    epoch_id: str,
    base: str | None,
    *,
    repo: Path | None,
    run_dir: RunDir,
    run_branch: str | None,
    backends: Backends,
    journal: JournalWriter,
    log_root: Path,
    sleep_fn: SleepFn,
    backoff_s: float,
    now_fn: NowFn,
) -> list[TaskResult]:
    """Run one epoch's fan-out to completion, RESTARTING the whole epoch after a
    node-#1 backoff if any task rate-limits (the partial work lives only in throwaway
    worktrees, razed by the restart). A read-only integration-tip worktree is checked
    out for the non-write tasks to read + cite."""

    read_needed = any(t.mode != "implement" for t in epoch.tasks)
    while True:
        read_root: Path | None = None
        if repo is not None and base is not None and read_needed:
            read_root = run_dir.worktrees_root / "_read_tip"
            wt.add_worktree_detached(repo, read_root, ref=base)
        try:
            results = _fan_out(
                epoch, epoch_id, base, repo=repo, run_dir=run_dir, backends=backends,
                journal=journal, read_root=read_root, now_fn=now_fn,
            )
        except WorkerRateLimited as exc:
            if read_root is not None and repo is not None:
                wt.remove_worktree(repo, read_root)
            detail = str(exc)
            journal.emit(
                lambda s: RateLimitedEvent(
                    seq=s, ts=now_fn(), role="worker", detail=detail
                )
            )
            _raze_epoch(run_dir, repo, run_branch, epoch_id, log_root)
            sleep_fn(backoff_s)
            continue
        if read_root is not None and repo is not None:
            wt.remove_worktree(repo, read_root)
        return results


@dataclass(frozen=True)
class _Staging:
    """The epoch's PRE-finalize integration disposition. ``conflict`` (set) is a hard
    error the close-out planner records in the baton (an ownership overlap or a merge
    conflict = the planner mis-scoped); ``staging_branch`` / ``staging_tip`` (set) hold
    the merged-but-not-yet-fast-forwarded work the close-out reads and ``_finalize_epoch``
    consumes. All-``None`` means nothing merged (finalize is a no-op fast-forward)."""

    conflict: str | None
    staging_branch: str | None
    staging_tip: str | None


def _ownership_overlap(
    repo: Path,
    base: str,
    passed: list[tuple[str, str, Task]],
) -> str | None:
    """Invariant #1: the PASSing implement tasks' realized ownership must be pairwise
    disjoint. Each task already wrote ONLY within its declared globs (the run_task
    scope check), so a file one task changed that also matches ANOTHER task's
    ownership globs is a true overlap (the planner gave colliding scopes). Returns the
    first overlap message, or ``None`` when disjoint."""

    changed = {tid: wt.changed_paths(repo, base, branch) for tid, branch, _ in passed}
    for i in range(len(passed)):
        for j in range(i + 1, len(passed)):
            tid_a, _, task_a = passed[i]
            tid_b, _, task_b = passed[j]
            own_a = list(task_a.file_ownership)
            own_b = list(task_b.file_ownership)
            for path in changed[tid_a]:
                if wt.path_in_scope(path, own_b):
                    return f"{tid_a} and {tid_b} both own {path}"
            for path in changed[tid_b]:
                if wt.path_in_scope(path, own_a):
                    return f"{tid_b} and {tid_a} both own {path}"
    return None


def _integrate_to_staging(
    epoch_id: str,
    base: str | None,
    *,
    repo: Path | None,
    run_dir: RunDir,
    run_branch: str | None,
    results: list[TaskResult],
    tasks_by_id: dict[str, Task],
) -> _Staging:
    """The disjoint-ownership merge (invariant #1), STOPPED before the fast-forward:
    verify the PASSing implement tasks own disjoint files, merge each wip branch (in
    task order) onto a fresh staging branch off the epoch base, and return the staging
    tip WITHOUT fast-forwarding the run branch and WITHOUT deleting the staging branch /
    wip (the close-out reads the staging tree, ``_finalize_epoch`` consumes it). An
    overlap or a merge conflict is a hard error the close-out records in the baton, the
    run branch left UNTOUCHED. No passers -> all-``None`` (finalize is a no-op)."""

    passed = [
        (r.task_id, r.branch, tasks_by_id[r.task_id])
        for r in results
        if r.outcome == "passed" and r.branch is not None
    ]
    if not passed:
        return _Staging(conflict=None, staging_branch=None, staging_tip=None)
    assert repo is not None and base is not None and run_branch is not None

    overlap = _ownership_overlap(repo, base, passed)
    if overlap is not None:
        for _, branch, _ in passed:
            if branch is not None:
                wt.delete_branch(repo, branch)
        wt.prune_tree(repo, run_dir.worktrees_root)
        return _Staging(
            conflict=f"ownership overlap: {overlap}",
            staging_branch=None,
            staging_tip=None,
        )

    staging = f"grind-wip/{run_dir.root.name}/{epoch_id.replace('/', '-')}/_staging"
    wt.delete_branch(repo, staging)
    wt.ensure_integration_branch(repo, staging, base)
    int_wt = run_dir.worktrees_root / "_staging"
    wt.add_worktree_on(repo, int_wt, branch=staging)
    for task_id, branch, _ in passed:
        assert branch is not None
        outcome = wt.merge_into(int_wt, branch)
        if not outcome.ok:
            wt.remove_worktree(repo, int_wt)
            wt.delete_branch(repo, staging)
            for _, b, _ in passed:
                if b is not None:
                    wt.delete_branch(repo, b)
            wt.prune_tree(repo, run_dir.worktrees_root)
            return _Staging(
                conflict=f"merge conflict on {task_id}: {outcome.conflict}",
                staging_branch=None,
                staging_tip=None,
            )
    wt.remove_worktree(repo, int_wt)
    staging_tip = wt.resolve_commit(repo, staging)
    wt.prune_tree(repo, run_dir.worktrees_root)
    # Deliberately NO fast-forward and NO branch deletion here: the staging tree is what
    # close-out reads, and ``_finalize_epoch`` does the durable fast-forward + cleanup.
    return _Staging(conflict=None, staging_branch=staging, staging_tip=staging_tip)


def _task_outcomes(
    results: list[TaskResult], tasks_by_id: dict[str, Task]
) -> tuple[TaskOutcome, ...]:
    """Project the in-memory ``TaskResult``s into pure close-out references: the
    deterministic terminal, the keyed-log handoff/verdict pointers, and the model's
    verbatim reason. Python labels NOTHING (partial / none / regression is the
    close-out planner's read of the pointed-to files)."""

    out: list[TaskOutcome] = []
    for r in results:
        task = tasks_by_id.get(r.task_id)
        out.append(
            TaskOutcome(
                task_id=r.task_id,
                mode=task.mode if task is not None else "",
                outcome=r.outcome,
                handoff_key=(
                    f"{r.task_id}/{HANDOFF_FILENAME}"
                    if r.handoff_path is not None
                    else None
                ),
                verdict_key=(
                    f"{r.task_id}/{CRITIC_VERDICT_FILENAME}"
                    if r.verdict is not None
                    else None
                ),
                reason=r.reason or (r.verdict.reason if r.verdict is not None else ""),
            )
        )
    return tuple(out)


def _finalize_epoch(
    staging: _Staging,
    baton_text: str,
    epoch_index: int,
    *,
    repo: Path | None,
    run_dir: RunDir,
    run_branch: str | None,
    journal: JournalWriter,
    now_fn: NowFn,
    strike_entries: list[StrikeLedgerEntry] | None = None,
) -> None:
    """The atomic durable commit point, run AFTER close-out wrote the baton: if work
    merged, fast-forward the run branch to the staging tip and delete the staging +
    every transient ``grind-wip/`` branch (this epoch's only); then persist the baton
    to ``E<n>/baton.md``, emit the strike-ledger snapshot (when the ladder changed this
    epoch), and emit ``EpochCompleted`` (which now IMPLIES "baton written" AND "strike
    snapshot final": both are emitted immediately before it, so an epoch that crashes
    before ``EpochCompleted`` leaves no counted strike snapshot). With no merged work
    (no passers / setup failure / conflict) there is nothing to fast-forward, but the
    epoch still completes WITH a baton."""

    epoch_id = f"E{epoch_index}"
    if staging.staging_tip is not None:
        assert repo is not None and run_branch is not None
        wt.fast_forward_branch(repo, run_branch, staging.staging_tip)
        for branch in wt.branches_with_prefix(repo, "grind-wip/"):
            wt.delete_branch(repo, branch)
        wt.prune_tree(repo, run_dir.worktrees_root)
    baton_path = run_dir.baton_path(epoch_index)
    baton_path.parent.mkdir(parents=True, exist_ok=True)
    baton_path.write_text(baton_text, encoding="utf-8")
    if strike_entries is not None:
        entries = strike_entries
        journal.emit(
            lambda s: StrikeLedger(
                seq=s, ts=now_fn(), epoch_id=epoch_id, entries=entries
            )
        )
    journal.emit(lambda s: EpochCompleted(seq=s, ts=now_fn(), epoch_id=epoch_id))


# --- log reaping + raze (the resume / restart cleanup) -------------------------


def _reap_epoch_logs(log_root: Path, epoch_id: str) -> None:
    """Delete one epoch's raw worker/critic stdout dirs (200-500 MB/task pure
    debugging scratch). The keyed log + events are kept forever; only the raw logs
    are ephemeral. ``ScriptWorker`` writes ``<slug>-<kind>`` dirs, so an epoch's logs
    share the ``<epoch-slug>-`` prefix (the trailing dash disambiguates E3 from E30)."""

    if not log_root.is_dir():
        return
    prefix = epoch_id.replace("/", "-") + "-"
    for entry in log_root.iterdir():
        if entry.name.startswith(prefix):
            if entry.is_dir():
                shutil.rmtree(entry, ignore_errors=True)
            else:
                entry.unlink(missing_ok=True)


def _reap_prior_logs(log_root: Path, epoch_index: int) -> None:
    """At each epoch's start, reap the PRIOR epoch's raw logs (keep only the latest
    epoch's full raw logs, so an overnight run cannot fill the disk)."""

    if epoch_index > 1:
        _reap_epoch_logs(log_root, f"E{epoch_index - 1}")


def _raze_epoch(
    run_dir: RunDir,
    repo: Path | None,
    run_branch: str | None,
    epoch_id: str,
    log_root: Path,
) -> None:
    """The crash-only cleanup primitive (resume AND the rate-limit restart): drop the
    throwaway worktrees + every transient ``grind-wip/`` branch (the durable run
    branch is never under that prefix), reap the incomplete epoch's partial keyed log
    + raw logs, and PRESERVE the completed-epoch keyed log + the append-only journal.
    The git tip needs no rewind (the run branch is already at the last clean
    boundary). Idempotent."""

    if repo is not None:
        wt.prune_tree(repo, run_dir.worktrees_root)
        for branch in wt.branches_with_prefix(repo, "grind-wip/"):
            wt.delete_branch(repo, branch)
    try:
        partial = run_dir.resolve(epoch_id)
    except ValueError:
        partial = None
    if partial is not None and partial.is_dir():
        shutil.rmtree(partial, ignore_errors=True)
    _reap_epoch_logs(log_root, epoch_id)


# --- terminal emitters ---------------------------------------------------------


def _park_event(
    pk: strikes.ParkedTask, epoch_id: str, now_fn: NowFn
) -> Callable[[int], TaskParked]:
    """A seq-assigning factory for the ``task_parked`` event (one parked lineage)."""

    return lambda s: TaskParked(
        seq=s, ts=now_fn(), epoch_id=epoch_id, task_id=pk.task_id,
        strikes=pk.strikes, reason=pk.reason, descriptor=pk.descriptor,
    )


def _result_reason(result: TaskResult) -> str:
    """The strike's recorded reason for a non-landed task: the escalation reason, else
    the critic verdict's reason, else a terse default (never empty in the ledger)."""

    if result.reason:
        return result.reason
    if result.verdict is not None and result.verdict.reason:
        return result.verdict.reason
    return "did not land"


def _append_parked(summary: str, run_dir: RunDir) -> str:
    """Append the parked-lineage block to a terminal summary so the operator sees "the
    rig could not close these N tasks". Reconstructed from the journal (the last
    completed strike snapshot), so it is resume-stable. A no-park run adds nothing (the
    summary stays byte-identical to today)."""

    entries = strikes.reconstruct_entries(read_events(run_dir.events_path))
    block = strikes.summarize_parked(entries)
    return f"{summary}\n\n{block}" if block else summary


def _emit_completed(journal: JournalWriter, now_fn: NowFn) -> None:
    journal.emit(lambda s: RunCompleted(seq=s, ts=now_fn()))


def _emit_ended(journal: JournalWriter, now_fn: NowFn, summary: str) -> None:
    journal.emit(lambda s: RunEnded(seq=s, ts=now_fn(), summary=summary))


# --- the boundary call (node #1 backoff) ---------------------------------------


def _plan_with_backoff(
    planner: Planner,
    context: PlannerContext,
    *,
    journal: JournalWriter,
    sleep_fn: SleepFn,
    backoff_s: float,
    now_fn: NowFn,
) -> Decision:
    """Call the planner, PARKING (~1/hr) and re-issuing on a rate limit (node #1).
    A non-rate-limit ``PlannerError`` escapes (the loop ends cleanly, node #2)."""

    while True:
        try:
            return planner.decide(context)
        except PlannerRateLimited as exc:
            detail = str(exc)
            journal.emit(
                lambda s: RateLimitedEvent(
                    seq=s, ts=now_fn(), role="planner", detail=detail
                )
            )
            sleep_fn(backoff_s)


# --- run-scoped SIGTERM/SIGINT reaping (resumable stop) ------------------------


class _Interrupted(BaseException):
    """A run-scoped SIGTERM / SIGINT: the handler already reaped the in-flight
    subprocess groups; the run boundary turns this into a RESUMABLE partial-end.

    A ``BaseException`` (like ``KeyboardInterrupt``), NOT an ``Exception``, so the
    epoch body's broad ``except Exception`` (the abort node) and the worker / critic
    transport ``except Exception`` never swallow it: it propagates straight to the
    run boundary, which ends cleanly (``ended``) and leaves resume to re-enter."""

    def __init__(self, signum: int) -> None:
        super().__init__(f"interrupted by signal {signum}")
        self.signum = signum


def _install_reaper_signals() -> dict[int, Any] | None:
    """Install the run-scoped SIGTERM/SIGINT handler (reap groups, then raise
    ``_Interrupted``); return the prior handlers to restore.

    Returns ``None`` (a no-op) when NOT on the main thread: ``signal.signal`` is
    illegal off the main thread, and a library caller that drives a run from a worker
    thread owns its own signal disposition, which is not ours to touch. Only the kill
    of PROCESSES happens here, never git or disk: resume owns scratch cleanup."""

    if threading.current_thread() is not threading.main_thread():
        return None

    def _handler(signum: int, frame: FrameType | None) -> None:
        reaper.reap_all()
        raise _Interrupted(signum)

    prior: dict[int, Any] = {}
    for sig in (signal.SIGTERM, signal.SIGINT):
        prior[sig] = signal.signal(sig, _handler)
    return prior


def _restore_reaper_signals(prior: dict[int, Any] | None) -> None:
    """Restore the handlers ``_install_reaper_signals`` replaced (no-op off-main)."""

    if prior is None:
        return
    for sig, handler in prior.items():
        signal.signal(sig, handler)


def _run_interruptible(
    body: Callable[[], RunResult],
    *,
    journal: JournalWriter,
    run_dir: RunDir,
    now_fn: NowFn,
) -> RunResult:
    """Run the epoch loop ``body`` under the run-scoped SIGTERM/SIGINT reaper, turning
    an interrupt into a RESUMABLE partial-end and restoring the prior handlers on exit.

    On ``_Interrupted`` (the handler already reaped the in-flight subprocess groups)
    this emits ``RunEnded`` (the existing "stopped, resumable" terminal, exit code 1)
    and returns ``ended`` with the completed-epoch count read back from the durable
    journal. It mutates NO git/disk state: the run branch only ever advanced on a
    completed epoch, the journal is append-only and fsynced per event, and resume owns
    all scratch cleanup. A subsequent ``resume_run`` re-enters exactly as today."""

    prior = _install_reaper_signals()
    try:
        return body()
    except _Interrupted as exc:
        summary = _append_parked(
            f"run interrupted by signal {exc.signum}; resume to continue", run_dir
        )
        _emit_ended(journal, now_fn, summary)
        return RunResult(
            "ended", summary, _completed_count(read_events(run_dir.events_path))
        )
    finally:
        _restore_reaper_signals(prior)


# --- the core loop -------------------------------------------------------------


def _drive(
    *,
    job: str,
    run_dir: RunDir,
    repo: Path | None,
    run_branch: str | None,
    planner: Planner,
    backends: Backends,
    max_epochs: int,
    log_root: Path,
    acceptance: AcceptanceCheck | None,
    sleep_fn: SleepFn,
    backoff_s: float,
    now_fn: NowFn,
    journal: JournalWriter,
    start_index: int,
) -> RunResult:
    """The shared epoch loop (entered fresh by ``start_run`` and re-entered by
    ``resume_run``). Drives boundaries from ``start_index`` until the planner ends, an
    unrecoverable planner failure forces a clean end, or ``max_epochs`` is hit. Each
    boundary the PLAN call reads the prior epoch's baton (rebuilt from disk); each
    epoch ends with the CLOSE-OUT call writing the new baton, which finalize persists.
    No in-memory failure context is threaded between epochs: the baton on disk is the
    only memory, so resume needs no reconstruction."""

    epoch_index = start_index
    consec_aborts = 0
    while epoch_index <= max_epochs:
        _reap_prior_logs(log_root, epoch_index)
        tip_ref = _resolve_tip(repo, run_branch)
        # Strike ladder: rebuild the struck-lineage ledger FROM DISK (resume-safe) and
        # render the carried nudge into the PLAN context BEFORE the planner decides.
        ledger = strikes.reconstruct_entries(read_events(run_dir.events_path))
        context = _build_context(
            job=job, repo=repo, run_dir=run_dir, run_branch=run_branch,
            tip_ref=tip_ref, epoch_index=epoch_index, max_epochs=max_epochs,
            carried=strikes.render_carried(ledger),
        )
        try:
            decision = _plan_with_backoff(
                planner, context, journal=journal, sleep_fn=sleep_fn,
                backoff_s=backoff_s, now_fn=now_fn,
            )
        except PlannerError as exc:
            summary = f"planner could not continue at E{epoch_index}: {exc}"
            summary = _append_parked(summary, run_dir)
            _emit_ended(journal, now_fn, summary)
            return RunResult("ended", summary, epoch_index - 1)

        if isinstance(decision, EndDecision):
            if acceptance is None or acceptance(context):
                _emit_completed(journal, now_fn)
                return RunResult(
                    "completed", _append_parked(decision.summary, run_dir),
                    epoch_index - 1,
                )
            summary = _append_parked(decision.summary, run_dir)
            _emit_ended(journal, now_fn, summary)
            return RunResult("ended", summary, epoch_index - 1)

        epoch = decision.epoch
        epoch_id = f"E{epoch_index}"
        base = tip_ref
        # The plan-time rung (DETERMINISTIC, pre-dispatch): match the proposed tasks to
        # the struck ledger and PARK (BLOCK) any lineage at strike 2 (its second
        # whole-ladder failure). Only the kept set is dispatched; the parked drops are
        # journaled here. Senior is reached in-epoch now, so there is no tier override.
        ladder = strikes.apply_ladder(epoch.tasks, ledger)
        for pk in ladder.parked:
            # ``emit`` invokes the factory synchronously, so the loop var is captured
            # before it advances (no late-binding hazard, no default-arg needed).
            journal.emit(_park_event(pk, epoch_id, now_fn))
        kept = list(ladder.tasks)
        parked_descriptors = tuple(pk.descriptor for pk in ladder.parked)
        try:
            if kept:
                run_epoch = epoch.model_copy(update={"tasks": kept})
                setup_err = _run_setup(
                    list(run_epoch.setup), repo=repo, run_dir=run_dir, base=base,
                )
            else:
                # Every proposed task is a parked lineage: nothing to grind. Skip setup
                # too (no work to prepare), close out, and advance so the planner re-plans.
                run_epoch = epoch
                setup_err = None
            journal.emit(
                lambda s: EpochStarted(
                    seq=s, ts=now_fn(), epoch_id=epoch_id, title=epoch.title,
                    tasks=[TaskRef(id=t.id, mode=t.mode) for t in kept],
                )
            )
            if kept and setup_err is None:
                results = _grind_epoch(
                    run_epoch, epoch_id, base, repo=repo, run_dir=run_dir,
                    run_branch=run_branch, backends=backends, journal=journal,
                    log_root=log_root, sleep_fn=sleep_fn, backoff_s=backoff_s,
                    now_fn=now_fn,
                )
                tasks_by_id = {f"{epoch_id}/{t.id}": t for t in kept}
                staging = _integrate_to_staging(
                    epoch_id, base, repo=repo, run_dir=run_dir, run_branch=run_branch,
                    results=results, tasks_by_id=tasks_by_id,
                )
            else:
                # A bad setup command will not pass on re-run (and an all-parked epoch
                # has nothing to grind); skip the grind, let the close-out baton record
                # it, and finalize so the planner sees it next boundary (NOT an abort).
                results = []
                tasks_by_id = {}
                staging = _Staging(conflict=None, staging_branch=None, staging_tip=None)
            closeout_ctx = CloseoutContext(
                job=job,
                repo=repo,
                run_dir=run_dir,
                staging_ref=staging.staging_tip or base,
                prior_baton=context.baton,
                epoch_index=epoch_index,
                epoch_id=epoch_id,
                title=epoch.title,
                task_outcomes=_task_outcomes(results, tasks_by_id),
                setup_error=setup_err,
                integration_conflict=staging.conflict,
                pending_additions=tuple(decision.pending),
                parked=parked_descriptors,
            )
            # CLOSE-OUT (node #1 parks): a RateLimited escapes to the handler below,
            # which razes + restarts the whole epoch (no advance). The baton is written
            # by finalize, BEFORE the EpochCompleted that marks the epoch done.
            baton_text = planner.close_out(closeout_ctx)
            # Recompute the struck ledger from this epoch's deterministic outcomes; emit
            # the new snapshot only when it CHANGED (so a no-carry run's journal stays
            # byte-identical, while a transition to empty is still recorded).
            new_ledger = strikes.next_ledger(
                ledger,
                landed=[tasks_by_id[r.task_id] for r in results if r.outcome == "passed"],
                failed=[
                    (tasks_by_id[r.task_id], _result_reason(r))
                    for r in results
                    if r.outcome != "passed"
                ],
            )
            snapshot = (
                strikes.to_event_entries(new_ledger) if new_ledger != ledger else None
            )
            _finalize_epoch(
                staging, baton_text, epoch_index, repo=repo, run_dir=run_dir,
                run_branch=run_branch, journal=journal, now_fn=now_fn,
                strike_entries=snapshot,
            )
            consec_aborts = 0
        except WorkerRateLimited:
            # Node #1 stays un-burned (defensive: _grind_epoch already parks on it).
            raise
        except PlannerRateLimited as exc:
            # Node #1, the CLOSE-OUT rate limit: RAZE the in-flight epoch (drop staging
            # + wip), PARK, then RESTART the SAME epoch whole (re-grind). The run is
            # never burned and no half-summarized epoch is left behind.
            detail = str(exc)
            journal.emit(
                lambda s: RateLimitedEvent(
                    seq=s, ts=now_fn(), role="planner", detail=detail
                )
            )
            _raze_epoch(run_dir, repo, run_branch, epoch_id, log_root)
            sleep_fn(backoff_s)
            continue
        except Exception as exc:
            # BONES node #2: an UNEXPECTED exception (a GitError/OSError escaping the
            # worktree/integration machinery, or a close-out transport hard error)
            # RAZES + RESTARTS the SAME epoch. An aborted epoch has NO baton, so it is
            # never completed and never advances. K consecutive aborts clean-end the
            # run so a persistent infra fault cannot infinite-loop an unattended run.
            consec_aborts += 1
            detail = (
                f"E{epoch_index} aborted on an unexpected "
                f"{type(exc).__name__}: {exc}"
            )
            _raze_epoch(run_dir, repo, run_branch, epoch_id, log_root)
            if consec_aborts >= MAX_CONSECUTIVE_ABORTS:
                summary = _append_parked(
                    f"{consec_aborts} consecutive epochs aborted on unexpected "
                    f"errors (persistent infra fault); last: {detail}",
                    run_dir,
                )
                _emit_ended(journal, now_fn, summary)
                return RunResult("ended", summary, epoch_index - 1)
            continue
        epoch_index += 1

    summary = _append_parked(
        f"max epochs ({max_epochs}) reached without a planned end", run_dir
    )
    _emit_ended(journal, now_fn, summary)
    return RunResult("ended", summary, epoch_index - 1)


# --- public callables (planner + backends injected) ----------------------------


def start_run(
    *,
    job_path: Path,
    run_dir: RunDir,
    repo: Path | None,
    planner: Planner,
    backends: Backends,
    max_epochs: int = DEFAULT_MAX_EPOCHS,
    log_root: Path | None = None,
    acceptance: AcceptanceCheck | None = None,
    sleep_fn: SleepFn = time.sleep,
    backoff_s: float = DEFAULT_BACKOFF_S,
    now_fn: NowFn = _now,
) -> RunResult:
    """Drive a FRESH run from its first boundary to a clean terminal. The run dir is
    created by the caller; this opens the journal, emits ``run_started``, and loops."""

    job = job_path.read_text(encoding="utf-8")
    run_id = run_dir.root.name
    run_branch = _run_branch(run_id) if repo is not None else None
    log_root = log_root if log_root is not None else run_dir.root / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    with JournalWriter(run_dir.events_path) as journal:
        journal.emit(
            lambda s: RunStarted(
                seq=s, ts=now_fn(), run_id=run_id, job_path=str(job_path),
                max_epochs=max_epochs,
            )
        )
        return _run_interruptible(
            lambda: _drive(
                job=job, run_dir=run_dir, repo=repo, run_branch=run_branch,
                planner=planner, backends=backends, max_epochs=max_epochs,
                log_root=log_root, acceptance=acceptance, sleep_fn=sleep_fn,
                backoff_s=backoff_s, now_fn=now_fn, journal=journal, start_index=1,
            ),
            journal=journal, run_dir=run_dir, now_fn=now_fn,
        )


def resume_run(
    *,
    run_dir: RunDir,
    repo: Path | None,
    planner: Planner,
    backends: Backends,
    max_epochs: int | None = None,
    log_root: Path | None = None,
    acceptance: AcceptanceCheck | None = None,
    sleep_fn: SleepFn = time.sleep,
    backoff_s: float = DEFAULT_BACKOFF_S,
    now_fn: NowFn = _now,
) -> RunResult:
    """Re-enter a killed / crashed / rate-limited run from its last clean boundary
    (the universal crash-only recovery primitive). Razes the incomplete epoch (no git
    rewind needed), appends a ``run_resumed`` marker, and re-enters the loop."""

    events = read_events(run_dir.events_path)
    started = next((e for e in events if isinstance(e, RunStarted)), None)
    if started is None:
        raise ValueError(f"no run_started in {run_dir.events_path}; nothing to resume")
    if any(isinstance(e, RunCompleted) for e in events):
        return RunResult("completed", "already completed", _completed_count(events))

    run_id = started.run_id
    job_path = Path(started.job_path)
    job = job_path.read_text(encoding="utf-8") if job_path.is_file() else ""
    eff_max = (
        max_epochs if max_epochs is not None
        else (started.max_epochs if started.max_epochs is not None else DEFAULT_MAX_EPOCHS)
    )
    run_branch = _run_branch(run_id) if repo is not None else None
    log_root = log_root if log_root is not None else run_dir.root / "logs"
    log_root.mkdir(parents=True, exist_ok=True)

    completed = _completed_count(events)
    start_index = completed + 1
    in_flight_id = f"E{start_index}"
    started_ids = {e.epoch_id for e in events if isinstance(e, EpochStarted)}
    razed = in_flight_id if in_flight_id in started_ids else None
    # Raze the in-flight epoch's debris INCLUDING any never-finalized E<n>/baton.md
    # (only a completed epoch persists a baton); the next PLAN re-reads the prior
    # completed epoch's baton from disk, so no failure context need be reconstructed.
    _raze_epoch(run_dir, repo, run_branch, in_flight_id, log_root)

    with JournalWriter(run_dir.events_path) as journal:
        journal.emit(
            lambda s: RunResumed(seq=s, ts=now_fn(), run_id=run_id, razed_epoch=razed)
        )
        return _run_interruptible(
            lambda: _drive(
                job=job, run_dir=run_dir, repo=repo, run_branch=run_branch,
                planner=planner, backends=backends, max_epochs=eff_max,
                log_root=log_root, acceptance=acceptance, sleep_fn=sleep_fn,
                backoff_s=backoff_s, now_fn=now_fn, journal=journal,
                start_index=start_index,
            ),
            journal=journal, run_dir=run_dir, now_fn=now_fn,
        )


def _completed_count(events: Sequence[Event]) -> int:
    """Distinct completed epochs in the journal (the resume start cursor)."""

    return len({e.epoch_id for e in events if isinstance(e, EpochCompleted)})


# --- CLI seams (config -> real planner + backends -> a process exit code) ------

#: Terminal status -> process exit code: a clean ``completed`` is success; a
#: ``ended`` partial-end is a non-zero "stopped, resumable" signal (the CLI surfaces
#: a config / no-such-run error as 2, distinct from a real partial-end).
_EXIT: dict[str, int] = {"completed": 0, "ended": 1}


def _default_run_id() -> str:
    """A UTC timestamp slug, e.g. ``20260624T142530Z`` (collision-resistant id)."""

    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _require_config(repo: Path) -> GrindstoneConfig:
    """Load + RCE-guard the repo config, or fail loudly toward ``grindstone init``.

    The core ships no rig-specific defaults (ARCHITECTURE.md), so an absent config is
    a hard error, and the loaded config is attacker-controlled (a cloned repo carries
    its own), so every configured ``script:`` is path-guarded before it is ever run.
    """

    cfg = load_config(repo)
    if cfg is None:
        raise FileNotFoundError(
            f"no .grindstone/config.yaml under {repo}; run `grindstone init` first "
            "(core ships no rig defaults)"
        )
    validate_script_paths(cfg)
    return cfg


def _build_planner(cfg: GrindstoneConfig, repo: Path) -> ScriptPlanner:
    """The real stateless planner: a ``ScriptPlanner`` over the ``planner`` role's
    request script (the script owns transport + model identity + GPU arbitration)."""

    rc = cfg.roles.planner
    transport = ScriptPlannerTransport(
        script=resolve_role_script("planner", rc),
        stop_script=models_script("stop.sh", rig=rc.rig),
        repo=repo,
        slots=rc.slots,
        timeout_s=rc.timeout_s,
    )
    return ScriptPlanner(transport=transport)


def _acceptance_for(cfg: GrindstoneConfig) -> AcceptanceCheck | None:
    """Invariant #2 from config: the job's ``done_when`` (run once in a tip worktree),
    or ``None`` to trust the planner's END when no acceptance is configured."""

    return make_acceptance(cfg.done_when) if cfg.done_when is not None else None


def run(job_path: Path, repo_root: Path, *, run_id: str | None = None) -> int:
    """Drive a FRESH job to a clean terminal: build the real planner + backends from
    the repo config, create the run dir, and loop. Returns a process exit code
    (``completed`` -> 0, ``ended`` -> 1). Raises ``FileNotFoundError`` / ``ValueError``
    on a missing job, an absent config, or an unsafe config (the CLI maps those to 2)."""

    repo = repo_root.resolve()
    cfg = _require_config(repo)
    run_id = run_id or _default_run_id()
    run_dir = create_run_dir(repo, run_id)
    reap_sibling_journals(run_dir)
    log_root = run_dir.root / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    print(f"run {run_id} -> {run_dir.root}")

    result = start_run(
        job_path=job_path,
        run_dir=run_dir,
        repo=repo,
        planner=_build_planner(cfg, repo),
        backends=build_backends(cfg, log_root=log_root),
        max_epochs=cfg.max_epochs if cfg.max_epochs is not None else DEFAULT_MAX_EPOCHS,
        log_root=log_root,
        acceptance=_acceptance_for(cfg),
    )
    write_journal(run_dir)
    print(f"{result.status}: {result.summary}")
    return _EXIT[result.status]


def resume(run_id: str, repo_root: Path) -> int:
    """Re-enter a killed / crashed / rate-limited run from its last clean boundary
    (BONES universal recovery primitive). Returns a process exit code. Raises
    ``FileNotFoundError`` (no config, or no such run) / ``ValueError`` (unsafe config),
    which the CLI maps to 2."""

    repo = repo_root.resolve()
    cfg = _require_config(repo)
    run_dir = RunDir(root=repo / ".grindstone" / "runs" / run_id)
    if not run_dir.events_path.is_file():
        raise FileNotFoundError(
            f"no run {run_id!r} under {repo} (no events.ndjson at {run_dir.events_path})"
        )
    log_root = run_dir.root / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    print(f"resume {run_id} -> {run_dir.root}")

    result = resume_run(
        run_dir=run_dir,
        repo=repo,
        planner=_build_planner(cfg, repo),
        backends=build_backends(cfg, log_root=log_root),
        max_epochs=cfg.max_epochs,
        log_root=log_root,
        acceptance=_acceptance_for(cfg),
    )
    write_journal(run_dir)
    print(f"{result.status}: {result.summary}")
    return _EXIT[result.status]
