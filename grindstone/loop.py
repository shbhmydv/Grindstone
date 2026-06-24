"""The epoch driver: the BONES state machine, the two failure nodes, and resume.

A synchronous, dumb Python loop; ALL sequencing judgment is delegated to the
planner. Each boundary rebuilds the planner's context fresh FROM DISK (the
integration tip + the durable keyed log + the job spec + the prior epoch's carried
failures), asks the planner for ONE decision, and disposes of it:

    EpochDecision -> run the planner-declared setup (the trusted host-mutation
        seam), fan the epoch's disjoint tasks out under ONE shared ``Backends``
        (the per-endpoint semaphores are the real concurrency bound), integrate the
        PASSing implement tasks by the disjoint-ownership merge invariant, and
        fast-forward the durable run branch ONLY on epoch completion (so its tip is
        ALWAYS a clean checkpoint resume can re-enter from).
    EndDecision  -> run the one final acceptance (invariant #2, an injected seam):
        pass -> ``completed``; otherwise persist the summary as the resume seed and
        end cleanly (``ended``, failure node #2).

Failure model (exactly two nodes):

  #1 RATE LIMIT (planner OR worker / critic): PARK, back off ~1/hr (injectable),
     then re-enter. A planner rate-limit re-issues the boundary call; a mid-epoch
     worker rate-limit RAZES the in-flight epoch's throwaway worktrees and RESTARTS
     the epoch whole (partial state is never trusted).
  #2 CANNOT CONTINUE (any other epoch failure): it becomes carried context the
     planner sees next boundary and steers around, or the planner ends cleanly. The
     ``max_epochs`` backstop is the INVOLUNTARY trigger of the same clean end.

RESUME is the universal crash-only recovery primitive: because the run branch only
fast-forwards on completion, the git tip needs no rewind. ``resume_run`` razes the
incomplete epoch's worktrees + wip branches + partial keyed log, PRESERVES the
completed-epoch keyed log + the append-only journal (it APPENDS a razed-epoch
marker, never truncates), and re-enters the loop from the planner call.

This module exposes the testable CORE callables (``start_run`` / ``resume_run``)
with the planner + backends injected; the CLI-facing ``run`` / ``resume`` (which
build the real planner + backends from config) are wired in a later part.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Literal, Protocol, Sequence

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
    EpochCarried,
    EpochCompleted,
    EpochStarted,
    HandoffAccepted,
    HandoffRejected,
    JournalWriter,
    RateLimited as RateLimitedEvent,
    RunCompleted,
    RunEnded,
    RunResumed,
    RunStarted,
    TaskDispatched,
    TaskDone,
    TaskRef,
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
    RateLimited as WorkerRateLimited,
    TaskResult,
    run_task,
)

#: The built-in epoch backstop when the config sets no ``max_epochs`` (BONES: the
#: cap is the involuntary trigger of the clean partial-end, never unbounded).
DEFAULT_MAX_EPOCHS = 40

#: BONES failure node #2 backstop: K CONSECUTIVE epochs aborting on an UNEXPECTED
#: error (a GitError/OSError escaping the worktree/integration machinery, not a
#: planned task failure) clean-ends the run, so a persistent infra fault cannot
#: infinite-loop an unattended run. A single transient fault is just carried.
MAX_CONSECUTIVE_ABORTS = 3

#: Node-#1 backoff (~1/hr). Injected as ``sleep_fn``/``backoff_s`` so tests park
#: without a real wall clock.
DEFAULT_BACKOFF_S = 3600.0

#: Wall-clock cap on each planner-declared setup command (the trusted-tier
#: host-mutation seam runs them via the shell before the tasks).
SETUP_TIMEOUT_S = 1800.0

#: The single phase prefix (BONES is epochs-only; the keyed-log grammar still wants
#: ``P*/E*/T*``, so every epoch lives under one fixed phase).
_PHASE = "P1"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


SleepFn = Callable[[float], None]
NowFn = Callable[[], str]


# --- the planner seam (the real planner is a later part) -----------------------


@dataclass(frozen=True)
class PlannerContext:
    """The bounded window the loop reconstructs FROM DISK each boundary and hands the
    stateless planner. The mock ignores it; the real planner (a later part) renders
    its prompt from it. Externalized context: re-derived, never accumulated."""

    job: str
    repo: Path | None
    run_dir: RunDir
    run_branch: str | None
    #: The integration-tip commit (the durable run-branch tip, or repo HEAD before
    #: the first epoch completes). ``None`` when there is no repo.
    tip_ref: str | None
    #: Names (not bodies) of every tracked file at the tip.
    tip_files: tuple[str, ...]
    #: Durable keyed-log references a task may name as ``inputs``.
    log_index: tuple[str, ...]
    #: The prior epoch's non-merged outcomes (blocked / escalated / conflict) the
    #: planner steers around or ends on.
    carried: tuple[str, ...]
    #: The 1-based epoch number about to be planned, and the backstop.
    epoch_index: int
    max_epochs: int


class Planner(Protocol):
    """The stateless one-shot planner the loop calls at each boundary. ``decide``
    returns ONE typed ``Decision`` (epoch or end), or raises ``RateLimited`` (node
    #1) / ``PlannerError`` (node #2)."""

    def decide(self, context: PlannerContext) -> Decision: ...


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
    ``run_task`` works in the fully-qualified ``P*/E*/T*`` for the keyed log."""

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

    def handoff_accepted(self, task_id: str) -> None:
        tid = _short_id(task_id)
        self._journal.emit(
            lambda s: HandoffAccepted(
                seq=s, ts=self._now(), epoch_id=self._epoch_id, task_id=tid
            )
        )

    def handoff_rejected(self, task_id: str, reason: str) -> None:
        tid = _short_id(task_id)
        self._journal.emit(
            lambda s: HandoffRejected(
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
    carried: tuple[str, ...],
    epoch_index: int,
    max_epochs: int,
) -> PlannerContext:
    tip_files: tuple[str, ...] = ()
    if repo is not None and tip_ref is not None:
        tip_files = tuple(wt.list_tree(repo, tip_ref))
    return PlannerContext(
        job=job,
        repo=repo,
        run_dir=run_dir,
        run_branch=run_branch,
        tip_ref=tip_ref,
        tip_files=tip_files,
        log_index=tuple(run_dir.log_index()),
        carried=carried,
        epoch_index=epoch_index,
        max_epochs=max_epochs,
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
    NEVER dirty the operator checkout. Project-LOCAL dependency installs (npm ci /
    pip install) do NOT belong here: this throwaway checkout is not the task
    worktrees, so an install run here would not reach them; an implement task
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
class _Integration:
    """The epoch's integration disposition. ``conflict`` (set) is a hard error
    surfaced to the planner next boundary (an ownership overlap or a merge conflict =
    the planner mis-scoped); ``tip`` (set) is the new run-branch tip on a clean
    fast-forward."""

    conflict: str | None
    tip: str | None


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


def _integrate(
    epoch_id: str,
    base: str | None,
    *,
    repo: Path | None,
    run_dir: RunDir,
    run_branch: str | None,
    results: list[TaskResult],
    tasks_by_id: dict[str, Task],
) -> _Integration:
    """The disjoint-ownership merge (invariant #1): verify the PASSing implement
    tasks own disjoint files, merge each wip branch (in task order) onto a fresh
    staging branch off the epoch base, then FAST-FORWARD the durable run branch to
    the staging tip. An overlap or a merge conflict aborts integration as a hard
    error (carried to the planner), leaving the run branch UNTOUCHED."""

    passed = [
        (r.task_id, r.branch, tasks_by_id[r.task_id])
        for r in results
        if r.outcome == "passed" and r.branch is not None
    ]
    if not passed:
        return _Integration(conflict=None, tip=None)
    assert repo is not None and base is not None and run_branch is not None

    overlap = _ownership_overlap(repo, base, passed)
    if overlap is not None:
        for _, branch, _ in passed:
            if branch is not None:
                wt.delete_branch(repo, branch)
        wt.prune_tree(repo, run_dir.worktrees_root)
        return _Integration(conflict=f"ownership overlap: {overlap}", tip=None)

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
            return _Integration(
                conflict=f"merge conflict on {task_id}: {outcome.conflict}", tip=None
            )
    wt.remove_worktree(repo, int_wt)
    staging_tip = wt.resolve_commit(repo, staging)
    wt.prune_tree(repo, run_dir.worktrees_root)
    wt.fast_forward_branch(repo, run_branch, staging_tip)
    wt.delete_branch(repo, staging)
    for _, branch, _ in passed:
        if branch is not None:
            wt.delete_branch(repo, branch)
    return _Integration(conflict=None, tip=staging_tip)


def _carry(results: list[TaskResult], integration: _Integration) -> tuple[str, ...]:
    """The prior epoch's non-merged outcomes, as context the next planner steers on."""

    out: list[str] = []
    for result in results:
        if result.outcome == "escalated":
            out.append(f"{result.task_id} escalated: {result.reason}")
    if integration.conflict is not None:
        out.append(integration.conflict)
    return tuple(out)


def _journal_carried(
    journal: JournalWriter, now_fn: NowFn, epoch_id: str, reasons: tuple[str, ...]
) -> None:
    """Journal each non-merged outcome (FIX 5): the in-memory carried tuple does not
    survive a crash, so resume re-reads these to repopulate the planner's context."""

    for reason in reasons:
        # emit() invokes the factory synchronously (before the loop advances), so
        # capturing ``reason`` directly is safe (no late-binding hazard) and keeps the
        # factory a clean Callable[[int], Event].
        journal.emit(
            lambda s: EpochCarried(
                seq=s, ts=now_fn(), epoch_id=epoch_id, reason=reason
            )
        )


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
        _reap_epoch_logs(log_root, f"{_PHASE}/E{epoch_index - 1}")


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
    start_carried: tuple[str, ...] = (),
) -> RunResult:
    """The shared epoch loop (entered fresh by ``start_run`` and re-entered by
    ``resume_run``). Drives boundaries from ``start_index`` until the planner ends,
    an unrecoverable planner failure forces a clean end, or ``max_epochs`` is hit.
    ``start_carried`` seeds the first boundary's carried context (empty for a fresh
    run; reconstructed from the journal on resume so the planner is not re-planned
    blind to why the prior epoch failed)."""

    epoch_index = start_index
    carried: tuple[str, ...] = start_carried
    consec_aborts = 0
    while epoch_index <= max_epochs:
        _reap_prior_logs(log_root, epoch_index)
        tip_ref = _resolve_tip(repo, run_branch)
        context = _build_context(
            job=job, repo=repo, run_dir=run_dir, run_branch=run_branch,
            tip_ref=tip_ref, carried=carried, epoch_index=epoch_index,
            max_epochs=max_epochs,
        )
        try:
            decision = _plan_with_backoff(
                planner, context, journal=journal, sleep_fn=sleep_fn,
                backoff_s=backoff_s, now_fn=now_fn,
            )
        except PlannerError as exc:
            summary = f"planner could not continue at E{epoch_index}: {exc}"
            _emit_ended(journal, now_fn, summary)
            return RunResult("ended", summary, epoch_index - 1)

        if isinstance(decision, EndDecision):
            if acceptance is None or acceptance(context):
                _emit_completed(journal, now_fn)
                return RunResult("completed", decision.summary, epoch_index - 1)
            _emit_ended(journal, now_fn, decision.summary)
            return RunResult("ended", decision.summary, epoch_index - 1)

        epoch = decision.epoch
        epoch_id = f"{_PHASE}/E{epoch_index}"
        base = tip_ref
        started_emitted = False
        try:
            setup_err = _run_setup(
                list(epoch.setup), repo=repo, run_dir=run_dir, base=base,
            )
            if setup_err is not None:
                msg = f"E{epoch_index} setup failed: {setup_err}"
                _journal_carried(journal, now_fn, epoch_id, (msg,))
                carried = carried + (msg,)
                consec_aborts = 0
                epoch_index += 1
                continue
            journal.emit(
                lambda s: EpochStarted(
                    seq=s, ts=now_fn(), epoch_id=epoch_id, title=epoch.title,
                    tasks=[TaskRef(id=t.id, mode=t.mode) for t in epoch.tasks],
                )
            )
            started_emitted = True
            results = _grind_epoch(
                epoch, epoch_id, base, repo=repo, run_dir=run_dir,
                run_branch=run_branch, backends=backends, journal=journal,
                log_root=log_root, sleep_fn=sleep_fn, backoff_s=backoff_s,
                now_fn=now_fn,
            )
            tasks_by_id = {f"{epoch_id}/{t.id}": t for t in epoch.tasks}
            integration = _integrate(
                epoch_id, base, repo=repo, run_dir=run_dir, run_branch=run_branch,
                results=results, tasks_by_id=tasks_by_id,
            )
            carried = _carry(results, integration)
            _journal_carried(journal, now_fn, epoch_id, carried)
            journal.emit(
                lambda s: EpochCompleted(seq=s, ts=now_fn(), epoch_id=epoch_id)
            )
            consec_aborts = 0
        except WorkerRateLimited:
            # Node #1 stays un-burned (defensive: _grind_epoch already parks on it).
            raise
        except Exception as exc:
            # BONES node #2: ANY other epoch failure (a GitError/OSError escaping the
            # worktree or integration machinery) becomes carried context the planner
            # steers around next boundary, never an uncaught crash of an unattended
            # run. The in-flight epoch's partial debris is razed.
            consec_aborts += 1
            detail = (
                f"E{epoch_index} aborted on an unexpected "
                f"{type(exc).__name__}: {exc}"
            )
            _raze_epoch(run_dir, repo, run_branch, epoch_id, log_root)
            if started_emitted:
                _journal_carried(journal, now_fn, epoch_id, (detail,))
                journal.emit(
                    lambda s: EpochCompleted(seq=s, ts=now_fn(), epoch_id=epoch_id)
                )
            if consec_aborts >= MAX_CONSECUTIVE_ABORTS:
                summary = (
                    f"{consec_aborts} consecutive epochs aborted on unexpected "
                    f"errors (persistent infra fault); last: {detail}"
                )
                _emit_ended(journal, now_fn, summary)
                return RunResult("ended", summary, epoch_index - 1)
            carried = carried + (detail,)
            epoch_index += 1
            continue
        epoch_index += 1

    summary = f"max epochs ({max_epochs}) reached without a planned end"
    if carried:
        summary += "; carried: " + " | ".join(carried)
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
        return _drive(
            job=job, run_dir=run_dir, repo=repo, run_branch=run_branch,
            planner=planner, backends=backends, max_epochs=max_epochs,
            log_root=log_root, acceptance=acceptance, sleep_fn=sleep_fn,
            backoff_s=backoff_s, now_fn=now_fn, journal=journal, start_index=1,
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
    in_flight_id = f"{_PHASE}/E{start_index}"
    started_ids = {e.epoch_id for e in events if isinstance(e, EpochStarted)}
    razed = in_flight_id if in_flight_id in started_ids else None
    carried = _reconstruct_carried(events)
    _raze_epoch(run_dir, repo, run_branch, in_flight_id, log_root)

    with JournalWriter(run_dir.events_path) as journal:
        journal.emit(
            lambda s: RunResumed(seq=s, ts=now_fn(), run_id=run_id, razed_epoch=razed)
        )
        return _drive(
            job=job, run_dir=run_dir, repo=repo, run_branch=run_branch,
            planner=planner, backends=backends, max_epochs=eff_max,
            log_root=log_root, acceptance=acceptance, sleep_fn=sleep_fn,
            backoff_s=backoff_s, now_fn=now_fn, journal=journal,
            start_index=start_index, start_carried=carried,
        )


def _reconstruct_carried(events: Sequence[Event]) -> tuple[str, ...]:
    """Repopulate the planner's carried context from the journal (FIX 5).

    The in-memory carried tuple does not survive a crash, so re-read the
    ``epoch_carried`` events of the LAST completed epoch (the prior boundary the
    re-planned epoch follows). Empty when nothing was carried (a clean boundary)."""

    last_completed: str | None = None
    for ev in events:
        if isinstance(ev, EpochCompleted):
            last_completed = ev.epoch_id
    if last_completed is None:
        return ()
    return tuple(
        ev.reason
        for ev in events
        if isinstance(ev, EpochCarried) and ev.epoch_id == last_completed
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
