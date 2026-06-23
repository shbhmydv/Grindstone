"""One epoch: fan out 1-8 independent tasks, scope-check, ff-integrate (ARCHITECTURE.md).

This is the rung that turns the S1 single-task machine into an epoch. Identity
(phase/epoch/base) is parameterized and the run/phase frame is owned by the
caller (S3: ``run_loop`` for production, a test helper for epoch-isolation
tests), ``run_epoch`` emits ONLY ``epoch_started`` … ``epoch_completed``
against a journal the caller opens, never the run frame (S3 ruling 8: the
synthesized scaffold is deleted, the multi-epoch loop owns ``run_started`` /
``phase_started`` / ``run_completed`` / ``run_escalated``). The task list is
REAL, taken from the typed ``ImplementEpochArgs`` / ``ArtifactEpochArgs``.

    epoch_started  -> fan out tasks on a bounded thread pool, each running the
      S1 per-task machine unchanged in shape (dispatch / validate / retry /
      escalate). Implement tasks grind in per-attempt worktrees; the core
      commits + scope-checks each success.
    done-predicate -> task queue empty AND nothing in flight (an explicit
      predicate over the epoch state, the S3 loop's gate, not a pool join).
    integration    -> merge every DONE task's branch, in task order, into the
      epoch integration branch (started fresh at the epoch base each epoch, so a
      stale same-named leftover never poisons it). Disjoint ownership + scope
      check then make conflicts impossible by construction, so ANY conflict
      aborts the epoch as ``integration_conflict`` (a structural bug, never
      retried).
    epoch_completed -> EpochOutcome (returned + serialized to outcome.json).

Determinism contract: per-task event ORDER is causal and tested; cross-task
interleaving is deliberately NOT asserted (concurrent tasks interleave in the
journal under the writer's lock, with strictly monotonic seq).

Kill-mid-epoch resume (ARCHITECTURE.md): per-task terminal states + integration
progress live in ``state.json`` (atomic rewrite every transition). On resume,
terminal tasks stay terminal (DONE tasks keep their branches + handoffs, never
re-run), in-flight tasks are burned exactly like S1 and continue from their
cursor, then integration finishes idempotently (already-merged branches are
ancestors of the integration branch and skipped).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import threading
import time
from typing import Callable, Literal, Mapping, Sequence, cast

from pydantic import BaseModel, ConfigDict

from grindstone import worktree as wt
from grindstone.config import PrepareConfig
from grindstone.contracts.models import (
    ArtifactEpochArgs,
    ImplementEpochArgs,
)
from grindstone.contracts.semantics import HandoffMode
from grindstone.events import (
    EpochCompleted,
    EpochStarted,
    JournalWriter,
    TaskRef,
)
from grindstone.rundir import RunDir, atomic_write_json
from grindstone.task_loop import (
    TIER0_ATTEMPTS,
    TaskCursorState,
    TaskIdentity,
    TaskOutcome,
    pending_cursor,
    run_task,
)
from grindstone.verify import TaskVerifier
from grindstone.worker import Task, WorkerTransport

EpochArgs = ImplementEpochArgs | ArtifactEpochArgs
#: Injected sleep for the session-limit hourly park, threaded run_loop -> epoch ->
#: task so tests inject a fake (no wall clock); defaults to real ``time.sleep``.
SleepFn = Callable[[float], None]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# --- durable epoch state (resume substrate) ------------------------------------


class IntegrationState(BaseModel):
    """Durable integration progress: which task branches are already merged."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    branch: str | None
    status: Literal["pending", "completed", "conflict", "skipped"]
    merged: list[str]
    conflict: str | None


class EpochState(BaseModel):
    """The whole epoch's durable cursor (atomic full rewrite every transition)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    phase_id: str
    epoch_id: str
    title: str
    mode: HandoffMode
    is_implement: bool
    #: For an IMPLEMENT epoch: the epoch base commit (the run tip the staging branch
    #: starts at; the scope check measures against it). For a NON-implement epoch
    #: (review/research/artifact, which integrate nothing): the READ-tip
    #: (integration-tip) commit the tasks read from, persisted so resume re-derives
    #: the same read tip. ``None`` when there is no tip yet or no repo.
    base: str | None
    integration: IntegrationState
    tasks: dict[str, TaskCursorState]


class _EpochStateStore:
    """Thread-safe owner of ``EpochState`` + its ``state.json`` (ruling 2).

    One lock guards every mutation + full atomic rewrite, so concurrent fan-out
    tasks can update their own cursors without racing the file.
    """

    def __init__(self, run_dir: RunDir, state: EpochState) -> None:
        self._run_dir = run_dir
        self._state = state
        self._lock = threading.Lock()
        self._flush()

    def _flush(self) -> None:
        atomic_write_json(self._run_dir.state_path, self._state.model_dump(mode="json"))

    @property
    def state(self) -> EpochState:
        return self._state

    def update_task(self, cursor: TaskCursorState) -> None:
        with self._lock:
            tasks = dict(self._state.tasks)
            tasks[cursor.task_id] = cursor
            self._state = self._state.model_copy(update={"tasks": tasks})
            self._flush()

    def update_integration(self, integration: IntegrationState) -> None:
        with self._lock:
            self._state = self._state.model_copy(update={"integration": integration})
            self._flush()


def epoch_done_predicate(state: EpochState) -> bool:
    """The done-predicate: task queue empty AND nothing in flight (ARCHITECTURE.md).

    Explicit over the epoch state so S3 can gate the planner call on it rather
    than on an implicit pool join.
    """

    queued = [c for c in state.tasks.values() if c.status == "pending"]
    in_flight = [c for c in state.tasks.values() if c.status == "running"]
    return not queued and not in_flight


# --- epoch outcome (raw material for S3's epoch report) ------------------------


@dataclass(frozen=True)
class TaskResult:
    """Flat per-task record for the epoch report."""

    task_id: str
    fq_task_id: str
    status: Literal["done", "failed"]
    attempts: int
    tier: str
    handoff_key: str | None
    failure_reason: str | None


@dataclass(frozen=True)
class IntegrationOutcome:
    status: Literal["completed", "conflict", "skipped"]
    branch: str | None
    merged: list[str]
    conflict: str | None


@dataclass(frozen=True)
class EpochOutcome:
    """Returned + serialized to ``<phase>/<epoch>/outcome.json``. Flat + small."""

    phase_id: str
    epoch_id: str
    status: Literal["completed", "integration_conflict"]
    tasks: list[TaskResult]
    integration: IntegrationOutcome

    def to_dict(self) -> dict[str, object]:
        return {
            "phase_id": self.phase_id,
            "epoch_id": self.epoch_id,
            "status": self.status,
            "tasks": [
                {
                    "task_id": t.task_id,
                    "fq_task_id": t.fq_task_id,
                    "status": t.status,
                    "attempts": t.attempts,
                    "tier": t.tier,
                    "handoff_key": t.handoff_key,
                    "failure_reason": t.failure_reason,
                }
                for t in self.tasks
            ],
            "integration": {
                "status": self.integration.status,
                "branch": self.integration.branch,
                "merged": self.integration.merged,
                "conflict": self.integration.conflict,
            },
        }

    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> "EpochOutcome":
        """Rebuild an outcome from its persisted ``outcome.json`` (resume seam).

        Inverse of ``to_dict``. Used on resume to re-run the G4 verification pass
        over a completed-but-unverified epoch (the outcome feeds the verifier brief),
        so a kill cannot let a completed epoch skip its semantic gate."""

        tasks_raw = cast(list[dict[str, object]], raw["tasks"])
        integ = cast(dict[str, object], raw["integration"])
        return cls(
            phase_id=cast(str, raw["phase_id"]),
            epoch_id=cast(str, raw["epoch_id"]),
            status=cast(Literal["completed", "integration_conflict"], raw["status"]),
            tasks=[
                TaskResult(
                    task_id=cast(str, t["task_id"]),
                    fq_task_id=cast(str, t["fq_task_id"]),
                    status=cast(Literal["done", "failed"], t["status"]),
                    attempts=cast(int, t["attempts"]),
                    tier=cast(str, t["tier"]),
                    handoff_key=cast("str | None", t["handoff_key"]),
                    failure_reason=cast("str | None", t["failure_reason"]),
                )
                for t in tasks_raw
            ],
            integration=IntegrationOutcome(
                status=cast(Literal["completed", "conflict", "skipped"], integ["status"]),
                branch=cast("str | None", integ["branch"]),
                merged=cast("list[str]", integ["merged"]),
                conflict=cast("str | None", integ["conflict"]),
            ),
        )


def _result_from_outcome(outcome: TaskOutcome) -> TaskResult:
    return TaskResult(
        task_id=outcome.identity.task_id,
        fq_task_id=outcome.identity.fq,
        status=outcome.status,
        attempts=outcome.attempts,
        tier=outcome.tier,
        handoff_key=outcome.handoff_key,
        failure_reason=outcome.reason,
    )


def _outcome_from_cursor(identity: TaskIdentity, cursor: TaskCursorState) -> TaskOutcome:
    """Reconstruct a terminal task's outcome from its durable cursor (resume)."""

    status: Literal["done", "failed"] = "done" if cursor.status == "done" else "failed"
    return TaskOutcome(
        identity=identity,
        status=status,
        tier=cursor.tier_name or "worker",
        attempts=cursor.attempt,
        handoff=None,
        handoff_key=f"{identity.fq}/handoff.json" if status == "done" else None,
        branch=cursor.branch,
        reason=cursor.reason,
    )


# --- epoch frame (epoch-level events only; the run frame is the caller's) ------


def _emit_epoch_started(
    journal: JournalWriter,
    *,
    phase_id: str,
    epoch_id: str,
    title: str,
    mode: HandoffMode,
    tasks: Sequence[Task],
) -> None:
    """Open the epoch: ``epoch_started`` with the REAL task list."""

    journal.emit(
        lambda s: EpochStarted(
            seq=s,
            ts=_now(),
            phase_id=phase_id,
            epoch_id=epoch_id,
            title=title,
            tasks=[TaskRef(id=t.id, mode=mode) for t in tasks],
        )
    )


# --- fan-out -------------------------------------------------------------------


def _fan_out(
    *,
    to_run: list[Task],
    identities: dict[str, TaskIdentity],
    resume_cursors: dict[str, TaskCursorState | None],
    mode: HandoffMode,
    run_dir: RunDir,
    journal: JournalWriter,
    store: _EpochStateStore,
    ladder: Sequence[tuple[str, WorkerTransport]],
    repo: Path | None,
    base: str | None,
    concurrency: int,
    tier0_attempts: int,
    force_senior: bool,
    prepare: PrepareConfig | None,
    sleep_fn: SleepFn,
    verifiers: Mapping[str, TaskVerifier] | None,
    epoch_hint: str | None = None,
) -> dict[str, TaskOutcome]:
    """Run ``to_run`` tasks concurrently; return outcomes keyed by short task id."""

    outcomes: dict[str, TaskOutcome] = {}
    if not to_run:
        return outcomes

    def _one(task: Task) -> TaskOutcome:
        return run_task(
            identity=identities[task.id],
            task=task,
            mode=mode,
            run_dir=run_dir,
            journal=journal,
            sink=store.update_task,
            ladder=ladder,
            repo=repo,
            base=base,
            tier0_attempts=tier0_attempts,
            resume_cursor=resume_cursors.get(task.id),
            force_senior=force_senior,
            prepare=prepare,
            epoch_hint=epoch_hint,
            sleep_fn=sleep_fn,
            verifiers=verifiers,
        )

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(_one, task): task.id for task in to_run}
        for fut in as_completed(futures):
            outcomes[futures[fut]] = fut.result()
    return outcomes


# --- integration ---------------------------------------------------------------


def _run_branch(run_dir: RunDir) -> str:
    """The ONE persistent run branch ``grind/{run_id}``: the only ref that survives
    between epochs. It fast-forwards as each epoch's staging branch passes. A leaf
    ref, so nothing else may live under ``grind/{run_id}/...`` (git ref dir/file
    conflict); the transient staging + attempt branches all live under ``grind-wip/``.
    """

    return f"grind/{run_dir.root.name}"


def _integrate(
    *,
    repo: Path,
    run_dir: RunDir,
    store: _EpochStateStore,
    branch: str,
    base: str,
    done_in_order: list[TaskOutcome],
) -> IntegrationOutcome:
    """Merge each DONE task's branch onto the epoch STAGING branch, then on success
    fast-forward the persistent run branch ``grind/{run_id}`` to the staging tip and
    delete every transient (staging + attempt) branch.

    ``branch`` is the transient ``grind-wip/.../_staging`` branch. A fresh
    integration (nothing merged yet) starts it from ``base`` (the current run tip the
    caller resolved): any stale same-named leftover is dropped first so prior content
    cannot poison the merge. On resume mid-integration (``merged`` non-empty) the
    staging branch carries real progress and is kept; an already-merged task branch is
    its ancestor (or deleted) and is skipped (idempotent). Any conflict ABORTS the
    epoch, leaving the run branch UNTOUCHED (given the fresh-base precondition +
    disjoint ownership it is structurally impossible). The returned
    ``IntegrationOutcome.branch`` is the RUN branch on success (so the run loop's
    ``last_integration_branch`` advances to the surviving ref), the staging branch on
    a conflict (nothing was promoted).
    """

    run_branch = _run_branch(run_dir)
    # A fresh integration (nothing merged yet) must start the staging branch from
    # `base` (the current run tip). The staging branch name is run-scoped, but a
    # crash before the prior run's staging branch was deleted could leave a same-named
    # leftover that would poison the merge with stale content (a phantom both-added
    # conflict). Drop it first. On resume mid-integration (merged non-empty) the branch
    # carries real progress: keep it (idempotent).
    if not store.state.integration.merged:
        wt.delete_branch(repo, branch)
    wt.ensure_integration_branch(repo, branch, base)
    merged: list[str] = list(store.state.integration.merged)

    if done_in_order:
        int_wt = run_dir.root / "worktrees" / "_staging"
        wt.add_worktree_on(repo, int_wt, branch=branch)
        for outcome in done_in_order:
            tid = outcome.identity.task_id
            task_branch = outcome.branch
            if task_branch is None:
                continue
            already = (
                tid in merged
                or not wt.branch_exists(repo, task_branch)
                or wt.is_ancestor(repo, task_branch, branch)
            )
            if already:
                if tid not in merged:
                    merged.append(tid)
                    store.update_integration(
                        IntegrationState(branch=branch, status="pending", merged=merged, conflict=None)
                    )
                continue
            result = wt.merge_into(int_wt, task_branch)
            if not result.ok:
                # Conflict: abort the epoch and leave the RUN branch UNTOUCHED. The
                # staging branch (and the attempt branches) are kept for the next
                # disposition; only the merge worktree is reclaimed.
                wt.remove_worktree(repo, int_wt)
                store.update_integration(
                    IntegrationState(
                        branch=branch, status="conflict", merged=merged, conflict=result.conflict
                    )
                )
                return IntegrationOutcome(
                    status="conflict", branch=branch, merged=merged, conflict=result.conflict
                )
            merged.append(tid)
            store.update_integration(
                IntegrationState(branch=branch, status="pending", merged=merged, conflict=None)
            )
        wt.remove_worktree(repo, int_wt)

    # Epoch succeeded: advance the persistent run branch to the staging tip (create
    # it there on the first epoch; a true fast-forward thereafter, the staging branch
    # was started off the run tip). Resolve the staging tip BEFORE pruning worktrees
    # (the ref still exists) so the run branch captures the merged work.
    staging_tip = wt.resolve_commit(repo, branch)
    # Prune the epoch's worktrees, THEN move/delete branches: git refuses to delete or
    # reuse a branch still checked out in a worktree (ruling 7). The run branch is a
    # leaf ref never checked out in an epoch worktree, so the fast-forward is safe.
    wt.prune_tree(repo, run_dir.root / "worktrees")
    wt.fast_forward_branch(repo, run_branch, staging_tip)
    # "Once things pass, they clean up and fall down": delete the now-absorbed
    # transient branches (the staging branch + every merged task attempt branch). The
    # work survives on the run branch; nothing under ``grind-wip/`` outlives the epoch.
    wt.delete_branch(repo, branch)
    for outcome in done_in_order:
        if outcome.branch:
            wt.delete_branch(repo, outcome.branch)
    store.update_integration(
        IntegrationState(branch=run_branch, status="completed", merged=merged, conflict=None)
    )
    return IntegrationOutcome(
        status="completed", branch=run_branch, merged=merged, conflict=None
    )


def _finish(
    *,
    journal: JournalWriter,
    store: _EpochStateStore,
    run_dir: RunDir,
    repo: Path | None,
    base: str | None,
    ordered_ids: list[str],
    identities: dict[str, TaskIdentity],
    outcomes: dict[str, TaskOutcome],
) -> EpochOutcome:
    """Integrate, build + persist the EpochOutcome, and close the journal out."""

    state = store.state
    done_in_order = [outcomes[tid] for tid in ordered_ids if outcomes[tid].status == "done"]

    if state.is_implement and state.integration.branch is not None:
        assert repo is not None and base is not None
        integration = _integrate(
            repo=repo,
            run_dir=run_dir,
            store=store,
            branch=state.integration.branch,
            base=base,
            done_in_order=done_in_order,
        )
    else:
        integration = IntegrationOutcome(status="skipped", branch=None, merged=[], conflict=None)

    epoch_status: Literal["completed", "integration_conflict"] = (
        "integration_conflict" if integration.status == "conflict" else "completed"
    )
    outcome = EpochOutcome(
        phase_id=state.phase_id,
        epoch_id=state.epoch_id,
        status=epoch_status,
        tasks=[_result_from_outcome(outcomes[tid]) for tid in ordered_ids],
        integration=integration,
    )
    path = run_dir.resolve(f"{state.phase_id}/{state.epoch_id}/outcome.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, outcome.to_dict())
    journal.emit(lambda s: EpochCompleted(seq=s, ts=_now(), epoch_id=state.epoch_id))
    return outcome


def _resolve_concurrency(concurrency: int | None, n_tasks: int) -> int:
    if concurrency is not None:
        if concurrency < 1:
            raise ValueError("concurrency must be >= 1")
        return concurrency
    return max(1, min(4, n_tasks))


# --- public entry points -------------------------------------------------------


def run_epoch(
    run_dir: RunDir,
    *,
    journal: JournalWriter,
    args: EpochArgs,
    mode: HandoffMode,
    ladder: Sequence[tuple[str, WorkerTransport]],
    repo: Path | None = None,
    phase_id: str = "P1",
    epoch_id: str = "E1",
    base: str | None = None,
    concurrency: int | None = None,
    tier0_attempts: int = TIER0_ATTEMPTS,
    prepare: PrepareConfig | None = None,
    epoch_hint: str | None = None,
    force_senior: bool = False,
    sleep_fn: SleepFn = time.sleep,
    verifiers: Mapping[str, TaskVerifier] | None = None,
) -> EpochOutcome:
    """Run ONE epoch (fresh start) to a terminal EpochOutcome.

    ``journal`` is the caller-owned run journal (this function emits only
    ``epoch_started`` … ``epoch_completed``). Implement epochs require ``repo``;
    ``base`` is the epoch base commit (the chain tip, ruling 4), defaulted to
    repo HEAD when omitted. Artifact epochs need no repo and integrate nothing.

    For a NON-implement epoch (research/review/artifact) ``base`` is NOT an epoch
    base (those epochs integrate nothing); it is the READ tip the tasks audit, the
    current integration tip resolved by the caller. The work built earlier in the
    run lives only on that tip (the operator checkout sits at the clean base), so a
    review/research task checks it out into a detached read-only worktree and reads
    THAT, not the stale operator tree (the live P5 hallucinated-review bug). It is
    persisted to ``EpochState.base`` so resume re-derives the same read tip. ``None``
    when there is no tip yet (a first epoch, an unborn HEAD) or no repo, in which case
    the task falls back to reading the operator checkout (the prior behavior).

    ``epoch_hint`` (a planner handle_failed_epoch ``retry`` corrective) is seeded
    into every task's failure context so the workers see it on their first
    attempt; ``force_senior`` starts EVERY task on the senior tier (the planner's
    tier bump / escalate_senior disposition), overriding each task's own per-task
    ``senior`` routing flag.
    """

    if not ladder:
        raise ValueError("ladder must have at least one tier")
    is_implement = isinstance(args, ImplementEpochArgs)
    if is_implement and repo is None:
        raise ValueError("implement epoch requires a git repo")

    tasks: list[Task] = list(args.tasks)
    ordered_ids = [t.id for t in tasks]
    run_id = run_dir.root.name
    identities = {t.id: TaskIdentity(run_id, phase_id, epoch_id, t.id) for t in tasks}
    if is_implement and repo is not None:
        base = base if base is not None else wt.head_commit(repo)
    # A non-implement epoch keeps ``base`` AS PASSED: it is the read tip the caller
    # resolved (the current integration tip), not an epoch base, and it is the handle
    # the tasks read against. None when no tip exists yet or no repo (fall back to the
    # operator checkout). Never defaulted to HEAD: HEAD is the stale operator tree.
    # The per-epoch STAGING branch (transient): merges fan out onto it off the
    # current run tip, then on success the run branch ``grind/{run_id}`` fast-forwards
    # to its tip and it is deleted. It lives under the transient ``grind-wip/``
    # namespace, never ``grind/{run_id}/...`` (which would dir/file-conflict with the
    # ``grind/{run_id}`` leaf run branch in git's ref store).
    integration_branch = (
        f"grind-wip/{run_id}/{phase_id}/{epoch_id}/_staging" if is_implement else None
    )

    state = EpochState(
        phase_id=phase_id,
        epoch_id=epoch_id,
        title=args.epoch_title,
        mode=mode,
        is_implement=is_implement,
        base=base,
        integration=IntegrationState(
            branch=integration_branch,
            status="pending" if is_implement else "skipped",
            merged=[],
            conflict=None,
        ),
        tasks={t.id: pending_cursor(identities[t.id], mode) for t in tasks},
    )

    store = _EpochStateStore(run_dir, state)
    _emit_epoch_started(
        journal,
        phase_id=phase_id,
        epoch_id=epoch_id,
        title=args.epoch_title,
        mode=mode,
        tasks=tasks,
    )
    outcomes = _fan_out(
        to_run=tasks,
        identities=identities,
        resume_cursors={t.id: None for t in tasks},
        mode=mode,
        run_dir=run_dir,
        journal=journal,
        store=store,
        ladder=ladder,
        repo=repo,
        base=base,
        concurrency=_resolve_concurrency(concurrency, len(tasks)),
        tier0_attempts=tier0_attempts,
        force_senior=force_senior,
        prepare=prepare,
        epoch_hint=epoch_hint,
        sleep_fn=sleep_fn,
        verifiers=verifiers,
    )
    if not epoch_done_predicate(store.state):
        raise RuntimeError("epoch loop ended before the done-predicate held")
    return _finish(
        journal=journal,
        store=store,
        run_dir=run_dir,
        repo=repo,
        base=base,
        ordered_ids=ordered_ids,
        identities=identities,
        outcomes=outcomes,
    )


def resume_epoch(
    run_dir: RunDir,
    *,
    journal: JournalWriter,
    args: EpochArgs,
    mode: HandoffMode,
    ladder: Sequence[tuple[str, WorkerTransport]],
    repo: Path | None = None,
    concurrency: int | None = None,
    tier0_attempts: int = TIER0_ATTEMPTS,
    prepare: PrepareConfig | None = None,
    sleep_fn: SleepFn = time.sleep,
    verifiers: Mapping[str, TaskVerifier] | None = None,
) -> EpochOutcome:
    """Re-enter a killed epoch from ``state.json`` against a caller-owned journal.

    Terminal tasks stay terminal (DONE tasks are never re-dispatched); in-flight
    tasks are burned and continue from their cursor; pending tasks run fresh;
    then integration finishes idempotently. The caller owns the run frame (it
    has already emitted ``run_resumed``); this only continues the epoch.
    """

    if not ladder:
        raise ValueError("ladder must have at least one tier")
    state = EpochState.model_validate_json(run_dir.state_path.read_text(encoding="utf-8"))
    if state.is_implement and repo is None:
        raise ValueError("implement epoch requires a git repo")

    task_by_id: dict[str, Task] = {t.id: t for t in args.tasks}
    ordered_ids = [t.id for t in args.tasks]
    identities = {
        tid: TaskIdentity(run_dir.root.name, state.phase_id, state.epoch_id, tid)
        for tid in state.tasks
    }

    store = _EpochStateStore(run_dir, state)

    outcomes: dict[str, TaskOutcome] = {}
    to_run: list[Task] = []
    resume_cursors: dict[str, TaskCursorState | None] = {}
    for tid, cursor in state.tasks.items():
        if cursor.status in ("done", "failed"):
            outcomes[tid] = _outcome_from_cursor(identities[tid], cursor)
        elif cursor.status == "running":
            to_run.append(task_by_id[tid])
            resume_cursors[tid] = cursor
        else:  # pending, never dispatched before the kill
            to_run.append(task_by_id[tid])
            resume_cursors[tid] = None

    live = _fan_out(
        to_run=to_run,
        identities=identities,
        resume_cursors=resume_cursors,
        mode=mode,
        run_dir=run_dir,
        journal=journal,
        store=store,
        ladder=ladder,
        repo=repo,
        base=state.base,
        concurrency=_resolve_concurrency(concurrency, max(1, len(to_run))),
        tier0_attempts=tier0_attempts,
        # force_senior is a fresh-dispatch escalation signal (handle_failed_epoch
        # tier bump); on resume each task routes by its own per-task ``senior``
        # flag, and an in-flight escalated task resumes from its recorded cursor.
        force_senior=False,
        prepare=prepare,
        sleep_fn=sleep_fn,
        verifiers=verifiers,
    )
    outcomes.update(live)
    if not epoch_done_predicate(store.state):
        raise RuntimeError("epoch resume ended before the done-predicate held")
    return _finish(
        journal=journal,
        store=store,
        run_dir=run_dir,
        repo=repo,
        base=state.base,
        ordered_ids=ordered_ids,
        identities=identities,
        outcomes=outcomes,
    )
