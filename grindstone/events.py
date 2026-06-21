"""Journal event vocabulary + NDJSON I/O + the replay fold.

ARCHITECTURE.md: the event stream alone must be sufficient to render the full
run -> phase -> epoch -> task tree with statuses, and ``planner_calls_per_run``
must be derivable. ``replay`` is the proof: it folds an event list into a
``RunTree`` snapshot. The vocabulary is frozen at S0, the TUI (S4) and resume
both consume it.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Annotated, Callable, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter


class _Event(BaseModel):
    """Common shape: monotonic ``seq``, ISO-8601 UTC ``ts``, ``event`` tag."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    seq: int
    ts: str


class PhaseRef(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    id: str
    title: str


class TaskRef(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    id: str
    mode: str


class RunStarted(_Event):
    event: Literal["run_started"] = "run_started"
    run_id: str
    job_path: str
    #: The planner-call backstop in force (None = off, a test-only seam). Carried
    #: in the journal so a watcher can show "n/N" without reading run state.
    max_planner_calls: int | None = None


class RunResumed(_Event):
    event: Literal["run_resumed"] = "run_resumed"
    run_id: str


class RunCompleted(_Event):
    event: Literal["run_completed"] = "run_completed"


class RunEscalated(_Event):
    event: Literal["run_escalated"] = "run_escalated"
    reason: str


class RunFailed(_Event):
    # The production safety valve (planner-call / epoch cap) tripped. Terminal,
    # but not an escalation, a harness bound the durable state also records. It
    # is a vocabulary event so the journal stays self-describing (the TUI exits).
    event: Literal["run_failed"] = "run_failed"
    reason: str


class FinalPolishApplied(_Event):
    # B5: codex's optional post-completion inline polish pass was KEPT, its edits
    # re-passed the SAME complete_run evidence, so the run completes on the polish
    # commit. ``commit`` is that adopted commit sha (the final branch now points at
    # it); ``changed_files`` is the polish diff's file list (`git add -A` is blind,
    # so the changed-file set is recorded for an auditable trail of what codex
    # touched). Defaulted so any pre-field journal still replays.
    event: Literal["final_polish_applied"] = "final_polish_applied"
    commit: str
    changed_files: list[str] = Field(default_factory=list)


class FinalPolishSkipped(_Event):
    # B5: the polish pass made no net change to the certified run, codex changed
    # nothing, its edits regressed the evidence (discarded), or the pass errored.
    # The original completion stands; ``reason`` records which (self-describing).
    event: Literal["final_polish_skipped"] = "final_polish_skipped"
    reason: str


class PlannerCallStarted(_Event):
    event: Literal["planner_call_started"] = "planner_call_started"


class PlannerCallSucceeded(_Event):
    event: Literal["planner_call_succeeded"] = "planner_call_succeeded"
    tool: str


class PlannerCallFailed(_Event):
    event: Literal["planner_call_failed"] = "planner_call_failed"
    classification: Literal["rate_limit", "transient", "hard"]


class SkeletonProposed(_Event):
    event: Literal["skeleton_proposed"] = "skeleton_proposed"
    phases: list[PhaseRef]


class PhasesRevised(_Event):
    event: Literal["phases_revised"] = "phases_revised"
    reason: str
    phases: list[PhaseRef]


class PhaseStarted(_Event):
    event: Literal["phase_started"] = "phase_started"
    phase_id: str


class PhasePassed(_Event):
    event: Literal["phase_passed"] = "phase_passed"
    phase_id: str


class PhaseEscalated(_Event):
    event: Literal["phase_escalated"] = "phase_escalated"
    phase_id: str


class EpochStarted(_Event):
    event: Literal["epoch_started"] = "epoch_started"
    phase_id: str
    epoch_id: str
    title: str
    tasks: list[TaskRef]


class EpochCompleted(_Event):
    event: Literal["epoch_completed"] = "epoch_completed"
    epoch_id: str


class EpochFailed(_Event):
    # An epoch finished with one or more FAILED tasks (the retry ladder was
    # exhausted). The run is now awaiting a focused handle_failed_epoch
    # disposition; ``failed_tasks`` names them so the journal renders the why.
    event: Literal["epoch_failed"] = "epoch_failed"
    phase_id: str
    epoch_id: str
    failed_tasks: list[str]


class FailedEpochHandled(_Event):
    # The planner's focused disposition of a failed epoch (or the deterministic
    # cap forcing one). ``action`` is retry / escalate_senior / halt / cap_halt;
    # ``detail`` is the planner's hint/diagnosis/reason (or the cap message).
    event: Literal["failed_epoch_handled"] = "failed_epoch_handled"
    phase_id: str
    epoch_id: str
    action: str
    detail: str


class InfraCheckDetected(_Event):
    # A gate check failed for an ENVIRONMENTAL reason (infra.classify_check_failure:
    # exit 127, missing tool/dependency, install failure), not a real assertion
    # failure. The core will auto-dispatch a senior infra-repair instead of
    # charging the worker. ``command`` is the failing check; ``reason`` is the
    # matched signature.
    event: Literal["infra_check_detected"] = "infra_check_detected"
    phase_id: str
    command: str
    reason: str


class InfraRepairDispatched(_Event):
    # A bounded, host-guarded senior infra-repair was dispatched against the gate's
    # tip worktree to make the environment satisfiable. ``attempt`` is 1-based;
    # ``cap`` is the configured infra_repair.attempts ceiling.
    event: Literal["infra_repair_dispatched"] = "infra_repair_dispatched"
    phase_id: str
    command: str
    attempt: int
    cap: int


class InfraRepairResolved(_Event):
    # A senior infra-repair fixed the environment: the previously infra-failing
    # gate checks now pass (re-run deterministically). ``attempt`` is the cycle
    # that resolved it.
    event: Literal["infra_repair_resolved"] = "infra_repair_resolved"
    phase_id: str
    attempt: int


class InfraRepairExhausted(_Event):
    # The infra-repair cap was reached and the gate is STILL infra-failing. The run
    # escalates to a human; ``command`` names the unsatisfiable tool/command so the
    # message is clear (not a vague worker failure).
    event: Literal["infra_repair_exhausted"] = "infra_repair_exhausted"
    phase_id: str
    command: str
    reason: str


class EpochVerificationStarted(_Event):
    # The end-of-epoch agentic verification pass (G4) began: the epoch cleared its
    # deterministic floor and carries criteria, so the local tier judges them against
    # the produced artifacts in a tip worktree. ``criteria`` is how many were judged.
    event: Literal["epoch_verification_started"] = "epoch_verification_started"
    phase_id: str
    epoch_id: str
    criteria: int


class EpochVerificationPassed(_Event):
    # The verification pass judged EVERY criterion met by the actual artifacts; the
    # epoch's semantic gate is clear and the run proceeds.
    event: Literal["epoch_verification_passed"] = "epoch_verification_passed"
    phase_id: str
    epoch_id: str


class EpochVerificationFailed(_Event):
    # The verification pass found an unmet criterion (or could not produce a valid
    # verdict, a fail-safe). The epoch is routed through the failed-epoch machinery
    # with the gaps as the planner-facing reason. ``gaps`` names what was unmet.
    event: Literal["epoch_verification_failed"] = "epoch_verification_failed"
    phase_id: str
    epoch_id: str
    gaps: list[str]


class TaskDispatched(_Event):
    event: Literal["task_dispatched"] = "task_dispatched"
    epoch_id: str
    task_id: str


class TaskRetried(_Event):
    event: Literal["task_retried"] = "task_retried"
    epoch_id: str
    task_id: str
    attempt: int


class TaskEscalated(_Event):
    event: Literal["task_escalated"] = "task_escalated"
    epoch_id: str
    task_id: str
    tier: str


class TaskDone(_Event):
    event: Literal["task_done"] = "task_done"
    epoch_id: str
    task_id: str


class TaskFailed(_Event):
    event: Literal["task_failed"] = "task_failed"
    epoch_id: str
    task_id: str


class HandoffRejected(_Event):
    event: Literal["handoff_rejected"] = "handoff_rejected"
    epoch_id: str
    task_id: str
    reason: str


Event = Annotated[
    Union[
        RunStarted,
        RunResumed,
        RunCompleted,
        RunEscalated,
        RunFailed,
        FinalPolishApplied,
        FinalPolishSkipped,
        PlannerCallStarted,
        PlannerCallSucceeded,
        PlannerCallFailed,
        SkeletonProposed,
        PhasesRevised,
        PhaseStarted,
        PhasePassed,
        PhaseEscalated,
        EpochStarted,
        EpochCompleted,
        EpochFailed,
        FailedEpochHandled,
        InfraCheckDetected,
        InfraRepairDispatched,
        InfraRepairResolved,
        InfraRepairExhausted,
        EpochVerificationStarted,
        EpochVerificationPassed,
        EpochVerificationFailed,
        TaskDispatched,
        TaskRetried,
        TaskEscalated,
        TaskDone,
        TaskFailed,
        HandoffRejected,
    ],
    Field(discriminator="event"),
]

_EVENT_ADAPTER: TypeAdapter[Event] = TypeAdapter(Event)


# --- NDJSON I/O ----------------------------------------------------------------


def _truncate_torn_tail(path: Path) -> None:
    """Drop a torn final line before appending.

    A crash mid-write can leave trailing bytes with no terminating newline.
    ``read_events`` TOLERATES such a line on read (it breaks), but if the writer
    then appends after it the new event FUSES onto the partial bytes into a single
    corrupt NON-final line that every later read rejects, turning a recoverable
    partial write into permanent loss of resume/replay. Truncating back to the
    last complete line (the last newline) keeps the next append on a clean boundary.
    """

    data = path.read_bytes()
    if not data or data.endswith(b"\n"):
        return
    cut = data.rfind(b"\n")  # -1 (no newline at all) -> cut+1 == 0 -> truncate empty
    with open(path, "rb+") as fh:
        fh.truncate(cut + 1)


class JournalWriter:
    """Append-only NDJSON writer: one event per line, flushed + fsynced.

    Opened in append mode (crash-safe). Enforces strictly increasing ``seq`` so
    a programming error cannot silently corrupt replay ordering.

    Internally thread-safe (S2 ruling 2): a single lock guards every write, so
    concurrent fan-out tasks may journal at the same time without corrupting the
    stream. ``emit`` is the concurrency-safe primitive, it assigns the next seq
    and writes the event **atomically under the lock**, so seq stays strictly
    monotonic even when two tasks race; their events simply interleave. ``append``
    (explicit-seq, single-threaded scaffold + tests) shares the same lock.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        if path.exists():
            _truncate_torn_tail(path)  # heal a crash-torn final line before appending
        existing = read_events(path) if path.exists() else []
        self._last_seq = existing[-1].seq if existing else -1
        self._fh = open(path, "a", encoding="utf-8")
        self._lock = threading.Lock()

    def _write_locked(self, event: Event) -> None:
        """Validate monotonicity + durably write one event. Caller holds the lock."""

        if event.seq <= self._last_seq:
            raise ValueError(
                f"non-monotonic seq: {event.seq} <= last {self._last_seq}"
            )
        self._fh.write(event.model_dump_json() + "\n")
        self._fh.flush()
        os.fsync(self._fh.fileno())
        self._last_seq = event.seq

    def append(self, event: Event) -> None:
        with self._lock:
            self._write_locked(event)

    def emit(self, factory: Callable[[int], Event]) -> Event:
        """Assign the next seq and write the event the factory builds, atomically.

        The seq is read and advanced inside the lock, so two concurrent callers
        can never be handed the same seq nor write out of order. Returns the
        written event (with its assigned seq) for the caller's bookkeeping.
        """

        with self._lock:
            event = factory(self._last_seq + 1)
            self._write_locked(event)
            return event

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> JournalWriter:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


def read_events(path: Path) -> list[Event]:
    """Read the journal, tolerating a truncated final line (crash mid-write).

    A crash can leave a half-written last line; that line is skipped, never
    raised. Corruption of any earlier line is a real error and propagates.
    """

    with open(path, "r", encoding="utf-8") as fh:
        lines = fh.readlines()
    out: list[Event] = []
    for i, raw in enumerate(lines):
        line = raw.rstrip("\n")
        if not line:
            continue
        try:
            out.append(_EVENT_ADAPTER.validate_json(line))
        except ValueError:
            if i == len(lines) - 1:
                break
            raise
    return out


# --- replay fold ---------------------------------------------------------------


@dataclass
class TaskNode:
    id: str
    mode: str
    status: str
    attempt: int
    started_ts: str | None = None
    ended_ts: str | None = None
    #: A transient note worth surfacing: a handoff-rejection reason (cleared on
    #: success) or the tier a task escalated to. Most-recent wins.
    note: str | None = None


@dataclass
class EpochNode:
    id: str
    title: str
    status: str
    tasks: list[TaskNode]
    started_ts: str | None = None
    ended_ts: str | None = None


@dataclass
class PhaseNode:
    id: str
    title: str
    status: str
    epochs: list[EpochNode]
    started_ts: str | None = None
    ended_ts: str | None = None


@dataclass
class RunTree:
    run_id: str
    job_path: str
    status: str
    planner_calls: int
    phases: list[PhaseNode] = field(default_factory=list)
    #: The planner-call backstop (from RunStarted), for an "n/N" header. None = off.
    planner_cap: int | None = None
    started_ts: str | None = None
    ended_ts: str | None = None
    #: ts of the most recent event seen, the journal's notion of "now", used as
    #: the elapsed-clock reference when no wall clock is injected (e.g. snapshots).
    last_ts: str | None = None
    #: RunEscalated / RunFailed reason (the terminal "why").
    escalation_reason: str | None = None
    #: True between a planner_call_started and its outcome (call in flight).
    planner_waiting: bool = False
    #: Classification of the most recent planner FAILURE, cleared on next success.
    last_planner_failure: str | None = None
    #: Tool of the most recent planner SUCCESS (the planner's last decision).
    last_planner_tool: str | None = None
    #: B5 final-polish outcome, "applied: <sha>" / "skipped: <reason>" / None
    #: when the optional polish pass never ran (off, or run not completed).
    final_polish: str | None = None


def _task(epoch: EpochNode, task_id: str) -> TaskNode:
    for task in epoch.tasks:
        if task.id == task_id:
            return task
    raise KeyError(f"task {task_id!r} not declared in epoch {epoch.id!r}")


def replay(events: list[Event]) -> RunTree:
    """Fold an event list into the run -> phase -> epoch -> task tree snapshot."""

    tree: RunTree | None = None
    phases_by_id: dict[str, PhaseNode] = {}
    epochs_by_id: dict[str, EpochNode] = {}

    for ev in events:
        if isinstance(ev, RunStarted):
            tree = RunTree(
                ev.run_id, ev.job_path, "running", 0,
                planner_cap=ev.max_planner_calls, started_ts=ev.ts, last_ts=ev.ts,
            )
            phases_by_id = {}
            epochs_by_id = {}
            continue
        if tree is None:
            raise ValueError(f"event {ev.event!r} before run_started")
        tree.last_ts = ev.ts

        if isinstance(ev, RunResumed):
            tree.status = "running"
        elif isinstance(ev, RunCompleted):
            tree.status = "completed"
            tree.ended_ts = ev.ts
        elif isinstance(ev, RunEscalated):
            tree.status = "escalated"
            tree.ended_ts = ev.ts
            tree.escalation_reason = ev.reason
        elif isinstance(ev, RunFailed):
            tree.status = "failed"
            tree.ended_ts = ev.ts
            tree.escalation_reason = ev.reason
        elif isinstance(ev, FinalPolishApplied):
            tree.final_polish = f"applied: {ev.commit[:12]} ({len(ev.changed_files)} files)"
        elif isinstance(ev, FinalPolishSkipped):
            tree.final_polish = f"skipped: {ev.reason}"
        elif isinstance(ev, PlannerCallStarted):
            tree.planner_calls += 1
            tree.planner_waiting = True
        elif isinstance(ev, PlannerCallSucceeded):
            tree.planner_waiting = False
            tree.last_planner_tool = ev.tool
            tree.last_planner_failure = None
        elif isinstance(ev, PlannerCallFailed):
            tree.planner_waiting = False
            tree.last_planner_failure = ev.classification
        elif isinstance(ev, SkeletonProposed):
            for ref in ev.phases:
                node = PhaseNode(ref.id, ref.title, "pending", [])
                tree.phases.append(node)
                phases_by_id[ref.id] = node
        elif isinstance(ev, PhasesRevised):
            kept = [ph for ph in tree.phases if ph.status != "pending"]
            kept_ids = {ph.id for ph in kept}
            for ref in ev.phases:
                if ref.id in kept_ids:
                    continue
                kept.append(PhaseNode(ref.id, ref.title, "pending", []))
            tree.phases = kept
            phases_by_id = {ph.id: ph for ph in kept}
        elif isinstance(ev, PhaseStarted):
            phase = phases_by_id[ev.phase_id]
            phase.status = "started"
            phase.started_ts = ev.ts
        elif isinstance(ev, PhasePassed):
            phase = phases_by_id[ev.phase_id]
            phase.status = "passed"
            phase.ended_ts = ev.ts
        elif isinstance(ev, PhaseEscalated):
            phase = phases_by_id[ev.phase_id]
            phase.status = "escalated"
            phase.ended_ts = ev.ts
        elif isinstance(ev, EpochStarted):
            tasks = [TaskNode(t.id, t.mode, "pending", 0) for t in ev.tasks]
            epoch = EpochNode(ev.epoch_id, ev.title, "started", tasks, started_ts=ev.ts)
            phases_by_id[ev.phase_id].epochs.append(epoch)
            epochs_by_id[ev.epoch_id] = epoch
        elif isinstance(ev, EpochCompleted):
            epoch = epochs_by_id[ev.epoch_id]
            epoch.status = "completed"
            epoch.ended_ts = ev.ts
        elif isinstance(ev, EpochFailed):
            epoch = epochs_by_id[ev.epoch_id]
            epoch.status = "failed"
            epoch.ended_ts = ev.ts
        elif isinstance(ev, FailedEpochHandled):
            epochs_by_id[ev.epoch_id].status = f"failed ({ev.action})"
        elif isinstance(ev, EpochVerificationStarted):
            epochs_by_id[ev.epoch_id].status = "verifying"
        elif isinstance(ev, EpochVerificationPassed):
            epochs_by_id[ev.epoch_id].status = "verified"
        elif isinstance(ev, EpochVerificationFailed):
            epoch = epochs_by_id[ev.epoch_id]
            epoch.status = "verification_failed"
            epoch.ended_ts = ev.ts
        elif isinstance(ev, TaskDispatched):
            task = _task(epochs_by_id[ev.epoch_id], ev.task_id)
            task.status = "dispatched"
            if task.started_ts is None:  # first dispatch; retries keep the origin
                task.started_ts = ev.ts
        elif isinstance(ev, TaskRetried):
            task = _task(epochs_by_id[ev.epoch_id], ev.task_id)
            task.status = "retried"
            task.attempt = ev.attempt
        elif isinstance(ev, TaskEscalated):
            task = _task(epochs_by_id[ev.epoch_id], ev.task_id)
            task.status = "escalated"
            task.ended_ts = ev.ts
            task.note = f"→ {ev.tier}"
        elif isinstance(ev, TaskDone):
            task = _task(epochs_by_id[ev.epoch_id], ev.task_id)
            task.status = "done"
            task.ended_ts = ev.ts
            task.note = None  # success clears any stale rejection note
        elif isinstance(ev, TaskFailed):
            task = _task(epochs_by_id[ev.epoch_id], ev.task_id)
            task.status = "failed"
            task.ended_ts = ev.ts
        elif isinstance(ev, HandoffRejected):
            # The worker's disk contract was rejected; surface why (until resolved).
            _task(epochs_by_id[ev.epoch_id], ev.task_id).note = ev.reason

    if tree is None:
        raise ValueError("empty event stream: no run_started")
    return tree
