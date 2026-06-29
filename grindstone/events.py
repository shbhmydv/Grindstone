"""Journal event vocabulary + NDJSON I/O + the replay fold.

The append-only ``events.ndjson`` is the single source of truth for a run: the
stream alone must render the full run -> epoch -> task tree with statuses, and
resume reads it to find the last clean boundary. ``replay`` folds an event list
into a ``RunTree`` snapshot.

The bones taxonomy is small (BONES "epochs only, no phases"): the run lifecycle,
the epoch and task lifecycle, the work gate, the critic ``verdict`` triage,
and the one backoff signal ``rate_limited``. No phases, no infra-repair, no
vision, no session-limit / failed-epoch state-machine events.
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


class TaskRef(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    id: str
    mode: str


# --- run lifecycle -------------------------------------------------------------


class RunStarted(_Event):
    event: Literal["run_started"] = "run_started"
    run_id: str
    job_path: str
    #: The epoch backstop in force (None = off). Carried so a watcher can show
    #: "n/N" epochs without reading run state.
    max_epochs: int | None = None


class RunResumed(_Event):
    # Re-entry from a killed / rate-limited / crashed run. ``razed_epoch`` records
    # the incomplete epoch the programmatic cleanup tore down (BONES resume): the
    # journal is appended, never truncated, so the marker is permanent.
    event: Literal["run_resumed"] = "run_resumed"
    run_id: str
    razed_epoch: str | None = None


class RunCompleted(_Event):
    # The planner ended the run as DONE and the one final acceptance passed.
    event: Literal["run_completed"] = "run_completed"


class RunEnded(_Event):
    # A CLEAN partial-end (BONES failure model #2): the planner wrote a phase
    # handoff / pending-summary instead of continuing. ``summary`` is the resume
    # seed for the next appendable run. Not an error, a deliberate stopping point.
    event: Literal["run_ended"] = "run_ended"
    summary: str


# --- epoch + task lifecycle ----------------------------------------------------


class EpochStarted(_Event):
    event: Literal["epoch_started"] = "epoch_started"
    epoch_id: str
    title: str
    tasks: list[TaskRef]


class EpochCompleted(_Event):
    # An epoch reached its durable boundary: the planner wrote its close-out BATON,
    # the passing tasks' work fast-forwarded the run branch, and the baton is persisted
    # at ``E<n>/baton.md``. ``epoch_completed`` now IMPLIES "baton written" (close-out
    # runs immediately before this), so an aborted epoch (no baton) is never completed.
    event: Literal["epoch_completed"] = "epoch_completed"
    epoch_id: str


class TaskDispatched(_Event):
    event: Literal["task_dispatched"] = "task_dispatched"
    epoch_id: str
    task_id: str


class TaskDone(_Event):
    event: Literal["task_done"] = "task_done"
    epoch_id: str
    task_id: str


# --- gate + triage -------------------------------------------------------------


class WorkGatePassed(_Event):
    event: Literal["work_gate_passed"] = "work_gate_passed"
    epoch_id: str
    task_id: str


class WorkGateRejected(_Event):
    event: Literal["work_gate_rejected"] = "work_gate_rejected"
    epoch_id: str
    task_id: str
    reason: str


class Verdict(_Event):
    # The agentic critic's triage of one task. ``outcome`` is PASS / RETRY /
    # ESCALATE; ``reason`` is its free-text note (surfaced to the planner or the
    # retry).
    event: Literal["verdict"] = "verdict"
    epoch_id: str
    task_id: str
    outcome: Literal["PASS", "RETRY", "ESCALATE"]
    reason: str = ""


class RateLimited(_Event):
    # BONES failure model #1: a rate-limit / quota refusal on a role. The loop
    # backs off (~1/hr) and retries; ``role`` is planner / worker / senior.
    event: Literal["rate_limited"] = "rate_limited"
    role: str
    detail: str = ""


# --- strike ladder (per-task repair-escalation) --------------------------------


class StrikeLedgerEntry(BaseModel):
    """One struck lineage's persisted shape inside a ``StrikeLedger`` snapshot: the
    carried-unfinished task identity (its ``ownership`` globs for implement, or its
    ``artifact_out`` for a non-write task) and the accumulated ``strikes`` (failed
    epochs). Matched to a next-epoch task by ownership OVERLAP / artifact identity, so
    the count survives a re-decomposition."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    ownership: list[str] = Field(default_factory=list)
    artifact_out: str | None = None
    mode: str
    strikes: int
    reason: str = ""


class StrikeLedger(_Event):
    # The authoritative snapshot of every currently-struck lineage AFTER an epoch's
    # close (the DETERMINISTIC, resume-safe persistence of the strike ladder: a resumed
    # run rebuilds the counts from the LAST snapshot, no in-memory state survives the
    # crash). Emitted only when the ledger is non-empty or transitions to empty, so a
    # run that never carries a task adds zero strike events (byte-identical journal).
    event: Literal["strike_ledger"] = "strike_ledger"
    epoch_id: str
    entries: list[StrikeLedgerEntry]


class TaskParked(_Event):
    # Strike 2: a task whose lineage failed the WHOLE in-epoch tier ladder twice is
    # REMOVED (BLOCKED) from the active set (not re-dispatched) so the run can still
    # reach a clean partial-end. The structured, operator-visible signal ("the rig
    # could not close this"); how an operator is notified lives OUTSIDE the engine.
    event: Literal["task_parked"] = "task_parked"
    epoch_id: str
    task_id: str
    strikes: int
    reason: str = ""
    descriptor: str = ""


class TierEscalated(_Event):
    # The IN-EPOCH tier ladder climbed: ``run_task`` exhausted a task's local stage and
    # re-dispatched it (on the carried wip) at ``to_tier`` within the same epoch.
    # ``attempt`` is the total dispatch number the escalated stage starts at.
    event: Literal["tier_escalated"] = "tier_escalated"
    epoch_id: str
    task_id: str
    to_tier: str
    attempt: int


# --- the critic FAILURE NODE (gate passed, but the critic could not emit a verdict) --


class CriticRecovered(_Event):
    # The critic answered in CHAT instead of writing ``verdict.json`` (the local-model
    # transport quirk), and the stdout fallback recovered a parseable lenient verdict.
    # Observability only: the work proceeds on the recovered verdict, no retry burned.
    event: Literal["critic_recovered"] = "critic_recovered"
    epoch_id: str
    task_id: str
    tier: str


class CriticEscalated(_Event):
    # The critic FAILURE NODE bumped a chatted-out (no parseable verdict) critic up to
    # the senior tier to judge the SAME already-passed work (no re-grind), after the
    # task-tier critic exhausted its own dispatches.
    event: Literal["critic_escalated"] = "critic_escalated"
    epoch_id: str
    task_id: str
    to_tier: str


class CriticFailed(_Event):
    # The critic FAILURE NODE exhausted every dispatch (task tier + senior) without a
    # parseable verdict: the task routes to the planner (the existing ``escalated``
    # path). ``snippet`` is a short echo of the model's actual chat output, for the
    # operator to see WHY the critic could not land a verdict.
    event: Literal["critic_failed"] = "critic_failed"
    epoch_id: str
    task_id: str
    tier: str
    snippet: str = ""


Event = Annotated[
    Union[
        RunStarted,
        RunResumed,
        RunCompleted,
        RunEnded,
        EpochStarted,
        EpochCompleted,
        TaskDispatched,
        TaskDone,
        WorkGatePassed,
        WorkGateRejected,
        Verdict,
        RateLimited,
        StrikeLedger,
        TaskParked,
        TierEscalated,
        CriticRecovered,
        CriticEscalated,
        CriticFailed,
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

    Opened in append mode (crash-safe). Enforces strictly increasing ``seq`` so a
    programming error cannot silently corrupt replay ordering. Internally
    thread-safe: a single lock guards every write, so concurrent fan-out tasks may
    journal at the same time without corrupting the stream. ``emit`` assigns the
    next seq and writes the event atomically under the lock, so seq stays strictly
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

        The seq is read and advanced inside the lock, so two concurrent callers can
        never be handed the same seq nor write out of order. Returns the written
        event (with its assigned seq) for the caller's bookkeeping.
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

    A crash can leave a half-written last line; that line is skipped, never raised.
    Corruption of any earlier line is a real error and propagates.
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
    started_ts: str | None = None
    ended_ts: str | None = None
    #: A transient note worth surfacing: a handoff-rejection reason or the critic's
    #: verdict reason (cleared on done). Most-recent wins.
    note: str | None = None
    #: The last critic triage outcome ("PASS" / "RETRY" / "ESCALATE"), RETAINED past
    #: ``done`` (unlike ``note``) so the watcher keeps showing the critic verdict.
    verdict: str | None = None


@dataclass
class EpochNode:
    id: str
    title: str
    status: str
    tasks: list[TaskNode]
    started_ts: str | None = None
    ended_ts: str | None = None


@dataclass
class ParkedNode:
    """A task the strike ladder PARKED (removed/BLOCKED from the active set at strike
    2): the operator-facing "the rig could not close this" record. Surfaced at the run
    level
    (the parked task was removed BEFORE its epoch's task tree was built, so it has no
    ``TaskNode``)."""

    task_id: str
    epoch_id: str
    strikes: int
    reason: str
    #: The lineage descriptor (owned files / artifact key), the dedup key across the
    #: epochs that re-propose the same parked lineage.
    descriptor: str = ""


@dataclass
class RunTree:
    run_id: str
    job_path: str
    status: str
    epochs: list[EpochNode] = field(default_factory=list)
    #: The epoch backstop (from RunStarted), for an "n/N" header. None = off.
    max_epochs: int | None = None
    started_ts: str | None = None
    ended_ts: str | None = None
    #: ts of the most recent event seen, the journal's notion of "now".
    last_ts: str | None = None
    #: The RunEnded pending-summary (clean partial-end resume seed), if ended.
    end_summary: str | None = None
    #: The most recent rate-limit signal "role: detail", cleared on the next epoch.
    last_rate_limit: str | None = None
    #: Tasks the strike ladder parked (strike 4), newest last. Deduped by lineage so a
    #: lineage the planner re-proposes each epoch surfaces once (highest strike kept).
    parked: list[ParkedNode] = field(default_factory=list)


def _task(epoch: EpochNode, task_id: str) -> TaskNode:
    for task in epoch.tasks:
        if task.id == task_id:
            return task
    raise KeyError(f"task {task_id!r} not declared in epoch {epoch.id!r}")


def replay(events: list[Event]) -> RunTree:
    """Fold an event list into the run -> epoch -> task tree snapshot."""

    tree: RunTree | None = None
    epochs_by_id: dict[str, EpochNode] = {}

    for ev in events:
        if isinstance(ev, RunStarted):
            tree = RunTree(
                ev.run_id, ev.job_path, "running",
                max_epochs=ev.max_epochs, started_ts=ev.ts, last_ts=ev.ts,
            )
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
        elif isinstance(ev, RunEnded):
            tree.status = "ended"
            tree.ended_ts = ev.ts
            tree.end_summary = ev.summary
        elif isinstance(ev, RateLimited):
            tree.last_rate_limit = f"{ev.role}: {ev.detail}" if ev.detail else ev.role
        elif isinstance(ev, TaskParked):
            # Dedup by lineage descriptor: the planner may re-propose a parked lineage
            # each epoch, so keep ONE entry (the latest, highest-strike) per descriptor.
            key = ev.descriptor or ev.task_id
            tree.parked = [p for p in tree.parked if (p.descriptor or p.task_id) != key]
            tree.parked.append(
                ParkedNode(ev.task_id, ev.epoch_id, ev.strikes, ev.reason, ev.descriptor)
            )
        elif isinstance(ev, EpochStarted):
            tasks = [TaskNode(t.id, t.mode, "pending") for t in ev.tasks]
            epoch = EpochNode(ev.epoch_id, ev.title, "started", tasks, started_ts=ev.ts)
            tree.epochs.append(epoch)
            epochs_by_id[ev.epoch_id] = epoch
            tree.last_rate_limit = None  # a fresh epoch supersedes a stale backoff
        elif isinstance(ev, EpochCompleted):
            epoch = epochs_by_id[ev.epoch_id]
            epoch.status = "completed"
            epoch.ended_ts = ev.ts
        elif isinstance(ev, TaskDispatched):
            task = _task(epochs_by_id[ev.epoch_id], ev.task_id)
            task.status = "dispatched"
            if task.started_ts is None:  # first dispatch; retries keep the origin
                task.started_ts = ev.ts
        elif isinstance(ev, WorkGatePassed):
            _task(epochs_by_id[ev.epoch_id], ev.task_id).status = "gate_passed"
        elif isinstance(ev, WorkGateRejected):
            task = _task(epochs_by_id[ev.epoch_id], ev.task_id)
            task.status = "gate_rejected"
            task.note = ev.reason
        elif isinstance(ev, Verdict):
            task = _task(epochs_by_id[ev.epoch_id], ev.task_id)
            task.status = f"verdict_{ev.outcome.lower()}"
            task.note = ev.reason or None
            task.verdict = ev.outcome  # retained past done (note is cleared, this is not)
        elif isinstance(ev, TaskDone):
            task = _task(epochs_by_id[ev.epoch_id], ev.task_id)
            task.status = "done"
            task.ended_ts = ev.ts
            task.note = None  # success clears any stale rejection / verdict note

    if tree is None:
        raise ValueError("empty event stream: no run_started")
    return tree
