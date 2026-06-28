"""The per-task STRIKE-LADDER: deterministic repair-escalation across epochs.

The bones loop already CARRIES an unfinished task forward (the planner re-issues it
next epoch from its baton backlog), and ``run_task`` now climbs the WHOLE in-epoch
tier ladder every epoch (local retries then a senior escalation on the carried wip,
all within one epoch). So escalation to senior no longer needs a cross-epoch rung -
every epoch already reaches the strongest tier. What the cross-epoch ladder still owns
is the TERMINAL: a lineage that fails the full in-epoch ladder epoch after epoch must
eventually STOP. This module is the small DETERMINISTIC state machine that blocks it.
Model proposes, state machine disposes:

* a STRIKE is one whole failed epoch for a task lineage (it was carried unfinished at
  the epoch close: its gate never passed / it escalated even off the senior tier).
* the ladder, keyed on the accumulated strike count C a lineage carries INTO an
  attempt: C == 1 -> keep, carry + a re-decomposition NUDGE (soft, in the
  baton/planner input) - the planner gets ONE chance to reframe or re-decompose;
  C >= 2 -> PARK (BLOCK) the lineage (remove it from the active set so the run can
  still reach a clean partial-end). C == 0 is the untouched first attempt, so a run
  that never carries a task behaves exactly as today.

The trick is LINEAGE INHERITANCE across re-decomposition: the planner controls task
ids and may split a carried task into new sub-tasks, so the count cannot live on the
task id. It lives on a struck-lineage LEDGER keyed by the task's DURABLE identity (the
owned-file globs for an implement task, the ``artifact_out`` key for a non-write one),
and a next-epoch task is matched to a struck lineage by ownership OVERLAP / artifact
identity. So a child that owns a subset of a struck parent's files inherits the
parent's strikes; relabelling cannot dodge the ladder.

The ledger is reconstructed each boundary from the journal's last ``StrikeLedger``
snapshot, so the count is RESUME-SAFE (no in-memory state survives a crash). Every
function here is PURE (no I/O, no git): the loop owns reconstruction, event emission,
and the dispatch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from grindstone import worktree as wt
from grindstone.contracts.models import Task
from grindstone.events import EpochCompleted, Event, StrikeLedger, StrikeLedgerEntry

#: The ladder's one rung, keyed on the strike count a lineage carries INTO an attempt:
#: the second whole-ladder failure BLOCKS (parks) the lineage. Senior is reached
#: in-epoch every epoch now, so there is no separate force-senior rung.
PARK_THRESHOLD = 2


@dataclass(frozen=True)
class StrikeEntry:
    """One struck lineage: the carried-unfinished task identity plus its accumulated
    ``strikes`` (failed epochs). ``ownership`` carries an implement task's owned globs
    (``artifact_out`` is ``None``); ``artifact_out`` carries a non-write task's log key
    (``ownership`` is empty). ``reason`` is the last failure's verbatim text."""

    ownership: tuple[str, ...]
    artifact_out: str | None
    mode: str
    strikes: int
    reason: str


@dataclass(frozen=True)
class CarriedItem:
    """The PLAN-time projection of one struck lineage, rendered into the planner input
    as the soft re-decomposition NUDGE. ``descriptor`` names the lineage (owned files /
    artifact key); ``parked`` tells the planner the state machine has BLOCKED an
    overlapping task (it was dropped) versus its one reframe/re-decompose chance."""

    descriptor: str
    mode: str
    strikes: int
    reason: str
    parked: bool


@dataclass(frozen=True)
class ParkedTask:
    """A task ``apply_ladder`` removed (BLOCKED) from the active set at strike 2 (its
    second whole-ladder failure)."""

    task_id: str
    strikes: int
    reason: str
    descriptor: str


@dataclass(frozen=True)
class LadderResult:
    """The plan-time transform's output: ``tasks`` is the kept set (unchanged - the
    ladder no longer mutates a task's tier) and ``parked`` the removed (BLOCKED) tasks."""

    tasks: tuple[Task, ...]
    parked: tuple[ParkedTask, ...]


# --- lineage matching (the inheritance mechanism) ------------------------------


def _ownership_overlap(a: Sequence[str], b: Sequence[str]) -> bool:
    """Do two ownership glob lists intersect? Symmetric: a glob of one list, read as a
    path, in scope of the other counts (so concrete-vs-concrete equality,
    concrete-vs-glob containment, and a shared subtree all overlap). Reuses the same
    scope test the disjoint-merge invariant uses, so the ladder's notion of "same
    lineage" matches the integration gate's notion of "same files"."""

    return any(wt.path_in_scope(g, list(b)) for g in a) or any(
        wt.path_in_scope(g, list(a)) for g in b
    )


def _matches(entry: StrikeEntry, task: Task) -> bool:
    """Does ``task`` descend from the struck ``entry``? Implement lineages match by
    ownership overlap; non-write lineages by ``artifact_out`` identity. The two
    identity spaces never cross-match."""

    if task.mode == "implement":
        return entry.mode == "implement" and _ownership_overlap(
            entry.ownership, task.file_ownership
        )
    return (
        entry.mode != "implement"
        and entry.artifact_out is not None
        and entry.artifact_out == task.artifact_out
    )


def inherited_strikes(entries: Sequence[StrikeEntry], task: Task) -> int:
    """The strike count ``task`` inherits: the MAX over every struck lineage it
    descends from (0 when none). Max, not sum, so a child overlapping one struck
    parent inherits exactly the parent's count."""

    return max((e.strikes for e in entries if _matches(e, task)), default=0)


def _descriptor(task: Task) -> str:
    if task.mode == "implement":
        return ", ".join(task.file_ownership)
    return task.artifact_out or ""


def _entry_descriptor(entry: StrikeEntry) -> str:
    if entry.mode == "implement":
        return ", ".join(entry.ownership)
    return entry.artifact_out or ""


# --- the plan-time transform (force senior / park) -----------------------------


def apply_ladder(
    tasks: Sequence[Task], entries: Sequence[StrikeEntry]
) -> LadderResult:
    """The DETERMINISTIC plan-time rung: match each proposed task to the struck ledger
    and dispose. C >= 2 -> park (BLOCK: drop from the active set); below -> keep as
    planned (NO tier mutation - senior is reached in-epoch). Pure; the loop emits the
    journal events + dispatches the kept set."""

    kept: list[Task] = []
    parked: list[ParkedTask] = []
    for task in tasks:
        c = inherited_strikes(entries, task)
        if c >= PARK_THRESHOLD:
            parked.append(
                ParkedTask(
                    task_id=task.id,
                    strikes=c,
                    reason=_matched_reason(entries, task),
                    descriptor=_descriptor(task),
                )
            )
        else:
            kept.append(task)
    return LadderResult(tuple(kept), tuple(parked))


def _matched_reason(entries: Sequence[StrikeEntry], task: Task) -> str:
    """The last failure reason of the highest-strike lineage ``task`` descends from."""

    matched = sorted(
        (e for e in entries if _matches(e, task)), key=lambda e: e.strikes
    )
    return matched[-1].reason if matched else ""


# --- the close-time recompute (the strike count) -------------------------------


def next_ledger(
    prior: Sequence[StrikeEntry],
    *,
    landed: Sequence[Task],
    failed: Sequence[tuple[Task, str]],
) -> list[StrikeEntry]:
    """Recompute the struck ledger AFTER an epoch, deterministically:

    1. STRIKE: a failed (carried) task inherits the max strikes over the prior lineages
       it overlaps (against the FULL prior ledger, so two re-decomposed siblings BOTH
       inherit the shared parent, and a partial re-decompose where one child lands and
       one fails still carries the failing child forward), increments by one, and
       SUPERSEDES those overlapped parents (the count moves onto the finer-grained
       child with no double-count and no stale parent).
    2. RESOLVE: a landed (passed) task clears every prior lineage it overlaps that a
       failed task did NOT already claim (so a still-failing portion of a split task is
       never silently cleared by its sibling landing).

    Unrelated prior entries (a parked lineage, an untouched carry) pass through. Pure;
    the loop persists the result as a ``StrikeLedger`` snapshot."""

    superseded: set[int] = set()
    new_entries: list[StrikeEntry] = []
    for task, reason in failed:
        matched = [e for e in prior if _matches(e, task)]
        carried = max((e.strikes for e in matched), default=0)
        superseded.update(id(e) for e in matched)
        new_entries.append(
            StrikeEntry(
                ownership=tuple(task.file_ownership),
                artifact_out=task.artifact_out,
                mode=task.mode,
                strikes=carried + 1,
                reason=reason,
            )
        )
    kept_prior = [
        e
        for e in prior
        if id(e) not in superseded and not any(_matches(e, t) for t in landed)
    ]
    return kept_prior + new_entries


# --- journal round-trip (resume-safe persistence) ------------------------------


def to_event_entries(entries: Sequence[StrikeEntry]) -> list[StrikeLedgerEntry]:
    """Project the in-memory ledger into the persisted ``StrikeLedger`` snapshot shape."""

    return [
        StrikeLedgerEntry(
            ownership=list(e.ownership),
            artifact_out=e.artifact_out,
            mode=e.mode,
            strikes=e.strikes,
            reason=e.reason,
        )
        for e in entries
    ]


def reconstruct_entries(events: Sequence[Event]) -> list[StrikeEntry]:
    """Rebuild the struck ledger from the journal's last COMPLETED ``StrikeLedger``
    snapshot (the resume-safe count: no in-memory strike state survives a crash, so the
    ladder is re-derived from disk each boundary). A snapshot is only counted once its
    epoch reached ``EpochCompleted``, so an orphan snapshot from an epoch that crashed
    after emitting it but before completing is IGNORED (the re-ground epoch cannot then
    double-count its own strikes). Empty when the run never struck a lineage."""

    completed = {ev.epoch_id for ev in events if isinstance(ev, EpochCompleted)}
    last: StrikeLedger | None = None
    for ev in events:
        if isinstance(ev, StrikeLedger) and ev.epoch_id in completed:
            last = ev
    if last is None:
        return []
    return [
        StrikeEntry(
            ownership=tuple(e.ownership),
            artifact_out=e.artifact_out,
            mode=e.mode,
            strikes=e.strikes,
            reason=e.reason,
        )
        for e in last.entries
    ]


# --- projections (the soft nudge + the run summary) ----------------------------


def render_carried(entries: Sequence[StrikeEntry]) -> tuple[CarriedItem, ...]:
    """Project the struck ledger into the planner's PLAN-time nudge (soft guidance):
    one item per struck lineage, flagged ``parked`` once the state machine has BLOCKED
    an overlapping task (strike >= 2), else its one reframe/re-decompose chance."""

    return tuple(
        CarriedItem(
            descriptor=_entry_descriptor(e),
            mode=e.mode,
            strikes=e.strikes,
            reason=e.reason,
            parked=e.strikes >= PARK_THRESHOLD,
        )
        for e in entries
    )


def summarize_parked(entries: Sequence[StrikeEntry]) -> str:
    """The run-terminal SUMMARY line for parked (BLOCKED) lineages (those at strike >=
    2): the operator-facing "the rig could not close these N tasks". Empty when none
    parked, so a run that parked nothing leaves its summary byte-identical to today."""

    parked = [e for e in entries if e.strikes >= PARK_THRESHOLD]
    if not parked:
        return ""
    lines = [
        f"  - {_entry_descriptor(e)} ({e.strikes} strikes): {e.reason}".rstrip(": ")
        for e in parked
    ]
    return "PARKED (the rig could not close these):\n" + "\n".join(lines)
