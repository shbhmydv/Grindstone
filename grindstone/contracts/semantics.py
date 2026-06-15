"""Semantic validators — rules JSON Schema cannot express.

Pure functions: they take the already-typed model plus explicit context
arguments (no global state) and return a list of violation strings (empty =
pass). The caller decides what to do with violations; these functions never
raise and never read the filesystem.
"""

from __future__ import annotations

import json
from typing import Literal

from grindstone.contracts.models import (
    ArtifactEpochArgs,
    ArtifactTask,
    EpochDecision,
    Handoff,
    ImplementEpochArgs,
    ImplementTask,
    Phase,
    ProposeSkeletonDecision,
    ReviewDecision,
    RevisePhasesDecision,
    _TaskBase,
)

HandoffMode = Literal["implement", "research", "review", "artifact"]

#: Total serialized handoff size cap (ARCHITECTURE.md): references, not payloads.
HANDOFF_MAX_BYTES = 8192


def canonical_bytes(model: Handoff) -> int:
    """Byte length of the handoff's canonical JSON form (the byte-cap measure).

    Canonical = sorted keys, no whitespace, ``None`` optionals dropped — a
    single deterministic form so the cap measures content, not formatting.
    """

    payload = model.model_dump(mode="json", exclude_none=True)
    return len(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())


def _inputs_exist(
    tasks: list[_TaskBase], existing_log_keys: frozenset[str]
) -> list[str]:
    out: list[str] = []
    for task in tasks:
        for key in task.inputs:
            if key not in existing_log_keys:
                out.append(f"task {task.id}: input log key not in keyed log: {key!r}")
    return out


def _unique_task_ids(tasks: list[_TaskBase]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for task in tasks:
        if task.id in seen:
            out.append(f"duplicate task id within epoch: {task.id!r}")
        seen.add(task.id)
    return out


def _unique_phase_ids(phases: list[Phase]) -> list[str]:
    """Phase ids must be unique within a skeleton. A duplicate would wedge phase
    advancement: ``_phase_index`` resolves an id to its FIRST occurrence, so the
    loop could never reach the later twin (an unbounded hang the planner-call cap
    cannot break). Rejected at the gate so a bad skeleton never runs."""

    seen: set[str] = set()
    out: list[str] = []
    for phase in phases:
        if phase.id in seen:
            out.append(f"duplicate phase id within skeleton: {phase.id!r}")
        seen.add(phase.id)
    return out


def _fixed_prefix(glob: str) -> str:
    """The literal leading segment of a glob, up to the first wildcard char."""

    chars: list[str] = []
    for ch in glob:
        if ch in "*?[":
            break
        chars.append(ch)
    return "".join(chars)


def _ownership_disjoint(tasks: list[ImplementTask]) -> list[str]:
    """Pairwise-disjoint file_ownership across tasks.

    Conservative rule (over-rejection acceptable, under-rejection not): two
    globs overlap if either's fixed prefix (the literal part before the first
    wildcard) is a prefix of the other's. Distinct sibling directories stay
    disjoint; a broad glob and anything beneath it overlap.
    """

    out: list[str] = []
    for i in range(len(tasks)):
        for j in range(i + 1, len(tasks)):
            for ga in tasks[i].file_ownership:
                for gb in tasks[j].file_ownership:
                    pa, pb = _fixed_prefix(ga), _fixed_prefix(gb)
                    if pa.startswith(pb) or pb.startswith(pa):
                        out.append(
                            f"file_ownership overlap between {tasks[i].id} "
                            f"({ga!r}) and {tasks[j].id} ({gb!r})"
                        )
    return out


def _review_targets(tasks: list[ArtifactTask]) -> list[str]:
    out: list[str] = []
    for task in tasks:
        if not task.targets:
            out.append(f"review task {task.id}: missing targets")
    return out


def epoch_decision_violations(
    decision: EpochDecision,
    *,
    existing_log_keys: frozenset[str],
    completed_phase_ids: frozenset[str],
) -> list[str]:
    """All semantic violations for one planner decision (empty = pass)."""

    out: list[str] = []
    args = decision.args
    if isinstance(args, (ImplementEpochArgs, ArtifactEpochArgs)):
        out += _inputs_exist(list(args.tasks), existing_log_keys)
        out += _unique_task_ids(list(args.tasks))
    if isinstance(args, ImplementEpochArgs):
        out += _ownership_disjoint(list(args.tasks))
    if isinstance(decision, ReviewDecision):
        out += _review_targets(list(decision.args.tasks))
    if isinstance(decision, ProposeSkeletonDecision):
        out += _unique_phase_ids(list(decision.args.phases))
    if isinstance(decision, RevisePhasesDecision):
        out += _unique_phase_ids(list(decision.args.phases))
        for phase in decision.args.phases:
            if phase.id in completed_phase_ids:
                out.append(f"revise_phases reuses completed phase id: {phase.id!r}")
    return out


def handoff_violations(
    handoff: Handoff, *, mode: HandoffMode, expected_task_id: str
) -> list[str]:
    """All semantic violations for one worker handoff (empty = pass)."""

    out: list[str] = []
    size = canonical_bytes(handoff)
    if size > HANDOFF_MAX_BYTES:
        out.append(f"handoff exceeds {HANDOFF_MAX_BYTES} bytes: {size}")
    if mode in ("research", "review") and not handoff.citations:
        out.append(f"{mode} handoff requires >= 1 citation")
    if handoff.task_id != expected_task_id:
        out.append(
            f"task_id {handoff.task_id!r} != dispatched id {expected_task_id!r}"
        )
    return out
