"""Semantic validators, rules JSON Schema cannot express.

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
    CmdCheck,
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

    Canonical = sorted keys, no whitespace, ``None`` optionals dropped, a
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


#: Globs that own the WHOLE repo: a fresh implement task carrying one of these
#: has not been decomposed at all (it claims everything), which defeats the
#: localize-the-failure point of file_ownership. Mirrors the whole-repo set
#: ``worktree.path_in_scope`` treats as matching every path, plus the bare
#: top-level ``*``. A scoped wildcard (``src/*``, ``src/**``) is NOT unbounded.
_UNBOUNDED_OWNERSHIP: frozenset[str] = frozenset({"**", "**/*", "*"})


def implement_task_size_violations(
    tasks: list[ImplementTask], *, max_files: int
) -> list[str]:
    """Deterministic per-task SIZE gate for a FRESH implement decomposition.

    Two structural rejections, both derived from the decision alone (no
    filesystem, no model trust), so the planner CANNOT emit an oversized task:

    - **Too broad**: a task whose ``file_ownership`` carries more than
      ``max_files`` globs is not a bounded slice; the planner is told to split it.
      ``max_files`` is TIER-AWARE: the caller passes the bound for the epoch's
      target tier (local default vs the larger senior default).
    - **Unbounded**: a whole-repo glob (``**`` / ``**/*`` / a bare ``*``) means
      "I did not decompose", it owns every path, so the merge-disjointness and
      scope checks lose all meaning. Rejected outright for a normal implement
      task. A scoped wildcard (``src/**``) is fine; only the whole-repo forms are.

    This gate is scoped to FRESH decomposition by the CALLER (it is skipped while
    a failed epoch is awaiting disposition): a ``handle_failed_epoch`` repair
    re-dispatches the ORIGINATING decision directly and may legitimately need
    broad scope, so it never re-enters this gate.
    """

    out: list[str] = []
    for task in tasks:
        unbounded = [g for g in task.file_ownership if g in _UNBOUNDED_OWNERSHIP]
        if unbounded:
            out.append(
                f"implement task {task.id} claims whole-repo ownership "
                f"({', '.join(repr(g) for g in unbounded)}): that is not a "
                f"decomposed slice, split it into tasks that each own a bounded "
                f"set of files"
            )
        elif len(task.file_ownership) > max_files:
            out.append(
                f"implement task {task.id} owns {len(task.file_ownership)} "
                f"file_ownership globs (> {max_files} for its tier): the task is "
                f"too big, split it into smaller tasks (or later epochs) so each "
                f"owns at most {max_files}"
            )
    return out


#: Program names whose JOB is to grep file CONTENT for a token. A check built on
#: one is a content-grep, a brittle proxy for semantic acceptance (it fails for
#: environmental reasons, e.g. the binary missing, and the gate rebalance forbids
#: it). Structural facts (build, test, type-check, file existence) are expressed
#: with the project's own commands; content/semantic acceptance is `criteria`.
_CONTENT_GREP_PROGRAMS: frozenset[str] = frozenset(
    {"rg", "grep", "egrep", "fgrep", "ag", "ack"}
)

#: Shell separators that start a fresh simple-command, so each segment's first
#: word (its program) is checked independently.
_SHELL_SEPARATORS = ("|", "&&", "||", ";")


def _segment_program(segment: str) -> str | None:
    """The program token of one simple-command segment, or ``None`` if empty.

    Leading ``NAME=value`` env-assignments are skipped (they precede the program
    in a simple command), and a path is reduced to its basename so ``/usr/bin/grep``
    resolves to ``grep``. Tokenization is whitespace-only (good enough to read the
    program name; we never execute the parse).
    """

    for token in segment.split():
        if "=" in token and not token.startswith("="):
            head = token.split("=", 1)[0]
            if head and all(c.isalnum() or c == "_" for c in head):
                continue  # an env-assignment prefix, the program is later
        return token.rsplit("/", 1)[-1]
    return None


def command_is_content_grep(cmd: str) -> bool:
    """True when ``cmd`` runs a content-grep in ANY of its simple-command segments.

    The command is split on the shell separators ``| && || ;``; each segment's
    program (its first word, past any ``NAME=value`` prefix and any path) is
    matched against the rg/grep/egrep/fgrep/ag/ack family. A filename that merely
    CONTAINS "grep" is never matched (the token must be the program itself).
    """

    segments = [cmd]
    for sep in _SHELL_SEPARATORS:
        segments = [part for seg in segments for part in seg.split(sep)]
    return any(
        _segment_program(seg) in _CONTENT_GREP_PROGRAMS for seg in segments
    )


def content_grep_check_violations(tasks: list[_TaskBase]) -> list[str]:
    """Reject any task ``done_when`` cmd check that is a content-grep.

    Deterministic checks are for STRUCTURAL facts (build, test, type-check, file
    existence); a content-grep (``rg`` / ``grep`` for a token) is a brittle proxy
    for semantic acceptance that fails for environmental reasons. The rejection
    steers the planner to express that acceptance as natural-language ``criteria``
    instead. Non-cmd checks (artifact_exists / vision_review) are unaffected.
    """

    out: list[str] = []
    for task in tasks:
        for check in task.done_when:
            if isinstance(check, CmdCheck) and command_is_content_grep(check.cmd):
                out.append(
                    f"task {task.id}: done_when check {check.cmd!r} is a content-grep; "
                    f"deterministic checks are for structural facts (build, test, "
                    f"type-check, file existence). Express content/semantic "
                    f"acceptance as natural-language `criteria` instead."
                )
    return out


def unknown_skill_violations(
    tasks: list[_TaskBase], *, known_skill_names: frozenset[str]
) -> list[str]:
    """Reject any task ``skills`` name not in the target repo's domain catalogue.

    A task selects domain skills (``.grindstone/skills/<name>.md``) by NAME; the
    planner cannot hallucinate one, every selected name must be a key the catalogue
    index advertised (``domain_skills.load_domain_skill_index``). The known set is
    threaded into the gate by the run loop; an EMPTY set (the default, and the case
    for a repo with no catalogue) means NO name is known, so any selected skill is
    rejected, which is exactly "no catalogue -> ``skills`` must be empty". The
    rejection names the offending task + skill so the re-ask steers the planner.
    """

    out: list[str] = []
    for task in tasks:
        for name in task.skills:
            if name not in known_skill_names:
                out.append(
                    f"task {task.id}: unknown domain skill {name!r} (not in the "
                    f"target repo's .grindstone/skills/index.md catalogue); select "
                    f"only skills the index advertises, or none"
                )
    return out


def epoch_decision_violations(
    decision: EpochDecision,
    *,
    existing_log_keys: frozenset[str],
    completed_phase_ids: frozenset[str],
    known_skill_names: frozenset[str] = frozenset(),
) -> list[str]:
    """All semantic violations for one planner decision (empty = pass)."""

    out: list[str] = []
    args = decision.args
    if isinstance(args, (ImplementEpochArgs, ArtifactEpochArgs)):
        out += _inputs_exist(list(args.tasks), existing_log_keys)
        out += _unique_task_ids(list(args.tasks))
        out += unknown_skill_violations(
            list(args.tasks), known_skill_names=known_skill_names
        )
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
