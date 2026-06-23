"""Hand-written Pydantic v2 mirror of ``schemas/epoch_decision.json`` and
``schemas/handoff.json``.

These are the typed structs the core works with: stringly JSON is parsed here
exactly once, at the boundary, and never crosses inward. Every model is frozen
and forbids unknown keys, so a parsed value is immutable and complete. Field
constraints mirror the JSON Schema character-level limits one-for-one; the
schema is the source of truth and the equivalence test guards against drift.
"""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StringConstraints,
    TypeAdapter,
    field_validator,
)

# --- scalar aliases (mirror schema $defs / pattern constraints) ----------------

#: Key into the run's durable keyed log; also the handoff downstream-needs shape.
LogKey = Annotated[
    str, StringConstraints(pattern=r"^[A-Za-z0-9][a-zA-Z0-9._/-]{0,127}$")
]


class _Frozen(BaseModel):
    """Immutable, closed model, the boundary invariant for every contract type."""

    model_config = ConfigDict(frozen=True, extra="forbid")


# --- checks (oneOf: a command or a required artifact) --------------------------


class CmdCheck(_Frozen):
    """Deterministic check: run a command, expect an exit code."""

    cmd: Annotated[str, StringConstraints(min_length=1, max_length=512)]
    expect_exit: int = 0


class ArtifactExistsCheck(_Frozen):
    """Deterministic check: a validated artifact must exist at a log key."""

    artifact_exists: LogKey


class VisionReviewSpec(_Frozen):
    """The taste-gate's payload: a screenshot path + the criteria to judge it by.

    ``screenshot`` is a path RELATIVE TO THE EVAL WORKTREE, a prior cmd check in
    the same criterion list renders the UI there (e.g. ``ui/screen.png``). The
    pattern forbids a leading ``/`` and any ``..`` segment so a planner-supplied
    path cannot escape the worktree (codex reads the image with ``-i``, an
    absolute/traversal path would be an arbitrary on-disk file read). ``criteria``
    is prose describing what polished/correct looks like; the core feeds both to
    codex, which returns a structured verdict (``vision_verdict``).
    """

    screenshot: Annotated[str, StringConstraints(min_length=1, max_length=256)]
    criteria: Annotated[str, StringConstraints(min_length=1, max_length=2048)]

    @field_validator("screenshot")
    @classmethod
    def _relative_no_traversal(cls, v: str) -> str:
        # Mirrors the schema's screenshot ``pattern`` (whose look-ahead the
        # Pydantic regex engine cannot express): a worktree-relative path with no
        # leading ``/`` and no ``..`` segment, so it can never escape the worktree.
        if v.startswith("/") or ".." in v.split("/"):
            raise ValueError(
                "screenshot must be a worktree-relative path with no '..' segments"
            )
        return v


class VisionReviewCheck(_Frozen):
    """Taste check (B3): codex looks at a rendered-UI screenshot + criteria and
    emits a pass/fail verdict, layered on top of the deterministic functional
    floor. The verdict is a re-read disk contract (``vision_verdict.json``), never
    stdout, identical in spirit to the worker handoff."""

    vision_review: VisionReviewSpec


Check = Union[CmdCheck, ArtifactExistsCheck, VisionReviewCheck]


# --- phases --------------------------------------------------------------------


class Phase(_Frozen):
    """Skeleton milestone: id, title, machine-checkable exit, epoch budget."""

    id: Annotated[str, StringConstraints(pattern=r"^P[1-9][0-9]?$")]
    title: Annotated[str, StringConstraints(min_length=1, max_length=120)]
    exit_criterion: Annotated[list[Check], Field(min_length=1, max_length=8)]
    epoch_budget: Annotated[int, Field(ge=1, le=20)]


# --- tasks ---------------------------------------------------------------------


class _TaskBase(_Frozen):
    id: Annotated[str, StringConstraints(pattern=r"^T[1-8]$")]
    goal: Annotated[str, StringConstraints(min_length=1, max_length=1024)]
    inputs: Annotated[list[LogKey], Field(max_length=12)] = Field(default_factory=list)
    done_when: Annotated[list[Check], Field(min_length=1, max_length=6)]
    #: Natural-language semantic acceptance statements, judged later by an agentic
    #: verification pass (gate rebalance), NOT by a shell command. Deterministic
    #: ``done_when`` / ``checks`` cover structural facts (build, test, type-check,
    #: file existence); content/semantic acceptance ("the plan maps every ramp to
    #: an RN equivalent") belongs here. Optional, defaults to an empty list; each
    #: entry is a non-empty prose statement (mirrors the schema's ``criteria``).
    criteria: Annotated[
        list[Annotated[str, StringConstraints(min_length=1, max_length=512)]],
        Field(max_length=8),
    ] = Field(default_factory=list)
    skills: Annotated[
        list[Annotated[str, StringConstraints(max_length=64)]], Field(max_length=6)
    ] = Field(default_factory=list)
    #: Per-task tier routing: True routes THIS task to the senior tier (judgment /
    #: taste work, e.g. layout, polish, an approach synthesis, a design-quality
    #: verdict), False (the default) runs it on the local rig (mechanical /
    #: factual work, e.g. scaffolding, tokens, boilerplate, web-search fact
    #: gathering, a structural review). Routing is per TASK, not per epoch, so one
    #: epoch can split a mechanical local slice from a taste senior slice and the
    #: senior quota is spent only where judgment is needed. A rig with no senior
    #: tier falls back to local. Optional, defaults False (a decision without the
    #: field parses unchanged); ``StrictBool`` mirrors the schema's ``boolean``
    #: type, so a non-bool is rejected at both layers. It also picks the size-gate
    #: file-count cap (senior tasks get the larger bound).
    senior: StrictBool = False


class ImplementTask(_TaskBase):
    """Write task: additionally owns disjoint path globs (merge-correctness)."""

    file_ownership: Annotated[
        list[Annotated[str, StringConstraints(min_length=1, max_length=256)]],
        Field(min_length=1, max_length=32),
    ]


class ArtifactTask(_TaskBase):
    """Non-write task (research/review/artifact): produces one artifact log key."""

    artifact_out: LogKey
    targets: (
        Annotated[
            list[Annotated[str, StringConstraints(max_length=256)]],
            Field(max_length=32),
        ]
        | None
    ) = None


# --- epoch args ----------------------------------------------------------------


class _EpochArgsBase(_Frozen):
    epoch_title: Annotated[str, StringConstraints(min_length=1, max_length=120)]
    rationale: Annotated[str, StringConstraints(max_length=2048)]


class ImplementEpochArgs(_EpochArgsBase):
    tasks: Annotated[list[ImplementTask], Field(min_length=1, max_length=8)]


class ArtifactEpochArgs(_EpochArgsBase):
    tasks: Annotated[list[ArtifactTask], Field(min_length=1, max_length=8)]


class SkeletonArgs(_Frozen):
    phases: Annotated[list[Phase], Field(min_length=2, max_length=10)]


class RevisePhasesArgs(_Frozen):
    reason: Annotated[str, StringConstraints(min_length=1, max_length=1024)]
    phases: Annotated[list[Phase], Field(min_length=1, max_length=10)]


class RetryFailedEpochArgs(_Frozen):
    """Retry the failed epoch, optionally with corrective guidance + a tier bump."""

    action: Literal["retry"]
    hint: Annotated[str, StringConstraints(min_length=1, max_length=2048)]
    escalate_tier: StrictBool = False


class EscalateSeniorFailedEpochArgs(_Frozen):
    """Hand the failed epoch to the senior tier with a diagnosis."""

    action: Literal["escalate_senior"]
    diagnosis: Annotated[str, StringConstraints(min_length=1, max_length=2048)]


class HaltFailedEpochArgs(_Frozen):
    """Stop the run for a human: the epoch is not satisfiable as specified."""

    action: Literal["halt"]
    reason: Annotated[str, StringConstraints(min_length=1, max_length=2048)]


#: The three focused dispositions of a FAILED epoch, discriminated on ``action``
#: (mirrors the schema's ``handle_failed_epoch_args`` oneOf). NOT a phase replan
#: (revise_phases) and NOT a fresh work epoch.
HandleFailedEpochArgs = Annotated[
    Union[
        RetryFailedEpochArgs,
        EscalateSeniorFailedEpochArgs,
        HaltFailedEpochArgs,
    ],
    Field(discriminator="action"),
]


class PhaseCompleteArgs(_Frozen):
    """The planner's judgement that the CURRENT phase is complete (deliverable met).

    ``summary`` says why the phase goal is satisfied; ``deliverables`` are the
    concrete repo-relative artifact paths that satisfy it, each existence-checked
    in the integration-tip tree before the phase ends (existence only, NOT a
    quality judgement). The path list is bounded like ``ImplementTask`` ``file_ownership``;
    a missing cited path bounces the decision back into planning, never halts."""

    summary: Annotated[str, StringConstraints(min_length=1, max_length=2048)]
    deliverables: Annotated[
        list[Annotated[str, StringConstraints(min_length=1, max_length=256)]],
        Field(min_length=1, max_length=32),
    ]


class EscalateRunArgs(_Frozen):
    reason: Annotated[str, StringConstraints(min_length=1, max_length=2048)]
    needed_from_human: (
        Annotated[str, StringConstraints(max_length=1024)] | None
    ) = None


class CompleteRunArgs(_Frozen):
    summary: Annotated[str, StringConstraints(min_length=1, max_length=2048)]
    evidence: Annotated[list[Check], Field(min_length=1, max_length=8)]


# --- decisions (discriminated union on ``tool``) -------------------------------


class ProposeSkeletonDecision(_Frozen):
    schema_version: Literal["1"]
    tool: Literal["propose_skeleton"]
    args: SkeletonArgs


class ImplementDecision(_Frozen):
    schema_version: Literal["1"]
    tool: Literal["implement"]
    args: ImplementEpochArgs


class ResearchDecision(_Frozen):
    schema_version: Literal["1"]
    tool: Literal["research"]
    args: ArtifactEpochArgs


class ReviewDecision(_Frozen):
    schema_version: Literal["1"]
    tool: Literal["review"]
    args: ArtifactEpochArgs


class ArtifactDecision(_Frozen):
    schema_version: Literal["1"]
    tool: Literal["artifact"]
    args: ArtifactEpochArgs


class RevisePhasesDecision(_Frozen):
    schema_version: Literal["1"]
    tool: Literal["revise_phases"]
    args: RevisePhasesArgs


class HandleFailedEpochDecision(_Frozen):
    schema_version: Literal["1"]
    tool: Literal["handle_failed_epoch"]
    args: HandleFailedEpochArgs


class PhaseCompleteDecision(_Frozen):
    schema_version: Literal["1"]
    tool: Literal["phase_complete"]
    args: PhaseCompleteArgs


class EscalateRunDecision(_Frozen):
    schema_version: Literal["1"]
    tool: Literal["escalate_run"]
    args: EscalateRunArgs


class CompleteRunDecision(_Frozen):
    schema_version: Literal["1"]
    tool: Literal["complete_run"]
    args: CompleteRunArgs


EpochDecision = Annotated[
    Union[
        ProposeSkeletonDecision,
        ImplementDecision,
        ResearchDecision,
        ReviewDecision,
        ArtifactDecision,
        RevisePhasesDecision,
        HandleFailedEpochDecision,
        PhaseCompleteDecision,
        EscalateRunDecision,
        CompleteRunDecision,
    ],
    Field(discriminator="tool"),
]

_DECISION_ADAPTER: TypeAdapter[EpochDecision] = TypeAdapter(EpochDecision)


def parse_decision(payload: object) -> EpochDecision:
    """Parse untrusted JSON into the typed decision union (raises on invalid)."""

    return _DECISION_ADAPTER.validate_python(payload)


# --- handoff -------------------------------------------------------------------


class WhatChanged(_Frozen):
    kind: Literal["file", "interface", "artifact"]
    ref: Annotated[str, StringConstraints(min_length=1, max_length=256)]


class Citation(_Frozen):
    file: Annotated[str, StringConstraints(min_length=1, max_length=256)]
    line: Annotated[int, Field(ge=1)] | None = None


class CheckResult(_Frozen):
    check: Annotated[str, StringConstraints(max_length=512)]
    exit_code: int


class Occupancy(_Frozen):
    compacted: bool
    subagent_splits: Annotated[int, Field(ge=0)]
    peak_context_tokens: Annotated[int, Field(ge=0)] | None = None


class Handoff(_Frozen):
    """Worker handoff written to disk; the disk file is the gate, not stdout."""

    schema_version: Literal["1"]
    task_id: Annotated[
        str, StringConstraints(pattern=r"^P[1-9][0-9]?/E[1-9][0-9]?/T[1-8]$")
    ]
    status: Literal["DONE", "FAILED", "PARTIAL"]
    what_changed: Annotated[list[WhatChanged], Field(max_length=16)] = Field(
        default_factory=list
    )
    resulting_state: Annotated[str, StringConstraints(min_length=1, max_length=1500)]
    downstream_needs: Annotated[list[LogKey], Field(max_length=8)] = Field(
        default_factory=list
    )
    not_done: Annotated[
        list[Annotated[str, StringConstraints(max_length=256)]], Field(max_length=8)
    ] = Field(default_factory=list)
    citations: Annotated[list[Citation], Field(max_length=12)] = Field(
        default_factory=list
    )
    # Cap 8, not the task's 6: an honest echo includes the planner's done_when
    # (max 6) plus the core-appended validator and implement-mode review gates.
    checks: Annotated[list[CheckResult], Field(max_length=8)]
    occupancy: Occupancy


_HANDOFF_ADAPTER: TypeAdapter[Handoff] = TypeAdapter(Handoff)


def parse_handoff(payload: object) -> Handoff:
    """Parse untrusted JSON into the typed handoff model (raises on invalid)."""

    return _HANDOFF_ADAPTER.validate_python(payload)


# --- vision verdict (B3 taste gate disk contract) ------------------------------


class VisionVerdict(_Frozen):
    """codex's verdict for a ``vision_review`` check, mirroring
    ``schemas/vision_verdict.json``: a strict ``pass`` boolean (``StrictBool``
    rejects a stringy/numeric verdict) and the reasons behind it. Parsed from the
    re-read verdict file at the boundary; the field is aliased to ``pass`` (a
    Python keyword) but read as ``.passed``."""

    passed: StrictBool = Field(alias="pass")
    reasons: Annotated[
        list[Annotated[str, StringConstraints(max_length=512)]], Field(max_length=16)
    ]


_VERDICT_ADAPTER: TypeAdapter[VisionVerdict] = TypeAdapter(VisionVerdict)


def parse_vision_verdict(payload: object) -> VisionVerdict:
    """Parse untrusted JSON into the typed vision verdict (raises on invalid)."""

    return _VERDICT_ADAPTER.validate_python(payload)


# --- epoch verdict (G4 agentic verification pass disk contract) ----------------


class CriterionJudgement(_Frozen):
    """One criterion's adversarial judgement, mirroring ``schemas/epoch_verdict.json``:
    the verbatim ``criterion``, a strict ``met`` boolean (``StrictBool`` rejects a
    stringy/numeric value), and the artifact ``evidence`` behind it.

    The free-text ``criterion`` / ``evidence`` are UNBOUNDED in length: the verdict is
    an agent INPUT delivered to the planner BY REFERENCE (the full ``verdict.json`` is
    persisted on disk and the planner reads the file), never byte-capped-and-embedded
    into a prompt, so a verbose verifier can never lose information or reject a verdict
    on length alone (the old whole-verdict-rejection bug is impossible)."""

    criterion: str
    met: StrictBool
    evidence: str


class EpochVerdict(_Frozen):
    """The local-tier verification pass's verdict for an epoch (G4), mirroring
    ``schemas/epoch_verdict.json``: a strict ``pass`` boolean, the per-criterion
    judgements, and the concrete ``gaps`` surfaced to the planner on a fail. Parsed
    from the re-read ``verdict.json`` at the boundary (stdout is never parsed); the
    field is aliased to ``pass`` (a Python keyword) but read as ``.passed``. The
    agentic pass can only FAIL an epoch the deterministic floor already cleared, so
    the core treats a missing/invalid verdict as a fail-safe (no rubber-stamp).

    The free-text ``gaps`` and ``digest`` are UNBOUNDED in length: the whole verdict is
    persisted on disk and delivered to the planner BY REFERENCE (it reads the file), so
    nothing is byte-capped-and-embedded into a prompt and no length can reject it. The
    structural validation (``pass`` is a bool, ``per_criterion`` is a list of the right
    shape) is unchanged; only the length caps are gone."""

    passed: StrictBool = Field(alias="pass")
    per_criterion: Annotated[list[CriterionJudgement], Field(max_length=16)]
    gaps: list[str]
    #: A descriptive steering summary the verifier emits in the SAME pass (G10): what
    #: the epoch actually produced, key structure/decisions, what is notably incomplete
    #: or risky, written for the planner choosing the NEXT epoch. It is NOT a grade and
    #: NEVER affects ``passed``; absent (older/malformed verdicts) defaults to "". It
    #: travels to the planner by FILE (the persisted verdict), never embedded in a prompt.
    digest: str = ""


_EPOCH_VERDICT_ADAPTER: TypeAdapter[EpochVerdict] = TypeAdapter(EpochVerdict)


def parse_epoch_verdict(payload: object) -> EpochVerdict:
    """Parse untrusted JSON into the typed epoch verdict (raises on invalid).

    STRUCTURAL validation only (``pass`` is a bool, ``per_criterion`` is a list of the
    right shape, no unknown keys): a long ``evidence`` / ``criterion`` / ``gaps`` /
    ``digest`` parses fine and is preserved IN FULL. The verdict is an agent input
    delivered by reference (persisted on disk, the planner reads it), never embedded in
    a prompt, so there is no length to cap and no truncation."""

    return _EPOCH_VERDICT_ADAPTER.validate_python(payload)
