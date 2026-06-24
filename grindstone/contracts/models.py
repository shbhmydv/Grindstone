"""Pydantic v2 wire contracts for the bones rewrite.

Three lenient structs the core works with, stringly JSON is parsed here exactly
once, at the boundary, and never crosses inward:

* ``Decision`` -- what the planner emits each boundary: an EPOCH (1..N tasks) or
  an END (a phase handoff / pending-summary that seeds the next appendable run).
* ``Handoff`` -- what a worker writes to disk in its CWD (the disk file is the
  gate, stdout is never parsed). Carries a self-reported ``BLOCKED`` status so an
  environment blocker routes straight to the planner, skipping the critic.
* ``Verdict`` -- the critic's triage: ``PASS`` | ``RETRY`` | ``ESCALATE`` plus a
  short free-text reason. Deliberately NOT a rigid multi-field schema, that shape
  is what a weak model fumbled (run 051645Z), rejecting work for a machinery fault.

Every model is frozen and forbids unknown keys, so a parsed value is immutable
and complete. The JSON Schemas in ``schemas/`` mirror these models; the schema is
the wire contract, Pydantic stays the source of truth.
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
    model_validator,
)

# --- scalar aliases (mirror schema $defs / pattern constraints) ----------------

#: Key into the run's durable keyed log; also the handoff downstream-needs shape.
#: A relative path under the run dir (``rundir.resolve`` enforces containment).
LogKey = Annotated[
    str, StringConstraints(pattern=r"^[A-Za-z0-9][a-zA-Z0-9._/-]{0,127}$")
]

#: The four worker intents. Picks the worker prompt/skill and the handoff rules
#: (research/review must cite). Carried on each task and echoed by the handoff.
HandoffMode = Literal["implement", "research", "review", "artifact"]

#: Total serialized handoff size cap (ARCHITECTURE.md): references, not payloads.
#: Kept here (formerly contracts/semantics.py) so ``check_handoff`` and any later
#: gate share one constant.
HANDOFF_MAX_BYTES = 8192


class _Frozen(BaseModel):
    """Immutable, closed model, the boundary invariant for every contract type."""

    model_config = ConfigDict(frozen=True, extra="forbid")


# --- decision: tasks -----------------------------------------------------------


class Task(_Frozen):
    """One unit of an epoch, fanned out to a worker in its own worktree.

    Minimal and lenient: a ``goal`` in prose (the worker's brief, including its own
    notion of done), a routing ``tier`` (local mechanical/checkable vs senior
    judgment/taste, the planner picks), an intent ``mode``, and the disk-shape
    fields the orchestrator needs to isolate + merge the work:

    * ``implement`` tasks declare ``file_ownership`` (>= 1 concrete path/glob), the
      disjoint-merge invariant is enforced over these and a worker may write only
      what it claimed.
    * ``research`` / ``review`` / ``artifact`` tasks declare ``artifact_out`` (the
      one log key the produced artifact lands at).

    ``skills`` selects domain skills by name from the target repo's catalogue
    (retrieve-not-concatenate); ``inputs`` names prior log keys this task reads.
    No rigid acceptance schema, semantic acceptance is judged agentically by the
    critic against the task's own claimed ``goal``.
    """

    id: Annotated[str, StringConstraints(pattern=r"^T[1-8]$")]
    mode: HandoffMode
    goal: Annotated[str, StringConstraints(min_length=1, max_length=2048)]
    #: Routing tier. ``local`` (default) is the Qwen rig (mechanical / checkable);
    #: ``senior`` is the Claude rig (judgment / taste). A rig with no senior tier
    #: falls back to local.
    tier: Literal["local", "senior"] = "local"
    file_ownership: Annotated[
        list[Annotated[str, StringConstraints(min_length=1, max_length=256)]],
        Field(max_length=32),
    ] = Field(default_factory=list)
    artifact_out: LogKey | None = None
    skills: Annotated[
        list[Annotated[str, StringConstraints(min_length=1, max_length=64)]],
        Field(max_length=6),
    ] = Field(default_factory=list)
    inputs: Annotated[list[LogKey], Field(max_length=12)] = Field(default_factory=list)

    @model_validator(mode="after")
    def _shape_by_mode(self) -> Task:
        # The one cross-field rule: an implement task owns files; a non-write task
        # produces an artifact at a log key. Kept minimal (not a criteria schema).
        if self.mode == "implement":
            if not self.file_ownership:
                raise ValueError(
                    f"task {self.id}: implement tasks must declare file_ownership"
                )
            if self.artifact_out is not None:
                raise ValueError(
                    f"task {self.id}: implement tasks do not declare artifact_out"
                )
        else:
            if self.artifact_out is None:
                raise ValueError(
                    f"task {self.id}: {self.mode} tasks must declare artifact_out"
                )
            if self.file_ownership:
                raise ValueError(
                    f"task {self.id}: {self.mode} tasks do not own files"
                )
        return self


class Epoch(_Frozen):
    """One epoch the planner proposes: a titled bundle of disjoint tasks.

    ``setup`` is the trusted-tier host-mutation seam (BONES safety boundary): the
    PLANNER (Claude) declares any install/setup commands the orchestrator runs
    before the tasks; the untrusted local worker never improvises host mutations.
    Empty by default.
    """

    title: Annotated[str, StringConstraints(min_length=1, max_length=120)]
    rationale: Annotated[str, StringConstraints(max_length=2048)] = ""
    tasks: Annotated[list[Task], Field(min_length=1, max_length=8)]
    setup: Annotated[
        list[Annotated[str, StringConstraints(min_length=1, max_length=512)]],
        Field(max_length=8),
    ] = Field(default_factory=list)

    @model_validator(mode="after")
    def _distinct_artifact_out(self) -> Epoch:
        # The artifact analogue of the disjoint-merge invariant: each non-write task
        # publishes to its artifact_out via a copy into the run dir, so two tasks
        # declaring the SAME key race on publish and the loser is silently clobbered
        # (both still return passed). Reject the collision here so the planner (which
        # self-validates decision.json through parse_decision) re-emits with distinct
        # keys, exactly as it must for colliding implement file_ownership.
        seen: set[str] = set()
        for task in self.tasks:
            out = task.artifact_out
            if out is None:
                continue
            if out in seen:
                raise ValueError(
                    f"two tasks declare the same artifact_out {out!r}; "
                    "artifact_out must be distinct across an epoch's tasks"
                )
            seen.add(out)
        return self


# --- decision: the two shapes (discriminated on ``kind``) ----------------------


class EpochDecision(_Frozen):
    """Propose one epoch to grind."""

    kind: Literal["epoch"]
    epoch: Epoch


class EndDecision(_Frozen):
    """End the run. ``summary`` is the phase handoff / pending-summary: the resume
    seed that lets the work continue as the next appendable run (a clean end, the
    planner's #2 disposition, or a satisfied done)."""

    kind: Literal["end"]
    summary: Annotated[str, StringConstraints(min_length=1, max_length=4096)]


Decision = Annotated[
    Union[EpochDecision, EndDecision], Field(discriminator="kind")
]

_DECISION_ADAPTER: TypeAdapter[Decision] = TypeAdapter(Decision)


def parse_decision(payload: object) -> Decision:
    """Parse untrusted JSON into the typed decision union (raises on invalid)."""

    return _DECISION_ADAPTER.validate_python(payload)


# --- handoff (the COPY bone, plus a BLOCKED self-report) -----------------------


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
    """Worker handoff written to disk; the disk file is the gate, not stdout.

    ``status`` adds ``BLOCKED`` to the bone: a worker that hits a hard environment
    blocker (missing dep, needs a host mutation it may not make) writes ``BLOCKED``
    so the orchestrator routes it STRAIGHT to the planner, skipping the critic (no
    point critiquing env-blocked work).
    """

    schema_version: Literal["1"]
    task_id: Annotated[
        str, StringConstraints(pattern=r"^P[1-9][0-9]?/E[1-9][0-9]?/T[1-8]$")
    ]
    status: Literal["DONE", "FAILED", "PARTIAL", "BLOCKED"]
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
    checks: Annotated[list[CheckResult], Field(max_length=8)]
    occupancy: Occupancy


_HANDOFF_ADAPTER: TypeAdapter[Handoff] = TypeAdapter(Handoff)


def parse_handoff(payload: object) -> Handoff:
    """Parse untrusted JSON into the typed handoff model (raises on invalid)."""

    return _HANDOFF_ADAPTER.validate_python(payload)


# --- verdict (the critic's triage, lenient by design) --------------------------

#: The critic ROUTES, it does not grade. PASS -> merge (notes carry forward);
#: RETRY -> the bounded same-worker retry (a defect the worker can plausibly fix);
#: ESCALATE -> the planner (anything the worker cannot fix: missing dep, ambiguous
#: spec, a decision, environmental).
VerdictOutcome = Literal["PASS", "RETRY", "ESCALATE"]


class Verdict(_Frozen):
    """The agentic critic's triage of one task. Lenient ON PURPOSE: an outcome enum
    plus free-text ``reason``, nothing a weak model can fumble into a schema-invalid
    rejection. ``reason`` defaults to empty so ``{"outcome": "PASS"}`` validates; it
    is unbounded because the verdict is delivered to the planner BY REFERENCE (read
    from its persisted file), never byte-capped into a prompt."""

    outcome: VerdictOutcome
    reason: str = ""


_VERDICT_ADAPTER: TypeAdapter[Verdict] = TypeAdapter(Verdict)


def parse_verdict(payload: object) -> Verdict:
    """Parse untrusted JSON into the typed critic verdict (raises on invalid)."""

    return _VERDICT_ADAPTER.validate_python(payload)
