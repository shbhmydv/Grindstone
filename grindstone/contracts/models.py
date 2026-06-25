"""Pydantic v2 wire contracts for the bones rewrite.

Two lenient structs the core works with, stringly JSON is parsed here exactly
once, at the boundary, and never crosses inward:

* ``Decision`` -- what the planner emits each boundary: an EPOCH (1..N tasks) or
  an END (a phase handoff / pending-summary that seeds the next appendable run).
* ``Verdict`` -- the critic's triage: ``PASS`` | ``RETRY`` | ``ESCALATE`` plus a
  short free-text reason. Deliberately NOT a rigid multi-field schema, that shape
  is what a weak model fumbled (run 051645Z), rejecting work for a machinery fault.

The worker's ``handoff.md`` is deliberately NOT modelled here: it is a FREE-FORM
prose report the worker writes for the critic, never a wire contract the state
machine parses or schema-gates (BONES: the machine disposes on DETERMINISTIC facts
-- the git diff + file existence -- plus the critic's lenient verdict, nothing
else). It is relocated to the keyed log verbatim and handed to the critic as text.

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

#: The four worker intents. Picks the worker prompt/skill and the critic's
#: grounding bar (research/review must cite real files). Carried on each task.
HandoffMode = Literal["implement", "research", "review", "artifact"]


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
    """Propose one epoch to grind.

    ``pending`` is the planner's ADDITIONS to the persisted cross-epoch work backlog:
    short prose lines of deferred FUTURE work (e.g. "refine the Welcome screen to taste
    (senior), after its scaffold lands"). The close-out (the sole baton writer) folds
    them into the baton's ``## Pending`` section so a multi-epoch plan survives the
    boundary without being re-derived; the additions MAY exceed the tasks scheduled this
    epoch. Optional + bounded like the sibling list fields; empty by default.
    """

    kind: Literal["epoch"]
    epoch: Epoch
    pending: Annotated[
        list[Annotated[str, StringConstraints(min_length=1, max_length=512)]],
        Field(max_length=16),
    ] = Field(default_factory=list)


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
