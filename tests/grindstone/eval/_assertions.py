"""Property oracle for the planner eval corpus (pure, hermetically unit-tested).

The eval tests drive a REAL model through a rig and then assert PROPERTIES of the
decision it produced: that it conforms to the gate, picks a legal tool, decomposes
within the size cap, lands a sane phase count. The bands are deliberately loose
(behavioral, not golden): a strong model and a weak one can both pass while an
unconditionally-broken plan cannot.

These helpers are PURE over a typed ``EpochDecision`` and reuse the production
gate (``validate_decision``) + oracle (``contracts.semantics``) wherever possible,
so the oracle never drifts from the gate the planner actually faces. They are unit
-tested against SYNTHETIC decisions in ``test_assertions.py`` (NOT eval-marked), so
the oracle itself is trusted before any live run leans on it.
"""

from __future__ import annotations

import json

from grindstone.contracts.gate import handoff_schema_errors
from grindstone.contracts.models import (
    EpochDecision,
    Handoff,
    ImplementDecision,
    ProposeSkeletonDecision,
    RevisePhasesDecision,
)
from grindstone.contracts.semantics import (
    HandoffMode,
    handoff_violations,
    implement_task_size_violations,
)
from grindstone.planner import (
    DEFAULT_LOCAL_MAX_TASK_FILES,
    DEFAULT_SENIOR_MAX_TASK_FILES,
    GateResult,
    select_planner_scenario,
    validate_decision,
)

#: The work tools a steady-state (mid-run) boundary may legally pick.
WORK_TOOLS: frozenset[str] = frozenset({"implement", "research", "review", "artifact"})


def decision_to_json(decision: EpochDecision) -> str:
    """The decision's canonical JSON envelope (what the gate re-parses)."""

    return json.dumps(decision.model_dump(mode="json"))


def conforms(
    decision: EpochDecision,
    *,
    existing_log_keys: frozenset[str] = frozenset(),
    completed_phase_ids: frozenset[str] = frozenset(),
    skeleton_exists: bool = True,
    phase_escalated: bool = False,
    failed_epoch_active: bool = False,
    has_senior: bool = False,
    local_max_task_files: int = DEFAULT_LOCAL_MAX_TASK_FILES,
    senior_max_task_files: int = DEFAULT_SENIOR_MAX_TASK_FILES,
) -> GateResult:
    """Re-run the production gate over a typed decision (defense in depth).

    The decision is already schema- and type-valid (it parsed), but the SEMANTIC
    gate (size, disjointness, content-grep, position legality) still applies. This
    re-serializes and feeds it through the exact ``validate_decision`` the run loop
    uses, so an eval assertion sees the same verdict the planner would.
    """

    return validate_decision(
        decision_to_json(decision),
        existing_log_keys=existing_log_keys,
        completed_phase_ids=completed_phase_ids,
        skeleton_exists=skeleton_exists,
        phase_escalated=phase_escalated,
        failed_epoch_active=failed_epoch_active,
        has_senior=has_senior,
        local_max_task_files=local_max_task_files,
        senior_max_task_files=senior_max_task_files,
    )


def assert_conforms(decision: EpochDecision, **gate_kwargs: object) -> None:
    """Assert the decision passes the production gate under ``gate_kwargs``."""

    result = conforms(decision, **gate_kwargs)  # type: ignore[arg-type]
    assert result.decision is not None, (
        f"decision did not conform to the gate: {result.errors}"
    )


def assert_tool(decision: EpochDecision, expected: str) -> None:
    """Assert the decision picked exactly ``expected`` as its tool."""

    assert decision.tool == expected, (
        f"expected tool {expected!r}, got {decision.tool!r}"
    )


def assert_tool_in(decision: EpochDecision, expected: frozenset[str]) -> None:
    """Assert the decision's tool is one of ``expected`` (a legal-set band)."""

    assert decision.tool in expected, (
        f"tool {decision.tool!r} not in the legal set {sorted(expected)}"
    )


def phase_count(decision: EpochDecision) -> int:
    """The number of phases in a skeleton / revise_phases decision.

    Raises ``TypeError`` for any other tool: phase_count is meaningless off a
    phase-structure decision, and a silent 0 would mask a wrong-tool answer.
    """

    if isinstance(decision, (ProposeSkeletonDecision, RevisePhasesDecision)):
        return len(decision.args.phases)
    raise TypeError(f"phase_count is only defined for phase decisions, got {decision.tool!r}")


def assert_phase_count_between(decision: EpochDecision, lo: int, hi: int) -> None:
    """Assert the skeleton's phase count is in the inclusive band ``[lo, hi]``."""

    n = phase_count(decision)
    assert lo <= n <= hi, f"phase count {n} not in band [{lo}, {hi}]"


def task_count(decision: EpochDecision) -> int:
    """The number of fan-out tasks in a work (implement/research/review/artifact)
    epoch. Raises ``TypeError`` off a non-task decision."""

    args = decision.args
    tasks = getattr(args, "tasks", None)
    if tasks is None:
        raise TypeError(f"task_count is only defined for work epochs, got {decision.tool!r}")
    return len(tasks)


def assert_every_implement_task_within(decision: EpochDecision, max_files: int) -> None:
    """Assert every task of an implement epoch is within the file-count cap.

    Reuses the production size oracle (``implement_task_size_violations``), so the
    band is exactly the gate's: a task over ``max_files`` globs, or one claiming
    whole-repo ownership, fails. A non-implement decision passes vacuously (the
    cap is implement-only)."""

    if not isinstance(decision, ImplementDecision):
        return
    violations = implement_task_size_violations(
        list(decision.args.tasks), max_files=max_files
    )
    assert not violations, f"implement task(s) over the size cap: {violations}"


# --- handoff oracles (the worker-boundary analogue of the decision oracles) ----
#
# These are PURE over a typed ``Handoff`` and reuse the PRODUCTION handoff gate
# (``handoff_schema_errors`` + ``handoff_violations``, the exact pair task_loop's
# ``_collect_handoff`` applies) so an eval assertion sees the same verdict the
# worker would. They are unit-tested against SYNTHETIC handoffs in
# ``test_assertions.py`` (NOT eval-marked), so the oracle is trusted before any
# live worker run leans on it.


def handoff_gate_errors(
    handoff: Handoff, *, mode: HandoffMode, task_id: str
) -> list[str]:
    """Re-run the production handoff gate (schema + semantics) over a typed handoff.

    The handoff is already type-valid (it parsed), but the gate still applies: the
    wire-schema structural rules (``handoff_schema_errors`` over the JSON envelope)
    plus the SEMANTIC rules (``handoff_violations``: byte cap, mode citation floor,
    task_id match). Reuses both production functions, so the band is exactly the
    gate's, never a re-implementation that could drift.
    """

    # exclude_none mirrors the CANONICAL wire form the worker writes (and the
    # production gate / check_handoff validate): the schema's integer fields reject
    # an explicit null, so a dumped-with-None optional (citation.line,
    # occupancy.peak_context_tokens) would spuriously fail the schema.
    payload = handoff.model_dump(mode="json", exclude_none=True)
    errors = list(handoff_schema_errors(payload))
    errors.extend(handoff_violations(handoff, mode=mode, expected_task_id=task_id))
    return errors


def assert_handoff_conforms(
    handoff: Handoff, *, mode: HandoffMode, task_id: str
) -> None:
    """Assert the handoff passes the production gate for ``mode`` + ``task_id``."""

    errors = handoff_gate_errors(handoff, mode=mode, task_id=task_id)
    assert not errors, f"handoff did not conform to the gate: {errors}"


def assert_handoff_status(handoff: Handoff, expected: str = "DONE") -> None:
    """Assert the handoff's terminal status is exactly ``expected``."""

    assert handoff.status == expected, (
        f"expected handoff status {expected!r}, got {handoff.status!r} "
        f"(not_done={handoff.not_done}, state={handoff.resulting_state!r})"
    )


def assert_handoff_citations_present(handoff: Handoff) -> None:
    """Assert the handoff grounds itself in >= 1 citation.

    The mode citation FLOOR (research/review require one) is enforced by the gate;
    this is the stricter band the worker corpus asserts directly: an honest handoff
    that did real work cites the files it touched/read, whatever the mode."""

    assert handoff.citations, "handoff carries no citations"


def assert_what_changed_shape(handoff: Handoff) -> None:
    """Assert every ``what_changed`` entry is a typed change with a non-empty ref.

    A typed ``Handoff`` already guarantees the ``kind`` enum + ref min-length, so
    this is defense in depth that pins the corpus' expectation explicitly: each
    entry names a real kind (file/interface/artifact) and a non-empty ref, never a
    free-prose string (the S2 RCA the schema now forbids)."""

    for entry in handoff.what_changed:
        assert entry.kind in {"file", "interface", "artifact"}, (
            f"what_changed kind {entry.kind!r} not a valid change kind"
        )
        assert entry.ref.strip(), "what_changed entry has an empty ref"


def assert_handoff_done_when_passed(handoff: Handoff) -> None:
    """Assert the handoff's own ``checks`` echo every done_when as passing (exit 0).

    The worker echoes each done_when result; a DONE handoff that echoes a non-zero
    exit is internally inconsistent. (The harness ALSO re-runs the real done_when
    out-of-band, the authoritative gate; this oracle pins the worker's self-report.)
    """

    failed = [c.check for c in handoff.checks if c.exit_code != 0]
    assert not failed, f"handoff echoes failing done_when checks: {failed}"


def assert_scenario_selected(
    *, skeleton_exists: bool, failed_epoch_active: bool, expected: str
) -> None:
    """Assert the state-machine scenario selector resolves to ``expected``.

    A pure check on ``select_planner_scenario`` (no model call): it pins that the
    durable signals a boundary carries route to the scenario skill the gate then
    enforces, so the corpus' scenario coverage is grounded in the same selector
    the run loop uses."""

    got = select_planner_scenario(
        skeleton_exists=skeleton_exists, failed_epoch_active=failed_epoch_active
    )
    assert got == expected, (
        f"scenario selector gave {got!r}, expected {expected!r} "
        f"(skeleton_exists={skeleton_exists}, failed_epoch_active={failed_epoch_active})"
    )
