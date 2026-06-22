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

from grindstone.contracts.models import (
    EpochDecision,
    ImplementDecision,
    ProposeSkeletonDecision,
    RevisePhasesDecision,
)
from grindstone.contracts.semantics import implement_task_size_violations
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
