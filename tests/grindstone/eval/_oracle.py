"""Property-band oracle for a real planner decision (NOT a golden).

A real model's decision is never byte-stable, so the corpus asserts PROPERTIES the
bones contract requires, loose enough that a strong and a weak model both pass, tight
enough that an unconditionally-broken plan (a wildcard mega-task, two tasks fighting
over one file, a malformed mode) cannot. Schema-conformance is already guaranteed (a
``Decision`` reached the oracle only because ``ScriptPlanner.decide`` parsed + gated
it), so the oracle adds the SEMANTIC bands the planner core promises.
"""

from __future__ import annotations

from grindstone.contracts.models import (
    Decision,
    EndDecision,
    EpochDecision,
    HandoffMode,
)
from grindstone.worker import TaskResult

#: The wildcard glob characters an implement task's file_ownership must NOT contain
#: (the planner core: "Enumerate concrete files; never claim a whole subtree or a
#: wildcard you cannot bound"). Concrete ownership is what makes the disjoint-merge
#: invariant decidable.
_WILDCARD_CHARS = "*?["

_MODES: tuple[HandoffMode, ...] = ("implement", "research", "review", "artifact")


def assert_decision_well_formed(decision: Decision) -> None:
    """The decision is exactly one of the two bones shapes."""

    assert isinstance(decision, (EpochDecision, EndDecision)), (
        f"decision is neither an epoch nor an end: {type(decision).__name__}"
    )


def assert_epoch_obeys_core_rules(decision: Decision) -> None:
    """An EPOCH obeys the planner core's per-task rules (an END trivially passes).

    The bands: 1..8 tasks; every task a valid mode + tier; an implement task owns >= 1
    CONCRETE file (no wildcard) and a non-write task names an artifact_out; and the
    implement tasks' ownership is pairwise DISJOINT (the disjoint-merge invariant the
    state machine will enforce, so a conforming plan is dispatchable as-is)."""

    if isinstance(decision, EndDecision):
        return
    tasks = decision.epoch.tasks
    assert 1 <= len(tasks) <= 8, f"epoch has {len(tasks)} tasks, outside 1..8"

    owned: dict[str, str] = {}
    for task in tasks:
        assert task.mode in _MODES, f"task {task.id}: bad mode {task.mode!r}"
        assert task.tier in ("local", "senior"), f"task {task.id}: bad tier {task.tier!r}"
        if task.mode == "implement":
            assert task.file_ownership, f"task {task.id}: implement owns no files"
            for glob in task.file_ownership:
                assert not any(c in glob for c in _WILDCARD_CHARS), (
                    f"task {task.id}: wildcard ownership {glob!r} (must enumerate "
                    "concrete files)"
                )
                prior = owned.get(glob)
                assert prior is None, (
                    f"ownership overlap: {glob!r} claimed by {prior} and {task.id}"
                )
                owned[glob] = task.id
        else:
            assert task.artifact_out is not None, (
                f"task {task.id}: {task.mode} task names no artifact_out"
            )


def assert_decision_conforms(decision: Decision) -> None:
    """The full boundary band: well-formed shape + the epoch core rules."""

    assert_decision_well_formed(decision)
    assert_epoch_obeys_core_rules(decision)


# --- worker task bands ----------------------------------------------------------


def assert_task_passed_with_verdict(result: TaskResult) -> None:
    """A tiny real worker task produced a gate-clean, critic-judged PASS.

    The capability band: a trivial task (write one file, write one findings note) on
    the local floor must PASS, which means its handoff VALIDATED (the disk-gate +
    Pydantic parse ran inside ``run_task``, so a non-None DONE handoff is a gate pass)
    and the independent CRITIC returned a verdict (the agentic judge actually ran and
    routed to PASS). A blocked / escalated tiny task is a real capability failure the
    corpus surfaces, with the handoff reason for the post-mortem."""

    reason = result.reason or (result.handoff.resulting_state if result.handoff else "")
    assert result.outcome == "passed", (
        f"a trivial task did not pass: {result.outcome!r} ({reason})"
    )
    assert result.handoff is not None and result.handoff.status == "DONE", (
        "no gate-clean DONE handoff was collected"
    )
    assert result.verdict is not None, "the critic returned no verdict"
    assert result.verdict.outcome == "PASS", (
        f"a trivial passed task drew a non-PASS verdict: {result.verdict.outcome}"
    )
