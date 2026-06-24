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


def _is_test_path(path: str) -> bool:
    """A heuristic for "this file is a test, not the code under test": a ``tests``
    path segment, or a filename in the ``test_*`` / ``*_test`` convention."""

    segments = path.split("/")
    name = segments[-1]
    stem = name.rsplit(".", 1)[0]
    return "tests" in segments or name.startswith("test_") or stem.endswith("_test")


def assert_no_fresh_test_code_split(decision: Decision) -> None:
    """On a FRESH boundary, the planner must not fan a pure-test task out alongside a
    task that owns the source it covers.

    Sibling tasks grind in isolated worktrees off the same base and cannot see each
    other's output, so testing code that another same-epoch task is still writing is
    the producer/consumer dependency the core forbids (a dependency means a LATER
    epoch). Scoped to the fresh boundary, where nothing is merged yet, so any
    test-task + source-task cofanout is necessarily a violation. Bundling tests AND
    their source into ONE task is allowed (that task is not pure-test), as is a
    source-only first epoch; both pass. An END trivially passes."""

    if isinstance(decision, EndDecision):
        return
    impl = [t for t in decision.epoch.tasks if t.mode == "implement"]
    pure_test = [t for t in impl if t.file_ownership and all(map(_is_test_path, t.file_ownership))]
    owns_source = [t for t in impl if any(not _is_test_path(f) for f in t.file_ownership)]
    collisions = [
        (tt.id, ss.id) for tt in pure_test for ss in owns_source if tt.id != ss.id
    ]
    assert not collisions, (
        f"fresh epoch fans a test task out alongside the code it covers {collisions}; "
        "tests depend on the source and belong in a LATER epoch, not a sibling task"
    )


# --- worker task bands ----------------------------------------------------------


def assert_task_passed_with_verdict(result: TaskResult) -> None:
    """A tiny real worker task produced a gate-clean, critic-judged PASS.

    The capability band: a trivial task (write one file, write one findings note) on
    the local floor must PASS, which means it cleared the DETERMINISTIC gate (a
    non-empty in-scope commit, or a present artifact, checked inside ``run_task``) and
    the independent CRITIC returned a PASS verdict (the agentic judge actually ran and
    routed). An escalated tiny task is a real capability failure the corpus surfaces,
    with the verdict reason for the post-mortem."""

    assert result.outcome == "passed", (
        f"a trivial task did not pass: {result.outcome!r} ({result.reason})"
    )
    assert result.verdict is not None, "the critic returned no verdict"
    assert result.verdict.outcome == "PASS", (
        f"a trivial passed task drew a non-PASS verdict: {result.verdict.outcome}"
    )
