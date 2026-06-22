"""Hermetic unit tests for the eval property oracle (NO model call, NOT eval-marked).

These pin the oracle helpers against SYNTHETIC typed decisions so the oracle is
trusted before any live eval leans on it: every band the corpus asserts is checked
here against a decision known to satisfy or violate it. Runs in the default suite.
"""

from __future__ import annotations

import pytest

from grindstone.contracts.models import EpochDecision, parse_decision
from tests.grindstone.conftest import (
    implement_decision,
    impl_task,
    research_decision,
    artifact_task,
    two_phase_skeleton,
)
from tests.grindstone.eval import _assertions as A


def _decision(payload: dict[str, object]) -> EpochDecision:
    return parse_decision(payload)


def _impl_task(tid: str, *globs: str) -> dict[str, object]:
    return {
        "id": tid,
        "goal": f"build {tid}",
        "done_when": [{"cmd": "true"}],
        "file_ownership": list(globs),
    }


# --- assert_conforms / conforms ------------------------------------------------


def test_conforms_accepts_valid_implement() -> None:
    decision = _decision(implement_decision(impl_task("T1", "src/a.py")))
    A.assert_conforms(decision, skeleton_exists=True)


def test_conforms_rejects_oversized_implement() -> None:
    # Seven owned globs blows the local cap of 5: the production size gate fails it.
    big = _decision(
        implement_decision(_impl_task("T1", *[f"src/m{i}.py" for i in range(7)]))
    )
    result = A.conforms(big, skeleton_exists=True, local_max_task_files=5)
    assert result.decision is None
    assert any("too big" in e or "file_ownership" in e for e in result.errors)
    with pytest.raises(AssertionError):
        A.assert_conforms(big, skeleton_exists=True, local_max_task_files=5)


# --- assert_tool / assert_tool_in ----------------------------------------------


def test_assert_tool_matches_and_rejects() -> None:
    decision = _decision(two_phase_skeleton())
    A.assert_tool(decision, "propose_skeleton")
    with pytest.raises(AssertionError):
        A.assert_tool(decision, "implement")


def test_assert_tool_in_band() -> None:
    decision = _decision(research_decision(artifact_task("T1")))
    A.assert_tool_in(decision, A.WORK_TOOLS)
    with pytest.raises(AssertionError):
        A.assert_tool_in(decision, frozenset({"implement"}))


# --- phase_count / assert_phase_count_between ----------------------------------


def test_phase_count_and_band() -> None:
    decision = _decision(two_phase_skeleton())
    assert A.phase_count(decision) == 2
    A.assert_phase_count_between(decision, 2, 6)
    with pytest.raises(AssertionError):
        A.assert_phase_count_between(decision, 3, 6)


def test_phase_count_off_non_phase_decision_raises() -> None:
    decision = _decision(implement_decision(impl_task("T1", "a.py")))
    with pytest.raises(TypeError):
        A.phase_count(decision)


# --- task_count ----------------------------------------------------------------


def test_task_count_on_work_epoch_and_off_skeleton() -> None:
    work = _decision(
        implement_decision(impl_task("T1", "a.py"), impl_task("T2", "b.py"))
    )
    assert A.task_count(work) == 2
    with pytest.raises(TypeError):
        A.task_count(_decision(two_phase_skeleton()))


# --- assert_every_implement_task_within ----------------------------------------


def test_every_implement_task_within_pass_and_fail() -> None:
    ok = _decision(implement_decision(impl_task("T1", "a.py")))
    A.assert_every_implement_task_within(ok, max_files=5)
    big = _decision(
        implement_decision(_impl_task("T1", *[f"src/m{i}.py" for i in range(7)]))
    )
    with pytest.raises(AssertionError):
        A.assert_every_implement_task_within(big, max_files=5)


def test_every_implement_task_within_vacuous_off_implement() -> None:
    # A non-implement decision has no file cap: the assertion passes vacuously.
    A.assert_every_implement_task_within(
        _decision(research_decision(artifact_task("T1"))), max_files=1
    )


# --- assert_scenario_selected --------------------------------------------------


def test_scenario_selector_covers_three_states() -> None:
    A.assert_scenario_selected(
        skeleton_exists=False, failed_epoch_active=False, expected="plan_skeleton"
    )
    A.assert_scenario_selected(
        skeleton_exists=True, failed_epoch_active=False, expected="plan_epoch"
    )
    A.assert_scenario_selected(
        skeleton_exists=True, failed_epoch_active=True, expected="repair_epoch"
    )
    with pytest.raises(AssertionError):
        A.assert_scenario_selected(
            skeleton_exists=False, failed_epoch_active=False, expected="plan_epoch"
        )
