"""Hermetic unit tests for the eval property oracle (NO model call, NOT eval-marked).

These pin the oracle helpers against SYNTHETIC typed decisions so the oracle is
trusted before any live eval leans on it: every band the corpus asserts is checked
here against a decision known to satisfy or violate it. Runs in the default suite.
"""

from __future__ import annotations

import pytest

from grindstone.contracts.models import EpochDecision, Handoff, parse_decision, parse_handoff
from tests.grindstone.conftest import (
    handoff_payload,
    implement_decision,
    impl_task,
    research_decision,
    artifact_task,
    two_phase_skeleton,
)
from tests.grindstone.eval import _assertions as A


def _decision(payload: dict[str, object]) -> EpochDecision:
    return parse_decision(payload)


def _handoff(**overrides: object) -> Handoff:
    return parse_handoff(handoff_payload(**overrides))  # type: ignore[arg-type]


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


# --- handoff oracles (the worker-boundary analogue) ----------------------------


def test_handoff_conforms_accepts_valid_done() -> None:
    h = _handoff(task_id="P1/E1/T1")
    A.assert_handoff_conforms(h, mode="implement", task_id="P1/E1/T1")
    assert A.handoff_gate_errors(h, mode="implement", task_id="P1/E1/T1") == []


def test_handoff_conforms_rejects_task_id_mismatch() -> None:
    h = _handoff(task_id="P1/E1/T1")
    errors = A.handoff_gate_errors(h, mode="implement", task_id="P1/E1/T2")
    assert any("dispatched id" in e for e in errors)
    with pytest.raises(AssertionError):
        A.assert_handoff_conforms(h, mode="implement", task_id="P1/E1/T2")


def test_handoff_conforms_enforces_research_citation_floor() -> None:
    # A research handoff with NO citations trips the production mode floor.
    h = _handoff(task_id="P1/E1/T1", citations=[])
    errors = A.handoff_gate_errors(h, mode="research", task_id="P1/E1/T1")
    assert any("citation" in e for e in errors)
    # The SAME handoff is fine under implement (no citation floor there).
    A.assert_handoff_conforms(h, mode="implement", task_id="P1/E1/T1")


def test_assert_handoff_status_matches_and_rejects() -> None:
    A.assert_handoff_status(_handoff(status="DONE"), "DONE")
    with pytest.raises(AssertionError):
        A.assert_handoff_status(_handoff(status="FAILED"), "DONE")


def test_assert_handoff_citations_present() -> None:
    A.assert_handoff_citations_present(_handoff(citations=[{"file": "a.py"}]))
    with pytest.raises(AssertionError):
        A.assert_handoff_citations_present(_handoff(citations=[]))


def test_assert_what_changed_shape_accepts_typed_entries() -> None:
    payload = handoff_payload(task_id="P1/E1/T1")
    payload["what_changed"] = [{"kind": "file", "ref": "greeting.txt"}]
    A.assert_what_changed_shape(parse_handoff(payload))


def test_assert_handoff_done_when_passed_pass_and_fail() -> None:
    ok = handoff_payload(task_id="P1/E1/T1")
    A.assert_handoff_done_when_passed(parse_handoff(ok))
    bad = handoff_payload(task_id="P1/E1/T1")
    bad["checks"] = [{"check": "test -f x", "exit_code": 1}]
    with pytest.raises(AssertionError):
        A.assert_handoff_done_when_passed(parse_handoff(bad))


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
