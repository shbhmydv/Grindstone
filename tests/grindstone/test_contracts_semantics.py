"""Semantic validators: the rules JSON Schema cannot express."""

from __future__ import annotations

import json
from pathlib import Path

from grindstone.contracts import (
    HANDOFF_MAX_BYTES,
    canonical_bytes,
    epoch_decision_violations,
    handoff_violations,
    parse_decision,
)
from grindstone.contracts.models import Handoff

from tests.grindstone.conftest import phase_dict, revise_decision, skeleton_decision

CORPUS = Path(__file__).parent / "corpus"
EMPTY: frozenset[str] = frozenset()


def _decision(name: str):
    return parse_decision(json.loads((CORPUS / "decision/valid" / name).read_text()))


# --- decision semantics --------------------------------------------------------


def test_inputs_must_exist_in_keyed_log() -> None:
    decision = _decision("implement_unknown_input.json")
    bad = epoch_decision_violations(
        decision, existing_log_keys=EMPTY, completed_phase_ids=EMPTY
    )
    assert any("ghost.md" in v for v in bad)
    good = epoch_decision_violations(
        decision,
        existing_log_keys=frozenset({"p9/e9/t9/ghost.md"}),
        completed_phase_ids=EMPTY,
    )
    assert good == []


def test_overlapping_file_ownership_rejected() -> None:
    decision = _decision("implement_overlapping_ownership.json")
    bad = epoch_decision_violations(
        decision, existing_log_keys=EMPTY, completed_phase_ids=EMPTY
    )
    assert any("overlap" in v for v in bad)


def test_disjoint_file_ownership_accepted() -> None:
    decision = _decision("implement.json")
    out = epoch_decision_violations(
        decision,
        existing_log_keys=frozenset({"p1/e1/spec.md"}),
        completed_phase_ids=EMPTY,
    )
    assert out == []


def test_duplicate_task_ids_rejected() -> None:
    decision = _decision("implement_duplicate_ids.json")
    bad = epoch_decision_violations(
        decision, existing_log_keys=EMPTY, completed_phase_ids=EMPTY
    )
    assert any("duplicate task id" in v for v in bad)


def test_duplicate_phase_ids_rejected() -> None:
    # Two phases sharing an id would wedge phase advancement in an unbounded hang
    # (_phase_index resolves an id to its FIRST occurrence), reject at the gate.
    dup = parse_decision(skeleton_decision(phase_dict("P1", title="a"), phase_dict("P1", title="b")))
    bad = epoch_decision_violations(dup, existing_log_keys=EMPTY, completed_phase_ids=EMPTY)
    assert any("duplicate phase id" in v for v in bad)
    # revise_phases with an internal duplicate is likewise rejected.
    dup_rev = parse_decision(revise_decision(phase_dict("P2", title="a"), phase_dict("P2", title="b")))
    bad_rev = epoch_decision_violations(dup_rev, existing_log_keys=EMPTY, completed_phase_ids=EMPTY)
    assert any("duplicate phase id" in v for v in bad_rev)
    # Unique phase ids pass clean.
    ok = parse_decision(skeleton_decision(phase_dict("P1"), phase_dict("P2")))
    assert epoch_decision_violations(ok, existing_log_keys=EMPTY, completed_phase_ids=EMPTY) == []


def test_review_requires_targets() -> None:
    missing = _decision("review_missing_targets.json")
    bad = epoch_decision_violations(
        missing, existing_log_keys=EMPTY, completed_phase_ids=EMPTY
    )
    assert any("targets" in v for v in bad)
    present = _decision("review.json")
    assert (
        epoch_decision_violations(
            present, existing_log_keys=EMPTY, completed_phase_ids=EMPTY
        )
        == []
    )


def test_revise_phases_may_not_reuse_completed_id() -> None:
    decision = _decision("revise_reuse_completed.json")
    bad = epoch_decision_violations(
        decision, existing_log_keys=EMPTY, completed_phase_ids=frozenset({"P1"})
    )
    assert any("completed phase id" in v for v in bad)
    ok = epoch_decision_violations(
        decision, existing_log_keys=EMPTY, completed_phase_ids=EMPTY
    )
    assert ok == []


# --- handoff semantics ---------------------------------------------------------


def _handoff_of_canonical_size(size: int) -> Handoff:
    base = {
        "schema_version": "1",
        "task_id": "P1/E1/T1",
        "status": "DONE",
        "what_changed": [{"kind": "file", "ref": "x" * 256} for _ in range(16)],
        "resulting_state": "x",
        "downstream_needs": [],
        "not_done": ["y" * 256 for _ in range(8)],
        "citations": [],
        "checks": [],
        "occupancy": {"compacted": False, "subagent_splits": 0},
    }
    for length in range(1, 1501):
        base["resulting_state"] = "a" * length
        handoff = Handoff.model_validate(base)
        if canonical_bytes(handoff) == size:
            return handoff
    raise AssertionError(f"could not build a handoff of exactly {size} bytes")


def test_handoff_at_cap_passes_over_cap_fails() -> None:
    at_cap = _handoff_of_canonical_size(HANDOFF_MAX_BYTES)
    assert canonical_bytes(at_cap) == 8192
    assert (
        handoff_violations(at_cap, mode="implement", expected_task_id="P1/E1/T1") == []
    )
    over_cap = _handoff_of_canonical_size(HANDOFF_MAX_BYTES + 1)
    assert canonical_bytes(over_cap) == 8193
    bad = handoff_violations(over_cap, mode="implement", expected_task_id="P1/E1/T1")
    assert any("exceeds" in v for v in bad)


def _load_handoff(name: str) -> Handoff:
    return Handoff.model_validate(
        json.loads((CORPUS / "handoff/valid" / name).read_text())
    )


def test_research_handoff_requires_citation() -> None:
    no_cite = _load_handoff("minimal_done.json")
    bad = handoff_violations(no_cite, mode="research", expected_task_id="P1/E1/T1")
    assert any("citation" in v for v in bad)
    # Same handoff under implement mode has no citation requirement.
    assert (
        handoff_violations(no_cite, mode="implement", expected_task_id="P1/E1/T1") == []
    )
    cited = _load_handoff("research_with_citation.json")
    assert (
        handoff_violations(cited, mode="research", expected_task_id="P1/E1/T1") == []
    )


def test_task_id_must_match_dispatched_id() -> None:
    handoff = _load_handoff("minimal_done.json")
    bad = handoff_violations(handoff, mode="implement", expected_task_id="P1/E1/T2")
    assert any("dispatched id" in v for v in bad)
