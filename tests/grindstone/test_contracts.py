"""Invariant tests for the lenient wire contracts (decision / verdict).

Stochastic-first (BONES): a handful of invariant tests, not hundreds. These pin
the contract shape every later part imports, plus the failure-routing entry point
(verdict ESCALATE) the failure model depends on, and guard the Pydantic models
against drift from the JSON Schemas the planner self-validates on. The worker's
``handoff.md`` is deliberately NOT a wire contract (free-form prose, never parsed),
so there is nothing to test here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from grindstone.contracts.models import (
    EndDecision,
    EpochDecision,
    Verdict,
    parse_decision,
    parse_verdict,
)

_SCHEMA_DIR = Path(__file__).resolve().parents[2] / "schemas"


def _decision_validator() -> Draft202012Validator:
    schema = json.loads((_SCHEMA_DIR / "epoch_decision.json").read_text())
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _verdict_validator() -> Draft202012Validator:
    schema = json.loads((_SCHEMA_DIR / "verdict.json").read_text())
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


# --- decision: the two shapes --------------------------------------------------


def _epoch_payload() -> dict[str, object]:
    return {
        "kind": "epoch",
        "epoch": {
            "title": "build the thing",
            "rationale": "first slice",
            "setup": ["npm ci"],
            "tasks": [
                {
                    "id": "T1",
                    "mode": "implement",
                    "goal": "write the cli",
                    "tier": "local",
                    "file_ownership": ["src/cli.py"],
                    "skills": ["py-style"],
                },
                {
                    "id": "T2",
                    "mode": "research",
                    "goal": "survey the api",
                    "tier": "senior",
                    "artifact_out": "E1/T2/report.md",
                },
            ],
        },
    }


def test_epoch_decision_parses_and_is_frozen() -> None:
    decision = parse_decision(_epoch_payload())
    assert isinstance(decision, EpochDecision)
    assert decision.epoch.tasks[0].mode == "implement"
    assert decision.epoch.tasks[1].tier == "senior"
    assert decision.epoch.setup == ["npm ci"]
    with pytest.raises(ValidationError):
        decision.epoch.tasks[0].id = "T9"  # frozen: mutation is rejected


def test_end_decision_carries_the_resume_summary() -> None:
    decision = parse_decision({"kind": "end", "summary": "did A and B; C is pending"})
    assert isinstance(decision, EndDecision)
    assert "pending" in decision.summary


def test_implement_task_requires_file_ownership() -> None:
    payload = _epoch_payload()
    payload["epoch"]["tasks"][0].pop("file_ownership")  # type: ignore[index]
    with pytest.raises(ValidationError):
        parse_decision(payload)


def test_artifact_task_requires_artifact_out() -> None:
    payload = _epoch_payload()
    payload["epoch"]["tasks"][1].pop("artifact_out")  # type: ignore[index]
    with pytest.raises(ValidationError):
        parse_decision(payload)


def test_implement_task_may_not_carry_artifact_out() -> None:
    payload = _epoch_payload()
    payload["epoch"]["tasks"][0]["artifact_out"] = "x/y.md"  # type: ignore[index]
    with pytest.raises(ValidationError):
        parse_decision(payload)


def test_unknown_key_is_rejected() -> None:
    payload = _epoch_payload()
    payload["epoch"]["surprise"] = 1  # type: ignore[index]
    with pytest.raises(ValidationError):
        parse_decision(payload)


def test_bad_discriminator_is_rejected() -> None:
    with pytest.raises(ValidationError):
        parse_decision({"kind": "nope", "summary": "x"})


def test_epoch_rejects_colliding_artifact_out() -> None:
    """The artifact analogue of disjoint-merge: two non-write tasks may not declare
    the SAME artifact_out, else concurrent publish races silently clobber the loser
    and both return passed. The epoch validator rejects the collision so the planner
    re-emits (it self-validates decision.json through parse_decision)."""

    payload: dict[str, object] = {
        "kind": "epoch",
        "epoch": {
            "title": "two reports",
            "tasks": [
                {"id": "T1", "mode": "research", "goal": "a", "artifact_out": "E1/out.md"},
                {"id": "T2", "mode": "review", "goal": "b", "artifact_out": "E1/out.md"},
            ],
        },
    }
    with pytest.raises(ValidationError):
        parse_decision(payload)


def test_epoch_allows_distinct_artifact_out() -> None:
    payload: dict[str, object] = {
        "kind": "epoch",
        "epoch": {
            "title": "two reports",
            "tasks": [
                {"id": "T1", "mode": "research", "goal": "a", "artifact_out": "E1/a.md"},
                {"id": "T2", "mode": "review", "goal": "b", "artifact_out": "E1/b.md"},
            ],
        },
    }
    decision = parse_decision(payload)
    assert isinstance(decision, EpochDecision)


def test_decision_schema_matches_pydantic() -> None:
    """A payload the Pydantic model accepts must also pass the wire schema, and the
    empty-epoch invalid payload must fail BOTH layers (no model/schema drift)."""

    validator = _decision_validator()
    good = _epoch_payload()
    assert list(validator.iter_errors(good)) == []
    parse_decision(good)

    bad = {"kind": "epoch", "epoch": {}}
    assert list(validator.iter_errors(bad)) != []
    with pytest.raises(ValidationError):
        parse_decision(bad)


# --- verdict: lenient triage ---------------------------------------------------


@pytest.mark.parametrize("outcome", ["PASS", "RETRY", "ESCALATE"])
def test_verdict_outcomes_parse(outcome: str) -> None:
    verdict = parse_verdict({"outcome": outcome, "reason": "because"})
    assert isinstance(verdict, Verdict)
    assert verdict.outcome == outcome
    assert verdict.reason == "because"


def test_verdict_reason_is_optional() -> None:
    """Lenient: a bare {"outcome": "PASS"} must validate (the weak-model fix)."""

    verdict = parse_verdict({"outcome": "PASS"})
    assert verdict.reason == ""


def test_verdict_unknown_outcome_rejected() -> None:
    with pytest.raises(ValidationError):
        parse_verdict({"outcome": "MAYBE"})


def test_verdict_extra_field_rejected() -> None:
    with pytest.raises(ValidationError):
        parse_verdict({"outcome": "PASS", "grade": 7})


def test_verdict_schema_matches_pydantic() -> None:
    validator = _verdict_validator()
    good = {"outcome": "ESCALATE", "reason": "missing dep"}
    assert list(validator.iter_errors(good)) == []
    parse_verdict(good)
    assert list(validator.iter_errors({"outcome": "MAYBE"})) != []
