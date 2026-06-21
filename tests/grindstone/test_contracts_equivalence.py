"""Equivalence guard: the JSON-Schema gate and the Pydantic models must agree
on accept/reject for every corpus fixture. A disagreement is structural drift
between the schema (source of truth) and the hand-written models.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from grindstone.contracts import (
    decision_schema_errors,
    handoff_schema_errors,
    parse_decision,
    parse_handoff,
)
from grindstone.contracts.models import CriterionJudgement, EpochVerdict

CORPUS = Path(__file__).parent / "corpus"
SCHEMAS = Path(__file__).resolve().parents[2] / "schemas"


def _cases(kind: str) -> list:
    out = []
    for validity in ("valid", "invalid"):
        for path in sorted((CORPUS / kind / validity).glob("*.json")):
            out.append(pytest.param(path, validity == "valid", id=f"{validity}/{path.stem}"))
    return out


@pytest.mark.parametrize("path,expect_valid", _cases("decision"))
def test_decision_layers_agree(path: Path, expect_valid: bool) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    schema_ok = not decision_schema_errors(payload)
    try:
        parse_decision(payload)
        pydantic_ok = True
    except ValidationError:
        pydantic_ok = False
    assert schema_ok == pydantic_ok, "schema gate and typed models disagree"
    assert schema_ok == expect_valid


@pytest.mark.parametrize("path,expect_valid", _cases("handoff"))
def test_handoff_layers_agree(path: Path, expect_valid: bool) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    schema_ok = not handoff_schema_errors(payload)
    try:
        parse_handoff(payload)
        pydantic_ok = True
    except ValidationError:
        pydantic_ok = False
    assert schema_ok == pydantic_ok, "schema gate and typed models disagree"
    assert schema_ok == expect_valid


def test_corpus_is_non_trivial() -> None:
    # Guard against an empty/half-populated corpus silently passing the suite.
    assert len(_cases("decision")) >= 12
    assert len(_cases("handoff")) >= 6


def test_parsed_decision_is_frozen() -> None:
    payload = json.loads((CORPUS / "decision/valid/implement.json").read_text())
    decision = parse_decision(payload)
    with pytest.raises(ValidationError):
        decision.tool = "research"  # type: ignore[misc]


@pytest.mark.parametrize("fixture", ["implement", "review"])
def test_visual_flag_backward_compatible_and_typed(fixture: str) -> None:
    """The taste-routing `visual` flag: optional (backward compat), bool-typed,
    and agreed-on by both validation layers."""

    base = json.loads((CORPUS / f"decision/valid/{fixture}.json").read_text())

    # Backward compat: a decision WITHOUT `visual` validates at both layers and
    # the parsed model defaults the flag to False.
    assert not decision_schema_errors(base)
    assert parse_decision(base).args.visual is False

    # Explicit visual:true is accepted by both layers and round-trips True.
    yes = {**base, "args": {**base["args"], "visual": True}}
    assert not decision_schema_errors(yes)
    assert parse_decision(yes).args.visual is True

    # A non-bool `visual` is rejected by BOTH layers (schema boolean / StrictBool).
    bad = {**base, "args": {**base["args"], "visual": "yes"}}
    assert decision_schema_errors(bad)
    with pytest.raises(ValidationError):
        parse_decision(bad)


def test_task_criteria_optional_and_round_trips() -> None:
    """The semantic-acceptance `criteria` field on a task: optional (backward
    compat, defaults to an empty list), a list of non-empty strings, and agreed
    on by both validation layers."""

    base = json.loads((CORPUS / "decision/valid/implement.json").read_text())

    # Backward compat: a task without `criteria` validates and defaults to [].
    assert not decision_schema_errors(base)
    assert parse_decision(base).args.tasks[0].criteria == []

    # A task carrying natural-language criteria validates at both layers and
    # round-trips the statements verbatim.
    tasks = [dict(t) for t in base["args"]["tasks"]]
    tasks[0] = {**tasks[0], "criteria": ["the tokenizer handles unicode identifiers"]}
    yes = {**base, "args": {**base["args"], "tasks": tasks}}
    assert not decision_schema_errors(yes)
    parsed = parse_decision(yes)
    assert parsed.args.tasks[0].criteria == [
        "the tokenizer handles unicode identifiers"
    ]

    # An empty-string criterion is rejected by BOTH layers (non-empty constraint).
    bad_tasks = [dict(t) for t in base["args"]["tasks"]]
    bad_tasks[0] = {**bad_tasks[0], "criteria": [""]}
    bad = {**base, "args": {**base["args"], "tasks": bad_tasks}}
    assert decision_schema_errors(bad)
    with pytest.raises(ValidationError):
        parse_decision(bad)


# --- epoch_verdict.json vs the EpochVerdict / CriterionJudgement models --------
#
# The verdict is a re-read disk contract parsed directly by parse_epoch_verdict
# (no runtime jsonschema gate, unlike decision/handoff), so the drift guard here
# introspects the schema FILE against the Pydantic model: properties, required,
# and types must stay in sync, and after G14-3 neither side caps the free-text
# fields with a maxLength. A property added on one side and not the other, or a
# reintroduced length cap, fails this test.

_VERDICT_SCHEMA = json.loads(
    (SCHEMAS / "epoch_verdict.json").read_text(encoding="utf-8")
)


def test_epoch_verdict_schema_and_model_agree() -> None:
    schema = _VERDICT_SCHEMA
    props = schema["properties"]
    model_fields = EpochVerdict.model_fields

    # Same property set on both sides (schema property name -> model field name:
    # the schema's `pass` is the model's `passed` aliased back to `pass`).
    assert set(props) == {"pass", "per_criterion", "gaps", "digest"}
    assert set(model_fields) == {"passed", "per_criterion", "gaps", "digest"}
    assert model_fields["passed"].alias == "pass"

    # Required matches: pass / per_criterion / gaps are required, digest optional.
    assert set(schema["required"]) == {"pass", "per_criterion", "gaps"}
    assert model_fields["passed"].is_required()
    assert model_fields["per_criterion"].is_required()
    assert model_fields["gaps"].is_required()
    assert not model_fields["digest"].is_required()
    assert model_fields["digest"].default == ""

    # Types match: bool / array-of-objects / array-of-strings / string.
    assert props["pass"]["type"] == "boolean"
    assert model_fields["passed"].annotation is bool
    assert props["per_criterion"]["type"] == "array"
    assert props["per_criterion"]["items"]["type"] == "object"
    assert props["gaps"]["type"] == "array"
    assert props["gaps"]["items"]["type"] == "string"
    assert props["digest"]["type"] == "string"

    # No unknown keys on either side.
    assert schema["additionalProperties"] is False
    assert EpochVerdict.model_config.get("extra") == "forbid"


def test_criterion_judgement_schema_and_model_agree() -> None:
    item = _VERDICT_SCHEMA["properties"]["per_criterion"]["items"]
    model_fields = CriterionJudgement.model_fields

    assert set(item["properties"]) == {"criterion", "met", "evidence"}
    assert set(model_fields) == {"criterion", "met", "evidence"}
    assert set(item["required"]) == {"criterion", "met", "evidence"}
    for name in ("criterion", "met", "evidence"):
        assert model_fields[name].is_required()

    assert item["properties"]["criterion"]["type"] == "string"
    assert item["properties"]["evidence"]["type"] == "string"
    assert item["properties"]["met"]["type"] == "boolean"
    assert model_fields["met"].annotation is bool
    assert item["additionalProperties"] is False
    assert CriterionJudgement.model_config.get("extra") == "forbid"


def test_epoch_verdict_free_text_has_no_length_caps() -> None:
    """G14-3 removed every maxLength on the verdict free-text fields (the verdict
    travels to the planner BY REFERENCE, never byte-capped into a prompt). Guard
    that neither the schema nor the model reintroduces a string length cap on
    criterion / evidence / gaps / digest."""

    schema = _VERDICT_SCHEMA
    item = schema["properties"]["per_criterion"]["items"]["properties"]
    assert "maxLength" not in item["criterion"]
    assert "maxLength" not in item["evidence"]
    assert "maxLength" not in schema["properties"]["gaps"]["items"]
    assert "maxLength" not in schema["properties"]["digest"]

    # The model side: round-tripping a very long string is preserved verbatim
    # (a StringConstraints max_length would raise instead).
    long = "x" * 10_000
    verdict = EpochVerdict.model_validate(
        {
            "pass": True,
            "per_criterion": [
                {"criterion": long, "met": True, "evidence": long}
            ],
            "gaps": [long],
            "digest": long,
        }
    )
    assert verdict.per_criterion[0].criterion == long
    assert verdict.per_criterion[0].evidence == long
    assert verdict.gaps[0] == long
    assert verdict.digest == long
