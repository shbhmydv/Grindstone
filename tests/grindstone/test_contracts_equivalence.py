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

CORPUS = Path(__file__).parent / "corpus"


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
