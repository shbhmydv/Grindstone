"""Runtime JSON-Schema gate.

Untrusted JSON (planner output, worker handoff) is validated against the
Draft 2020-12 schemas in ``schemas/`` BEFORE it is parsed into typed models.
The schema is the single source of truth; this layer is the structural gate and
the equivalence test pins the typed models to it.
"""

from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator

_SCHEMA_DIR = Path(__file__).resolve().parents[2] / "schemas"


def _validator(name: str) -> Draft202012Validator:
    schema = json.loads((_SCHEMA_DIR / name).read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


_DECISION_VALIDATOR = _validator("epoch_decision.json")
_HANDOFF_VALIDATOR = _validator("handoff.json")


def decision_schema_errors(payload: object) -> list[str]:
    """Return schema-validation error messages for a decision (empty = valid)."""

    return [e.message for e in _DECISION_VALIDATOR.iter_errors(payload)]


def handoff_schema_errors(payload: object) -> list[str]:
    """Return schema-validation error messages for a handoff (empty = valid)."""

    return [e.message for e in _HANDOFF_VALIDATOR.iter_errors(payload)]
