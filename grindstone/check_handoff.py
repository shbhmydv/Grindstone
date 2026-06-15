"""Generator for the worker-facing handoff validator.

The core's gate (``contracts/gate.py``) only judges a handoff AFTER the worker
exits — an invalid handoff costs a whole new worker subprocess. To let a worker
self-correct, the core drops a small validator in the worker's CWD; the worker
runs it and loops until clean before handing back.

``generate_check_script`` emits that validator as a **self-contained, stdlib-only**
Python source string parametrized by ``task_id`` and ``mode``: target repos have
neither ``grindstone`` nor ``jsonschema`` on the path, so the script imports
neither. The structural rules are NOT hand-mirrored — ``schemas/handoff.json``
itself is baked into the script as a literal and walked by a small stdlib
validator covering exactly the keywords the schema uses (type, required,
properties, additionalProperties, items, enum, const, maxItems, min/maxLength,
pattern, minimum). Hand-mirroring was the S2-gate RCA of 2026-06-12: the mirror
silently lacked type checks for ``what_changed``/``not_done``/``downstream_needs``,
so workers passed self-validation and burned attempts at the core gate.

Hand-coded here are only the context rules the schema cannot express: exact
``task_id`` match, the mode citation requirement (``semantics.py``), citation
files existing on disk, and the canonical-size cap. The corpus-equivalence test
remains the fence pinning script verdicts to the core gate.
"""

from __future__ import annotations

import json
from pathlib import Path

from grindstone.contracts.semantics import HANDOFF_MAX_BYTES, HandoffMode

#: The filename the script is written as, and the done_when command that re-runs
#: it. Co-located with the generator so the core (task_loop) reuses both.
CHECK_SCRIPT_NAME = "check_handoff.py"
CHECK_COMMAND = f"python3 {CHECK_SCRIPT_NAME}"

#: Modes whose handoff must carry >= 1 citation (mirrors ``handoff_violations``).
_CITATION_MODES = ("research", "review")

#: The wire schema, loaded once at import (same artifact the core gate compiles)
#: and re-serialized compact for baking into the generated script.
_SCHEMA_TEXT = json.dumps(
    json.loads(
        (Path(__file__).resolve().parents[1] / "schemas" / "handoff.json").read_text(
            encoding="utf-8"
        )
    ),
    sort_keys=True,
    separators=(",", ":"),
)

# The invariant body of the generated validator. Baked constants are prepended
# by the generator; everything below is mode/task-agnostic and stdlib-only.
_SCRIPT_BODY = '''
import json
import re
import sys
from pathlib import Path

_SCHEMA = json.loads(_SCHEMA_TEXT)


def _type_ok(value, expected):
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    return True


def _walk(value, schema, path, out):
    """Validate value against the subset of JSON-Schema keywords handoff.json
    uses. A type mismatch stops descent at this node (children would be noise)."""

    label = path or "handoff"
    expected = schema.get("type")
    if expected is not None and not _type_ok(value, expected):
        out.append("%s must be of type %s" % (label, expected))
        return
    if "const" in schema and value != schema["const"]:
        out.append("%s must be %r, got %r" % (label, schema["const"], value))
    if "enum" in schema and value not in schema["enum"]:
        out.append("%s must be one of %s, got %r" % (label, schema["enum"], value))
    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            out.append("%s is shorter than %d chars" % (label, schema["minLength"]))
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            out.append("%s is longer than %d chars" % (label, schema["maxLength"]))
        if "pattern" in schema and not re.search(schema["pattern"], value):
            out.append("%s does not match %s" % (label, schema["pattern"]))
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            out.append("%s is below minimum %s" % (label, schema["minimum"]))
    if isinstance(value, dict):
        props = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in value:
                out.append("%s: missing required field: %s" % (label, key))
        if schema.get("additionalProperties") is False:
            for key in sorted(value):
                if key not in props:
                    out.append("%s: unknown field: %s" % (label, key))
        for key, sub in props.items():
            if key in value:
                _walk(value[key], sub, path + "." + key if path else key, out)
    if isinstance(value, list):
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            out.append("%s has more than %d items" % (label, schema["maxItems"]))
        items = schema.get("items")
        if isinstance(items, dict):
            for idx, item in enumerate(value):
                _walk(item, items, "%s[%d]" % (label, idx), out)


def _strip_none(value):
    """Drop None entries recursively — the canonical form excludes them."""

    if isinstance(value, dict):
        return {k: _strip_none(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_strip_none(v) for v in value]
    return value


def main():
    base = Path(__file__).resolve().parent
    src = base / "handoff.json"
    if not src.is_file():
        print("handoff.json not found in %s" % base)
        return 1
    try:
        data = json.loads(src.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        print("handoff.json is not valid JSON: %s" % exc)
        return 1

    violations = []
    _walk(data, _SCHEMA, "", violations)

    if isinstance(data, dict):
        if "task_id" in data and data["task_id"] != _EXPECTED_TASK_ID:
            violations.append(
                "task_id %r != dispatched id %r" % (data["task_id"], _EXPECTED_TASK_ID)
            )
        citations = data.get("citations") or []
        if _REQUIRE_CITATION and len(citations) < 1:
            violations.append("%s handoff requires >= 1 citation" % _MODE)
        if isinstance(citations, list):
            roots = [base] + ([Path(_REPO_ROOT)] if _REPO_ROOT else [])
            for cite in citations:
                if isinstance(cite, dict) and isinstance(cite.get("file"), str):
                    rel = cite["file"]
                    found = False
                    for root in roots:
                        cand = (root / rel).resolve()
                        if cand.is_file() and any(
                            cand.is_relative_to(r) for r in roots
                        ):
                            found = True
                            break
                    if not found:
                        violations.append(
                            "citation file missing or outside allowed roots: %s" % rel
                        )

    canonical = json.dumps(
        _strip_none(data), sort_keys=True, separators=(",", ":")
    ).encode()
    if len(canonical) > _MAX_BYTES:
        violations.append("handoff exceeds %d bytes: %d" % (_MAX_BYTES, len(canonical)))

    for line in violations:
        print(line)
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
'''


def generate_check_script(
    *, task_id: str, mode: HandoffMode, repo_root: Path | None = None
) -> str:
    """Return the validator source for one dispatched task (pure — no I/O).

    The dispatched ``task_id``, the human-readable ``mode``, whether the mode
    requires a citation, the byte cap, the wire schema and the citation roots
    are baked in as literals; the body is invariant and stdlib-only.
    ``repo_root`` adds the target repo as a second allowed citation root for
    research/review/artifact attempts (their scratch is a plain dir, so repo
    files they investigate live OUTSIDE the CWD — gate-5 P0); implement
    attempts pass ``None`` because their CWD already IS a repo checkout and
    the operator checkout must stay out of bounds.
    """

    baked_root = str(repo_root.resolve()) if repo_root is not None else None
    header = (
        "# Generated by grindstone.check_handoff for one attempt — DO NOT EDIT.\n"
        f"_EXPECTED_TASK_ID = {task_id!r}\n"
        f"_MODE = {mode!r}\n"
        f"_REQUIRE_CITATION = {mode in _CITATION_MODES!r}\n"
        f"_MAX_BYTES = {HANDOFF_MAX_BYTES}\n"
        f"_REPO_ROOT = {baked_root!r}\n"
        f"_SCHEMA_TEXT = {_SCHEMA_TEXT!r}\n"
    )
    return header + _SCRIPT_BODY
