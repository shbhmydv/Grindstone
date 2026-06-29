"""Planner-facing decision validator: ground the planner like the workers.

The worker's deliverable is gated on deterministic facts (its committed diff /
produced artifact) plus the critic, so its free-form ``handoff.md`` needs no
self-validator; the planner, by contrast, emits structured JSON the core MUST parse,
so it gets a self-validate-on-disk loop. It writes ``decision.json`` into its
worktree CWD, runs
``python3 check_decision.py decision.json``, and fixes every violation the script
prints until it exits 0. The core then reads the already-validated file back as the
disk contract and re-runs ``parse_decision`` as defense in depth (parsing stays in
core), so the planner stops burning blind re-asks on a schema it guessed wrong.

This validator runs on the grindstone HOST, so it does NOT re-implement the schema:
it re-execs the REAL core gate (``parse_decision``).
``parse_decision`` is the single source of truth; ``write_validator`` drops a tiny
stdlib wrapper that re-execs it through the grindstone interpreter. The tolerant
JSON extractor lives here too (it is the other half of "parse untrusted planner
output"); the core imports it for its own read-back.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from grindstone.contracts.models import parse_decision

#: The filename the wrapper is written as (the planner re-runs it on its decision).
CHECK_SCRIPT_NAME = "check_decision.py"
#: Where the planner writes its candidate decision (the disk contract read back).
DECISION_FILE = "decision.json"


# --- tolerant JSON extraction (the core does the authoritative parse) ----------


def _balanced_object_spans(text: str) -> list[tuple[int, int]]:
    """Yield ``(start, end)`` of every TOP-LEVEL ``{...}`` region, string-aware.

    Brace counting that ignores braces inside JSON string literals (honouring
    backslash escapes), so a model that emits reasoning then a fenced object still
    yields the real object spans. Nested objects ride their enclosing span.
    """

    spans: list[tuple[int, int]] = []
    i, n = 0, len(text)
    while i < n:
        if text[i] != "{":
            i += 1
            continue
        depth, in_str, esc, j = 0, False, False, i
        while j < n:
            c = text[j]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            elif c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    spans.append((i, j + 1))
                    break
            j += 1
        i = j + 1
    return spans


def _select_json_object(text: str, discriminator: str) -> str | None:
    """Return the source of the JSON object carrying ``discriminator``, or ``None``.

    The generic free-text sniffer shared by ``extract_decision_json`` (discriminator
    ``kind``) and ``extract_verdict_json`` (discriminator ``outcome``): scan every
    balanced top-level object, keep those that parse, and prefer the LAST one carrying
    the discriminator key (a model may reason before its final answer). Falls back to
    the last parsing object, then to ``None`` when nothing parses.
    """

    discriminated: list[str] = []
    any_object: list[str] = []
    for start, end in _balanced_object_spans(text):
        sub = text[start:end]
        try:
            obj = json.loads(sub)
        except ValueError:
            continue
        if isinstance(obj, dict):
            any_object.append(sub)
            if discriminator in obj:
                discriminated.append(sub)
    if discriminated:
        return discriminated[-1]
    if any_object:
        return any_object[-1]
    return None


def extract_decision_json(text: str) -> str | None:
    """Recover the planner's decision object from free text (it may reason first),
    preferring the LAST object carrying the ``kind`` discriminator."""

    return _select_json_object(text, "kind")


def extract_verdict_json(text: str) -> str | None:
    """Recover the critic's lenient verdict object from free-text chat (a local model
    that answered in prose instead of writing ``verdict.json``), preferring the LAST
    object carrying the ``outcome`` discriminator. The verdict twin of
    ``extract_decision_json`` (both share ``_select_json_object``)."""

    return _select_json_object(text, "outcome")


# --- the gate (re-execs the real core validator) -------------------------------


def _run(decision_path: Path) -> tuple[int, str]:
    """Validate ``decision_path`` against the core gate (``parse_decision``).

    Returns ``(0, "OK ...")`` when the decision conforms, else ``(1, "<violations>")``
    with the human-readable reason the planner must fix.
    """

    try:
        text = decision_path.read_text(encoding="utf-8")
    except OSError as exc:
        return 1, f"cannot read {decision_path.name}: {exc}"
    json_text = extract_decision_json(text)
    if json_text is None:
        return 1, f"{decision_path.name} carries no JSON decision object"
    try:
        payload = json.loads(json_text)
    except ValueError as exc:
        return 1, f"{decision_path.name} is not valid JSON: {exc}"
    try:
        parse_decision(payload)
    except ValueError as exc:
        return 1, str(exc)
    return 0, "OK: decision conforms to the epoch-decision schema."


def generate_check_script(*, grindstone_python: str) -> str:
    """Return the wrapper source the planner runs (pure, no I/O).

    A self-contained stdlib script that re-execs the real core gate through the
    grindstone interpreter, so the planner's local verdict is byte-identical to the
    core gate it faces after handing back. The interpreter path is baked in; the
    body is invariant.
    """

    return (
        "# Generated by grindstone.check_decision for one planner boundary. DO NOT EDIT.\n"
        "# Re-execs the real grindstone decision gate (parse_decision) so the planner\n"
        "# self-corrects its epoch JSON against the SAME validator the core applies.\n"
        "import subprocess\n"
        "import sys\n"
        f"_GRINDSTONE_PYTHON = {grindstone_python!r}\n"
        "_TARGET = sys.argv[1] if len(sys.argv) > 1 else 'decision.json'\n"
        "sys.exit(\n"
        "    subprocess.run(\n"
        "        [_GRINDSTONE_PYTHON, '-m', 'grindstone.check_decision', _TARGET]\n"
        "    ).returncode\n"
        ")\n"
    )


def write_validator(workdir: Path, *, grindstone_python: str) -> None:
    """Drop ``check_decision.py`` into the planner CWD so it can self-validate.

    The caller ensures ``workdir`` exists (the boundary's worktree always does).
    """

    (workdir / CHECK_SCRIPT_NAME).write_text(
        generate_check_script(grindstone_python=grindstone_python), encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="grindstone.check_decision")
    parser.add_argument("decision", type=Path, help="the candidate decision JSON file")
    ns = parser.parse_args(argv)
    code, message = _run(ns.decision)
    print(message)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
