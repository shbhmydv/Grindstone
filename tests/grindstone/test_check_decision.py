"""Tests for the planner-facing decision validator (``grindstone.check_decision``).

Mirrors the worker's ``check_handoff`` contract: the planner writes ``decision.json``
in its worktree, runs this validator, and loops until it exits 0 before grindstone
reads the file back. Unlike the worker validator (stdlib-only, target repos lack
grindstone), the planner runs on the grindstone host, so the script re-execs the
REAL core gate (``validate_decision``) instead of re-implementing it. The fence is
corpus-equivalence: the script verdict must never disagree with the core gate.
"""

from __future__ import annotations

import json
from pathlib import Path

from grindstone import check_decision
from grindstone.planner import extract_decision_json, validate_decision
from tests.grindstone.conftest import (
    artifact_task,
    implement_decision,
    impl_task,
    two_phase_skeleton,
)

# A real dogfood halt: Claude stuffed a >120-char brief into epoch_title AND
# flattened task-level fields (goal/done_when/criteria) onto the epoch args.
_FLATTENED = {
    "schema_version": "1",
    "tool": "research",
    "args": {
        "epoch_title": "Produce dom_plan.md: " + "a code-grounded plan. " * 8,
        "rationale": "ground the web target in the real screens",
        "tasks": [
            {
                "id": "T1",
                "goal": "produce dom_plan.md",
                "done_when": [{"cmd": "true"}],
                "artifact_out": "P1/E1/T1/dom_plan.md",
            }
        ],
        "goal": "Produce dom_plan.md covering deps, app.json, playwright",
        "done_when": "the plan covers deps, app.json, playwright",
        "criteria": "thorough",
    },
}


def _ctx(**over: object) -> dict[str, object]:
    base: dict[str, object] = {
        "existing_log_keys": [],
        "completed_phase_ids": [],
        "skeleton_exists": False,
        "phase_escalated": False,
        "failed_epoch_active": False,
        "has_senior": False,
        "local_max_task_files": 6,
        "senior_max_task_files": 12,
    }
    base.update(over)
    return base


def _run(tmp_path: Path, decision: object, ctx: dict[str, object]) -> tuple[int, str]:
    dpath = tmp_path / "decision.json"
    cpath = tmp_path / "decision_context.json"
    dpath.write_text(json.dumps(decision), encoding="utf-8")
    cpath.write_text(json.dumps(ctx), encoding="utf-8")
    return check_decision._run(dpath, cpath)


def test_accepts_valid_first_skeleton(tmp_path: Path) -> None:
    code, msg = _run(tmp_path, two_phase_skeleton(), _ctx(skeleton_exists=False))
    assert code == 0
    assert "OK" in msg


def test_rejects_flattened_overlong_epoch(tmp_path: Path) -> None:
    code, msg = _run(tmp_path, _FLATTENED, _ctx(skeleton_exists=True))
    assert code == 1
    # Both real failure modes surface to the planner, verbatim from the schema.
    assert "too long" in msg
    assert "criteria" in msg or "unexpected" in msg or "unknown" in msg


def test_corpus_equivalence_with_core_gate(tmp_path: Path) -> None:
    """The script verdict pins to ``validate_decision`` for every payload."""

    corpus: list[tuple[object, dict[str, object]]] = [
        (two_phase_skeleton(), _ctx(skeleton_exists=False)),
        (_FLATTENED, _ctx(skeleton_exists=True)),
        (implement_decision(impl_task("T1", "a.py")), _ctx(skeleton_exists=True)),
        ({"not": "a decision"}, _ctx(skeleton_exists=True)),
        ("{ broken json", _ctx(skeleton_exists=True)),
    ]
    for decision, ctx in corpus:
        text = decision if isinstance(decision, str) else json.dumps(decision)
        core = validate_decision(
            extract_decision_json(text),
            existing_log_keys=frozenset(),
            completed_phase_ids=frozenset(),
            skeleton_exists=bool(ctx["skeleton_exists"]),
        )
        code, _ = _run(tmp_path, decision, ctx)
        assert (code == 0) == (core.decision is not None), decision


def test_generated_wrapper_reexecs_grindstone(tmp_path: Path) -> None:
    check_decision.write_validator(
        tmp_path, context=_ctx(), grindstone_python="/x/.venv/bin/python"
    )
    script = (tmp_path / check_decision.CHECK_SCRIPT_NAME).read_text(encoding="utf-8")
    assert "/x/.venv/bin/python" in script
    assert "grindstone.check_decision" in script
    assert (tmp_path / check_decision.CONTEXT_FILE).is_file()


def test_artifact_task_epoch_round_trips(tmp_path: Path) -> None:
    decision = {
        "schema_version": "1",
        "tool": "research",
        "args": {
            "epoch_title": "scope the DOM plan",
            "rationale": "ground the web target in the real screens",
            "tasks": [artifact_task("T1", out="P1/E1/T1/dom_plan.md")],
        },
    }
    code, _ = _run(tmp_path, decision, _ctx(skeleton_exists=True))
    assert code == 0
