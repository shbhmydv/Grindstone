"""The real-worker capability corpus (live, ``@pytest.mark.eval``, RED-until-run).

Each test drives the REAL ``run_task`` once through ``run_worker_task`` on the ``rig``
fixture (the local floor and the cloud ceiling) and asserts PROPERTIES of the result
via the ``_oracle`` bands: a trivial task must PASS, which means it cleared the
DETERMINISTIC gate (a non-empty in-scope commit, or a present artifact) and the
independent CRITIC returned a PASS verdict. The thesis mirrors the boundary corpus:
if the local floor handles a trivial implement + research task cleanly, the cloud
ceiling will too.

These need a live endpoint, so they are excluded from the default suite (the ``eval``
marker) and skipped unless ``GRINDSTONE_EVAL_RIG`` names the rig. Run them live with:

    GRINDSTONE_EVAL_RIG=local  .venv/bin/python -m pytest tests/grindstone/eval -m eval
    GRINDSTONE_EVAL_RIG=claude .venv/bin/python -m pytest tests/grindstone/eval -m eval
"""

from __future__ import annotations

import pytest

from grindstone.contracts.models import Task
from tests.grindstone.eval import _oracle as O
from tests.grindstone.eval._task import run_worker_task


@pytest.mark.eval
def test_implement_task_creates_file(rig: str) -> None:
    """A tiny implement task: create one file with exact content. BAND: the task
    PASSES the deterministic gate (a non-empty in-scope commit) and the critic
    verdict. file_ownership is a single concrete file, so the worker has no
    decomposition latitude to get wrong."""

    task = Task(
        id="T1",
        mode="implement",
        goal=(
            "Create a file named exactly `greeting.txt` in your worktree whose entire "
            "contents are the five characters `HELLO` (a trailing newline is fine). Do "
            "not create or edit any other file."
        ),
        file_ownership=["greeting.txt"],
    )
    result = run_worker_task(task=task, rig=rig)
    O.assert_task_passed_with_verdict(result)


@pytest.mark.eval
def test_research_task_writes_findings(rig: str) -> None:
    """A tiny research task: write a short grounded findings note. BAND: the task
    PASSES the deterministic gate (the artifact exists) and the critic verdict (the
    critic enforces the research grounding floor)."""

    task = Task(
        id="T1",
        mode="research",
        goal=(
            "Write a short findings note to `findings.md` in your current directory: "
            "3-5 bullets on what a command-line TODO app needs (a persistent store, "
            "add/list/done commands, a test suite). Ground at least one claim with a "
            "citation to `findings.md` itself."
        ),
        artifact_out="findings.md",
    )
    result = run_worker_task(task=task, rig=rig)
    O.assert_task_passed_with_verdict(result)


@pytest.mark.eval
def test_non_write_task_nested_artifact_out_publishes(rig: str) -> None:
    """Regression for the Run 5.5 E11 seed bug (the ``artifact_out`` non-write
    boundary). The planner emits a NESTED run-dir key like ``E11/T2/verdict.json``; the
    worker grinds in an isolated scratch CWD that knows nothing of the run-dir layout.
    Pre-fix the worker was told "produce at log key ``E1/T1/review.json``", wrote the
    BASENAME at its CWD root (what a weak local model naturally does), and the gate
    demanded the doubly-nested key inside scratch, so it was rejected with
    'artifact_out not produced in CWD' (10 live local-tier rejections). Post-fix (A) the
    instructed write basename and the gated basename agree, so the artifact lands and
    Python publishes it at the nested run-dir key. BAND: the task PASSES the
    deterministic gate + critic, and ``artifact_key`` is the nested key the planner
    chose. A NON-RESERVED basename (Fix B reserves ``verdict.json`` and the other
    control names), so this is a realistic-but-legal planner emission."""

    task = Task(
        id="T1",
        mode="research",
        goal=(
            "Write a short findings note as a JSON object to `review.json` in your "
            "current directory: a top-level \"summary\" string and a \"points\" array "
            "of 3-5 short strings on what a command-line TODO app needs (a persistent "
            "store, add/list/done commands, a test suite). Ground the summary by "
            "referencing `review.json` itself."
        ),
        artifact_out="E1/T1/review.json",
    )
    result = run_worker_task(task=task, rig=rig)
    O.assert_task_passed_with_verdict(result)
    assert result.artifact_key == "E1/T1/review.json", (
        "the non-write deliverable did not publish at its nested run-dir key: "
        f"{result.artifact_key!r}"
    )
