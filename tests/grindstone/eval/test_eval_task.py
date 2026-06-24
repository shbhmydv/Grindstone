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
