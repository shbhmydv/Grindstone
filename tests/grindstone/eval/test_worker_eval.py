"""The worker property eval corpus (live, @pytest.mark.eval, RED-until-run).

Each test drives a REAL worker through ``run_worker_task`` on the ``rig`` fixture
(the local floor and the cloud ceiling) and asserts PROPERTIES of the handoff it
produced + the tree it left, never exact goldens. The thesis mirrors the planner
corpus: if the local floor produces conforming, correctly-grounded handoffs that
pass their own done_when, the cloud ceiling will too. The bands are the production
gate's own (``_collect_handoff`` already ran inside the harness and would have
raised), plus a few stricter corpus expectations (DONE status, citations present).

One test per epoch mode: an IMPLEMENT task (write a file + prove it), a RESEARCH
task (write a findings artifact + cite it), and a REVIEW task (judge a small
target + cite it). These need a live endpoint, so they are excluded from the
default suite (the ``eval`` marker) and skipped unless ``GRINDSTONE_EVAL_RIG``
names the rig. The parent runs the live baseline (``GRINDSTONE_EVAL_RIG=local``).
"""

from __future__ import annotations

import pytest

from grindstone.contracts.models import ArtifactTask, CmdCheck, ImplementTask
from tests.grindstone.eval import _assertions as A
from tests.grindstone.eval._worker import run_worker_task

# --- IMPLEMENT: create a file with exact content -------------------------------


@pytest.mark.eval
def test_implement_task_creates_file(rig: str) -> None:
    """An implement task that writes ``greeting.txt`` containing exactly ``HELLO``.

    BAND: the handoff is DONE and conforms to the production gate (so it is
    dispatchable as-is); the done_when (``test -f`` + an exact-content grep) passed
    (the harness re-ran them in the worktree and would have raised otherwise); and
    the worker grounded its claim in >= 1 citation. file_ownership is a single file,
    so the worker has no decomposition latitude to get wrong."""

    task = ImplementTask(
        id="T1",
        goal=(
            "Create a file named exactly `greeting.txt` in the current directory "
            "whose entire contents are the five characters `HELLO` (you may add a "
            "trailing newline). Do not create any other file."
        ),
        done_when=[
            CmdCheck(cmd="test -f greeting.txt"),
            CmdCheck(cmd="grep -qx HELLO greeting.txt"),
        ],
        criteria=["greeting.txt contains exactly HELLO"],
        file_ownership=["greeting.txt"],
    )
    handoff = run_worker_task(task=task, rig=rig, mode="implement")
    A.assert_handoff_status(handoff, "DONE")
    A.assert_handoff_conforms(handoff, mode="implement", task_id="P1/E1/T1")
    A.assert_handoff_done_when_passed(handoff)
    A.assert_what_changed_shape(handoff)
    A.assert_handoff_citations_present(handoff)


# --- RESEARCH: write a short findings artifact + cite it -----------------------


@pytest.mark.eval
def test_research_task_writes_findings(rig: str) -> None:
    """A research task that writes a short findings artifact into its CWD.

    BAND: the handoff is DONE and conforms under ``mode='research'`` (the gate
    enforces the >= 1 citation floor there); the done_when (the artifact exists +
    is non-empty) passed; and the corpus citation band holds. The findings content
    is not golden, only that a grounded artifact was produced."""

    task = ArtifactTask(
        id="T1",
        goal=(
            "Write a short findings note to `findings.md` in your current "
            "directory: 3-5 bullet points summarizing what a command-line TODO app "
            "needs (a persistent store, add/list/done commands, a test suite). "
            "Ground at least one claim with a citation to `findings.md` itself."
        ),
        done_when=[
            CmdCheck(cmd="test -s findings.md"),
        ],
        criteria=["findings.md summarizes the TODO-app requirements"],
        artifact_out="findings.md",
    )
    handoff = run_worker_task(task=task, rig=rig, mode="research")
    A.assert_handoff_status(handoff, "DONE")
    A.assert_handoff_conforms(handoff, mode="research", task_id="P1/E1/T1")
    A.assert_handoff_done_when_passed(handoff)
    A.assert_handoff_citations_present(handoff)


# --- REVIEW: judge a small target + cite it ------------------------------------


@pytest.mark.eval
def test_review_task_judges_target(rig: str) -> None:
    """A review task that judges a small target file and writes a verdict artifact.

    The harness seeds the target INTO the scratch CWD via the worker's own
    instructions: a review task investigates files, so the goal pins the worker to
    first write the target, then review it (a self-contained corpus item that needs
    no external repo). BAND: the handoff is DONE and conforms under
    ``mode='review'`` (citation floor enforced); the done_when (the review note
    exists) passed; and the corpus citation band holds."""

    task = ArtifactTask(
        id="T1",
        goal=(
            "Review the Python snippet below for correctness and write your verdict "
            "to `review.md` in your current directory. First write the snippet "
            "verbatim to `target.py` so you have a file to cite, then review it.\n\n"
            "    def add(a, b):\n"
            "        return a - b   # claims to add\n\n"
            "Your review.md must state whether `add` is correct and cite `target.py` "
            "(or `review.md`) as grounding."
        ),
        done_when=[
            CmdCheck(cmd="test -s review.md"),
        ],
        criteria=["review.md judges the snippet's correctness"],
        artifact_out="review.md",
    )
    handoff = run_worker_task(task=task, rig=rig, mode="review")
    A.assert_handoff_status(handoff, "DONE")
    A.assert_handoff_conforms(handoff, mode="review", task_id="P1/E1/T1")
    A.assert_handoff_done_when_passed(handoff)
    A.assert_handoff_citations_present(handoff)
