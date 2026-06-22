"""The planner property eval corpus (live, @pytest.mark.eval, RED-until-run).

Each test drives a REAL planner through ``run_planner_boundary`` on the ``rig``
fixture (the local floor and the cloud ceiling) and asserts PROPERTIES of the
decision, never exact goldens. The thesis: if the local floor emits conforming,
correctly-shaped plans, the cloud ceiling will too. The bands are deliberately
loose so a strong and a weak model can both pass, while an unconditionally-broken
plan (wrong tool, undecomposed mega-task, malformed schema) cannot.

These need a live endpoint, so they are excluded from the default suite (the
``eval`` marker) and skipped unless ``GRINDSTONE_EVAL_RIG`` names the rig. The
parent runs the live baseline (``GRINDSTONE_EVAL_RIG=local`` first).
"""

from __future__ import annotations

import pytest

from grindstone.contracts.models import Phase, parse_decision
from grindstone.planner import DEFAULT_LOCAL_MAX_TASK_FILES, FailedEpochInfo
from tests.grindstone.conftest import phase_dict, skeleton_decision
from tests.grindstone.eval import _assertions as A
from tests.grindstone.eval._boundary import run_planner_boundary

# --- job specs (short, inline, spanning the planner's mode space) --------------

#: An implement-heavy app: the planner should skeleton it then build code.
TODO_APP_JOB = """\
Build a tiny command-line TODO app in Python (single package `todo/`).

Requirements:
- `todo add <text>` appends a task to a JSON store on disk.
- `todo list` prints the open tasks, numbered.
- `todo done <n>` marks task n complete.
- A pytest suite covers add / list / done.

Keep it small and idiomatic; one module per command is fine.
"""

#: A small library/CLI: a different implement shape (pure functions + a thin CLI).
SLUGIFY_LIB_JOB = """\
Build a small Python library `slugify` that turns arbitrary text into URL slugs.

Requirements:
- `slugify(text: str, *, max_length: int | None = None) -> str`: lowercase,
  spaces and punctuation collapsed to single hyphens, leading/trailing hyphens
  stripped, optional truncation at a word boundary.
- A thin `python -m slugify <text>` CLI prints the slug.
- A pytest suite covers unicode, punctuation, and the max_length boundary.
"""

#: A research/report spec: the planner should reach for research/artifact epochs.
RESEARCH_REPORT_JOB = """\
Produce a research report `report.md` comparing three Python web frameworks
(Flask, FastAPI, Django) for building a small JSON API.

Requirements:
- Cover routing, request validation, async support, and the testing story.
- Cite the official docs for each claim.
- End with a recommendation for a 3-endpoint internal service.
"""

#: A two-phase skeleton handed to the mid-run + failed-epoch boundaries (P1 done).
_SKELETON_DICT = skeleton_decision(
    phase_dict("P1", title="scaffold the package + store"),
    phase_dict("P2", title="implement the commands + tests"),
)


def _skeleton() -> list[Phase]:
    """The shared two-phase skeleton as typed ``Phase`` objects."""

    decision = parse_decision(_SKELETON_DICT)
    return list(decision.args.phases)  # type: ignore[union-attr]


# --- first boundary: propose_skeleton ------------------------------------------


@pytest.mark.eval
@pytest.mark.parametrize("job", [TODO_APP_JOB, SLUGIFY_LIB_JOB, RESEARCH_REPORT_JOB])
def test_first_boundary_proposes_skeleton(rig: str, job: str) -> None:
    """No skeleton yet -> the only legal tool is propose_skeleton, with a sane
    phase count.

    BAND: phase count in [2, 6]. Lower bound 2 is the schema floor (a skeleton
    needs >=2 phases); upper bound 6 is a generous ceiling for a small app, a
    skeleton over six phases for a TODO app / one-function library is
    over-decomposed and a real planning smell. The scenario selector is pinned to
    ``plan_skeleton`` for this state."""

    A.assert_scenario_selected(
        skeleton_exists=False, failed_epoch_active=False, expected="plan_skeleton"
    )
    decision = run_planner_boundary(job_spec=job, rig=rig, skeleton=None)
    A.assert_tool(decision, "propose_skeleton")
    A.assert_conforms(decision, skeleton_exists=False)
    A.assert_phase_count_between(decision, 2, 6)


# --- mid-run boundary: a work epoch --------------------------------------------


@pytest.mark.eval
@pytest.mark.parametrize("job", [TODO_APP_JOB, SLUGIFY_LIB_JOB])
def test_mid_run_boundary_emits_work_epoch(rig: str, job: str) -> None:
    """Skeleton given + P1 completed -> a work tool whose tasks are decomposed.

    BAND: tool in {implement, research, review, artifact} (the legal work set);
    the full core gate passes (disjoint file_ownership, legal inputs, position
    legality); and every implement task is within the local file cap. These are
    the gate's own bands, so a plan that conforms here is dispatchable as-is."""

    A.assert_scenario_selected(
        skeleton_exists=True, failed_epoch_active=False, expected="plan_epoch"
    )
    decision = run_planner_boundary(
        job_spec=job,
        rig=rig,
        skeleton=_skeleton(),
        completed_phase_ids=("P1",),
        has_senior=False,
    )
    A.assert_tool_in(decision, A.WORK_TOOLS)
    A.assert_conforms(
        decision,
        skeleton_exists=True,
        completed_phase_ids=frozenset({"P1"}),
    )
    A.assert_every_implement_task_within(decision, DEFAULT_LOCAL_MAX_TASK_FILES)


# --- failed-epoch boundary: handle_failed_epoch --------------------------------


@pytest.mark.eval
def test_failed_epoch_boundary_disposes(rig: str) -> None:
    """A failed epoch awaiting disposition -> the only legal tool is
    handle_failed_epoch (retry / escalate_senior / halt).

    BAND: tool == handle_failed_epoch and the decision conforms under
    ``failed_epoch_active=True`` (the gate rejects any other tool here). The action
    itself is not pinned: retry, escalate, and halt are all legal dispositions of a
    failing build gate, so over-constraining the action would be a golden, not a
    property."""

    A.assert_scenario_selected(
        skeleton_exists=True, failed_epoch_active=True, expected="repair_epoch"
    )
    failed = FailedEpochInfo(
        epoch_id="P2/E1",
        failed_tasks=[("T1", "pytest failed: 2 of 5 tests error on import")],
        failed_checks=["python -m pytest -q (exit 1)"],
        passing_handoffs=[("T2", "added todo/store.py")],
        disposed_count=0,
        cap=3,
    )
    decision = run_planner_boundary(
        job_spec=TODO_APP_JOB,
        rig=rig,
        skeleton=_skeleton(),
        completed_phase_ids=("P1",),
        failed_epoch=failed,
    )
    A.assert_tool(decision, "handle_failed_epoch")
    A.assert_conforms(
        decision,
        skeleton_exists=True,
        failed_epoch_active=True,
        completed_phase_ids=frozenset({"P1"}),
    )
