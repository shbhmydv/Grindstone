"""The real-planner capability corpus (live, ``@pytest.mark.eval``, RED-until-run).

Each test drives the REAL ``ScriptPlanner`` once through ``run_planner_boundary`` on
the ``rig`` fixture (the local floor and the cloud ceiling) and asserts PROPERTIES of
the decision via the ``_oracle`` bands, never exact goldens. The thesis: if the local
floor emits a conforming, correctly-shaped decision, the cloud ceiling will too.

These need a live endpoint, so they are excluded from the default suite (the ``eval``
marker) and skipped unless ``GRINDSTONE_EVAL_RIG`` names the rig. Run them live with:

    GRINDSTONE_EVAL_RIG=local  .venv/bin/python -m pytest tests/grindstone/eval -m eval
    GRINDSTONE_EVAL_RIG=claude .venv/bin/python -m pytest tests/grindstone/eval -m eval
"""

from __future__ import annotations

import pytest

from tests.grindstone.eval import _oracle as O
from tests.grindstone.eval._boundary import run_planner_boundary

# --- job specs (short, inline, spanning the planner's mode space) --------------

#: An implement-heavy app: the planner should propose a bounded, disjoint epoch.
TODO_APP_JOB = """\
Build a tiny command-line TODO app in Python (single package `todo/`).

Requirements:
- `todo add <text>` appends a task to a JSON store on disk.
- `todo list` prints the open tasks, numbered.
- `todo done <n>` marks task n complete.
- A pytest suite covers add / list / done.

Keep it small and idiomatic; one module per command is fine.
"""

#: A research/report spec: the planner should reach for a research / artifact task.
RESEARCH_REPORT_JOB = """\
Produce a research report `report.md` comparing three Python web frameworks
(Flask, FastAPI, Django) for building a small JSON API.

Requirements:
- Cover routing, request validation, async support, and the testing story.
- Cite the official docs for each claim.
- End with a recommendation for a 3-endpoint internal service.
"""

#: A decomposable UI build: a MECHANICAL slice (tokens) + a TASTE slice (the screen),
#: which a well-shaped epoch fans into disjoint, concretely-owned tasks.
UI_BUILD_JOB = """\
Build a small React Native home screen for a notes app (Expo, TypeScript).

Requirements:
- A `src/tokens.ts` exporting the color + spacing scale (a Material 3 palette).
- A `src/HomeScreen.tsx` home screen: a calm layout with ONE primary action
  (a "New note" button, >=44px target), a list of recent notes, generous spacing.
- The tokens are mechanical; the screen's layout and feel are a matter of taste.
"""


@pytest.mark.eval
@pytest.mark.parametrize("job", [TODO_APP_JOB, RESEARCH_REPORT_JOB, UI_BUILD_JOB])
def test_first_boundary_decision_conforms(rig: str, job: str) -> None:
    """A FRESH boundary: the planner emits exactly one bones decision (epoch or end)
    that obeys the core rules.

    BAND (properties, not goldens): the decision is well-formed; if it is an epoch it
    has 1..8 tasks, every task a valid mode + tier, every implement task owns >= 1
    CONCRETE file (no wildcard), and the implement tasks' ownership is pairwise
    disjoint. A broken plan (a wildcard mega-task, an ownership collision, a malformed
    mode) cannot pass; a strong and a weak model both can."""

    decision = run_planner_boundary(job_spec=job, rig=rig)
    O.assert_decision_conforms(decision)


@pytest.mark.eval
def test_boundary_with_carried_failure_conforms(rig: str) -> None:
    """A boundary carrying a prior-epoch failure: the planner must STILL emit a
    conforming decision (steer around the blocker or end on it), never a malformed or
    overlapping plan. Same property band as a fresh boundary."""

    decision = run_planner_boundary(
        job_spec=TODO_APP_JOB,
        rig=rig,
        carried=("P1/E1/T1 escalated: pytest is not installed in the environment",),
        epoch_index=2,
    )
    O.assert_decision_conforms(decision)
