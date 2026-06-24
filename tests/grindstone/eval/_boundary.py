"""Drive ONE real planner boundary on a rig, exactly as the loop does.

``run_planner_boundary`` is the corpus's single live planner entry point: it builds
the SAME ``PlannerContext`` the epoch driver rebuilds from disk at a boundary, then
calls the REAL ``ScriptPlanner`` (over the real ``ScriptPlannerTransport`` rig) once.
``ScriptPlanner.decide`` does everything production does, render the prompt, ground +
self-validate ``decision.json`` on disk in a ``_planner_tip`` checkout, read the
result back by the ``decision.json`` > ``--out`` > stdout priority, and gate it with
``parse_decision``, so the returned ``Decision`` is precisely what the orchestrator
would accept. The corpus then asserts PROPERTIES of it (the ``_oracle`` bands).

This REUSES the production planner whole (no run-loop helper is duplicated, the bones
planner is already a single stateless ``decide`` call), so there is zero drift from
what a live run sees.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from grindstone import worktree as wt
from grindstone.config import RoleConfig, models_script, resolve_role_script
from grindstone.contracts.models import Decision
from grindstone.loop import PlannerContext
from grindstone.planner import ScriptPlanner
from grindstone.rundir import RunDir
from grindstone.script_planner import ScriptPlannerTransport
from tests.grindstone.conftest import init_git_repo


def run_planner_boundary(
    *,
    job_spec: str,
    rig: str,
    baton: str = "",
    epoch_index: int = 1,
    max_epochs: int = 25,
    timeout: float = 900,
) -> Decision:
    """Drive one real planner boundary on ``rig`` and return the validated decision.

    Builds a throwaway committed repo + run dir, assembles the boundary
    ``PlannerContext`` (the integration tip = the repo HEAD, an empty keyed log, the
    given prior-epoch baton), and runs the real ``ScriptPlanner`` once. Returns the
    gate-clean typed ``Decision`` (``ScriptPlanner.decide`` raises ``PlannerError`` /
    ``RateLimited`` if the rig cannot produce one)."""

    script = resolve_role_script("planner", RoleConfig(rig=rig, slots=1, timeout_s=timeout))
    stop = models_script("stop.sh", rig=rig)

    with tempfile.TemporaryDirectory(prefix="gs-eval-boundary-") as scratch_str:
        scratch = Path(scratch_str)
        repo = init_git_repo(scratch / "repo")
        run_dir = RunDir(root=scratch / "rundir")
        run_dir.root.mkdir(parents=True)

        tip = wt.head_commit(repo)
        context = PlannerContext(
            job=job_spec,
            repo=repo,
            run_dir=run_dir,
            run_branch=f"grind/{run_dir.root.name}",
            tip_ref=tip,
            log_index=(),
            baton=baton,
            epoch_index=epoch_index,
            max_epochs=max_epochs,
        )
        transport = ScriptPlannerTransport(
            script=script, stop_script=stop, repo=repo, slots=1, timeout_s=timeout
        )
        planner = ScriptPlanner(transport=transport)
        try:
            return planner.decide(context)
        finally:
            # Unregister the in-repo _planner_tip worktree before the tempdir is torn
            # down (it lives under run_dir.root, inside the repo's worktree registry).
            tip_wt = run_dir.root / "_planner_tip"
            if tip_wt.is_dir():
                wt.remove_worktree(repo, tip_wt)
