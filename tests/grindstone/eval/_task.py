"""Drive ONE real worker task on a rig, through the production ``run_task``.

``run_worker_task`` is the worker analogue of ``run_planner_boundary``: given a typed
``Task`` + a rig, it builds the real per-backend ``Backends`` map (one ``ScriptWorker``
per endpoint, the same the CLI ladder builds) and runs the production ``run_task``
end to end, the isolated worktree / scratch, the rig dispatch, the disk-gate handoff
collection, and the independent tier-matched CRITIC. It returns the ``TaskResult`` so
the corpus can assert PROPERTIES (the handoff validated, the critic returned a
verdict). REUSES ``run_task`` + ``build_backends`` verbatim, so there is zero drift.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from grindstone import worktree as wt
from grindstone.config import GrindstoneConfig, RoleConfig, RolesConfig
from grindstone.contracts.models import Task
from grindstone.rundir import RunDir
from grindstone.script_worker import build_backends
from grindstone.worker import TaskResult, run_task
from tests.grindstone.conftest import init_git_repo


def run_worker_task(*, task: Task, rig: str, timeout: float = 900) -> TaskResult:
    """Drive one real worker task on ``rig`` through ``run_task`` and return its
    result.

    Builds a throwaway committed repo (the epoch base) + run dir + a ``Backends`` map
    whose ``worker`` (and fallback ``senior``) role is the given rig, then runs the
    production ``run_task`` for ``P1/E1/T1``. An implement task grinds in an isolated
    worktree off the repo tip; a non-write task grounds its citations against the repo.
    """

    config = GrindstoneConfig(
        roles=RolesConfig(
            planner=RoleConfig(rig=rig, slots=1, timeout_s=timeout),
            worker=RoleConfig(rig=rig, slots=1, timeout_s=timeout),
            senior=None,
        )
    )

    with tempfile.TemporaryDirectory(prefix="gs-eval-task-") as scratch_str:
        scratch = Path(scratch_str)
        repo = init_git_repo(scratch / "repo")
        run_dir = RunDir(root=scratch / "rundir")
        run_dir.root.mkdir(parents=True)
        log_root = scratch / "logs"
        log_root.mkdir()

        backends = build_backends(config, log_root=log_root)
        base = wt.head_commit(repo)
        return run_task(
            task, "P1/E1/T1", run_dir=run_dir, repo=repo, base=base, backends=backends
        )
