"""Replicate ONE real worker task the way ``task_loop`` does, for the eval.

``run_worker_task`` is the worker-corpus analogue of ``_boundary.run_planner_boundary``:
given a typed task + a rig, it sets up the SAME attempt context ``task_loop._dispatch_attempt``
sets up (a real git worktree for an ``ImplementTask``, a plain scratch dir for a
non-implement task, ``check_handoff.py`` armed in the worker CWD), invokes the rig's
real ``worker_request.sh`` through the SAME production transport (``ScriptWorker``),
reads back ``handoff.json``, and gates it through the SAME production gate the loop
applies (``task_loop._collect_handoff``: schema + typed parse + semantic rules +
grounding + a re-run of the task's ``done_when`` in the worktree). It returns the
validated, typed ``Handoff`` (or raises ``WorkerBoundaryError`` with the raw handoff +
stderr for debugging).

WHAT IS REUSED vs DUPLICATED (the ``_boundary.py`` discipline):

  * REUSED (no drift): ``ScriptWorker`` (the exact production transport the CLI
    ladder builds), ``task_loop._install_attempt_checks`` (arms ``check_handoff.py``
    + appends the validator/review done_when), and ``task_loop._collect_handoff``
    (the authoritative handoff gate + done_when re-run, raising ``_AttemptFailed``).
    These need only a ``RunDir`` + a ``TaskIdentity`` + a scratch path, all cheap to
    construct here, so reusing them is strictly better than re-implementing the gate
    (the planner boundary had to duplicate because its run-loop helpers needed a live
    run state in flight; the worker helpers do not).
  * DUPLICATED (the small attempt scaffold ``_dispatch_attempt`` owns and a live run
    state would otherwise supply): the worktree/scratch setup + epoch base capture,
    the ``WorkerRequest`` assembly, and the rig-script wiring (a ``ScriptWorker`` over
    the resolved ``worker_request.sh`` with a log root). The harness deliberately does
    NOT commit/scope-check/publish the artifact (the parts of ``_dispatch_attempt``
    AFTER the gate): the corpus judges the worker's produced handoff + tree, not the
    core's git integration, which the epoch-loop tests already cover.

``mode`` is a REQUIRED-IN-EFFECT input the task type alone cannot supply: an
``ArtifactTask`` backs research/review/artifact alike (``WorkerRequest`` carries the
mode for exactly this reason), so the corpus passes it explicitly. It defaults to
``implement`` for an ``ImplementTask`` and ``artifact`` for an ``ArtifactTask`` when
unset. No production code is touched.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from grindstone.config import models_script, resolve_role_script, RoleConfig
from grindstone.contracts.models import Handoff, ImplementTask
from grindstone.contracts.semantics import HandoffMode
from grindstone.rundir import RunDir
from grindstone.script_worker import ScriptWorker
from grindstone.worker import Task, WorkerRequest, WorkerTransport
from grindstone import worktree as wt

# Reuse the production attempt-scaffold + gate verbatim (zero drift). These are
# module-private in task_loop on purpose, importing them in a test helper is the
# anti-drift choice the eval philosophy prizes (reuse the real gate, never mirror it).
from grindstone.task_loop import (
    TaskIdentity,
    _AttemptFailed,
    _collect_handoff,
    _install_attempt_checks,
)


class WorkerBoundaryError(AssertionError):
    """A worker boundary that produced no gate-clean handoff.

    Carries the gate/transport failure reason plus the raw ``handoff.json`` the
    worker left on disk (or ``None``) and any captured stderr, so a failing eval is
    debuggable without a re-run."""

    def __init__(
        self, message: str, *, raw_handoff: str | None = None, stderr: str | None = None
    ) -> None:
        super().__init__(message)
        self.raw_handoff = raw_handoff
        self.stderr = stderr


def _default_mode(task: Task) -> HandoffMode:
    """The mode a task type implies when the caller leaves it unset."""

    return "implement" if isinstance(task, ImplementTask) else "artifact"


def _init_temp_repo(path: Path) -> Path:
    """A minimal committed git repo to root an implement worktree from.

    Mirrors ``_boundary._init_temp_repo``: the worker worktree is branched from this
    repo's tip (the epoch base analogue), so it only needs a born HEAD + the same
    ignore hygiene a real target repo carries (so a worker's ``__pycache__`` never
    reads as out-of-scope debris)."""

    path.mkdir(parents=True)
    (path / ".gitignore").write_text(".grindstone/\n__pycache__/\n", encoding="utf-8")
    (path / "README.md").write_text("# eval worker repo\n", encoding="utf-8")
    env_git = ["git", "-c", "user.email=eval@grindstone.test", "-c", "user.name=eval"]
    subprocess.run(["git", "init", "-q", "-b", "main", str(path)], check=True)
    subprocess.run(env_git + ["-C", str(path), "add", "."], check=True)
    subprocess.run(
        env_git + ["-C", str(path), "commit", "-q", "-m", "seed"], check=True
    )
    return path


def run_worker_task(
    *,
    task: Task,
    rig: str,
    mode: HandoffMode | None = None,
    repo: Path | None = None,
    inputs: dict[str, Path] | None = None,
    timeout: float = 900,
) -> Handoff:
    """Drive one real worker task on ``rig`` and return the validated handoff.

    Replicates ``task_loop._dispatch_attempt`` up to (not including) the git
    commit/scope/artifact-publish: set up the attempt CWD (a fresh worktree off a
    repo tip for an ``ImplementTask``, else a plain scratch dir), arm
    ``check_handoff.py`` + the validator/review done_when, invoke the rig's real
    ``worker_request.sh`` via ``ScriptWorker``, then gate the handoff through the
    production ``_collect_handoff`` (schema + parse + semantics + grounding + a
    done_when re-run in the worktree). Raises ``WorkerBoundaryError`` (with the raw
    handoff + stderr) when no gate-clean DONE handoff is produced.
    """

    effective_mode: HandoffMode = mode if mode is not None else _default_mode(task)
    implement = isinstance(task, ImplementTask)
    resolved_inputs = dict(inputs) if inputs is not None else {}

    with tempfile.TemporaryDirectory(prefix="gs-eval-worker-") as scratch_str:
        tmp = Path(scratch_str)
        run_dir = RunDir(root=tmp / "rundir")
        run_dir.root.mkdir()
        log_root = tmp / "worker_logs"
        scratch = tmp / "scratch"
        # A stable in-tree identity so the dispatched fully-qualified task_id is a
        # valid handoff task_id (P1/E1/T<n>); only the task id varies per task.
        identity = TaskIdentity(
            run_id="eval", phase_id="P1", epoch_id="E1", task_id=task.id
        )

        owns_repo = repo is None
        if implement:
            repo = repo if repo is not None else _init_temp_repo(tmp / "repo")
            base = wt.head_commit(repo)
            branch = f"grind/eval/{identity.fq}-a1"
            wt.add_worktree(repo, scratch, branch=branch, base=base)
        else:
            scratch.mkdir(parents=True, exist_ok=True)

        # Arm check_handoff.py + the augmented done_when EXACTLY as task_loop does
        # (non-implement gets the repo as a second citation root; implement bakes
        # None because its CWD already IS a repo checkout).
        augmented = _install_attempt_checks(
            task, scratch, effective_mode, identity.fq, repo
        )

        worker: WorkerTransport = ScriptWorker(
            script=resolve_role_script("worker", RoleConfig(rig=rig, slots=1, timeout_s=timeout)),
            stop_script=models_script("stop.sh"),
            slots=1,
            timeout_s=timeout,
            log_root=log_root,
        )
        request = WorkerRequest(
            task=augmented,
            task_id=identity.fq,
            inputs=resolved_inputs,
            scratch=scratch,
            attempt=1,
            failure_context=[],
            mode=effective_mode,
            repo_map=None,
        )

        try:
            worker.run(request)
        except Exception as exc:  # transport boundary (rate limit / process error / kill)
            raw = _read_raw_handoff(scratch)
            try:
                if implement and repo is not None:
                    wt.remove_worktree(repo, scratch)
            except Exception:
                pass
            raise WorkerBoundaryError(
                f"rig {rig!r} worker transport failed: {type(exc).__name__}: {exc}",
                raw_handoff=raw,
                stderr=str(exc),
            ) from exc

        try:
            handoff = _collect_handoff(
                request, augmented, effective_mode, run_dir, identity, repo
            )
        except _AttemptFailed as failure:
            raw = _read_raw_handoff(scratch)
            raise WorkerBoundaryError(
                f"rig {rig!r} handoff did not pass the gate: {failure.reason}",
                raw_handoff=raw,
                stderr=None,
            ) from failure
        finally:
            if implement and owns_repo is False and repo is not None:
                # Unregister a worktree we placed in a CALLER-owned repo (an
                # owned temp repo is cleaned wholesale with the tempdir).
                try:
                    wt.remove_worktree(repo, scratch)
                except Exception:
                    pass

        return handoff


def _read_raw_handoff(scratch: Path) -> str | None:
    """The raw ``handoff.json`` the worker left (for a failing-eval post-mortem)."""

    src = scratch / "handoff.json"
    if src.is_file():
        try:
            return src.read_text(encoding="utf-8")
        except OSError:
            return None
    return None
