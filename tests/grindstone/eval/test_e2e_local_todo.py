"""Full-run local E2E: build the TODO app with EVERY role on ``rig: local``.

The owner's explicit Batch-2 ask: one end-to-end grindstone run that builds a real
TODO-app spec with the planner, worker, AND senior all pinned to the all-local Qwen
rig (``rig: local``), then proves the PRODUCED app actually works, not that it
matches a golden tree. Structurally it mirrors ``test_real_gate_s5`` (init -> config
-> CLI ``run`` -> assert on the integration branch), but it is a REAL build (not the
S4 toy job) and it is NOT parametrized over a rig: the whole point is to exercise the
local floor as planner+worker+senior in one run.

It is ``@pytest.mark.eval`` (excluded from the default suite) and skips unless
``GRINDSTONE_EVAL_RIG`` names ``local`` AND a healthy llama-server is reachable. The
parent runs the live baseline. The per-run planner valve is generous (a real build
takes more boundaries than the S4 toy), but still bounded so an unattended spin
cannot drain anything.
"""

from __future__ import annotations

import functools
import os
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import pytest

from grindstone import cli
from grindstone import worktree as wt
from grindstone.cli import main
from grindstone.rundir import RunDir
from grindstone.run_loop import RunState, run_grind

from tests.grindstone.conftest import init_git_repo, tracked_files
from tests.grindstone.eval.conftest import EVAL_RIG_ENV, _enabled_rigs
from tests.grindstone.eval.test_planner_eval import TODO_APP_JOB

pytestmark = pytest.mark.eval

DEFAULT_ENDPOINT = "http://localhost:8080"
#: Generous per-run planner-call valve: a real multi-phase build needs more
#: boundaries than the S4 toy (skeleton + several epochs + verification + complete),
#: but still bounded so an unattended revision spin cannot drain the subscription.
PLANNER_VALVE = 40


def _endpoint() -> str:
    return os.environ.get("GRINDSTONE_TEST_ENDPOINT", DEFAULT_ENDPOINT)


def _healthy(endpoint: str) -> bool:
    for path in ("/health", "/healthz"):
        try:
            with urlopen(endpoint.rstrip("/") + path, timeout=2.0) as resp:
                if 200 <= resp.status < 300:
                    return True
        except (OSError, URLError):
            continue
    return False


def _config_yaml() -> str:
    """A config pinning planner + worker + senior all to the all-local Qwen rig via
    the ``rig:`` shorthand (each role resolves ``models/local/<role>_request.sh``).
    Generous wall-clock per role for a real build on the local stack."""

    return (
        "roles:\n"
        "  planner:\n"
        "    rig: local\n"
        "    slots: 1\n"
        "    timeout_s: 1200\n"
        "  worker:\n"
        "    rig: local\n"
        "    slots: 2\n"
        "    timeout_s: 1800\n"
        "  senior:\n"
        "    rig: local\n"
        "    slots: 1\n"
        "    timeout_s: 1800\n"
    )


def _run(checkout: Path, *args: str, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=str(checkout),
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )


def test_e2e_local_todo_app_works(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if "local" not in _enabled_rigs():
        pytest.skip(
            f"local rig not enabled; set {EVAL_RIG_ENV}=local to drive this E2E live"
        )
    endpoint = _endpoint()
    if not _healthy(endpoint):
        pytest.skip(f"E2E SKIPPED: no healthy llama-server at {endpoint}")

    repo = init_git_repo(tmp_path / "repo")
    assert main(["init", "--repo", str(repo)]) == 0
    (repo / ".grindstone" / "config.yaml").write_text(_config_yaml(), encoding="utf-8")
    (repo / "job.md").write_text(TODO_APP_JOB, encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=str(repo), check=True, capture_output=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "job"],
        cwd=str(repo), check=True, capture_output=True,
    )

    # Thread the generous TEST valve onto the run_grind the CLI calls (no valve flag).
    monkeypatch.setattr(
        cli, "run_grind", functools.partial(run_grind, max_planner_calls=PLANNER_VALVE)
    )

    run_id = "e2e-local-todo"
    started = time.monotonic()
    code = main(["run", str(repo / "job.md"), "--repo", str(repo), "--run-id", run_id])
    elapsed = time.monotonic() - started

    run_dir = RunDir(root=repo / ".grindstone" / "runs" / run_id)
    state = RunState.model_validate_json(run_dir.run_state_path.read_text())
    branch = state.last_integration_branch
    files = sorted(tracked_files(repo, branch)) if branch else []
    print(
        f"\n[E2E local todo] {elapsed:.1f}s exit={code} status={state.status} "
        f"planner_calls={state.planner_call_count} branch={branch}"
    )
    print(f"[E2E local todo] final branch files: {files}")
    print(f"[E2E local todo] terminal reason: {state.terminal_reason}")

    assert code == 0, (
        f"E2E run exit {code}, status={state.status} reason={state.terminal_reason}"
    )
    assert state.status == "completed", state.terminal_reason
    assert branch, "run completed but recorded no integration branch"
    # PROPERTY (not golden): a `todo/` package landed in the integration result.
    assert any(p == "todo" or p.startswith("todo/") for p in files), files

    # Materialize the produced tree to PROVE the app works: a detached worktree of
    # the integration tip (read-only, never touches the operator checkout).
    checkout = tmp_path / "checkout"
    wt.add_worktree_detached(repo, checkout, ref=branch)
    try:
        env = {**os.environ, "PYTHONPATH": str(checkout), "PYTHONDONTWRITEBYTECODE": "1"}

        # The app's OWN pytest suite (the job mandates one) passes against the tree.
        suite = _run(checkout, "-m", "pytest", "-q", env=env)
        assert suite.returncode == 0, (
            f"the produced app's pytest suite failed:\n{suite.stdout}\n{suite.stderr}"
        )

        # `todo add` / `todo list` / `todo done` work end to end on the produced tree.
        add = _run(checkout, "-m", "todo", "add", "write the tests", env=env)
        assert add.returncode == 0, f"`todo add` failed:\n{add.stdout}\n{add.stderr}"

        listing = _run(checkout, "-m", "todo", "list", env=env)
        assert listing.returncode == 0, f"`todo list` failed:\n{listing.stderr}"
        assert "write the tests" in listing.stdout, listing.stdout

        done = _run(checkout, "-m", "todo", "done", "1", env=env)
        assert done.returncode == 0, f"`todo done` failed:\n{done.stdout}\n{done.stderr}"
    finally:
        wt.remove_worktree(repo, checkout)
