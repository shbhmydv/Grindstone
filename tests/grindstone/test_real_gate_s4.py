"""S4 E2E GATE (marker ``real``, excluded by default), REQUIRED to PASS.

A full job end-to-end: real ``codex exec`` planner + real ``ScriptWorker`` workers
(:8082, concurrency 2) on a throwaway repo, with a job that needs >=2 phases of
real work, P1 builds ``greet.py`` + a passing ``test_greet.py`` (run with
``python3 -m pytest``); P2 documents it in ``README.md``. The run MUST reach
``complete_run`` with verified evidence. The 12-planner-call valve is TEST-only.

The gate is also the S3 Gate-B flakiness diagnosis: it captures every planner
decision (tool + epoch title), every task outcome with per-attempt failure
reasons, total wall time, planner_calls, and the final branch listing, printed
with ``-s`` so the evidence trail survives even on success.

Run: ``rtk proxy python3 -m pytest tests/grindstone/test_real_gate_s4.py -m real -s``
Needs a healthy llama-server (GRINDSTONE_TEST_ENDPOINT, default :8082) AND an
authed ``codex`` CLI; skips loudly if either is absent.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import pytest

from grindstone.events import (
    EpochStarted,
    HandoffRejected,
    PhaseEscalated,
    PhasePassed,
    PlannerCallSucceeded,
    TaskFailed,
    read_events,
    replay,
)
from grindstone.rundir import create_run_dir
from grindstone.run_loop import RunState, run_grind
from grindstone.script_planner import ScriptPlanner
from grindstone.script_worker import ScriptWorker

from tests.grindstone.conftest import init_git_repo, tracked_files

pytestmark = pytest.mark.real

# The role scripts (models/*.sh) own transport/model/GPU; the test only points
# at them. parents[2] = repo root (this file is <root>/tests/grindstone/).
_MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
DEFAULT_ENDPOINT = "http://localhost:8082"
PLANNER_VALVE = 12  # TEST-only safety valve

_JOB = (
    "Goal: build and document a tiny Python greeting module, across TWO phases.\n"
    "\n"
    "Phase 1, build: create `greet.py` defining a function `greet(name)` that\n"
    "returns the string `Hello, <name>!`, and a `test_greet.py` with a passing\n"
    "test for it. The tests must pass when run with `python3 -m pytest -q`.\n"
    "\n"
    "Phase 2, document: create `README.md` documenting greet.py and how to run\n"
    "its test.\n"
    "\n"
    "Each phase's exit_criterion must be deterministic checks (a command with an\n"
    "expected exit code, or a required artifact). When both phases are done,\n"
    "complete_run with evidence that re-runs the test and checks README.md exists.\n"
)


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


def _codex_available() -> bool:
    return shutil.which("codex") is not None


def test_gate_s4_full_job_real_codex_and_pi(tmp_path: Path) -> None:
    endpoint = _endpoint()
    if not _codex_available():
        pytest.skip("GATE S4 SKIPPED: no `codex` CLI on PATH")
    if not _healthy(endpoint):
        pytest.skip(f"GATE S4 SKIPPED: no healthy llama-server at {endpoint}")

    repo = init_git_repo(tmp_path / "repo")
    (repo / "job.md").write_text(_JOB, encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=str(repo), check=True, capture_output=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "job"],
        cwd=str(repo), check=True, capture_output=True,
    )

    run_dir = create_run_dir(repo, "s4-gate")
    planner = ScriptPlanner(
        script=_MODELS_DIR / "codex" / "planner_request.sh",
        stop_script=_MODELS_DIR / "_common" / "stop.sh",
        repo=repo, slots=1, timeout_s=600.0
    )
    worker = ScriptWorker(
        script=_MODELS_DIR / "personal" / "worker_request.sh",
        stop_script=_MODELS_DIR / "_common" / "stop.sh",
        slots=2,
        timeout_s=1800.0,
        log_root=run_dir.root / "worker_logs",
    )

    started = time.monotonic()
    outcome = run_grind(
        run_dir,
        job_path=str(repo / "job.md"),
        planner=planner,
        ladder=[("worker", worker)],
        repo=repo,
        concurrency=2,
        max_planner_calls=PLANNER_VALVE,
    )
    elapsed = time.monotonic() - started

    events = read_events(run_dir.events_path)
    tree = replay(events)
    state = RunState.model_validate_json(run_dir.run_state_path.read_text())

    # --- evidence capture (the deliverable, even on success) -------------------
    titles = {e.epoch_id: e.title for e in events if isinstance(e, EpochStarted)}
    decisions = [e.tool for e in events if isinstance(e, PlannerCallSucceeded)]
    print(f"\n[S4 GATE] {elapsed:.1f}s status={outcome.status} "
          f"planner_calls={outcome.planner_calls} epochs={outcome.epochs_run} "
          f"branch={outcome.final_branch}")
    print(f"[S4 GATE] decision tools: {decisions}")
    for phase in tree.phases:
        print(f"[S4 GATE] phase {phase.id} ({phase.title}) {phase.status}")
        for ep in phase.epochs:
            print(f"[S4 GATE]   epoch {ep.id} ({titles.get(ep.id, ep.title)}) {ep.status}: "
                  + ", ".join(f"{t.id}={t.status}/a{t.attempt}" for t in ep.tasks))
    passed = [e.phase_id for e in events if isinstance(e, PhasePassed)]
    escalated = [e.phase_id for e in events if isinstance(e, PhaseEscalated)]
    print(f"[S4 GATE] phases passed: {passed}  escalated: {escalated}")
    # Per-task / per-attempt failure reasons (diagnoses S3 Gate-B flakiness).
    for e in events:
        if isinstance(e, HandoffRejected):
            print(f"[S4 GATE]   reject {e.epoch_id}/{e.task_id}: {e.reason}")
        elif isinstance(e, TaskFailed):
            print(f"[S4 GATE]   FAILED {e.epoch_id}/{e.task_id}")
    print(f"[S4 GATE] terminal reason: {state.terminal_reason}")
    if outcome.final_branch is not None:
        print(f"[S4 GATE] final branch files: {tracked_files(repo, outcome.final_branch)}")

    # --- the gate: completion with verified evidence ---------------------------
    assert outcome.status == "completed", (
        f"S4 gate did not complete: status={outcome.status} reason={outcome.reason}"
    )
    assert state.skeleton is not None and len(state.skeleton) >= 2, "fewer than two phases"
    assert "complete_run" in decisions, "never reached complete_run"
    assert outcome.final_branch is not None
    files = set(tracked_files(repo, outcome.final_branch))
    assert {"greet.py", "test_greet.py", "README.md"} <= files, files
