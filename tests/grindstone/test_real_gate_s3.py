"""S3 GATES (marker ``real``, excluded by default). Both cost real quota.

Gate A: ONE real ``codex exec`` call with a constructed S3 input for a toy job
        → a schema-valid decision comes back (any tool).
Gate B: a full ``run_grind`` on a throwaway git repo, real codex planner + real
        pi workers (:8082, concurrency 2), toy two-file job → run to terminal.

Run: ``rtk proxy python3 -m pytest tests/grindstone/test_real_gate_s3.py -m real -s``
Gate B needs a healthy llama-server (GRINDSTONE_TEST_ENDPOINT, default :8082);
it skips loudly if absent. The safety valve (6 planner calls / 4 epochs) is
TEST-harness only.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import pytest

from grindstone.contracts.gate import decision_schema_errors
from grindstone.contracts.models import parse_decision
from grindstone.events import read_events, replay
from grindstone.planner import build_planner_input, extract_decision_json
from grindstone.rundir import create_run_dir
from grindstone.run_loop import RunState, run_grind
from grindstone.script_planner import ScriptPlanner
from grindstone.script_worker import ScriptWorker

from tests.grindstone.conftest import init_git_repo, tracked_files

pytestmark = pytest.mark.real

# The role scripts (models/*.sh) own transport/model/GPU; the tests only point
# at them. parents[2] = repo root (this file is <root>/tests/grindstone/).
_MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
DEFAULT_ENDPOINT = "http://localhost:8082"
_JOB = (
    "Goal: create exactly two small text files at the repo root via INDEPENDENT "
    "tasks, then complete the run.\n"
    "- file `alpha.txt` must contain the single word ALPHA.\n"
    "- file `beta.txt` must contain the single word BETA.\n"
    "The two files have disjoint ownership and can be built in one epoch (or "
    "two). When both exist, complete_run with evidence that greps each file.\n"
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


# --- Gate A: one real codex decision -------------------------------------------


def test_gate_a_codex_returns_schema_valid_decision(tmp_path: Path) -> None:
    repo = init_git_repo(tmp_path / "repo")
    transport = ScriptPlanner(
        script=_MODELS_DIR / "codex" / "planner_request.sh",
        stop_script=_MODELS_DIR / "_common" / "stop.sh",
        repo=repo, slots=1, timeout_s=300.0
    )
    prompt = build_planner_input(
        job=_JOB, skeleton=None, phase_id=None, epoch_counter=0,
        log_index=[], last_epoch_rows=None, reask_errors=[],
    )
    started = time.monotonic()
    raw = transport.plan(prompt)
    elapsed = time.monotonic() - started

    json_text = extract_decision_json(raw)
    print(f"\n[GATE A] codex call took {elapsed:.1f}s")
    print(f"[GATE A] extracted decision JSON:\n{json_text}")
    assert json_text is not None, f"no decision JSON extractable from:\n{raw!r}"
    payload = json.loads(json_text)
    errors = decision_schema_errors(payload)
    assert errors == [], f"schema errors: {errors}\npayload: {payload}"
    decision = parse_decision(payload)
    print(f"[GATE A] tool = {decision.tool}")


# --- Gate B: full run, real codex planner + real pi workers --------------------


def test_gate_b_full_run_real_codex_and_pi(tmp_path: Path) -> None:
    endpoint = _endpoint()
    if not _healthy(endpoint):
        pytest.skip(f"GATE B SKIPPED: no healthy llama-server at {endpoint}")
    repo = init_git_repo(tmp_path / "repo")
    (repo / "job.md").write_text(_JOB, encoding="utf-8")
    import subprocess

    subprocess.run(["git", "add", "-A"], cwd=str(repo), check=True, capture_output=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "job"],
        cwd=str(repo), check=True, capture_output=True,
    )
    run_dir = create_run_dir(repo, "s3-gate-b")
    planner = ScriptPlanner(
        script=_MODELS_DIR / "codex" / "planner_request.sh",
        stop_script=_MODELS_DIR / "_common" / "stop.sh",
        repo=repo, slots=1, timeout_s=300.0
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
        max_planner_calls=6,  # TEST-harness safety valve
        max_epochs=4,
    )
    elapsed = time.monotonic() - started

    events = read_events(run_dir.events_path)
    tree = replay(events)
    state = RunState.model_validate_json(run_dir.run_state_path.read_text())
    print(f"\n[GATE B] {elapsed:.1f}s status={outcome.status} planner_calls="
          f"{outcome.planner_calls} epochs={outcome.epochs_run} branch={outcome.final_branch}")
    for phase in tree.phases:
        print(f"[GATE B] phase {phase.id} ({phase.title}) {phase.status}")
        for ep in phase.epochs:
            print(f"[GATE B]   epoch {ep.id} ({ep.title}) {ep.status}: "
                  + ", ".join(f"{t.id}={t.status}/a{t.attempt}" for t in ep.tasks))
    decisions = [(e.event, getattr(e, "tool", getattr(e, "classification", "")))
                 for e in events if e.event.startswith("planner_call")]
    print(f"[GATE B] planner calls: {decisions}")
    print(f"[GATE B] terminal reason: {state.terminal_reason}")
    if outcome.status == "completed":
        files = set(tracked_files(repo, outcome.final_branch or "HEAD"))
        print(f"[GATE B] COMPLETED. final branch files: {sorted(files)}")
    else:
        print(f"[GATE B] stopped at: {outcome.reason}")

    # S3's contract is that the LOOP works end-to-end against the REAL planner and
    # REAL workers: a skeleton is proposed, epochs are planned + integrated, and
    # the run reaches a terminal RunOutcome (never hangs/crashes). Whether the
    # local worker lands every toy task within the brief's tight test valve is a
    # model-capability matter, not the loop's contract, so completion is the
    # logged goal, while the gate asserts the invariants S3 owns.
    assert outcome.status in {"completed", "escalated", "failed"}
    assert state.skeleton is not None, "planner never proposed a skeleton"
    assert outcome.epochs_run >= 1, "no epoch was planned + run"
    assert outcome.final_branch is not None, "no epoch integrated onto a branch"
    assert replay(events).status in {"completed", "escalated", "running"}
    if outcome.status == "completed":
        files = set(tracked_files(repo, outcome.final_branch))
        assert {"alpha.txt", "beta.txt"} <= files, files
