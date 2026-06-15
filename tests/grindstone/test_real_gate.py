"""S2 GATE (marker ``real``, excluded by default): a 3-task implement epoch on a
throwaway git repo, driven by real ``ScriptWorker`` (local role script) workers
against a live llama-server, concurrency 2.

Three disjoint write tasks (alpha/beta/gamma) each create a distinct file with
distinct real content. The full epoch path must hold against real models: the
disk contract, per-attempt worktrees, the ownership scope check, commit-on-
success, fast-forward integration, the journal, and outcome.json. Probe /health
first and skip loudly if the server is down. Endpoint via
GRINDSTONE_TEST_ENDPOINT (default :8082, the coder GPU).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import pytest

from grindstone.contracts.models import CmdCheck, ImplementTask
from grindstone.events import RunStarted, read_events, replay
from grindstone.rundir import create_run_dir
from grindstone.script_worker import ScriptWorker

from tests.grindstone.conftest import (
    implement_epoch,
    init_git_repo,
    run_one_epoch,
    tracked_files,
)

pytestmark = pytest.mark.real

# The role script (models/local_request.sh) owns transport/model/GPU; the test
# only points at it. parents[2] = repo root (this file is <root>/tests/grindstone/).
_MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
DEFAULT_ENDPOINT = "http://localhost:8082"

_SPEC = [("T1", "alpha.txt", "ALPHA"), ("T2", "beta.txt", "BETA"), ("T3", "gamma.txt", "GAMMA")]


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


def _task(task_id: str, fname: str, content: str) -> ImplementTask:
    return ImplementTask(
        id=task_id,
        goal=(
            f"Create a file named {fname} in the current directory whose contents "
            f"are exactly the word {content}. Then write handoff.json citing {fname}."
        ),
        done_when=[
            CmdCheck(cmd=f"test -f {fname}"),
            CmdCheck(cmd=f"grep -q {content} {fname}"),
        ],
        file_ownership=[fname],
    )


def test_real_pi_three_task_epoch(tmp_path: Path) -> None:
    endpoint = _endpoint()
    if not _healthy(endpoint):
        pytest.skip(
            f"GATE SKIPPED: no healthy llama-server at {endpoint} "
            "(start scripts/launch_coder.sh; set GRINDSTONE_TEST_ENDPOINT to override)"
        )
    repo = init_git_repo(tmp_path / "repo")
    run_dir = create_run_dir(repo, "real-gate")
    tasks = [_task(*spec) for spec in _SPEC]
    worker = ScriptWorker(
        script=_MODELS_DIR / "local_request.sh",
        slots=2,
        timeout_s=1800.0,
        log_root=run_dir.root / "worker_logs",
    )

    started = time.monotonic()
    outcome = run_one_epoch(
        run_dir,
        args=implement_epoch(*tasks),
        mode="implement",
        ladder=[("local", worker)],
        repo=repo,
        concurrency=2,
    )
    elapsed = time.monotonic() - started

    detail = {
        t.task_id: (t.status, t.attempts, t.tier, t.failure_reason) for t in outcome.tasks
    }
    assert outcome.status == "completed", (
        f"real gate failed after {elapsed:.1f}s: {detail}; integration={outcome.integration}"
    )
    assert [t.status for t in outcome.tasks] == ["done", "done", "done"], detail

    branch = outcome.integration.branch
    assert branch is not None
    files = set(tracked_files(repo, branch))
    assert {"alpha.txt", "beta.txt", "gamma.txt"} <= files, files

    events = read_events(run_dir.events_path)
    assert sum(isinstance(e, RunStarted) for e in events) == 1
    tree = replay(events)
    assert tree.status == "completed"
    assert {t.status for t in tree.phases[0].epochs[0].tasks} == {"done"}

    payload = json.loads(run_dir.resolve("P1/E1/outcome.json").read_text())
    assert payload["status"] == "completed"
    assert payload["integration"]["merged"] == ["T1", "T2", "T3"]
