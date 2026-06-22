"""S5 E2E GATE (marker ``real``, excluded by default), REQUIRED to PASS.

The CONFIG-PATH gate: a full job driven through the CLI exactly as an operator
would, ``grindstone init`` scaffolds ``.grindstone/config.yaml``, then
``grindstone run`` loads a local-only role config, builds the REAL planner +
REAL ``ScriptWorker`` from it, and runs the S4 toy job to completion. This proves
the init → config → CLI → loop plumbing (ruling 5), not new orchestration; the
job shape is S4's.

It is parametrized over the planner rig so the SAME toy job, config builder,
valve, and assertions cover BOTH planner presets end to end: ``codex`` (override
preset, ``codex exec``) and ``claude`` (the SHIPPED DEFAULT,
``models/claude/planner_request.sh`` runs ``claude -p``). Each param proves its
planner emits schema-conforming epoch decisions across the FULL lifecycle
(propose_skeleton, implement, complete_run), not just one call. Only the planner
script path + required CLI differ per param; the local llama-server tier is the
same. Each param skips loudly if its planner CLI is absent, so codex coverage
stays intact even when ``claude`` is unavailable and vice versa.

The 12-planner-call valve is TEST-only and is threaded by monkeypatching the
``run_grind`` symbol the CLI calls (the CLI exposes no valve flag, it is not a
real loop bound), so the live wiring path stays intact while quota stays capped.

Run: ``rtk proxy python3 -m pytest tests/grindstone/test_real_gate_s5.py -m real -s``
Needs a healthy llama-server (GRINDSTONE_TEST_ENDPOINT, default :8080) AND the
selected planner CLI (``codex`` and/or ``claude``); skips loudly if absent.
"""

from __future__ import annotations

import functools
import os
import shutil
import subprocess
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import pytest

from grindstone import cli
from grindstone.cli import main
from grindstone.rundir import RunDir
from grindstone.run_loop import RunState, run_grind

from tests.grindstone.conftest import init_git_repo, tracked_files

pytestmark = pytest.mark.real

DEFAULT_ENDPOINT = "http://localhost:8080"
#: The rig's role scripts (this repo's sibling models/ folder). The script owns
#: provider/model/GPU now, the gate only needs the path, not the model identity.
_MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
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


def _config_yaml(models_dir: Path, planner_script: str) -> str:
    """A local-only role config pinned to this rig's scripts (no senior tier, so
    a healthy-server toy job never escalates to an unauthed cloud rung). Only the
    planner script differs per rig, the local tier is the same llama-server path."""

    return (
        "roles:\n"
        "  planner:\n"
        f"    script: {models_dir}/{planner_script}\n"
        "    slots: 1\n"
        "    timeout_s: 600\n"
        "  worker:\n"
        f"    script: {models_dir}/personal/worker_request.sh\n"
        "    slots: 2\n"
        "    timeout_s: 1800\n"
    )


#: Each planner rig the gate proves end to end: (id, planner script, required CLI).
#: ``codex`` is a tracked preset, ``claude`` is the SHIPPED DEFAULT planner
#: (``models/claude/planner_request.sh`` runs ``claude -p``). Each param skips
#: loudly if its CLI is absent, so the codex coverage stays intact regardless.
_PLANNER_RIGS = [
    pytest.param("codex/planner_request.sh", "codex", id="codex"),
    pytest.param("claude/planner_request.sh", "claude", id="claude"),
]


@pytest.mark.parametrize("planner_script,planner_cli", _PLANNER_RIGS)
def test_gate_s5_cli_config_path_real(
    planner_script: str,
    planner_cli: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    endpoint = _endpoint()
    if shutil.which(planner_cli) is None:
        pytest.skip(f"GATE S5 SKIPPED: no `{planner_cli}` CLI on PATH")
    if not _healthy(endpoint):
        pytest.skip(f"GATE S5 SKIPPED: no healthy llama-server at {endpoint}")

    repo = init_git_repo(tmp_path / "repo")
    # init via the CLI (scaffold config + gitignore), then write the run config
    # the gate needs (local-only, pinned to this rig's role scripts).
    assert main(["init", "--repo", str(repo)]) == 0
    (repo / ".grindstone" / "config.yaml").write_text(
        _config_yaml(_MODELS_DIR, planner_script), encoding="utf-8"
    )
    (repo / "job.md").write_text(_JOB, encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=str(repo), check=True, capture_output=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "job"],
        cwd=str(repo), check=True, capture_output=True,
    )

    # Thread the TEST-only valve onto the run_grind the CLI calls (no valve flag).
    monkeypatch.setattr(
        cli, "run_grind", functools.partial(run_grind, max_planner_calls=PLANNER_VALVE)
    )

    run_id = f"s5-gate-{planner_cli}"
    started = time.monotonic()
    code = main(["run", str(repo / "job.md"), "--repo", str(repo), "--run-id", run_id])
    elapsed = time.monotonic() - started

    run_dir = RunDir(root=repo / ".grindstone" / "runs" / run_id)
    state = RunState.model_validate_json(run_dir.run_state_path.read_text())
    files = sorted(tracked_files(repo, state.last_integration_branch)) if state.last_integration_branch else []
    print(
        f"\n[S5 GATE {planner_cli}] {elapsed:.1f}s exit={code} status={state.status} "
        f"planner_calls={state.planner_call_count} branch={state.last_integration_branch}"
    )
    print(f"[S5 GATE {planner_cli}] final branch files: {files}")
    print(f"[S5 GATE {planner_cli}] terminal reason: {state.terminal_reason}")

    assert code == 0, (
        f"S5 gate ({planner_cli}) exit code {code}, "
        f"status={state.status} reason={state.terminal_reason}"
    )
    assert state.status == "completed"
    assert {"greet.py", "test_greet.py", "README.md"} <= set(files), files
    # The post-mortem journal was rendered from the real run's events.
    assert run_dir.journal_path.exists(), "journal.md not written"
    journal = run_dir.journal_path.read_text(encoding="utf-8")
    assert f"# Run {run_id}" in journal and "completed" in journal, journal[:200]
