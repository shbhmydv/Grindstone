"""The real rig dispatch (``ScriptWorker``) + the ``build_backends`` wiring.

Driven by tiny FAKE ``*_request.sh`` scripts (no model): they honor the file
contract (write the disk artifact into ``--worktree``) so the subprocess dispatch,
the rate-limit mapping, and the timeout supervisor are exercised end-to-end.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from grindstone.contracts.models import Task, parse_handoff
from grindstone.script_worker import ScriptWorker, build_backends
from grindstone.worker import RateLimited, TransportError, WorkerRequest

_HANDOFF_JSON = """{{"schema_version":"1","task_id":"{tid}","status":"DONE",
"resulting_state":"fake worker done","what_changed":[],"downstream_needs":[],
"not_done":[],"citations":[],"checks":[{{"check":"x","exit_code":0}}],
"occupancy":{{"compacted":false,"subagent_splits":0}}}}"""


def _script(path: Path, body: str) -> Path:
    path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def _arg(name: str) -> str:
    # Parse --worktree out of "$@" in bash for the fake scripts.
    return (
        'wt=""; while [[ $# -gt 0 ]]; do case "$1" in '
        f'--{name}) wt="$2"; shift 2;; *) shift;; esac; done\n'
    )


def _request(scratch: Path) -> WorkerRequest:
    task = Task(id="T1", mode="implement", goal="x", file_ownership=["a.py"])
    return WorkerRequest(
        task=task, task_id="P1/E1/T1", mode="implement", scratch=scratch
    )


def _worker(script: Path, tmp_path: Path, timeout_s: float = 30.0) -> ScriptWorker:
    return ScriptWorker(
        script=script,
        stop_script=tmp_path / "stop.sh",  # absent -> in-process kill fallback
        timeout_s=timeout_s,
        log_root=tmp_path / "logs",
    )


def test_script_worker_writes_handoff(tmp_path: Path) -> None:
    body = _arg("worktree") + (
        "cat > \"$wt/handoff.json\" <<'EOF'\n"
        + _HANDOFF_JSON.format(tid="P1/E1/T1")
        + "\nEOF\n"
    )
    script = _script(tmp_path / "worker_request.sh", body)
    scratch = tmp_path / "wt"
    scratch.mkdir()
    _worker(script, tmp_path).run(_request(scratch))
    handoff = parse_handoff(
        __import__("json").loads((scratch / "handoff.json").read_text())
    )
    assert handoff.status == "DONE"


def test_script_worker_maps_rate_limit(tmp_path: Path) -> None:
    script = _script(
        tmp_path / "worker_request.sh",
        'echo "Error: 429 rate limit exceeded" >&2\nexit 1\n',
    )
    scratch = tmp_path / "wt"
    scratch.mkdir()
    with pytest.raises(RateLimited):
        _worker(script, tmp_path).run(_request(scratch))


def test_script_worker_plain_failure_is_transport_error(tmp_path: Path) -> None:
    script = _script(tmp_path / "worker_request.sh", 'echo boom >&2\nexit 3\n')
    scratch = tmp_path / "wt"
    scratch.mkdir()
    with pytest.raises(TransportError):
        _worker(script, tmp_path).run(_request(scratch))


def test_script_worker_timeout(tmp_path: Path) -> None:
    script = _script(tmp_path / "worker_request.sh", "sleep 10\n")
    scratch = tmp_path / "wt"
    scratch.mkdir()
    with pytest.raises(TransportError):
        _worker(script, tmp_path, timeout_s=1.0).run(_request(scratch))


def test_build_backends_shares_endpoint_for_fallback_senior(tmp_path: Path) -> None:
    # An all-local rig (no senior role): both tiers must resolve to the SAME endpoint
    # so the single local slot is never double-booked.
    from grindstone.config import GrindstoneConfig

    config = GrindstoneConfig.model_validate(
        {
            "roles": {
                "planner": {"rig": "local", "slots": 1, "timeout_s": 60},
                "worker": {"rig": "local", "slots": 1, "timeout_s": 60},
            }
        }
    )
    backends = build_backends(config, log_root=tmp_path / "logs")
    # Acquire sequentially (the shared 1-slot semaphore would deadlock if nested):
    # both tiers must hand back the SAME transport object (one endpoint, one slot).
    with backends.slot("local") as a:
        local_transport = a
    with backends.slot("senior") as b:
        senior_transport = b
    assert local_transport is senior_transport
