"""Session-limit handling: the long quota-window limit (Claude/codex session
limit) must PARK the run and retry hourly WITHOUT burning the transient /
rate-limit / attempt budgets, on both the planner and the worker path.

The incident: the claude CLI prints "You've hit your session limit · resets
2:20am (Asia/Kolkata)" to STDOUT (not stderr), and every classifier only
inspected stderr with a ``rate.?limit|429|quota|usage limit`` pattern that does
NOT match "session limit". So the long limit was misclassified as a transient
transport error and burned the retry budget in seconds. These tests pin the
detection, the new ``SessionLimited`` exception + classification, and the
hourly-park retry policy on both paths.
"""

from __future__ import annotations

import stat
from pathlib import Path

import pytest

from grindstone.planner import (
    MAX_SESSION_LIMIT_WAITS,
    SESSION_LIMIT_RETRY_S,
    RateLimited,
    SessionLimited,
    classify_failure,
)
from grindstone.script_planner import ScriptPlanner
from grindstone.script_worker import ScriptWorker
from grindstone.worker import is_session_limit
from grindstone.contracts.models import CmdCheck, ImplementTask
from grindstone.worker import WorkerRequest


# --- Layer 2: detection helper -------------------------------------------------


def test_is_session_limit_true_for_claude_stdout_signature() -> None:
    # The exact incident string the claude CLI prints to STDOUT.
    assert is_session_limit(
        "You've hit your session limit · resets 2:20am (Asia/Kolkata)"
    )


def test_is_session_limit_true_for_usage_limit() -> None:
    assert is_session_limit("usage limit reached")


def test_is_session_limit_true_for_limit_with_resets() -> None:
    # claude's alternate phrasing: a limit that "resets" at a wall-clock time.
    assert is_session_limit("Your limit will reset; resets at 9pm")


def test_is_session_limit_false_for_ordinary_text() -> None:
    assert not is_session_limit("some unexpected failure")
    assert not is_session_limit("")


def test_is_session_limit_false_for_plain_rate_limit_429() -> None:
    # A transient 429 must stay on the rate_limit (short backoff) path, NOT be
    # promoted to the hourly session-limit path.
    assert not is_session_limit("Error: rate limit exceeded (429)")


# --- Layer 1 + 3: exception + classification -----------------------------------


def test_session_limited_is_a_rate_limited_subclass() -> None:
    # Subclassing RateLimited keeps every existing ``except RateLimited`` catching
    # it; the policy layer distinguishes via isinstance.
    assert issubclass(SessionLimited, RateLimited)


def test_classify_failure_routes_session_limit_before_rate_limit() -> None:
    assert classify_failure(SessionLimited("session limit")) == "session_limit"
    # ordinary 429 still classifies as rate_limit (short backoff)
    assert classify_failure(RateLimited("429")) == "rate_limit"


def test_session_limit_retry_constants() -> None:
    assert SESSION_LIMIT_RETRY_S == 3600.0
    assert MAX_SESSION_LIMIT_WAITS == 24


# --- Layer 2: script_planner detection (the exact regression) ------------------


def _make_script(path: Path, body: str) -> Path:
    path.write_text("#!/usr/bin/env bash\n" + body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


def test_script_planner_session_limit_on_stdout_raises_session_limited(
    tmp_path: Path,
) -> None:
    # The regression: the session-limit line is printed to STDOUT (not stderr),
    # and the planner exits non-zero. It MUST raise SessionLimited, not fall
    # through to a transient TransportError.
    repo = tmp_path / "repo"
    repo.mkdir()
    script = _make_script(
        tmp_path / "planner_request.sh",
        "echo \"You've hit your session limit · resets 2:20am (Asia/Kolkata)\"\n"
        "exit 1\n",
    )
    planner = ScriptPlanner(
        script=script, stop_script=tmp_path / "stop.sh", repo=repo, slots=1, timeout_s=10.0
    )
    with pytest.raises(SessionLimited):
        planner.plan("p")


def test_script_planner_plain_rate_limit_stays_rate_limited(tmp_path: Path) -> None:
    # Guard against over-promotion: a plain 429 on stdout/stderr must stay
    # RateLimited (and NOT a SessionLimited).
    repo = tmp_path / "repo"
    repo.mkdir()
    script = _make_script(
        tmp_path / "planner_request.sh",
        'echo "Error: rate limit exceeded (429)" >&2\nexit 1\n',
    )
    planner = ScriptPlanner(
        script=script, stop_script=tmp_path / "stop.sh", repo=repo, slots=1, timeout_s=10.0
    )
    with pytest.raises(RateLimited) as ei:
        planner.plan("p")
    assert not isinstance(ei.value, SessionLimited)


# --- Layer 2: script_worker detection ------------------------------------------


def _request(scratch: Path) -> WorkerRequest:
    return WorkerRequest(
        task=ImplementTask(
            id="T1",
            goal="do the thing",
            done_when=[CmdCheck(cmd="test -f out.txt")],
            file_ownership=["out.txt"],
        ),
        task_id="P1/E1/T1",
        inputs={},
        scratch=scratch,
        attempt=1,
        failure_context=[],
        mode="implement",
    )


def test_script_worker_session_limit_on_stdout_raises_session_limited(
    tmp_path: Path,
) -> None:
    # Same incident, worker path: the session-limit line is on STDOUT and the
    # worker exits non-zero. It MUST raise SessionLimited (a burned attempt would
    # consume the task retry ladder against a multi-hour limit).
    script = _make_script(
        tmp_path / "worker_request.sh",
        "echo \"You've hit your session limit · resets 2:20am (Asia/Kolkata)\"\n"
        "exit 1\n",
    )
    worker = ScriptWorker(
        script=script,
        stop_script=tmp_path / "stop.sh",
        slots=1,
        timeout_s=10.0,
        log_root=tmp_path / "l",
    )
    with pytest.raises(SessionLimited):
        worker.run(_request(tmp_path / "wt"))
