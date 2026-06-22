"""ScriptWorker against tiny fake role scripts (no pi). Asserts the file-contract
args, the stderr→exception mapping, timeout→stop.sh, and the semaphore bound."""

from __future__ import annotations

import stat
import threading
from pathlib import Path

import pytest

from grindstone.contracts.models import CmdCheck, ImplementTask
from grindstone.script_worker import ScriptWorker
from grindstone.worker import RateLimited, TransportError, WorkerRequest, WorkerTimeout


def _make_script(path: Path, body: str) -> Path:
    path.write_text("#!/usr/bin/env bash\n" + body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


def _request(scratch: Path, *, task_id: str = "P1/E1/T1", attempt: int = 1) -> WorkerRequest:
    return WorkerRequest(
        task=ImplementTask(
            id="T1",
            goal="do the thing",
            done_when=[CmdCheck(cmd="test -f out.txt")],
            file_ownership=["out.txt"],
        ),
        task_id=task_id,
        inputs={},
        scratch=scratch,
        attempt=attempt,
        failure_context=[],
        mode="implement",
    )


def test_success_relays_handoff_and_passes_contract_args(tmp_path: Path) -> None:
    argv_file = tmp_path / "argv.txt"
    script = _make_script(
        tmp_path / "worker_request.sh",
        f"""set -euo pipefail
printf '%s\\n' "$@" > "{argv_file}"
worktree="" handle=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --worktree) worktree="$2"; shift 2 ;;
    --handle-out) handle="$2"; shift 2 ;;
    *) shift ;;
  esac
done
echo $$ > "$handle"
echo '{{"ok":1}}' > "$worktree/handoff.json"
exit 0
""",
    )
    scratch = tmp_path / "wt"
    scratch.mkdir()
    log_root = tmp_path / "logs"

    worker = ScriptWorker(
        script=script,
        stop_script=tmp_path / "stop.sh",
        slots=1,
        timeout_s=10.0,
        log_root=log_root,
    )
    worker.run(_request(scratch))

    # handoff relayed into the worktree (the disk contract)
    assert (scratch / "handoff.json").is_file()

    # the file-contract args reached the script
    argv = argv_file.read_text(encoding="utf-8").splitlines()
    assert "--worktree" in argv
    assert argv[argv.index("--worktree") + 1] == str(scratch)
    prompt_arg = argv[argv.index("--prompt") + 1]
    assert Path(prompt_arg).is_file()  # prompt file allocated under log_root
    assert "--log-dir" in argv
    assert "--handle-out" in argv
    assert argv[argv.index("--timeout") + 1] == "10"
    # the prompt file carries the built worker prompt
    assert "do the thing" in Path(prompt_arg).read_text(encoding="utf-8")


def test_nonzero_rate_limit_stderr_maps_to_rate_limited(tmp_path: Path) -> None:
    script = _make_script(
        tmp_path / "worker_request.sh",
        'echo "Error: rate limit exceeded" >&2\nexit 1\n',
    )
    worker = ScriptWorker(
        script=script, stop_script=tmp_path / "stop.sh", slots=1, timeout_s=10.0, log_root=tmp_path / "l"
    )
    with pytest.raises(RateLimited):
        worker.run(_request(tmp_path))


def test_nonzero_other_stderr_maps_to_transport_error(tmp_path: Path) -> None:
    script = _make_script(
        tmp_path / "worker_request.sh",
        'echo "some unexpected failure" >&2\nexit 1\n',
    )
    worker = ScriptWorker(
        script=script, stop_script=tmp_path / "stop.sh", slots=1, timeout_s=10.0, log_root=tmp_path / "l"
    )
    with pytest.raises(TransportError):
        worker.run(_request(tmp_path))


def test_timeout_raises_and_invokes_stop_sh(tmp_path: Path) -> None:
    stop_marker = tmp_path / "stop_called"
    # stop.sh sibling: records it was called AND actually reaps the group so the
    # supervisor returns fast (mirrors the real stop.sh kill-group contract).
    _make_script(
        tmp_path / "stop.sh",
        f"""set -euo pipefail
echo called > "{stop_marker}"
handle=""
while [[ $# -gt 0 ]]; do
  case "$1" in --handle) handle="$2"; shift 2 ;; *) shift ;; esac
done
if [[ -f "$handle" ]]; then
  pgid="$(cat "$handle")"
  [[ -n "$pgid" ]] && kill -KILL -- "-$pgid" 2>/dev/null || true
fi
exit 0
""",
    )
    script = _make_script(
        tmp_path / "worker_request.sh",
        """set -euo pipefail
handle=""
while [[ $# -gt 0 ]]; do
  case "$1" in --handle-out) handle="$2"; shift 2 ;; *) shift ;; esac
done
echo $$ > "$handle"
sleep 30
""",
    )
    worker = ScriptWorker(
        script=script,
        stop_script=tmp_path / "stop.sh",
        slots=1,
        timeout_s=1.0,
        log_root=tmp_path / "l",
    )
    with pytest.raises(WorkerTimeout):
        worker.run(_request(tmp_path / "wt"))
    assert stop_marker.is_file(), "stop.sh was not invoked on timeout"


def test_semaphore_bounds_concurrency(tmp_path: Path) -> None:
    slots = 2
    counter = tmp_path / "counter"
    maxfile = tmp_path / "maxc"
    lock = tmp_path / "lock"
    counter.write_text("0", encoding="utf-8")
    maxfile.write_text("0", encoding="utf-8")

    script = _make_script(
        tmp_path / "worker_request.sh",
        f"""set -euo pipefail
worktree="" handle=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --worktree) worktree="$2"; shift 2 ;;
    --handle-out) handle="$2"; shift 2 ;;
    *) shift ;;
  esac
done
exec 9>"{lock}"
flock 9
cur=$(( $(cat "{counter}") + 1 )); echo "$cur" > "{counter}"
if [[ "$cur" -gt "$(cat "{maxfile}")" ]]; then echo "$cur" > "{maxfile}"; fi
flock -u 9
sleep 0.3
flock 9
echo "$(( $(cat "{counter}") - 1 ))" > "{counter}"
flock -u 9
echo $$ > "$handle"
echo '{{}}' > "$worktree/handoff.json"
exit 0
""",
    )
    worker = ScriptWorker(
        script=script, stop_script=tmp_path / "stop.sh", slots=slots, timeout_s=10.0, log_root=tmp_path / "l"
    )

    n = 6
    errors: list[BaseException] = []

    def _go(i: int) -> None:
        scratch = tmp_path / f"wt{i}"
        scratch.mkdir()
        try:
            worker.run(_request(scratch, task_id=f"P1/E1/T{i}", attempt=1))
        except BaseException as exc:  # noqa: BLE001, surface to the asserting thread
            errors.append(exc)

    threads = [threading.Thread(target=_go, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors
    observed_max = int(maxfile.read_text(encoding="utf-8").strip())
    assert observed_max <= slots, f"max concurrent {observed_max} exceeded slots {slots}"
    assert observed_max >= 1


def test_stop_script_is_the_explicit_path_not_a_sibling(tmp_path: Path) -> None:
    # The role script can resolve from a preset/override dir with no sibling
    # stop.sh, so stop.sh is passed EXPLICITLY (resolved by the CLI), never assumed
    # beside the role script.
    script = tmp_path / "personal" / "worker_request.sh"
    script.parent.mkdir()
    script.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    stop = tmp_path / "_common" / "stop.sh"
    worker = ScriptWorker(
        script=script, stop_script=stop, slots=1, timeout_s=1.0, log_root=tmp_path
    )
    assert worker._stop_script == stop
    assert worker._stop_script != script.parent / "stop.sh"


def test_log_artifacts_land_under_log_root_not_run_dir(tmp_path: Path) -> None:
    script = _make_script(
        tmp_path / "worker_request.sh",
        """set -euo pipefail
handle="" worktree=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --handle-out) handle="$2"; shift 2 ;;
    --worktree) worktree="$2"; shift 2 ;;
    *) shift ;;
  esac
done
echo $$ > "$handle"
echo '{}' > "$worktree/handoff.json"
exit 0
""",
    )
    scratch = tmp_path / "wt"
    scratch.mkdir()
    log_root = tmp_path / "logroot"
    worker = ScriptWorker(
        script=script, stop_script=tmp_path / "stop.sh", slots=1, timeout_s=10.0, log_root=log_root
    )
    worker.run(_request(scratch))
    # a per-attempt dir with the handle file exists under log_root, not in scratch
    attempt_dir = log_root / "P1-E1-T1-attempt-1"
    assert (attempt_dir / "handle").is_file()
    assert not (scratch / "handle").exists()
