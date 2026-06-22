"""ScriptPlanner against tiny fake planner scripts (no codex), plus the relocated
``extract_decision_json`` extractor cases (now imported from grindstone.planner).
"""

from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from grindstone.planner import (
    PlannerHardError,
    RateLimited,
    TransportError,
    WorkerTimeout,
    extract_decision_json,
)
from grindstone.script_planner import ScriptPlanner


def _make_script(path: Path, body: str) -> Path:
    path.write_text("#!/usr/bin/env bash\n" + body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


# --- transport: file contract + return ----------------------------------------


def test_plan_returns_out_file_and_passes_contract_args(tmp_path: Path) -> None:
    argv_file = tmp_path / "argv.txt"
    prompt_copy = tmp_path / "prompt_seen.txt"
    repo = tmp_path / "repo"
    repo.mkdir()
    script = _make_script(
        tmp_path / "planner_request.sh",
        f"""set -euo pipefail
printf '%s\\n' "$@" > "{argv_file}"
out="" handle="" prompt=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --out) out="$2"; shift 2 ;;
    --handle-out) handle="$2"; shift 2 ;;
    --prompt) prompt="$2"; shift 2 ;;
    *) shift ;;
  esac
done
cp "$prompt" "{prompt_copy}"
echo $$ > "$handle"
echo 'DECISION TEXT' > "$out"
exit 0
""",
    )
    planner = ScriptPlanner(
        script=script, stop_script=tmp_path / "stop.sh", repo=repo, slots=1, timeout_s=10.0
    )
    result = planner.plan("the prompt")
    assert result.strip() == "DECISION TEXT"

    argv = argv_file.read_text(encoding="utf-8").splitlines()
    assert argv[argv.index("--repo") + 1] == str(repo)
    assert "--prompt" in argv and "--out" in argv and "--handle-out" in argv
    assert argv[argv.index("--timeout") + 1] == "10"
    # the prompt reached the script via a real file
    assert prompt_copy.read_text(encoding="utf-8") == "the prompt"


def test_rate_limit_stderr_maps_to_rate_limited(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    script = _make_script(
        tmp_path / "planner_request.sh",
        'echo "429 usage limit exceeded" >&2\nexit 1\n',
    )
    planner = ScriptPlanner(
        script=script, stop_script=tmp_path / "stop.sh", repo=repo, slots=1, timeout_s=10.0
    )
    with pytest.raises(RateLimited):
        planner.plan("p")


def test_auth_stderr_maps_to_planner_hard_error(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    script = _make_script(
        tmp_path / "planner_request.sh",
        'echo "401 Unauthorized: please run codex login" >&2\nexit 1\n',
    )
    planner = ScriptPlanner(
        script=script, stop_script=tmp_path / "stop.sh", repo=repo, slots=1, timeout_s=10.0
    )
    with pytest.raises(PlannerHardError):
        planner.plan("p")


def test_other_nonzero_stderr_maps_to_transport_error(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    script = _make_script(
        tmp_path / "planner_request.sh",
        'echo "transient network hiccup" >&2\nexit 1\n',
    )
    planner = ScriptPlanner(
        script=script, stop_script=tmp_path / "stop.sh", repo=repo, slots=1, timeout_s=10.0
    )
    with pytest.raises(TransportError):
        planner.plan("p")


def test_timeout_raises_and_invokes_stop_sh(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    stop_marker = tmp_path / "stop_called"
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
        tmp_path / "planner_request.sh",
        """set -euo pipefail
handle=""
while [[ $# -gt 0 ]]; do
  case "$1" in --handle-out) handle="$2"; shift 2 ;; *) shift ;; esac
done
echo $$ > "$handle"
sleep 30
""",
    )
    planner = ScriptPlanner(
        script=script, stop_script=tmp_path / "stop.sh", repo=repo, slots=1, timeout_s=1.0
    )
    with pytest.raises(WorkerTimeout):
        planner.plan("p")
    assert stop_marker.is_file(), "stop.sh was not invoked on timeout"


def test_stop_script_is_the_explicit_path_not_a_sibling(tmp_path: Path) -> None:
    # The planner script can resolve from a preset dir (e.g. codex/) with no sibling
    # stop.sh, so stop.sh is passed EXPLICITLY (resolved by the CLI), never assumed
    # beside the planner script.
    script = tmp_path / "codex" / "planner_request.sh"
    script.parent.mkdir()
    script.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    stop = tmp_path / "_common" / "stop.sh"
    planner = ScriptPlanner(
        script=script, stop_script=stop, repo=tmp_path, slots=1, timeout_s=1.0
    )
    assert planner._stop_script == stop
    assert planner._stop_script != script.parent / "stop.sh"


# --- relocated extractor (grindstone.planner.extract_decision_json) ------------

_DECISION = {
    "schema_version": "1",
    "tool": "complete_run",
    "args": {"summary": "ok", "evidence": [{"cmd": "true"}]},
}
_BODY = json.dumps(_DECISION)


def test_extract_bare_object() -> None:
    assert json.loads(extract_decision_json(_BODY) or "") == _DECISION


def test_extract_fenced_block() -> None:
    text = f"Here is my decision:\n```json\n{_BODY}\n```\n"
    assert json.loads(extract_decision_json(text) or "") == _DECISION


def test_extract_prose_wrapped() -> None:
    text = f"Let me reason about this carefully.\n\n{_BODY}\n\nThat is the call."
    assert json.loads(extract_decision_json(text) or "") == _DECISION


def test_extract_prefers_last_tool_object() -> None:
    text = (
        'I considered {"note": "a {nested} brace and a \\" quote"} earlier.\n'
        f"Final answer:\n```json\n{_BODY}\n```"
    )
    assert json.loads(extract_decision_json(text) or "") == _DECISION


def test_extract_ignores_braces_inside_strings() -> None:
    payload = {
        "schema_version": "1",
        "tool": "escalate_run",
        "args": {"reason": "value with } and { braces"},
    }
    text = f"prose {json.dumps(payload)} more prose"
    assert json.loads(extract_decision_json(text) or "") == payload


def test_extract_none_when_no_json() -> None:
    assert extract_decision_json("no json here, just prose.") is None
    assert extract_decision_json("") is None
    assert extract_decision_json("{ this is broken json") is None
