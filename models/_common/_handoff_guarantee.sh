# _handoff_guarantee.sh, sourced by the worker request scripts. NOT executable /
# no shebang: it only defines a function.
#
# The disk contract (ARCHITECTURE.md): the worker signals its result ONLY by
# writing handoff.json into its worktree; a MISSING handoff makes grindstone
# retry blindly. The RCA of a real dogfood failure: a worker ran out of turn
# budget mid-investigation and its agent loop ENDED before the "write handoff"
# step, producing ZERO output -> grindstone saw "no handoff.json" and retried
# twice. This helper closes that gap: after the agent exits, if it left no
# handoff, we SYNTHESIZE a schema-valid FAILED handoff carrying a diagnosis +
# log tails, so the planner gets "FAILED + why" instead of a silent retry loop.
#
# We do NOT clobber a handoff the agent wrote itself (success path untouched),
# and we never synthesize on a genuine infra failure (429/rate limit): those
# keep propagating a non-zero exit so grindstone's transport raises RateLimited.
#
# guarantee_handoff <worktree> <prompt_text> <log_out> <log_err> <rc>:
#   ensures <worktree>/handoff.json exists; returns 0. Best-effort: any failure
#   here must not crash the role script (a present-but-imperfect handoff still
#   beats a missing one), so the python step is guarded.

# Number of trailing log bytes folded into the diagnosis (kept small: the whole
# handoff is capped at 8192 bytes by the generated validator).
_HANDOFF_LOG_TAIL_BYTES=600

guarantee_handoff() {
  local worktree="$1" prompt_text="$2" log_out="$3" log_err="$4" rc="$5"

  # The agent wrote its own handoff: leave it exactly as-is (success or its own
  # truthful FAILED/PARTIAL). Never clobber a real result.
  if [[ -f "$worktree/handoff.json" ]]; then
    return 0
  fi

  # No handoff and no python to build one: degrade to a stderr note, let the
  # caller's missing-handoff path run (no worse than before this helper).
  if ! command -v python3 >/dev/null 2>&1; then
    echo "guarantee_handoff: no handoff.json and no python3 to synthesize one" >&2
    return 0
  fi

  # The task_id is not passed to the role script; recover it from the prompt
  # (build_worker_prompt emits `<task id="P*/E*/T*">` as the first line). If we
  # cannot find a well-formed id we leave it empty: the synthesized handoff then
  # fails the core gate on task_id, still a reasoned failed attempt, not a blind
  # retry.
  local task_id
  task_id="$(printf '%s' "$prompt_text" \
    | grep -oE 'P[1-9][0-9]?/E[1-9][0-9]?/T[1-8]' | head -n1 || true)"

  WORKTREE="$worktree" TASK_ID="$task_id" RC="$rc" \
  LOG_OUT="$log_out" LOG_ERR="$log_err" TAIL_BYTES="$_HANDOFF_LOG_TAIL_BYTES" \
  python3 - <<'PY' || echo "guarantee_handoff: synthesis failed (leaving no handoff)" >&2
import json
import os
from pathlib import Path

worktree = Path(os.environ["WORKTREE"])
task_id = os.environ.get("TASK_ID", "")
rc = os.environ.get("RC", "?")
tail_bytes = int(os.environ.get("TAIL_BYTES", "600"))


def tail(path: str) -> str:
    try:
        data = Path(path).read_bytes()
    except OSError:
        return ""
    return data[-tail_bytes:].decode("utf-8", "replace").strip()


err = tail(os.environ.get("LOG_ERR", ""))
out = tail(os.environ.get("LOG_OUT", ""))
diag = (
    "agent terminated without writing handoff.json (exit rc=%s); the agent "
    "loop ended before the final handoff step (out of turn/budget or a crash). "
    "stderr tail: %s | stdout tail: %s"
) % (rc, err or "(empty)", out or "(empty)")

handoff = {
    "schema_version": "1",
    "task_id": task_id,
    "status": "FAILED",
    "resulting_state": diag[:1500],
    "what_changed": [],
    "not_done": [
        "agent produced no handoff; work is incomplete and unverified",
        ("rc=%s; see attempt logs for the full agent transcript" % rc)[:256],
    ],
    "downstream_needs": [],
    "checks": [],
    "occupancy": {"compacted": False, "subagent_splits": 0},
}

# Write atomically-ish: a temp in the same dir then replace, so a concurrent
# reader never sees a half-written file.
tmp = worktree / "handoff.json.synth.tmp"
tmp.write_text(json.dumps(handoff), encoding="utf-8")
tmp.replace(worktree / "handoff.json")
PY
  return 0
}
