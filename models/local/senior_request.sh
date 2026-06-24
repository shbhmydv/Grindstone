#!/usr/bin/env bash
# senior_request.sh, the all-local `senior` role: the escalation tier. For an
# all-local rig the senior tier is the SAME local model in the senior slot, driven
# through pi exactly like the local worker, but with the SENIOR role wording so it
# investigates more thoroughly and INDEPENDENTLY re-derives the claims it judges.
#
# This is the committed, env-parameterized all-local senior. The provider/model are
# pure env defaults (no operator hostnames or paths); override via
# $GRINDSTONE_SENIOR_PROVIDER / $GRINDSTONE_SENIOR_MODEL. Same grindstone contract
# as worker_request.sh; claims no GPU arbitration (the local endpoint's own
# concurrency bound governs).
#
# The agent commits its work / writes its artifact in the worktree (the gate) plus a
# free-form handoff.md report; we propagate pi's exit code and forward stderr so the
# caller can grep `rate|limit|429`. local and senior differ only by ROLE WORDING (below).
set -euo pipefail

# Portable timeout prefix (resolves `timeout`, else `gtimeout`, else none).
source "$(dirname "$0")/../_common/_timeout_prefix.sh"

worktree="" prompt="" log_dir="" handle_out="" timeout=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --worktree)   worktree="$2"; shift 2 ;;
    --prompt)     prompt="$2";   shift 2 ;;
    --log-dir)    log_dir="$2";  shift 2 ;;
    --handle-out) handle_out="$2"; shift 2 ;;
    --timeout)    timeout="$2";  shift 2 ;;
    *) echo "senior_request: unknown arg: $1" >&2; exit 2 ;;
  esac
done

for req in worktree prompt log_dir handle_out; do
  if [[ -z "${!req}" ]]; then
    echo "senior_request: missing required --${req//_/-}" >&2
    exit 2
  fi
done

# Resolve paths to absolute BEFORE we cd into the worktree.
worktree="$(cd "$worktree" && pwd)"
mkdir -p "$log_dir"; log_dir="$(cd "$log_dir" && pwd)"
mkdir -p "$(dirname "$handle_out")"
handle_out="$(cd "$(dirname "$handle_out")" && pwd)/$(basename "$handle_out")"

# Model identity is THIS script's concern. For an all-local rig the senior tier is
# the same local model as the worker, in the senior slot. Generic local defaults
# only; override via $GRINDSTONE_SENIOR_PROVIDER / $GRINDSTONE_SENIOR_MODEL.
provider="${GRINDSTONE_SENIOR_PROVIDER:-local-reviewer}"; model="${GRINDSTONE_SENIOR_MODEL:-qwen-3-6-27b-dense}"

# Pin pi-subagents spawned by this senior (the implement plan's `reviewer`) to THIS
# endpoint's model, same rationale as the local worker: pi-subagents reads the
# nearest `.pi/settings.json` up the tree, so writing it into the worktree keeps the
# reviewer on the local model instead of a cloud default. Grindstone strips this
# file (by relpath) before commit, so it never enters the diff.
mkdir -p "$worktree/.pi"
cat > "$worktree/.pi/settings.json" <<EOF
{
  "subagents": {
    "agentOverrides": {
      "reviewer": {
        "model": "$provider/$model"
      }
    }
  }
}
EOF

# Killable process-group id, written before grinding (start_new_session group).
pgid="$(ps -o pgid= -p $$ | tr -d '[:space:]')"
echo "$pgid" > "$handle_out"

# CWD = worktree (where the agent writes its work + handoff.md); fence git's upward repo
# discovery at the worktree's parent (ports the GIT_CEILING_DIRECTORIES scar).
export GIT_CEILING_DIRECTORIES="$(dirname "$worktree")"
cd "$worktree"

log_out="$log_dir/agent.stdout.log"
log_err="$log_dir/agent.stderr.log"

build_timeout_prefix "$timeout"

# The `senior` role is the escalation tier: investigate thoroughly and
# INDEPENDENTLY re-derive the claims it judges rather than merely confirm expected
# sections exist. (Local senior has no web tools; it leans on deeper investigation
# of the repo itself.) pi runs agentically in the worktree (full tools), runs its
# own checks, and writes handoff.md.
sys_append="You are the SENIOR escalation worker for a grindstone task. Investigate thoroughly and INDEPENDENTLY re-derive the claims you judge rather than merely confirm expected sections exist. Work only inside this worktree (your CWD): write every file with a path RELATIVE to your CWD, never an absolute path and never outside it, COMMIT your work, run whatever checks convince you it works, and write a short free-form handoff.md report for the reviewer exactly as the task instructs."

# The prompt is fed to pi on STDIN (pi reads the message from stdin in --print
# mode), never as an argv string: a large prior-failure context could otherwise
# exceed the kernel's MAX_ARG_STRLEN (~128KB) and the CLI dies before launching
# ("Argument list too long"). Stdin makes the prompt size irrelevant.
set +e
"${timeout_prefix[@]}" pi \
  --provider "$provider" \
  --model "$model" \
  --mode json \
  --print \
  --no-session \
  --append-system-prompt "$sys_append" \
  < "$prompt" > "$log_out" 2> "$log_err"
rc=$?
set -e

cat "$log_err" >&2 || true
cat "$log_out" || true

# A genuine infra failure (rate limit / 429 / a long quota-window SESSION limit)
# must still propagate a non-zero exit so grindstone's transport raises
# RateLimited / SessionLimited and parks, never synthesize a FAILED handoff over it
# (that would mask the retryable condition as a hard fail and burn the retry ladder).
# A session/usage limit can land on STDOUT, so we grep BOTH logs; the pattern also
# covers the long session/usage limit, and we re-echo it to stderr so the Python
# transport's stdout+stderr inspection catches it.
if [[ "$rc" -ne 0 ]] \
   && grep -qiE 'rate.?limit|429|session limit|usage limit' "$log_err" "$log_out" 2>/dev/null; then
  echo "senior_request: pi exited $rc (provider=$provider model=$model, rate/session-limited)" >&2
  grep -hiE 'rate.?limit|429|session limit|usage limit' "$log_err" "$log_out" 2>/dev/null | head -3 >&2 || true
  exit "$rc"
fi

# Otherwise propagate pi's exit code. The worker is gated on its committed diff /
# produced artifact, NOT a handoff file, so a non-rate-limit non-zero exit is a real
# transport failure: grindstone retries the attempt, then escalates to the planner.
if [[ "$rc" -ne 0 ]]; then
  echo "senior_request: pi exited $rc (provider=$provider model=$model)" >&2
fi
exit "$rc"
