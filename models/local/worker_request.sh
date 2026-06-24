#!/usr/bin/env bash
# worker_request.sh, the all-local `worker` role. Runs pi one-shot against a local
# GPU endpoint (a Qwen-class dense model served on :8080 via the pi provider).
#
# This is the committed, env-parameterized generalization of the operator's
# models/personal/worker_request.sh: a fresh cloner with pi installed and a local
# endpoint wired into pi's provider config runs with zero per-machine edits. The
# provider/model are pure env defaults (no operator hostnames or paths); override
# via $GRINDSTONE_LOCAL_PROVIDER / $GRINDSTONE_LOCAL_MODEL.
#
# This script owns the whole grindstone<->local-model boundary: model identity,
# transport (pi) and the killable process group. Grindstone passes only a worktree,
# a prompt file, a log dir, a handle-out path and a timeout; it never learns the
# transport or the model behind the `worker` role.
#
# The agent writes handoff.json into the worktree; that file is the ONLY result
# channel, stdout is never parsed. We propagate pi's exit code and forward its
# stderr to ours so the caller can grep `rate|limit|429`.
set -euo pipefail

# Portable timeout prefix (resolves `timeout`, else `gtimeout`, else none).
source "$(dirname "$0")/../_common/_timeout_prefix.sh"
# guarantee_handoff: synthesize a FAILED handoff if the agent left none.
source "$(dirname "$0")/../_common/_handoff_guarantee.sh"

worktree="" prompt="" log_dir="" handle_out="" timeout=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --worktree)   worktree="$2"; shift 2 ;;
    --prompt)     prompt="$2";   shift 2 ;;
    --log-dir)    log_dir="$2";  shift 2 ;;
    --handle-out) handle_out="$2"; shift 2 ;;
    --timeout)    timeout="$2";  shift 2 ;;
    *) echo "worker_request: unknown arg: $1" >&2; exit 2 ;;
  esac
done

for req in worktree prompt log_dir handle_out; do
  if [[ -z "${!req}" ]]; then
    echo "worker_request: missing required --${req//_/-}" >&2
    exit 2
  fi
done

# Resolve paths to absolute BEFORE we cd into the worktree.
worktree="$(cd "$worktree" && pwd)"
prompt_text="$(cat "$prompt")"
mkdir -p "$log_dir"; log_dir="$(cd "$log_dir" && pwd)"
mkdir -p "$(dirname "$handle_out")"
handle_out="$(cd "$(dirname "$handle_out")" && pwd)/$(basename "$handle_out")"

# Model identity is THIS script's concern. Generic local defaults only: pi routes
# the `local-reviewer` provider to your local endpoint (e.g. a Qwen-class dense
# model on :8080) via its own config. Override for your own rig via
# $GRINDSTONE_LOCAL_PROVIDER / $GRINDSTONE_LOCAL_MODEL (the pi --provider/--model
# your agent routes to your local endpoint).
provider="${GRINDSTONE_LOCAL_PROVIDER:-local-reviewer}"; model="${GRINDSTONE_LOCAL_MODEL:-qwen-3-6-27b-dense}"

# Pin pi-subagents spawned by this worker (the implement plan's `reviewer`) to
# THIS endpoint's model. pi-subagents does NOT inherit our --provider/--model; it
# reads the nearest `.pi/settings.json` up the tree and treats that dir as the
# project root, so writing it into the worktree makes the reviewer run on the same
# local model instead of silently falling back to the cloud default. Grindstone
# strips this file (by relpath) before commit, so it never enters the diff.
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

# Write the killable process-group id BEFORE grinding so stop.sh can always reap
# us. Grindstone launches this script with start_new_session=True, so pi (and the
# subagents it fans out) inherit this group.
pgid="$(ps -o pgid= -p $$ | tr -d '[:space:]')"
echo "$pgid" > "$handle_out"

# CWD = worktree (where the agent writes handoff.json); fence git's upward repo
# discovery at the worktree's parent (ports the GIT_CEILING_DIRECTORIES scar).
export GIT_CEILING_DIRECTORIES="$(dirname "$worktree")"
cd "$worktree"

log_out="$log_dir/agent.stdout.log"
log_err="$log_dir/agent.stderr.log"

# Honor --timeout as a backstop (grindstone also supervises wall-clock).
build_timeout_prefix "$timeout"

# The `worker` role is the on-rig grinder: build the task and verify it. A local
# model needs the worktree-isolation contract stated at the SYSTEM level too (RCA:
# an unframed local worker wrote files with ABSOLUTE paths into the real repo, so
# nothing landed in the worktree the orchestrator gates and every build escalated
# to the senior tier). The composed prompt carries the full motivated contract;
# this is the system-level reinforcement its claude/senior siblings already had.
sys_append="You are the LOCAL grinder for a grindstone task. Work only inside this worktree (your CWD): write every file with a path RELATIVE to your CWD, never an absolute path and never outside it. Make the change, run the done_when checks, and write handoff.json exactly as the task instructs."

# The prompt is fed to pi on STDIN (pi reads the message from stdin in --print
# mode), never as an argv string: a large prior-failure context could otherwise
# exceed the kernel's MAX_ARG_STRLEN (~128KB) and the CLI dies before launching
# ("Argument list too long"). Stdin makes the prompt size irrelevant. The agent
# does not read stdin again, so no interactive hang.
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

# Tee the agent's stderr to ours (log already holds it) so the caller can map the
# failure reason; stdout is never parsed but we surface it for debugging too.
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
  echo "worker_request: pi exited $rc (provider=$provider model=$model, rate/session-limited)" >&2
  grep -hiE 'rate.?limit|429|session limit|usage limit' "$log_err" "$log_out" 2>/dev/null | head -3 >&2 || true
  exit "$rc"
fi

# Otherwise GUARANTEE a handoff before returning (no-op if pi wrote one, else a
# synthesized schema-valid FAILED handoff with a diagnosis + log tails), then
# exit 0 so grindstone reads it instead of retrying a missing handoff blind.
guarantee_handoff "$worktree" "$prompt_text" "$log_out" "$log_err" "$rc"

if [[ "$rc" -ne 0 ]]; then
  echo "worker_request: pi exited $rc (provider=$provider model=$model); synthesized/kept a FAILED handoff" >&2
fi
exit 0
