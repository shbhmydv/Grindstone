#!/usr/bin/env bash
# local_request.sh, the `local` worker role. Runs pi one-shot on a local GPU.
#
# This script owns the whole grindstone<->local-model boundary: model identity,
# transport (pi) and the killable process group. The `local` role is ONE model,
# Qwen3.6-27B UD-Q6_K_XL spanning both GPUs on a single :8080 endpoint whose
# --parallel 2 slots ARE the concurrency bound (no GPU arbitration). Grindstone
# passes only a worktree, a prompt file, a log dir, a handle-out path and a
# timeout, it never learns the transport or the model behind the `local` role.
#
# The agent writes handoff.json into the worktree; that file is the ONLY result
# channel, stdout is never parsed. We propagate pi's exit code and forward its
# stderr to ours so the caller can grep `rate|limit|429`.
set -euo pipefail

# Portable timeout prefix (resolves `timeout`, else `gtimeout`, else none).
source "$(dirname "$0")/_timeout_prefix.sh"

worktree="" prompt="" log_dir="" handle_out="" timeout=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --worktree)   worktree="$2"; shift 2 ;;
    --prompt)     prompt="$2";   shift 2 ;;
    --log-dir)    log_dir="$2";  shift 2 ;;
    --handle-out) handle_out="$2"; shift 2 ;;
    --timeout)    timeout="$2";  shift 2 ;;
    *) echo "local_request: unknown arg: $1" >&2; exit 2 ;;
  esac
done

for req in worktree prompt log_dir handle_out; do
  if [[ -z "${!req}" ]]; then
    echo "local_request: missing required --${req//_/-}" >&2
    exit 2
  fi
done

# Resolve paths to absolute BEFORE we cd into the worktree.
worktree="$(cd "$worktree" && pwd)"
prompt_text="$(cat "$prompt")"
mkdir -p "$log_dir"; log_dir="$(cd "$log_dir" && pwd)"
mkdir -p "$(dirname "$handle_out")"
handle_out="$(cd "$(dirname "$handle_out")" && pwd)/$(basename "$handle_out")"

# Model identity is THIS script's concern. The local role is ONE model:
# Qwen3.6-27B UD-Q6_K_XL on a single :8080 endpoint spanning both GPUs (no
# per-attempt GPU claim, the endpoint's --parallel 2 slots are the concurrency
# bound). pi routes `local-reviewer` to :8080 via its own config.
# Override for your own rig via $GRINDSTONE_LOCAL_PROVIDER / $GRINDSTONE_LOCAL_MODEL
# (the pi --provider/--model your agent routes to your local endpoint).
provider="${GRINDSTONE_LOCAL_PROVIDER:-local-reviewer}"; model="${GRINDSTONE_LOCAL_MODEL:-qwen-3-6-27b-dense}"

# Pin pi-subagents spawned by this worker (the implement plan's `reviewer`) to
# THIS GPU's model. pi-subagents does NOT inherit our --provider/--model; it reads
# the nearest `.pi/settings.json` up the tree and treats that dir as the project
# root, so writing it into the worktree makes the reviewer run on the same local
# model instead of silently falling back to the cloud default. Grindstone strips
# this file (by relpath) before commit, so it never enters the diff.
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

# Honor --timeout as a backstop (grindstone also supervises wall-clock). Use
# --foreground so pi stays in OUR process group, keeping stop.sh's group kill and
# this handle consistent.
build_timeout_prefix "$timeout"

set +e
"${timeout_prefix[@]}" pi \
  --provider "$provider" \
  --model "$model" \
  --mode json \
  --print \
  --no-session \
  "$prompt_text" \
  < /dev/null > "$log_out" 2> "$log_err"
rc=$?
set -e

# Tee the agent's stderr to ours (log already holds it) so the caller can map the
# failure reason; stdout is never parsed but we surface it for debugging too.
cat "$log_err" >&2 || true
cat "$log_out" || true

if [[ "$rc" -ne 0 ]]; then
  echo "local_request: pi exited $rc (provider=$provider model=$model)" >&2
  exit "$rc"
fi
exit 0
