#!/usr/bin/env bash
# senior_request.sh, the `senior` worker role: the CLOUD tier, driven through the
# OPENCODE AGENT (not pi) so the model gets opencode's web tools (Exa websearch +
# webfetch). Senior handles research/review escalations, which need real web
# search; that is the whole reason this role exists. Same grindstone contract as
# local_request.sh; claims NO GPU (cloud has no local device to arbitrate).
#
# The agent writes handoff.json into the worktree (the ONLY result channel); we
# propagate opencode's exit code and forward stderr so the caller can grep
# `rate|limit|429`. The per-attempt OPENCODE_DB lives in the LOG dir (never the
# worktree) for two reasons: opencode's SQLite state then can't leak into a commit,
# and each attempt gets its own db so concurrent senior runs never hit SQLITE_BUSY
# (opencode's single-global-db lock, the documented per-instance workaround).
set -euo pipefail

# Portable timeout prefix (resolves `timeout`, else `gtimeout`, else none).
source "$(dirname "$0")/_timeout_prefix.sh"

# Cloud model identity is THIS script's concern. opencode addresses the Zen
# (OpenCode Go) kimi model exactly as `opencode-go/kimi-k2.6`. Override for your
# own rig via $GRINDSTONE_SENIOR_MODEL (any `opencode -m` target).
model="${GRINDSTONE_SENIOR_MODEL:-opencode-go/kimi-k2.6}"

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

worktree="$(cd "$worktree" && pwd)"
prompt_text="$(cat "$prompt")"
mkdir -p "$log_dir"; log_dir="$(cd "$log_dir" && pwd)"
mkdir -p "$(dirname "$handle_out")"
handle_out="$(cd "$(dirname "$handle_out")" && pwd)/$(basename "$handle_out")"

# Killable process-group id, written before grinding (start_new_session group).
pgid="$(ps -o pgid= -p $$ | tr -d '[:space:]')"
echo "$pgid" > "$handle_out"

export GIT_CEILING_DIRECTORIES="$(dirname "$worktree")"
cd "$worktree"

log_out="$log_dir/agent.stdout.log"
log_err="$log_dir/agent.stderr.log"

build_timeout_prefix "$timeout"

# OPENCODE_ENABLE_EXA turns on the built-in Exa websearch tool (no key, hosted
# Exa service). --dangerously-skip-permissions auto-approves websearch/webfetch/
# write/bash so the headless run never blocks on a prompt. --dir keeps work in the
# worktree, where the agent writes handoff.json. OPENCODE_DB lives in the log dir.
set +e
OPENCODE_ENABLE_EXA=true OPENCODE_DB="$log_dir/opencode.db" \
  "${timeout_prefix[@]}" opencode run \
  --dangerously-skip-permissions \
  --dir "$worktree" \
  -m "$model" \
  "$prompt_text" \
  < /dev/null > "$log_out" 2> "$log_err"
rc=$?
set -e

cat "$log_err" >&2 || true
cat "$log_out" || true

if [[ "$rc" -ne 0 ]]; then
  echo "senior_request: opencode exited $rc (model=$model)" >&2
  exit "$rc"
fi
exit 0
