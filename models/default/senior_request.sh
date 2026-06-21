#!/usr/bin/env bash
# senior_request.sh, the DEFAULT `senior` worker role: the escalation tier. Runs
# Claude (Opus) headless via `claude -p` one-shot IN the task worktree, with edit
# + exec permissions, same grindstone contract as local_request.sh. Senior handles
# research/review escalations and visual epochs, so it leans on deeper
# investigation (Claude's built-in web search/fetch tools, available under full
# permissions) and independent re-derivation.
#
# This is the shipped default rig: a fresh cloner with Claude Code installed runs
# with zero setup. An operator's own senior tier (e.g. a cloud agent with a
# different web-search stack) goes in models/override/senior_request.sh (gitignored,
# highest priority).
#
# The agent writes handoff.json into the worktree (the ONLY result channel); we
# propagate claude's exit code and forward stderr so the caller can grep
# `rate|limit|429`. local and senior use the SAME model (Opus); they differ only by
# ROLE WORDING (below), per the owner's "Opus handling all" decision.
set -euo pipefail

# Portable timeout prefix (resolves `timeout`, else `gtimeout`, else none).
source "$(dirname "$0")/_timeout_prefix.sh"

# Model identity is THIS script's concern. Override via $GRINDSTONE_SENIOR_MODEL.
model="${GRINDSTONE_SENIOR_MODEL:-opus}"

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
prompt_text="$(cat "$prompt")"
mkdir -p "$log_dir"; log_dir="$(cd "$log_dir" && pwd)"
mkdir -p "$(dirname "$handle_out")"
handle_out="$(cd "$(dirname "$handle_out")" && pwd)/$(basename "$handle_out")"

# Killable process-group id, written before grinding (start_new_session group).
pgid="$(ps -o pgid= -p $$ | tr -d '[:space:]')"
echo "$pgid" > "$handle_out"

# CWD = worktree (where the agent writes handoff.json); fence git's upward repo
# discovery at the worktree's parent (ports the GIT_CEILING_DIRECTORIES scar).
export GIT_CEILING_DIRECTORIES="$(dirname "$worktree")"
cd "$worktree"

log_out="$log_dir/agent.stdout.log"
log_err="$log_dir/agent.stderr.log"

build_timeout_prefix "$timeout"

# The `senior` role is the escalation tier: investigate thoroughly, use web search
# when researching, and INDEPENDENTLY re-derive claims rather than confirm them.
# The worktree is an isolated, throwaway checkout, so --dangerously-skip-permissions
# (full tool access incl. web search/fetch + Bash) is safe and required for a
# headless run that never blocks on a permission prompt.
sys_append="You are the SENIOR escalation worker for a grindstone task. Investigate thoroughly: use web search when researching, and INDEPENDENTLY re-derive the claims you judge rather than merely confirm expected sections exist. Work only inside this worktree (your CWD), run the done_when checks, and write handoff.json exactly as the task instructs."

set +e
"${timeout_prefix[@]}" claude -p \
  --model "$model" \
  --dangerously-skip-permissions \
  --append-system-prompt "$sys_append" \
  "$prompt_text" \
  < /dev/null > "$log_out" 2> "$log_err"
rc=$?
set -e

cat "$log_err" >&2 || true
cat "$log_out" || true

if [[ "$rc" -ne 0 ]]; then
  echo "senior_request: claude exited $rc (model=$model)" >&2
  exit "$rc"
fi
exit 0
