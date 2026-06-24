#!/usr/bin/env bash
# senior_request.sh, the DEFAULT `senior` worker role: the escalation tier. Runs
# Claude (Opus) headless via `claude -p` one-shot IN the task worktree, with edit
# + exec permissions, same grindstone contract as worker_request.sh. Senior handles
# research/review escalations and visual epochs, so it leans on deeper
# investigation (Claude's built-in web search/fetch tools, available under full
# permissions) and independent re-derivation.
#
# This is the shipped default rig: a fresh cloner with Claude Code installed runs
# with zero setup. An operator's own senior tier (e.g. a cloud agent with a
# different web-search stack) goes in models/personal/senior_request.sh (gitignored,
# highest priority).
#
# The agent commits its work / writes its artifact in the worktree (the gate) plus a
# free-form handoff.md report; we propagate claude's exit code and forward stderr so the
# caller can grep `rate|limit|429`. local and senior use the SAME model (Opus); they differ only by
# ROLE WORDING (below), per the owner's "Opus handling all" decision.
set -euo pipefail

# Portable timeout prefix (resolves `timeout`, else `gtimeout`, else none).
source "$(dirname "$0")/../_common/_timeout_prefix.sh"

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
mkdir -p "$log_dir"; log_dir="$(cd "$log_dir" && pwd)"
mkdir -p "$(dirname "$handle_out")"
handle_out="$(cd "$(dirname "$handle_out")" && pwd)/$(basename "$handle_out")"

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

# The `senior` role is the escalation tier: investigate thoroughly, use web search
# when researching, and INDEPENDENTLY re-derive claims rather than confirm them.
# The worktree is an isolated, throwaway checkout, so --dangerously-skip-permissions
# (full tool access incl. web search/fetch + Bash) is safe and required for a
# headless run that never blocks on a permission prompt.
sys_append="You are the SENIOR escalation worker for a grindstone task. Investigate thoroughly: use web search when researching, and INDEPENDENTLY re-derive the claims you judge rather than merely confirm expected sections exist. Work only inside this worktree (your CWD): write every file with a path RELATIVE to your CWD, never an absolute path and never outside it, COMMIT your work, run the done_when checks, and write a short free-form handoff.md report for the reviewer exactly as the task instructs."

# The prompt is fed to claude on STDIN (`claude -p` reads the prompt from stdin),
# never as an argv string: a large prior-failure context could otherwise exceed
# the kernel's MAX_ARG_STRLEN (~128KB) and the CLI dies before launching
# ("Argument list too long"). Stdin makes the prompt size irrelevant.
set +e
"${timeout_prefix[@]}" claude -p \
  --model "$model" \
  --dangerously-skip-permissions \
  --append-system-prompt "$sys_append" \
  < "$prompt" > "$log_out" 2> "$log_err"
rc=$?
set -e

cat "$log_err" >&2 || true
cat "$log_out" || true

# A genuine infra failure (rate limit / 429 / a long quota-window SESSION limit)
# must still propagate a non-zero exit so grindstone's transport raises
# RateLimited / SessionLimited and parks, never synthesize a FAILED handoff over
# it (that would mask the retryable condition as a hard fail and burn the retry
# ladder). The claude CLI prints "session limit" to STDOUT, not stderr, so we grep
# BOTH logs; the pattern also covers the long session/usage limit. We re-echo the
# signature to stderr so the Python transport's stdout+stderr inspection catches it.
if [[ "$rc" -ne 0 ]] \
   && grep -qiE 'rate.?limit|429|session limit|usage limit' "$log_err" "$log_out" 2>/dev/null; then
  echo "senior_request: claude exited $rc (model=$model, rate/session-limited)" >&2
  grep -hiE 'rate.?limit|429|session limit|usage limit' "$log_err" "$log_out" 2>/dev/null | head -3 >&2 || true
  exit "$rc"
fi

# Otherwise propagate claude's exit code. The worker is gated on its committed diff /
# produced artifact, NOT a handoff file, so a non-rate-limit non-zero exit is a real
# transport failure: grindstone retries the attempt, then escalates to the planner.
if [[ "$rc" -ne 0 ]]; then
  echo "senior_request: claude exited $rc (model=$model)" >&2
fi
exit "$rc"
