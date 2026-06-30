#!/usr/bin/env bash
# senior_request.sh, the `codex` rig senior role: the escalation TIER on GPT-5.5 via
# `codex exec`. Its reason to exist is QUOTA load-balancing: with a Claude planner, putting
# the senior on codex spreads the two paid tiers across two separate subscriptions, so a
# vision-heavy Claude planner and the execution tier no longer burn ONE Claude session cap
# in lockstep. Same grindstone EXECUTOR contract as the claude / local senior: grind
# agentically IN the throwaway task worktree with workspace-write, COMMIT the work (the
# deterministic gate is the committed in-scope diff, never a handoff file), and write a
# free-form handoff.md for the critic.
#
# codex exec has no --append-system-prompt, so unlike the claude / local rigs we cannot
# pass the role wording as a side channel. The worktree + handoff contract already rides in
# the grindstone-built prompt BODY (worker.py's _WORKTREE_CONTRACT + _HANDOFF_BLOCK), so we
# only PREPEND the SENIOR escalation wording + an explicit git-commit instruction ahead of
# that prompt on stdin (parity with the claude / local senior --append-system-prompt).
#
# codex grinds in the worktree with `-s workspace-write` (it may edit + git-commit there)
# and `-C "$worktree"` as its working root; the throwaway worktree is discarded after the
# gate, so write access there is safe. We propagate codex's exit code and surface any
# rate / 429 / usage / session-limit signature to stderr so the transport raises
# RateLimited / SessionLimited and PARKS rather than masking it as a hard fail.
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

# The SENIOR role wording + an explicit git-commit instruction, prepended to the prompt
# (codex has no system-prompt flag; the worktree + handoff contract is already in the
# prompt body, but the body does not itself say "git commit", which the deterministic gate
# requires). Mirrors the claude / local senior --append-system-prompt intent.
sys_append="You are the SENIOR escalation worker for a grindstone task. Investigate thoroughly and INDEPENDENTLY re-derive the claims you judge rather than merely confirm expected sections exist. Work only inside this worktree (your CWD): write every file with a path RELATIVE to your CWD, never an absolute path and never outside it. When your change is ready you MUST stage and COMMIT it with git (git add -A && git commit) inside this worktree - the orchestrator gates on your committed diff, so uncommitted work is invisible and discarded. Run whatever checks convince you it works, and write a short free-form handoff.md report for the reviewer exactly as the task instructs."

# Assemble the final stdin: the senior wording, then the grindstone-built prompt. We feed
# it as a FILE on stdin (`codex exec -` reads instructions from stdin), never as an argv
# string: a large prior-failure context could otherwise exceed the kernel's MAX_ARG_STRLEN
# (~128KB) and the CLI dies before launching ("Argument list too long"). A temp file (vs a
# pipe) mirrors the proven planner redirect and keeps codex's stdin a regular file.
full_prompt="$(mktemp)"
trap 'rm -f "$full_prompt"' EXIT
{ printf '%s\n\n' "$sys_append"; cat "$prompt"; } > "$full_prompt"

# codex grinds IN the writable task worktree: -s workspace-write lets it edit + git-commit
# there, -C "$worktree" makes that worktree its working root (it reads + writes only the
# throwaway checkout). stdout (codex's streamed run / final message) -> log_out, stderr ->
# log_err, both cat back so the Python transport inspects them for a limit signature.
set +e
"${timeout_prefix[@]}" codex exec \
  --ephemeral \
  --skip-git-repo-check \
  -s workspace-write \
  -C "$worktree" \
  - \
  < "$full_prompt" > "$log_out" 2> "$log_err"
rc=$?
set -e

cat "$log_err" >&2 || true
cat "$log_out" || true

# A genuine infra failure (rate limit / 429 / a long quota-window SESSION limit) must still
# propagate a non-zero exit so grindstone's transport raises RateLimited / SessionLimited
# and parks, never synthesize a FAILED handoff over it (that would mask the retryable
# condition as a hard fail and burn the retry ladder). codex can print a limit to stdout or
# stderr, so we grep BOTH logs and re-echo the signature to stderr for the Python
# transport's stdout+stderr inspection.
if [[ "$rc" -ne 0 ]] \
   && grep -qiE 'rate.?limit|429|session limit|usage limit' "$log_err" "$log_out" 2>/dev/null; then
  echo "senior_request: codex exec exited $rc (rate/session-limited)" >&2
  grep -hiE 'rate.?limit|429|session limit|usage limit' "$log_err" "$log_out" 2>/dev/null | head -3 >&2 || true
  exit "$rc"
fi

# Otherwise propagate codex's exit code. The worker is gated on its committed diff /
# produced artifact, NOT a handoff file, so a non-rate-limit non-zero exit is a real
# transport failure: grindstone retries the attempt, then escalates to the planner.
if [[ "$rc" -ne 0 ]]; then
  echo "senior_request: codex exec exited $rc (worktree=$worktree)" >&2
fi
exit "$rc"
