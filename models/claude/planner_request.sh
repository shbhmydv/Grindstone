#!/usr/bin/env bash
# planner_request.sh, the DEFAULT `planner` role. Runs Claude (Opus) headless via
# `claude -p` one-shot to decide the next epoch as a single JSON object matching
# the epoch-decision schema.
#
# Two modes, chosen by whether grindstone passes --workdir (the writable planner
# worktree):
#   * SELF-VALIDATE (--workdir given): grind IN the worktree like a worker. Write
#     the decision to ./decision.json, run `python3 check_decision.py decision.json`
#     (grindstone armed it there with THIS boundary's gate context), FIX every
#     violation it prints, and loop until it exits 0. That gate-clean file is the
#     disk contract grindstone reads back. The worktree is a throwaway checkout, so
#     full tool access (--dangerously-skip-permissions) is safe and required for a
#     headless run: any file edit here is discarded, only decision.json is read.
#     This is what stops the planner from burning blind re-asks on a schema it
#     guessed wrong (a real dogfood halt: a flattened, overlong epoch).
#   * READ-ONLY (no --workdir): the legacy fallback (artifact-only run / unborn
#     HEAD). cwd = the target repo, only Read/Grep/Glob granted, the final message
#     is written to --out; grindstone extracts + validates it itself.
#
# This is the shipped default rig: a fresh cloner with Claude Code installed runs
# with zero setup. The alternative codex-based planner lives at models/codex/ and
# is opt-in via `grindstone init --rig codex`; an operator's own planner goes in
# models/personal/ (gitignored, highest priority).
#
# Grindstone passes the target repo, a prompt file, an --out path, a handle-out
# path, a timeout and (in self-validate mode) a --workdir. We propagate claude's
# exit code and forward its stderr so the caller can map the failure reason
# (rate|limit|429, auth/login).
set -euo pipefail

# Portable timeout prefix (resolves `timeout`, else `gtimeout`, else none).
source "$(dirname "$0")/../_common/_timeout_prefix.sh"

# Model identity is THIS script's concern. The owner's decision is Opus for every
# role; `opus` is the alias for the latest Opus. Override for your own rig via
# $GRINDSTONE_PLANNER_MODEL (any `claude --model` target).
model="${GRINDSTONE_PLANNER_MODEL:-opus}"

repo="" prompt="" out="" handle_out="" timeout="" workdir="" purpose="plan"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo)       repo="$2";   shift 2 ;;
    --prompt)     prompt="$2"; shift 2 ;;
    --out)        out="$2";    shift 2 ;;
    --handle-out) handle_out="$2"; shift 2 ;;
    --timeout)    timeout="$2"; shift 2 ;;
    --workdir)    workdir="$2"; shift 2 ;;
    --purpose)    purpose="$2"; shift 2 ;;
    *) echo "planner_request: unknown arg: $1" >&2; exit 2 ;;
  esac
done

for req in repo prompt out handle_out; do
  if [[ -z "${!req}" ]]; then
    echo "planner_request: missing required --${req//_/-}" >&2
    exit 2
  fi
done

repo="$(cd "$repo" && pwd)"
mkdir -p "$(dirname "$out")"
out="$(cd "$(dirname "$out")" && pwd)/$(basename "$out")"
mkdir -p "$(dirname "$handle_out")"
handle_out="$(cd "$(dirname "$handle_out")" && pwd)/$(basename "$handle_out")"

# Killable process-group id, written before grinding (start_new_session group).
pgid="$(ps -o pgid= -p $$ | tr -d '[:space:]')"
echo "$pgid" > "$handle_out"

build_timeout_prefix "$timeout"

# Fence git's upward repo discovery at the target repo's parent so tooling cannot
# wander above the repo (ports the GIT_CEILING_DIRECTORIES scar).
export GIT_CEILING_DIRECTORIES="$(dirname "$repo")"

err_tmp="$(mktemp)"
trap 'rm -f "$err_tmp"' EXIT

# The prompt is fed to claude on STDIN (`claude -p` reads the prompt from stdin),
# never as an argv string: a large constructed planner input (job spec + skeleton
# + repo memory + repo map) could otherwise exceed the kernel's MAX_ARG_STRLEN
# (~128KB) and the CLI dies before launching ("Argument list too long"). Stdin
# makes the prompt size irrelevant.
set +e
if [[ -n "$workdir" ]]; then
  # SELF-VALIDATE in the throwaway worktree (cwd = the boundary's _planner_tip
  # checkout). Full tool access is safe here: the worktree is discarded after the
  # call, so any edit evaporates and only decision.json is read back. The contract
  # is the disk file, not stdout; --out still captures the agent log for debugging.
  workdir="$(cd "$workdir" && pwd)"
  if [[ "$purpose" == "closeout" ]]; then
    # CLOSE-OUT: read the staging tree + the keyed-log handoffs/verdicts and write the
    # updated living BATON. Free-form prose (NEVER parsed, like the handoff); no
    # decision.json, no self-validate loop. Vision is load-bearing: claude's Read views
    # images natively, so the planner judges any rendered UI / screenshot with its eyes.
    sys_append="You are the grindstone planner closing out the epoch you just ran. Your CWD is a throwaway checkout of the epoch's staging tree (the work that merged): read and grep it, and READ the keyed-log handoffs + verdicts named in your prompt (VIEW any images; you can see). Then write your updated living plan to ./baton.md and stop. baton.md is your ONLY output; do not print it. Follow the section skeleton in the prompt."
  else
    sys_append="You are the grindstone planner. Decide the SINGLE next step as one JSON object: either {\"kind\":\"epoch\",\"epoch\":{...}} or {\"kind\":\"end\",\"summary\":\"...\"}. Your CWD is a throwaway worktree checkout of the current code: read and grep it to ground your plan, but any file you change here is discarded. Steps you MUST follow: (1) write your decision JSON to ./decision.json; (2) run \`python3 check_decision.py decision.json\`; (3) if it prints violations, FIX decision.json (an epoch carries title and a tasks array of 1 to 8 disjoint tasks; each task carries id/mode/goal/tier, an implement task carries file_ownership and a research/review/artifact task carries artifact_out; you author NO done_when or check commands) and re-run; (4) repeat until it exits 0 with no violations. decision.json, gate-clean, is your ONLY output. Do not print the decision."
  fi
  ( cd "$workdir" && "${timeout_prefix[@]}" claude -p \
    --model "$model" \
    --output-format text \
    --dangerously-skip-permissions \
    --append-system-prompt "$sys_append" ) \
    < "$prompt" > "$out" 2> "$err_tmp"
  rc=$?
else
  # READ-ONLY planning. cwd = the target repo, ONLY the read-only navigation tools
  # (Read/Grep/Glob) granted; no Edit/Write/Bash is allowlisted and we do NOT pass
  # --dangerously-skip-permissions, so in headless `-p` mode any edit/exec tool is
  # denied (it cannot prompt) and the planner cannot mutate the repo. The final
  # message is captured to --out for the core extractor.
  sys_append="Output ONLY a single JSON object matching the grindstone epoch-decision schema: either {\"kind\":\"epoch\",\"epoch\":{...}} or {\"kind\":\"end\",\"summary\":\"...\"}. No prose, no markdown code fences, no commentary before or after the object."
  ( cd "$repo" && "${timeout_prefix[@]}" claude -p \
    --model "$model" \
    --output-format text \
    --allowedTools Read Grep Glob \
    --append-system-prompt "$sys_append" ) \
    < "$prompt" > "$out" 2> "$err_tmp"
  rc=$?
fi
set -e

cat "$err_tmp" >&2 || true

if [[ "$rc" -ne 0 ]]; then
  echo "planner_request: claude exited $rc (repo=$repo)" >&2
  # The claude CLI prints a rate/429/session/usage limit to its STDOUT, which we
  # redirected to "$out" (the agent log), NOT to this script's stderr. Grindstone's
  # transport classifies on the script's stdout+stderr, so surface any limit
  # signature from "$out" to stderr here so the transport raises RateLimited /
  # SessionLimited (and PARKS) instead of misreading a long limit as a transient
  # transport error and burning the retry budget.
  grep -hiE 'rate.?limit|429|session limit|usage limit' "$out" 2>/dev/null | head -3 >&2 || true
  exit "$rc"
fi
exit 0
