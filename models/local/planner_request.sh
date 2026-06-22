#!/usr/bin/env bash
# planner_request.sh, the all-local `planner` role. Runs pi one-shot against a
# local GPU endpoint (a Qwen-class dense model on :8080 via the pi provider) to
# decide the next epoch as a single JSON object matching the epoch-decision schema.
#
# A local planner has never existed before this rig; it mirrors the shipped Claude
# planner (models/claude/planner_request.sh) one-for-one in arg parsing, the two
# modes and the disk contract, but drives `pi` instead of `claude`. pi is agentic
# (it can edit files and run bash, exactly like the local worker), so it can grind
# the self-validate loop in the throwaway worktree just as Claude does.
#
# Two modes, chosen by whether grindstone passes --workdir (the writable planner
# worktree):
#   * SELF-VALIDATE (--workdir given): grind IN the worktree like a worker. Write
#     the decision to ./decision.json, run `python3 check_decision.py decision.json`
#     (grindstone armed it there with THIS boundary's gate context), FIX every
#     violation it prints, and loop until it exits 0. That gate-clean file is the
#     disk contract grindstone reads back. The worktree is a throwaway checkout, so
#     full tool access (pi's default read/edit/write/bash) is safe and required for
#     a headless run: any file edit here is discarded, only decision.json is read.
#   * READ-ONLY (no --workdir): the legacy fallback (artifact-only run / unborn
#     HEAD). cwd = the target repo, pi is restricted to read-only tools
#     (read,grep,find,ls), the final assistant message is written to --out;
#     grindstone extracts + validates it itself.
#
# Generic local defaults only (no operator hostnames or paths); override via
# $GRINDSTONE_PLANNER_PROVIDER / $GRINDSTONE_PLANNER_MODEL.
#
# Grindstone passes the target repo, a prompt file, an --out path, a handle-out
# path, a timeout and (in self-validate mode) a --workdir. We propagate pi's exit
# code and forward its stderr so the caller can map the failure reason
# (rate|limit|429).
set -euo pipefail

# Portable timeout prefix (resolves `timeout`, else `gtimeout`, else none).
source "$(dirname "$0")/../_common/_timeout_prefix.sh"

# Model identity is THIS script's concern. Generic local defaults: pi routes the
# `local-reviewer` provider to your local endpoint via its own config. Override for
# your own rig via $GRINDSTONE_PLANNER_PROVIDER / $GRINDSTONE_PLANNER_MODEL.
provider="${GRINDSTONE_PLANNER_PROVIDER:-local-reviewer}"; model="${GRINDSTONE_PLANNER_MODEL:-qwen-3-6-27b-dense}"

repo="" prompt="" out="" handle_out="" timeout="" workdir=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo)       repo="$2";   shift 2 ;;
    --prompt)     prompt="$2"; shift 2 ;;
    --out)        out="$2";    shift 2 ;;
    --handle-out) handle_out="$2"; shift 2 ;;
    --timeout)    timeout="$2"; shift 2 ;;
    --workdir)    workdir="$2"; shift 2 ;;
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

# The prompt is fed to pi on STDIN (pi reads the message from stdin in --print
# mode), never as an argv string: a large constructed planner input (job spec +
# skeleton + repo memory + repo map) could otherwise exceed the kernel's
# MAX_ARG_STRLEN (~128KB) and the CLI dies before launching ("Argument list too
# long"). Stdin makes the prompt size irrelevant. pi has a system-prompt flag
# (--append-system-prompt), so the schema + self-validate INSTRUCTIONS go there
# (appended to pi's default coding-assistant prompt), exactly as the Claude planner
# passes its --append-system-prompt; the prompt body stays on stdin untouched.
set +e
if [[ -n "$workdir" ]]; then
  # SELF-VALIDATE in the throwaway worktree (cwd = the boundary's _planner_tip
  # checkout). Full tool access (pi's default read/edit/write/bash) is safe here:
  # the worktree is discarded after the call, so any edit evaporates and only
  # decision.json is read back. The contract is the disk file, not stdout; --out
  # still captures the agent log for debugging.
  workdir="$(cd "$workdir" && pwd)"
  sys_append="You are the grindstone planner. Decide the SINGLE next epoch as one JSON object matching the epoch-decision schema (schema_version, tool, args). Your CWD is a throwaway worktree checkout of the current code: read and grep it to ground your plan, but any file you change here is discarded. Steps you MUST follow: (1) write your decision JSON to ./decision.json; (2) run \`python3 check_decision.py decision.json\`; (3) if it prints violations, FIX decision.json (the schema is two levels: an epoch carries epoch_title, rationale and a tasks array; per-task fields like id/goal/done_when live INSIDE each task, never on the epoch) and re-run; (4) repeat until it exits 0 with no violations. decision.json, gate-clean, is your ONLY output. Do not print the decision."
  ( cd "$workdir" && "${timeout_prefix[@]}" pi \
    --provider "$provider" \
    --model "$model" \
    --mode text \
    --print \
    --no-session \
    --append-system-prompt "$sys_append" ) \
    < "$prompt" > "$out" 2> "$err_tmp"
  rc=$?
else
  # READ-ONLY planning. cwd = the target repo, pi restricted to the read-only
  # navigation tools (read,grep,find,ls) via --tools, so it cannot mutate the repo.
  # The final assistant message is captured to --out for the core extractor.
  sys_append="Output ONLY a single JSON object that matches the grindstone epoch-decision schema (schema_version, tool, args). No prose, no markdown code fences, no commentary before or after the object."
  ( cd "$repo" && "${timeout_prefix[@]}" pi \
    --provider "$provider" \
    --model "$model" \
    --mode text \
    --print \
    --no-session \
    --tools read,grep,find,ls \
    --append-system-prompt "$sys_append" ) \
    < "$prompt" > "$out" 2> "$err_tmp"
  rc=$?
fi
set -e

cat "$err_tmp" >&2 || true

if [[ "$rc" -ne 0 ]]; then
  echo "planner_request: pi exited $rc (provider=$provider model=$model repo=$repo)" >&2
  exit "$rc"
fi
exit 0
