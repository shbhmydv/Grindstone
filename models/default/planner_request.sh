#!/usr/bin/env bash
# planner_request.sh, the DEFAULT `planner` role. Runs Claude (Opus) headless via
# `claude -p` one-shot against the TARGET repo, READ-ONLY. The agent's final
# message is written to --out (a disk contract); grindstone reads --out and does
# the tolerant extract_decision_json + validation itself (parsing stays in core,
# stdout is never scraped for the result).
#
# This is the shipped default rig: a fresh cloner with Claude Code installed runs
# with zero setup. The alternative codex-based planner lives at models/codex/ and
# is opt-in via `grindstone init --rig codex`; an operator's own planner goes in
# models/override/ (gitignored, highest priority).
#
# Grindstone passes only the target repo, a prompt file, an --out path, a
# handle-out path and a timeout. We propagate claude's exit code and forward its
# stderr so the caller can map the failure reason (rate|limit|429, auth/login).
set -euo pipefail

# Portable timeout prefix (resolves `timeout`, else `gtimeout`, else none).
source "$(dirname "$0")/_timeout_prefix.sh"

# Model identity is THIS script's concern. The owner's decision is Opus for every
# role; `opus` is the alias for the latest Opus. Override for your own rig via
# $GRINDSTONE_PLANNER_MODEL (any `claude --model` target).
model="${GRINDSTONE_PLANNER_MODEL:-opus}"

repo="" prompt="" out="" handle_out="" timeout=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo)       repo="$2";   shift 2 ;;
    --prompt)     prompt="$2"; shift 2 ;;
    --out)        out="$2";    shift 2 ;;
    --handle-out) handle_out="$2"; shift 2 ;;
    --timeout)    timeout="$2"; shift 2 ;;
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

# READ-ONLY planning. We run with cwd = the target repo and grant ONLY the
# read-only navigation tools (Read/Grep/Glob); no Edit/Write/Bash is allowlisted,
# and we do NOT pass --dangerously-skip-permissions, so in headless `-p` mode any
# edit/exec tool is denied (it cannot prompt) and the planner cannot mutate the
# repo. The grindstone-constructed prompt already carries the full input (job
# spec, skeleton, repo memory, repo map); the repo tools are a parity aid.
#
# --output-format text returns the raw final message; we capture stdout to --out.
# The append-system-prompt reinforces the JSON-only contract the prompt requests.
sys_append="Output ONLY a single JSON object that matches the grindstone epoch-decision schema (schema_version, tool, args). No prose, no markdown code fences, no commentary before or after the object."

# The prompt is fed to claude on STDIN (`claude -p` reads the prompt from stdin),
# never as an argv string: a large constructed planner input (job spec + skeleton
# + repo memory + repo map) could otherwise exceed the kernel's MAX_ARG_STRLEN
# (~128KB) and the CLI dies before launching ("Argument list too long"). Stdin
# makes the prompt size irrelevant; --allowedTools is no longer adjacent to a
# positional, so its greedy variadic parse is also safe.
set +e
( cd "$repo" && "${timeout_prefix[@]}" claude -p \
  --model "$model" \
  --output-format text \
  --allowedTools Read Grep Glob \
  --append-system-prompt "$sys_append" ) \
  < "$prompt" > "$out" 2> "$err_tmp"
rc=$?
set -e

cat "$err_tmp" >&2 || true

if [[ "$rc" -ne 0 ]]; then
  echo "planner_request: claude exited $rc (repo=$repo)" >&2
  exit "$rc"
fi
exit 0
