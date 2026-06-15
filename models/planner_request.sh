#!/usr/bin/env bash
# planner_request.sh, the `planner` role. Runs GPT-5.5 via `codex exec` one-shot
# against the TARGET repo, read-only. The agent's final message is written to
# --out (a disk contract); grindstone reads --out and does the tolerant
# extract_decision_json + validation itself (parsing stays in core).
#
# Grindstone passes only the target repo, a prompt file, an --out path, a
# handle-out path and a timeout. We propagate codex's exit code and forward its
# stderr so the caller can map the failure reason (rate|limit|429, auth/login).
set -euo pipefail

# Portable timeout prefix (resolves `timeout`, else `gtimeout`, else none).
source "$(dirname "$0")/_timeout_prefix.sh"

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
prompt_text="$(cat "$prompt")"
mkdir -p "$(dirname "$out")"
out="$(cd "$(dirname "$out")" && pwd)/$(basename "$out")"
mkdir -p "$(dirname "$handle_out")"
handle_out="$(cd "$(dirname "$handle_out")" && pwd)/$(basename "$handle_out")"

# Killable process-group id, written before grinding (start_new_session group).
pgid="$(ps -o pgid= -p $$ | tr -d '[:space:]')"
echo "$pgid" > "$handle_out"

build_timeout_prefix "$timeout"

err_tmp="$(mktemp)"
trap 'rm -f "$err_tmp"' EXIT

# The locked invocation (verified against grindstone/codex_planner.py @ cc05198).
set +e
"${timeout_prefix[@]}" codex exec \
  --ephemeral \
  --skip-git-repo-check \
  -s read-only \
  -C "$repo" \
  -o "$out" \
  "$prompt_text" \
  < /dev/null > /dev/null 2> "$err_tmp"
rc=$?
set -e

cat "$err_tmp" >&2 || true

if [[ "$rc" -ne 0 ]]; then
  echo "planner_request: codex exec exited $rc (repo=$repo)" >&2
  exit "$rc"
fi
exit 0
