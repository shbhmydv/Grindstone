#!/usr/bin/env bash
# planner_request.sh, the `planner` role. Runs GPT-5.5 via `codex exec` one-shot to
# decide the next epoch (PLAN) or write the updated living baton (CLOSE-OUT).
#
# Like the claude rig, it grinds IN the writable planner worktree grindstone passes as
# --workdir (the throwaway `_planner_tip` checkout of the integration tip / staging
# tree): codex runs there with `-s workspace-write` and its working root set to that
# worktree (-C "$workdir"), so it can WRITE its decision.json + baton.md into it exactly
# like claude does. The worktree is discarded after the call, so write access there is
# safe (it cannot touch the real tree or the run branch). grindstone reads the result
# back by priority decision.json / baton.md > --out > stdout, so --out stays the
# fallback channel and parsing/validation stays in core. The check_decision.py validator
# grindstone arms in the worktree lets codex self-validate its PLAN decision on disk
# (write -> check -> fix -> loop) just like claude.
#
# Grindstone passes the target repo, a prompt file, an --out path, a handle-out path, a
# timeout, the --workdir worktree and the --purpose (plan|closeout). We propagate codex's
# exit code and forward its stderr so the caller can map the failure reason
# (rate|limit|429, auth/login).
set -euo pipefail

# Portable timeout prefix (resolves `timeout`, else `gtimeout`, else none).
source "$(dirname "$0")/../_common/_timeout_prefix.sh"

repo="" prompt="" out="" handle_out="" timeout="" workdir="" purpose="plan"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo)       repo="$2";   shift 2 ;;
    --prompt)     prompt="$2"; shift 2 ;;
    --out)        out="$2";    shift 2 ;;
    --handle-out) handle_out="$2"; shift 2 ;;
    --timeout)    timeout="$2"; shift 2 ;;
    --workdir)    workdir="$2"; shift 2 ;;
    # --purpose (plan|closeout) is accepted for parity with the claude rig; the prompt
    # carries the role, so the invocation is purpose-agnostic (workspace-write either way).
    --purpose)    purpose="$2"; shift 2 ;;
    *) echo "planner_request: unknown arg: $1" >&2; exit 2 ;;
  esac
done
: "${purpose:?}"  # referenced for documentation; the invocation is purpose-agnostic

for req in repo prompt out handle_out workdir; do
  if [[ -z "${!req}" ]]; then
    echo "planner_request: missing required --${req//_/-}" >&2
    exit 2
  fi
done

repo="$(cd "$repo" && pwd)"
workdir="$(cd "$workdir" && pwd)"
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

# The locked invocation. codex grinds IN the writable planner worktree: -s
# workspace-write lets it land decision.json / baton.md there, and -C "$workdir" makes
# that worktree its working root (so it reads + writes the throwaway checkout, never the
# real tree). The prompt is fed on STDIN (`codex exec -` reads instructions from stdin),
# never as an argv string: a large constructed planner input could otherwise exceed the
# kernel's MAX_ARG_STRLEN (~128KB) and the CLI dies before launching ("Argument list too
# long"). Stdin makes the prompt size irrelevant.
set +e
( cd "$workdir" && "${timeout_prefix[@]}" codex exec \
  --ephemeral \
  --skip-git-repo-check \
  -s workspace-write \
  -C "$workdir" \
  -o "$out" \
  - \
  ) < "$prompt" > /dev/null 2> "$err_tmp"
rc=$?
set -e

cat "$err_tmp" >&2 || true

if [[ "$rc" -ne 0 ]]; then
  echo "planner_request: codex exec exited $rc (repo=$repo)" >&2
  # codex usually reports a rate/429/usage/session limit on stderr (already cat to
  # ours above), but a limit could also land in the final-message log "$out".
  # Grindstone classifies on the script's stdout+stderr, so surface any limit
  # signature from "$out" to stderr too, so the transport raises RateLimited /
  # SessionLimited (and PARKS) rather than burning the retry budget on a long limit.
  grep -hiE 'rate.?limit|429|session limit|usage limit' "$out" 2>/dev/null | head -3 >&2 || true
  exit "$rc"
fi
exit 0
