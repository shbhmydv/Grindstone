#!/usr/bin/env bash
# codex_polish.sh — the B5 FINAL-POLISH pass. Runs codex in WORKSPACE-WRITE against
# a finished, gated worktree and asks it to make tasteful finishing-touch edits per
# the supplied criteria. Unlike the read-only scripts, codex EDITS files IN PLACE
# here; there is NO --output-schema / -o verdict. The safety gate is NOT codex's
# word — grindstone re-runs the run's complete_run evidence against the polished
# commit and KEEPS the edits only if it still passes (run_loop._final_polish).
#
# Mirrors vision_review.sh (codex exec -C <repo>, prompt-before-i ordering, killable
# pgid handle, exit-code propagation) but flips the sandbox to workspace-write (with
# network access explicitly pinned off). An optional --screenshot is forwarded via -i.
set -euo pipefail

# Portable timeout prefix (resolves `timeout`, else `gtimeout`, else none).
source "$(dirname "$0")/_timeout_prefix.sh"

repo="" criteria_file="" screenshot="" handle_out="" timeout=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo)          repo="$2";          shift 2 ;;
    --criteria-file) criteria_file="$2"; shift 2 ;;
    --screenshot)    screenshot="$2";    shift 2 ;;
    --handle-out)    handle_out="$2";    shift 2 ;;
    --timeout)       timeout="$2";       shift 2 ;;
    *) echo "codex_polish: unknown arg: $1" >&2; exit 2 ;;
  esac
done

for req in repo criteria_file handle_out; do
  if [[ -z "${!req}" ]]; then
    echo "codex_polish: missing required --${req//_/-}" >&2
    exit 2
  fi
done

repo="$(cd "$repo" && pwd)"
criteria_text="$(cat "$criteria_file")"
mkdir -p "$(dirname "$handle_out")"
handle_out="$(cd "$(dirname "$handle_out")" && pwd)/$(basename "$handle_out")"

# The optional screenshot is RELATIVE TO THE WORKTREE; resolve + validate it's an
# image (codex -i accepts PNG/JPEG only) before codex is ever invoked.
screenshot_abs=""
if [[ -n "$screenshot" ]]; then
  # Reject an absolute path or a `..` path segment before joining onto $repo —
  # the same boundary guard as vision_review.sh / the Python contract (Chunk 2),
  # so a crafted screenshot path can never escape the repo.
  case "$screenshot" in
    /*) echo "codex_polish: screenshot must be repo-relative, not absolute: $screenshot" >&2; exit 2 ;;
  esac
  case "/$screenshot/" in
    */../*) echo "codex_polish: screenshot must not contain a '..' path segment: $screenshot" >&2; exit 2 ;;
  esac
  screenshot_abs="$repo/$screenshot"
  if [[ ! -f "$screenshot_abs" ]]; then
    echo "codex_polish: screenshot not found: $screenshot (in $repo)" >&2
    exit 2
  fi
  case "${screenshot_abs,,}" in
    *.png|*.jpg|*.jpeg) ;;
    *) echo "codex_polish: screenshot must be a PNG/JPEG image: $screenshot" >&2; exit 2 ;;
  esac
fi

# The polish-pass prompt. The state machine, not the model, decides whether the
# edits are kept (the evidence re-run gates them); codex only proposes the polish.
prompt="You are doing a FINAL polish pass on a finished, working repo. Make tasteful finishing-touch improvements (visual/code polish) per the criteria. Do NOT break existing behavior or tests. Edit files in place.

Criteria:
${criteria_text}"

# Killable process-group id, written before grinding (start_new_session group).
pgid="$(ps -o pgid= -p $$ | tr -d '[:space:]')"
echo "$pgid" > "$handle_out"

build_timeout_prefix "$timeout"

err_tmp="$(mktemp)"
trap 'rm -f "$err_tmp"' EXIT

# workspace-write: codex may edit the worktree. `codex exec` is non-interactive
# (no TTY -> approval policy is already "never"), so NO -a/--ask-for-approval flag
# is passed (and `codex exec` rejects -a). The prompt POSITIONAL must precede -i
# (codex exec arg-ordering gotcha). The -C cwd is already the writable root, so
# no --add-dir is needed. workspace-write's network is off by default but the
# default is INHERITED — a global ~/.codex/config.toml could flip it on — so we
# pin sandbox_workspace_write.network_access=false explicitly (defense-in-depth).
img_args=()
if [[ -n "$screenshot_abs" ]]; then
  img_args=(-i "$screenshot_abs")
fi

set +e
"${timeout_prefix[@]}" codex exec \
  --ephemeral \
  --skip-git-repo-check \
  -s workspace-write \
  --config sandbox_workspace_write.network_access=false \
  -C "$repo" \
  "$prompt" \
  "${img_args[@]}" \
  < /dev/null > /dev/null 2> "$err_tmp"
rc=$?
set -e

cat "$err_tmp" >&2 || true

if [[ "$rc" -ne 0 ]]; then
  echo "codex_polish: codex exec exited $rc (repo=$repo)" >&2
  exit "$rc"
fi
exit 0
