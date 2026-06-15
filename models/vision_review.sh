#!/usr/bin/env bash
# vision_review.sh — the B3 TASTE GATE. Runs codex one-shot against a SCREENSHOT
# of the rendered UI (read-only) and asks it to judge visual polish/correctness
# against the supplied criteria. codex's verdict is written to --out (a disk
# contract: {"pass":bool,"reasons":[..]} per --output-schema); grindstone re-reads
# and re-validates --out itself — stdout is never parsed. Mirrors
# planner_request.sh (codex exec --ephemeral -s read-only -C <repo> -o <out>),
# adding -i <screenshot> and --output-schema <verdict schema>.
#
# We propagate codex's exit code and forward its stderr so the caller can map the
# failure reason (rate|limit|429, auth/login). A non-image screenshot is a hard
# error before codex is ever invoked (codex -i accepts PNG/JPEG only).
set -euo pipefail

# Portable timeout prefix (resolves `timeout`, else `gtimeout`, else none).
source "$(dirname "$0")/_timeout_prefix.sh"

repo="" screenshot="" criteria_file="" schema="" out="" handle_out="" timeout=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo)          repo="$2";          shift 2 ;;
    --screenshot)    screenshot="$2";    shift 2 ;;
    --criteria-file) criteria_file="$2"; shift 2 ;;
    --schema)        schema="$2";        shift 2 ;;
    --out)           out="$2";           shift 2 ;;
    --handle-out)    handle_out="$2";    shift 2 ;;
    --timeout)       timeout="$2";       shift 2 ;;
    *) echo "vision_review: unknown arg: $1" >&2; exit 2 ;;
  esac
done

for req in repo screenshot criteria_file schema out handle_out; do
  if [[ -z "${!req}" ]]; then
    echo "vision_review: missing required --${req//_/-}" >&2
    exit 2
  fi
done

repo="$(cd "$repo" && pwd)"
# The screenshot is REPO-RELATIVE (codex -C root). Reject an absolute path or a
# `..` path segment before joining it onto $repo — the Python contract already
# rejects these (Chunk 2), this is the same guard at the script boundary so a
# crafted path can never escape the repo. Then resolve it to an absolute path
# for -i and validate it exists and is an image.
case "$screenshot" in
  /*) echo "vision_review: screenshot must be repo-relative, not absolute: $screenshot" >&2; exit 2 ;;
esac
case "/$screenshot/" in
  */../*) echo "vision_review: screenshot must not contain a '..' path segment: $screenshot" >&2; exit 2 ;;
esac
screenshot_abs="$repo/$screenshot"
if [[ ! -f "$screenshot_abs" ]]; then
  echo "vision_review: screenshot not found: $screenshot (in $repo)" >&2
  exit 2
fi
case "${screenshot_abs,,}" in
  *.png|*.jpg|*.jpeg) ;;
  *) echo "vision_review: screenshot must be a PNG/JPEG image: $screenshot" >&2; exit 2 ;;
esac

criteria_text="$(cat "$criteria_file")"
mkdir -p "$(dirname "$out")"
out="$(cd "$(dirname "$out")" && pwd)/$(basename "$out")"
mkdir -p "$(dirname "$handle_out")"
handle_out="$(cd "$(dirname "$handle_out")" && pwd)/$(basename "$handle_out")"

# The taste-reviewer prompt. The state machine, not the model, decides done;
# codex only emits the structured verdict per the schema.
prompt="You are a UI taste reviewer. The attached image is the rendered UI. Judge its visual polish and correctness against these criteria:

${criteria_text}

Emit the verdict per the output schema: set pass=true ONLY if the rendered UI meets the bar, and list concrete reasons (especially for any failure)."

# Killable process-group id, written before grinding (start_new_session group).
pgid="$(ps -o pgid= -p $$ | tr -d '[:space:]')"
echo "$pgid" > "$handle_out"

build_timeout_prefix "$timeout"

err_tmp="$(mktemp)"
trap 'rm -f "$err_tmp"' EXIT

# The prompt POSITIONAL must precede -i (codex exec arg-ordering gotcha).
set +e
"${timeout_prefix[@]}" codex exec \
  --ephemeral \
  --skip-git-repo-check \
  -s read-only \
  -C "$repo" \
  --output-schema "$schema" \
  -o "$out" \
  "$prompt" \
  -i "$screenshot_abs" \
  < /dev/null > /dev/null 2> "$err_tmp"
rc=$?
set -e

cat "$err_tmp" >&2 || true

if [[ "$rc" -ne 0 ]]; then
  echo "vision_review: codex exec exited $rc (repo=$repo)" >&2
  exit "$rc"
fi
exit 0
