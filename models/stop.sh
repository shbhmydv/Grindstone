#!/usr/bin/env bash
# stop.sh — reap a role script's process group. Reads the pgid written to the
# handle file by *_request.sh, SIGTERMs the whole group, waits briefly, then
# escalates to SIGKILL. This is the v7 kill-group scar moved into models/.
#
# Idempotent / no-op if the group is already gone (the handle is stale, missing,
# empty, or the process already exited) — always exits 0 in those cases.
set -euo pipefail

handle=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --handle) handle="$2"; shift 2 ;;
    *) echo "stop.sh: unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$handle" ]]; then
  echo "stop.sh: missing required --handle" >&2
  exit 2
fi

# Missing/empty handle -> nothing to reap.
[[ -f "$handle" ]] || exit 0
pgid="$(tr -d '[:space:]' < "$handle" 2>/dev/null || true)"
[[ -n "$pgid" ]] || exit 0
# Guard against a garbage handle (must be a bare number to target a group).
[[ "$pgid" =~ ^[0-9]+$ ]] || exit 0

# Already gone? no-op.
if ! kill -0 -- "-$pgid" 2>/dev/null; then
  exit 0
fi

kill -TERM -- "-$pgid" 2>/dev/null || true
sleep 2
if kill -0 -- "-$pgid" 2>/dev/null; then
  kill -KILL -- "-$pgid" 2>/dev/null || true
fi
exit 0
