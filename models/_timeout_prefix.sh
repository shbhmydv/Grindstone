# _timeout_prefix.sh — portable timeout-prefix resolution, sourced by the five
# request scripts. NOT executable / no shebang: it only defines a function.
#
# All request scripts honor --timeout as a wall-clock backstop via a `timeout`
# command prefix. Linux has GNU `timeout`; macOS has NONE in its base system —
# coreutils installs it as `gtimeout` (brew). GNU gtimeout accepts the SAME
# `--foreground --signal=TERM` flags, so only the BINARY NAME needs resolving.
# If neither exists we degrade gracefully: an empty prefix (no timeout backstop)
# plus a one-line stderr warning. Linux stays byte-identical (timeout present ->
# same prefix as before).
#
# build_timeout_prefix <timeout>: sets the GLOBAL array `timeout_prefix` to the
# command prefix (empty when <timeout> is empty or no timeout binary is found).
build_timeout_prefix() {
  local timeout="$1"
  timeout_prefix=()
  if [[ -z "$timeout" ]]; then
    return 0
  fi
  local bin=""
  if command -v timeout >/dev/null 2>&1; then
    bin="timeout"
  elif command -v gtimeout >/dev/null 2>&1; then
    bin="gtimeout"
  fi
  if [[ -z "$bin" ]]; then
    echo "${0##*/}: no timeout/gtimeout on PATH; running without a timeout backstop" >&2
    return 0
  fi
  timeout_prefix=("$bin" --foreground --signal=TERM "$timeout")
}
