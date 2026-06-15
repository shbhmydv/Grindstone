"""Repo-memory READ seam — the planner's ``<repo_memory>`` digest source.

S5 ruling 1 / ARCHITECTURE.md. This module is the READ HALF of the deferred write-back
repo-memory organ, and ONLY the read half. ``load_digest`` reads a per-repo
``.grindstone/memory/digest.md`` that the run-loop freezes into ``RunState`` at
run start (so the stable head stays byte-identical for the whole run) and the
planner renders into its ``<repo_memory>`` slot.

The WRITE-BACK hook (run-end extraction that grows the digest from failure data)
is DELIBERATELY NOT BUILT here. ARCHITECTURE.md defers the organ until 3-5 real runs
justify it: you cannot know what to remember before failure data exists, a
memory loop is a compounding amplifier best attached to an already-trusted
system, and a present-but-empty extraction function would be dead code under the
zero-dead-code rule. Adding the writer behind this fixed read seam later is
cheap; the seam is the whole point.

Reads NEVER fail a run: a missing digest yields ``None`` (the common case —
most repos never grow one), and an oversized digest is truncated to a byte cap
with a visible marker rather than blowing the planner's prefix budget.
"""

from __future__ import annotations

from pathlib import Path

#: Byte cap on the rendered digest (16 KiB). The digest rides in the byte-stable
#: head of every planner call; an unbounded one would silently bloat the prefix
#: (ARCHITECTURE.md: prefix growth is a design smell). Truncate, never reject.
DIGEST_MAX_BYTES = 16 * 1024

#: Appended when a digest is truncated, so the planner can SEE it was clipped
#: rather than silently planning against a half-sentence.
TRUNCATION_MARKER = "\n…[repo-memory digest truncated at 16 KiB]"

#: Repo-relative path of the digest (sibling of ``runs/`` under ``.grindstone``).
_DIGEST_REL = Path(".grindstone") / "memory" / "digest.md"


def load_digest(repo_root: Path) -> str | None:
    """Return the repo-memory digest text for ``repo_root``, or ``None``.

    ``None`` when ``.grindstone/memory/digest.md`` is absent (the usual case).
    When present, its UTF-8 text — truncated to ``DIGEST_MAX_BYTES`` with
    ``TRUNCATION_MARKER`` appended (on a UTF-8 boundary, so a multi-byte
    codepoint straddling the cap never raises) when it would otherwise overflow.
    """

    path = Path(repo_root) / _DIGEST_REL
    try:
        raw = path.read_bytes()
    except (FileNotFoundError, NotADirectoryError, IsADirectoryError):
        return None
    if len(raw) <= DIGEST_MAX_BYTES:
        return raw.decode("utf-8", errors="replace")
    budget = DIGEST_MAX_BYTES - len(TRUNCATION_MARKER.encode("utf-8"))
    kept = raw[:budget].decode("utf-8", errors="ignore")
    return kept + TRUNCATION_MARKER
