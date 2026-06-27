"""The target-repo-owned REPO MAP overlay: a single, OPTIONAL navigation file the
TARGET repo carries to orient the planner and the workers in its tree (where things
live, how the package is laid out, which modules matter).

A sibling of the always-on strategy overlay (``strategy_skill.py``): same single-file,
never-selected, never-gated, fixed-filename shape, but it rides BOTH the planner input
AND the worker/senior prompt (navigation helps whoever greps the tree, plan or grind).
An absent repo map is a graceful no-op everywhere (the common case) and the constructed
prompts are then byte-for-byte identical to shipping no map at all.

Layout (target-repo-owned, committed-or-gitignored by the repo, never by grindstone):

    <repo>/.grindstone/repomap.md   the one optional repo-navigation overlay

The filename is FIXED (never model-supplied), so there is no traversal surface. It is
injected on every planner and worker boundary, so it MUST be size-bounded so a runaway
file cannot blow the context window.
"""

from __future__ import annotations

from pathlib import Path

#: Repo-relative path of the single optional repo-map overlay.
REPOMAP_REL = Path(".grindstone") / "repomap.md"

#: Hard byte cap (~4k tokens), matching the strategy overlay. The map rides every
#: planner + worker boundary, so an oversized file is truncated (NOT rejected): it
#: stays advisory, never fatal.
MAX_BYTES = 16_384


def load_repo_map(repo: Path | None) -> str:
    """The repo's optional REPO MAP overlay text, or ``""`` when absent.

    Never raises (graceful no-op, like ``load_strategy``): a missing repo, a missing /
    unreadable / non-file ``repomap.md``, or a decode error all yield ``""``. Size-bounded
    to ``MAX_BYTES`` (truncated on a UTF-8 boundary), since it is injected on every planner
    and worker boundary. The fixed filename means no traversal surface. The returned text
    is stripped.
    """

    if repo is None:
        return ""
    path = Path(repo) / REPOMAP_REL
    if not path.is_file():
        return ""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return ""
    if len(text.encode("utf-8")) > MAX_BYTES:
        text = text.encode("utf-8")[:MAX_BYTES].decode("utf-8", "ignore")
    return text.strip()
