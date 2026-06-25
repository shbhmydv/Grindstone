"""The target-repo-owned PLANNER STRATEGY overlay: a single, always-on advisory
file the TARGET repo carries to tune HOW its planner sequences, splits, and
prioritises work (cadence / focus / decomposition emphasis).

Distinct from the DOMAIN skills (``domain_skills.py``): those are a multi-entry,
index-driven catalogue the planner SELECTS per task by name and the core DELIVERS
into the WORKER prompt. A strategy skill is SINGLE-per-repo, never selected, never
gated by name, planner-ONLY, and ALWAYS ON: it rides every PLAN and CLOSE-OUT call
as an advisory overlay on top of grindstone's byte-stable operating preamble. An
absent strategy file is a graceful no-op everywhere (the common case).

Layout (target-repo-owned, committed-or-gitignored by the repo, never by grindstone):

    <repo>/.grindstone/strategy.md   the one always-on planner strategy overlay

The filename is FIXED (never planner-supplied), so there is no traversal surface at
all - strictly safer than ``load_domain_skill``. Because it is injected
UNCONDITIONALLY on every planner call (unlike domain skills, which are retrieved
only when selected), it MUST be size-bounded so a runaway file cannot blow the
planner context.
"""

from __future__ import annotations

from pathlib import Path

#: Repo-relative path of the single always-on planner strategy overlay.
STRATEGY_REL = Path(".grindstone") / "strategy.md"

#: Hard byte cap (~4k tokens). The strategy rides EVERY plan + closeout call, so an
#: oversized file is truncated (NOT rejected): it stays advisory, never fatal.
MAX_BYTES = 16_384


def load_strategy(repo: Path | None) -> str:
    """The repo's always-on PLANNER strategy overlay text, or ``""`` when absent.

    Never raises (graceful no-op, like ``load_domain_skill_index``): a missing repo,
    a missing / unreadable / non-file ``strategy.md``, or a decode error all yield
    ``""``. Size-bounded to ``MAX_BYTES`` (truncated on a UTF-8 boundary), since it is
    injected unconditionally on every planner boundary. The fixed filename means no
    traversal surface. The returned text is stripped.
    """

    if repo is None:
        return ""
    path = Path(repo) / STRATEGY_REL
    if not path.is_file():
        return ""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return ""
    if len(text.encode("utf-8")) > MAX_BYTES:
        text = text.encode("utf-8")[:MAX_BYTES].decode("utf-8", "ignore")
    return text.strip()
