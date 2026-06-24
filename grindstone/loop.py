"""The epoch driver (STUB).

BONES state machine (epochs only, no phases): at each boundary the planner sees
{job, integrated tip, digest of done work} and proposes an epoch (1..N disjoint
tasks, tier-routed) OR ends; tasks fan out in their own worktrees, a critic
triages each, passing tasks merge to the run branch (which only ever fast-forwards
on epoch completion, so its tip is always at a clean boundary). Resume = cleanup +
re-plan from that boundary.

This is the SEAM the CLI calls; the real driver (planner input construction, task
exec, critic, disjoint merge, resume cleanup) is built in a later part.
"""

from __future__ import annotations

from pathlib import Path

#: The built-in epoch backstop when the config sets no ``max_epochs`` (BONES: the
#: cap is the involuntary trigger of the clean partial-end, never unbounded).
DEFAULT_MAX_EPOCHS = 40


def run(job_path: Path, repo_root: Path, *, run_id: str | None = None) -> int:
    """Drive a job to a clean terminal (STUB: built in a later part)."""

    raise NotImplementedError(
        "the epoch loop is built in a later part of the bones rewrite"
    )


def resume(run_id: str, repo_root: Path) -> int:
    """Re-enter a killed run from its last clean boundary (STUB: later part)."""

    raise NotImplementedError(
        "resume is built in a later part of the bones rewrite"
    )
