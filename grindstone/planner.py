"""Planner transport seam (STUB).

BONES: the planner is STATELESS per call. It sees {job, integrated tip, digest of
done work} reconstructed fresh from disk, self-validates its ``decision.json`` in
its own writable tip, and emits ONE decision (an epoch, or an end). This module
defines the stable SEAM (the ``plan`` protocol + the two-node failure taxonomy);
the real input construction + dispatch is built in a later part of the rewrite.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


class PlannerError(Exception):
    """Any planner transport failure (auth, malformed output, hang). Routes to the
    run's clean partial-end if it cannot be recovered (BONES failure model #2)."""


class RateLimited(PlannerError):
    """A rate-limit / quota refusal (BONES failure model #1): back off (~1/hr) and
    re-issue the planner call at the boundary; nothing is burned."""


class PlannerTransport(Protocol):
    """The stateless one-shot planner the loop calls at each boundary."""

    def plan(self, prompt: str, *, workdir: Path | None = None) -> str: ...


def plan(prompt: str, *, workdir: Path | None = None) -> str:
    """Emit one decision (STUB: the real transport is built in a later part)."""

    raise NotImplementedError(
        "the planner transport is built in a later part of the bones rewrite"
    )
