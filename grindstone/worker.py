"""Worker transport seam (STUB).

ARCHITECTURE / BONES: a transport receives a fully-resolved ``WorkerRequest``,
does its work in the request's scratch CWD, and writes ``handoff.json`` THERE.
The disk file is the only output channel; ``run`` returns nothing. Raising signals
a transport-level failure the task loop maps to a failed attempt.

This module defines the stable SEAM (the request shape + the two-node failure
taxonomy + the implement-mode review-gate constant) so the contracts/config/event
spine and the test doubles can be built now. The real transport (subprocess rig,
timeout supervisor, handoff relocation) is built in a later part of the rewrite.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from grindstone.contracts.models import HandoffMode, Task

#: The implement-mode review gate: the checked artifact a fresh-context review
#: must produce in the attempt CWD. Shared by the loop (which appends the gate)
#: and the prompt builder. (Verified mechanism: a review demanded as a gated
#: artifact fires; prose instructions do not.)
REVIEW_FILENAME = "review.md"


class TransportError(Exception):
    """Any worker transport failure. The loop treats it as a failed attempt that,
    if unrecoverable, routes to the planner (BONES failure model #2)."""


class RateLimited(TransportError):
    """A rate-limit / quota refusal (BONES failure model #1): the loop backs off
    (~1/hr) and retries; mid-epoch the in-flight epoch is restarted whole."""


@dataclass(frozen=True)
class WorkerRequest:
    """One fully-resolved dispatch: the typed ``task``, its full keyed-log
    ``task_id`` (``P*/E*/T*``), the worker ``mode``, the scratch CWD it writes its
    ``handoff.json`` into, and the selected domain-skill names to compose."""

    task: Task
    task_id: str
    mode: HandoffMode
    scratch: Path
    skills: tuple[str, ...] = ()


class WorkerTransport(Protocol):
    """The uniform task-in / disk-out boundary the loop dispatches through."""

    def run(self, request: WorkerRequest) -> None: ...


def run(request: WorkerRequest) -> None:
    """Grind one task (STUB: the real transport is built in a later part)."""

    raise NotImplementedError(
        "the worker transport is built in a later part of the bones rewrite"
    )
