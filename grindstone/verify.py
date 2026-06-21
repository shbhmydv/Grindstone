"""End-of-epoch agentic verification pass (gate-rebalance G4).

After every task in an epoch clears its deterministic floor (per-task ``done_when``
re-run by the core) and the epoch would otherwise complete, IF the epoch carries any
natural-language ``criteria`` the core runs ONE adversarial verification pass on the
LOCAL tier, in a worktree of the epoch's integration tip with the declared
dependencies materialized. The verifier is a SEPARATE invocation from the worker,
given only the epoch goal + criteria + the produced artifacts, told to find gaps and
DEFAULT TO FAIL. Its only output is ``verdict.json``, a re-read disk contract the
core re-validates with Pydantic (stdout is never parsed), exactly like
``vision_review``.

The agentic pass can ONLY fail an epoch the deterministic floor already cleared, it
never rubber-stamps past the floor. A verdict that cannot be produced (the transport
raised, no ``verdict.json``, an invalid one) is itself a FAIL (``VerificationError``):
verification defaults to fail-safe, never to a silent pass.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Protocol

from grindstone.contracts.models import (
    CmdCheck,
    EpochVerdict,
    ImplementTask,
    parse_epoch_verdict,
)
from grindstone.worker import (
    VERDICT_FILENAME,
    VerificationBrief,
    WorkerRequest,
    WorkerTransport,
)

#: DoS sanity backstop on the ``verdict.json`` disk read (the principle's item F). The
#: verdict is read in FULL (no content cap) and delivered to the planner by reference;
#: this is purely a robustness guard that REJECTS (fail-safe, never truncates) an
#: absurd/corrupt multi-megabyte file before it is read into memory. Generous so it only
#: ever fires on a pathological file, never on real verifier output.
VERDICT_MAX_BYTES = 8 * 1024 * 1024


class VerificationError(Exception):
    """The verification pass could not produce a valid verdict (→ epoch FAIL)."""


class EpochVerifier(Protocol):
    """The seam the run loop calls to verify one completed epoch.

    ``worktree`` is the eval CWD (a checkout of the epoch's integration tip with deps
    materialized); ``brief`` carries the epoch goal + criteria + produced artifacts.
    ``verdict_dest`` is the STABLE run-dir path the produced ``verdict.json`` is
    relocated to (so it survives the worktree teardown and is delivered to the planner
    BY REFERENCE via the ``<workspace>`` manifest). Returns the parsed verdict or raises
    ``VerificationError``.
    """

    def verify(
        self,
        *,
        worktree: Path,
        brief: VerificationBrief,
        task_id: str,
        verdict_dest: Path,
    ) -> EpochVerdict: ...


class WorkerEpochVerifier:
    """``EpochVerifier`` backed by the LOCAL tier ``WorkerTransport`` (G4).

    The verification pass is just another local-tier dispatch behind the uniform
    worker boundary: a ``WorkerRequest`` carrying the ``VerificationBrief`` (so the
    transport builds the adversarial prompt via ``build_verification_prompt``), run
    in the prepared tip worktree. The result channel is ``verdict.json`` in that
    worktree, NOT a handoff, so this adapter re-reads + Pydantic-validates it after
    the transport returns. No model identity here, the local transport owns it.
    """

    def __init__(self, transport: WorkerTransport) -> None:
        self._transport = transport

    def verify(
        self,
        *,
        worktree: Path,
        brief: VerificationBrief,
        task_id: str,
        verdict_dest: Path,
    ) -> EpochVerdict:
        request = WorkerRequest(
            task=_verification_task(),
            task_id=task_id,
            inputs={},
            scratch=worktree,
            attempt=1,
            failure_context=[],
            mode="review",
            verification=brief,
        )
        try:
            self._transport.run(request)
        except Exception as exc:  # a transport raise is a FAIL-SAFE, never a pass
            raise VerificationError(
                f"verification transport failed: {type(exc).__name__}: {exc}"
            ) from exc
        verdict_file = worktree / VERDICT_FILENAME
        if not verdict_file.is_file():
            raise VerificationError(f"verifier wrote no {VERDICT_FILENAME}")
        # DoS sanity backstop (item F): reject (never truncate) an absurd verdict file
        # before reading it. Normal verifier output is far under the limit and read in
        # full; this only fires on a pathological/corrupt file (fail-safe -> epoch FAIL).
        try:
            size = verdict_file.stat().st_size
        except OSError as exc:
            raise VerificationError(f"{VERDICT_FILENAME} unreadable: {exc}") from exc
        if size > VERDICT_MAX_BYTES:
            raise VerificationError(
                f"{VERDICT_FILENAME} is {size} bytes, over the {VERDICT_MAX_BYTES}-byte "
                f"DoS guard; rejecting (fail-safe), not truncating"
            )
        try:
            payload = json.loads(verdict_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise VerificationError(
                f"{VERDICT_FILENAME} is not valid JSON: {exc}"
            ) from exc
        try:
            verdict = parse_epoch_verdict(payload)
        except ValueError as exc:
            raise VerificationError(f"{VERDICT_FILENAME} invalid: {exc}") from exc
        # Relocate the full verdict to its STABLE run-dir keyed-log path so it survives
        # the eval-worktree teardown and reaches the planner BY REFERENCE (the
        # <workspace> manifest). The relocated copy is the durable record; the parsed
        # value is returned for the loop's immediate routing.
        verdict_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(verdict_file, verdict_dest)
        return verdict


def _verification_task() -> ImplementTask:
    """The minimal placeholder task a verification request carries.

    The real brief rides ``WorkerRequest.verification``; the prompt builder branches
    on it before reading this task. It exists only to satisfy the request's typed
    ``task`` slot with a non-empty ``done_when`` (the verifier never runs it)."""

    return ImplementTask(
        id="T1",
        goal="verify the epoch's criteria against the produced artifacts",
        done_when=[CmdCheck(cmd="true")],
        file_ownership=["**"],
    )
