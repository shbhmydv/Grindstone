"""Per-task agentic verification, at the tier that BUILT the task.

After a task clears its deterministic floor (its ``done_when`` re-run, scope, and
grounding all pass) the core runs ONE adversarial verification of THAT task against
ITS OWN natural-language ``criteria``, at the task's tier (a ``senior`` task is
verified by the senior verifier, every other task by the local verifier). The
verifier is a FRESH instance, an independent critic: it sees only the task goal +
criteria + produced artifact, never the builder's session/context. It runs in the
task's own scratch (the implement worktree, or a checkout the caller prepares for a
non-write task), reads the real files, hunts for gaps, and DEFAULTS TO FAIL.

Its only output is ``verdict.json``, a re-read disk contract the core re-validates
with Pydantic (stdout is never parsed); a transport raise or a missing/invalid verdict
is itself a FAIL (``VerificationError``), never a silent pass. The pass can ONLY fail a
task the deterministic floor already cleared.

On a RE-verification (a retry of a task whose prior agentic verification failed) the
brief carries the verifier's OWN prior verdict (``brief.prior_verdict``): the new pass
CONFIRMS the previously-failed criteria are now closed and REGRESSION-checks the
previously-passed ones, so the gap set shrinks monotonically and cannot oscillate.
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
    """The verification pass could not produce a valid verdict (→ task FAIL)."""


class TaskVerifier(Protocol):
    """The seam the task loop calls to verify ONE task at its tier.

    ``scratch`` is the task's own CWD (the implement worktree post-commit, or a
    non-write task's checkout): the verifier reads the produced files there. ``brief``
    carries the task goal + that task's criteria + the produced artifact pointer, and
    (on a re-verification) the verifier's prior verdict as the convergence anchor.
    ``verdict_dest`` is the STABLE run-dir path the produced ``verdict.json`` is
    relocated to (so it survives the worktree teardown and reaches the planner by
    reference). Returns the parsed verdict or raises ``VerificationError``.
    """

    def verify(
        self,
        *,
        scratch: Path,
        brief: VerificationBrief,
        task_id: str,
        verdict_dest: Path,
    ) -> EpochVerdict: ...


class WorkerTaskVerifier:
    """``TaskVerifier`` backed by a single-tier ``WorkerTransport``.

    The verification pass is just another dispatch behind the uniform worker
    boundary: a ``WorkerRequest`` carrying the ``VerificationBrief`` (so the transport
    builds the adversarial prompt via ``build_verification_prompt``), run in the task's
    scratch. The result channel is ``verdict.json`` in that scratch, NOT a handoff, so
    this adapter re-reads + Pydantic-validates it after the transport returns. The TIER
    is the transport: the task loop selects the local-tier or senior-tier verifier to
    match the tier that built the task, so the critic is at the builder's strength.
    """

    def __init__(self, transport: WorkerTransport) -> None:
        self._transport = transport

    def verify(
        self,
        *,
        scratch: Path,
        brief: VerificationBrief,
        task_id: str,
        verdict_dest: Path,
    ) -> EpochVerdict:
        request = WorkerRequest(
            task=_verification_task(),
            task_id=task_id,
            inputs={},
            scratch=scratch,
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
        verdict_file = scratch / VERDICT_FILENAME
        if not verdict_file.is_file():
            raise VerificationError(f"verifier wrote no {VERDICT_FILENAME}")
        # DoS sanity backstop (item F): reject (never truncate) an absurd verdict file
        # before reading it. Normal verifier output is far under the limit and read in
        # full; this only fires on a pathological/corrupt file (fail-safe -> task FAIL).
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
        # the scratch/worktree teardown and reaches the planner BY REFERENCE (the
        # <workspace> manifest). The relocated copy is the durable record; the parsed
        # value is returned for the loop's immediate routing + the next anchor.
        verdict_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(verdict_file, verdict_dest)
        # Remove the scratch verdict so a chained retry's worktree never inherits a stale
        # verdict.json (it would otherwise read as an out-of-scope write at commit).
        verdict_file.unlink(missing_ok=True)
        return verdict


def _verification_task() -> ImplementTask:
    """The minimal placeholder task a verification request carries.

    The real brief rides ``WorkerRequest.verification``; the prompt builder branches
    on it before reading this task. It exists only to satisfy the request's typed
    ``task`` slot with a non-empty ``done_when`` (the verifier never runs it)."""

    return ImplementTask(
        id="T1",
        goal="verify the task's criteria against the produced artifact",
        done_when=[CmdCheck(cmd="true")],
        file_ownership=["**"],
    )
