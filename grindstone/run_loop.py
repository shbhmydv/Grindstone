"""The multi-epoch run loop: stateless planner ⇄ deterministic core (ARCHITECTURE.md).

This is S3's spine. A stateless one-shot planner is called with constructed
input, returns ONE decision as constrained JSON, the core validates and executes
it, and the result feeds the next call, until ``complete_run`` (terminal
success) or escalation (terminal, needs a human)::

    loop:
      input    = stable_head(skeleton) + volatile_tail(state, last_epoch, request)
      raw      = transport.plan(input)                 # codex exec | mock
      decision = extract -> schema -> typed -> semantic ; invalid -> re-ask (<=2)
      dispatch on tool name:
        propose_skeleton -> store skeleton (legal only while none exists)
        implement|research|review|artifact -> run_epoch -> epoch report feeds next
        revise_phases    -> replace skeleton tail (S3: whole skeleton, none done)
        escalate_run     -> terminal escalation
        complete_run     -> re-run evidence deterministically; pass -> success,
                            fail -> reject + re-ask with the failing evidence

Phases stay THIN at S3 (brief): the skeleton is stored, validated, fed to the
stable head, and the run lives in its FIRST phase, epoch ids increment within
it (E1, E2, …). Exit-criteria evaluation, budgets, and phase escalation are S4.

Epoch chaining (ruling 4): epoch N+1's base is epoch N's integration tip when
one exists, else repo HEAD; the run's final branch is the last integration
branch (merging to the user's branch stays manual).

Run-level durability (rulings 6/7): ``RunState`` is rewritten atomically to a
file DISTINCT from the epoch's ``state.json``. A kill while a planner call is
in flight leaves ``status=awaiting_planner`` and nothing else on disk (planner
calls are side-effect-free), so resume simply RE-ISSUES the call, no burn,
unlike workers. A kill mid-epoch leaves ``status=running_epoch`` + the pending
decision; resume delegates to ``resume_epoch`` then continues the loop.
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Literal, Sequence, cast

from pydantic import BaseModel, ConfigDict

from grindstone import worktree as wt
from grindstone.config import FloorConfig, InfraRepairConfig, PrepareConfig
from grindstone.infra import InfraClassification, classify_check_failure
from grindstone.prepare import PrepareError, materialize_env
from grindstone.planner import extract_decision_json
from grindstone.contracts.models import (
    ArtifactEpochArgs,
    ArtifactExistsCheck,
    Check,
    CmdCheck,
    CompleteRunDecision,
    EpochDecision,
    EpochVerdict,
    EscalateRunDecision,
    HaltFailedEpochArgs,
    HandleFailedEpochDecision,
    ImplementEpochArgs,
    ImplementTask,
    Phase,
    ProposeSkeletonDecision,
    RetryFailedEpochArgs,
    RevisePhasesDecision,
    VisionReviewCheck,
    parse_decision,
)
from grindstone.script_polish import Polisher
from grindstone.script_vision import VisionReviewError, VisionReviewer
from grindstone.verify import EpochVerifier, VerificationError
from grindstone.contracts.semantics import HandoffMode
from grindstone.epoch_loop import EpochArgs, EpochOutcome, resume_epoch, run_epoch
from grindstone.journal import reap_sibling_journals, write_journal
from grindstone.memory import load_digest
from grindstone.events import (
    EpochFailed,
    EpochVerificationFailed,
    EpochVerificationPassed,
    EpochVerificationStarted,
    FailedEpochHandled,
    FinalPolishApplied,
    FinalPolishSkipped,
    InfraCheckDetected,
    InfraRepairDispatched,
    InfraRepairExhausted,
    InfraRepairResolved,
    JournalWriter,
    PhaseEscalated,
    PhasePassed,
    PhaseRef,
    PhasesRevised,
    PhaseStarted,
    PlannerCallFailed,
    PlannerCallStarted,
    PlannerCallSucceeded,
    RunCompleted,
    RunEscalated,
    RunFailed,
    RunResumed,
    RunStarted,
    SkeletonProposed,
    read_events,
)
from grindstone.planner import (
    MAX_RATE_LIMIT_WAITS,
    MAX_REASKS,
    DEFAULT_LOCAL_MAX_TASK_FILES,
    DEFAULT_SENIOR_MAX_TASK_FILES,
    MAX_TRANSIENT_RETRIES,
    FailedEpochInfo,
    PhaseTailInfo,
    PlannerTransport,
    WorkspaceInfo,
    backoff_delay,
    build_planner_input,
    classify_failure,
    flatten_last_epoch,
    validate_decision,
)
from grindstone.repomap import build_repo_map
from grindstone.rundir import RunDir, atomic_write_json
from grindstone.task_loop import TIER0_ATTEMPTS
from grindstone.worker import (
    InfraRepairBrief,
    VerificationBrief,
    WorkerRequest,
    WorkerTransport,
)

RunStatus = Literal["awaiting_planner", "running_epoch", "completed", "escalated", "failed"]
SleepFn = Callable[[float], None]
Ladder = Sequence[tuple[str, WorkerTransport]]

#: Cap on the integration-tip file listing surfaced in the tail (ruling 3b):
#: a reference, not a payload, the full count is always reported alongside.
TIP_LISTING_CAP = 200

#: Deterministic default for the per-phase failed-epoch cap (Part C): after this
#: many failed epochs in one phase the state machine FORCES a halt-to-human
#: regardless of the planner, so the dogfood spin-loop (15 identical repairs)
#: can never recur. Config (``max_failed_epochs_per_phase``) overrides it.
DEFAULT_MAX_FAILED_EPOCHS_PER_PHASE = 3

_EPOCH_MODE: dict[str, HandoffMode] = {
    "implement": "implement",
    "research": "research",
    "review": "review",
    "artifact": "artifact",
}


def _now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


# --- durable run state + returned outcome (ruling 7) ---------------------------


class RunState(BaseModel):
    """The run-level cursor (atomic full rewrite to ``run_state_path``)."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    job_path: str
    job_text: str
    status: RunStatus
    skeleton: list[Phase] | None
    current_phase_id: str | None
    epoch_counter: int
    planner_call_count: int
    rate_limit_waits: int
    last_integration_branch: str | None
    pending_decision: dict[str, Any] | None
    terminal_reason: str | None
    #: Repo-memory digest frozen at run start (S5 seam): ``None`` when the repo
    #: has no ``.grindstone/memory/digest.md``. Held here so the planner's stable
    #: head stays byte-identical run-long even though the file could change on
    #: disk; defaulted so S3/S4-crafted states still load.
    repo_memory: str | None = None
    # --- S4 phase machinery (defaulted so S3-crafted states still load) --------
    #: Per-phase epoch index → the epoch id ``E{n+1}``; resets to 0 on phase
    #: advance (NOT on revise, keeping it monotonic avoids E-dir collisions).
    phase_epoch_index: int = 0
    #: Epochs charged to the current phase's budget; resets on advance AND revise.
    phase_budget_used: int = 0
    #: Phase ids whose exit criterion has passed (skeleton order), never reused.
    passed_phase_ids: list[str] = []
    #: True while the current phase is under a budget-exhaustion escalation demand.
    phase_escalation_active: bool = False
    #: The epoch decision that produced a FAILED epoch awaiting a focused
    #: handle_failed_epoch disposition (retry re-dispatches it), plus the captured
    #: failure context (failed task reasons, failed phase checks WITH output, the
    #: handoffs that claimed pass). ``None`` = no failed epoch is pending. Defaulted
    #: so older states still load.
    pending_failed_epoch: dict[str, Any] | None = None
    #: FAILED epochs disposed of within the CURRENT phase (the deterministic
    #: spin-loop cap, Part C); resets on phase advance AND revise.
    phase_failed_epochs: int = 0
    #: The DURABLE marker that a just-completed epoch carrying ``criteria`` still owes
    #: the G4 agentic verification pass (the originating ``decision`` + ``epoch_id``).
    #: Set when ``_record_epoch`` flips the epoch to ``awaiting_planner``; CLEARED only
    #: once verification reaches a terminal classification (a Passed/Failed event, an
    #: infra-failure, or no criteria to judge). On resume, a set marker WITHOUT a
    #: terminal verification event in the journal means a kill landed AFTER the epoch
    #: persisted but BEFORE the gate finished -> the pass is RE-RUN before the next
    #: planner boundary, so verification can never be skipped by loop position. A
    #: verdict already produced + recorded (terminal event present) is reused, never
    #: regenerated. ``None`` = nothing owes verification. Defaulted so older states load.
    pending_verification: dict[str, Any] | None = None


@dataclass(frozen=True)
class FinalPolish:
    """The B5 final-polish wiring the CLI assembles from config (off when absent).

    ``polisher`` runs codex inline against a writable worktree of the final branch;
    ``criteria`` is the polish brief; ``screenshot_rel`` is an optional
    worktree-relative image for a visual pass. Passed to ``run_grind`` /
    ``resume_grind``; ``None`` there = the pass never runs (the default).
    """

    polisher: Polisher
    criteria: str
    screenshot_rel: str | None = None


@dataclass(frozen=True)
class RunOutcome:
    """The returned result of a run (ruling 7)."""

    status: Literal["completed", "escalated", "failed"]
    reason: str | None
    summary: str | None
    planner_calls: int
    epochs_run: int
    final_branch: str | None


class _RunStateStore:
    """Single-writer owner of ``RunState`` + its file (the loop is synchronous)."""

    def __init__(self, run_dir: RunDir, state: RunState) -> None:
        self._run_dir = run_dir
        self._state = state
        self._flush()

    @property
    def state(self) -> RunState:
        return self._state

    def update(self, **fields: Any) -> None:
        self._state = self._state.model_copy(update=fields)
        self._flush()

    def _flush(self) -> None:
        atomic_write_json(self._run_dir.run_state_path, self._state.model_dump(mode="json"))


# --- internal control signals (unwind the transport-retry inner loop) ----------


class _Escalate(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class _Valve(Exception):
    """The TEST-only safety valve tripped (max planner calls / epochs)."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class _Boundary:
    kind: Literal["decision", "escalate", "valve"]
    decision: EpochDecision | None = None
    reason: str | None = None


# --- deterministic check evaluation (evidence + phase exit criteria) -----------


def _check_label(check: Check) -> str:
    if isinstance(check, CmdCheck):
        return f"cmd `{check.cmd}`"
    if isinstance(check, ArtifactExistsCheck):
        return f"artifact_exists:{check.artifact_exists}"
    return f"vision_review:{check.vision_review.screenshot}"


def _safe_output(stdout: str, stderr: str) -> str:
    """The text-safe combined output of a failed command, in FULL (no byte cap).

    Joins stdout + stderr and scrubs the result text-safe: control bytes other than
    tab/newline are stripped so an odd-byte build log can never corrupt the persisted
    record. The FULL output is kept (the principle: agent inputs are delivered by
    reference on disk, never truncated-and-embedded). Empty when the command said
    nothing.
    """

    parts = [p for p in (stdout, stderr) if p]
    combined = "\n".join(parts).strip()
    if not combined:
        return ""
    raw = combined.encode("utf-8", errors="replace")
    text = raw.decode("utf-8", errors="replace")
    return "".join(c for c in text if c in "\t\n" or c >= " ")


def _persist_check_output(
    run_dir: RunDir, scratch_name: str, index: int, cmd: str, output: str
) -> Path | None:
    """Durably record a failed check's FULL captured output under the run dir.

    A flat text file per (eval scratch, check index) so the failure trail survives the
    run, is auditable alongside the journal, and is delivered to the planner BY
    REFERENCE (the label carries this path; the planner reads the file for the full
    failing output). Best-effort: a persist failure returns ``None`` (no path to
    surface) and never breaks check evaluation."""

    try:
        out_dir = run_dir.root / "check_output" / scratch_name
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"c{index}.txt"
        path.write_text(f"$ {cmd}\n{output}\n", encoding="utf-8")
        return path
    except OSError:
        return None


def _vision_result(
    check: VisionReviewCheck,
    *,
    cwd: Path,
    run_dir: RunDir,
    scratch_name: str,
    index: int,
    reviewer: VisionReviewer | None,
) -> tuple[str, bool]:
    """Run one taste check through the configured reviewer; ``(label, passed)``.

    Deterministic FAIL, never a crash, when no reviewer is wired, the screenshot
    a prior check should have produced is absent, or the script/verdict path fails
    (``VisionReviewError``). On a clean PASS the label is the bare check; on a
    FAIL the verdict reasons are surfaced into the label (the failing-evidence
    re-ask + the journal both read it).
    """

    spec = check.vision_review
    label = _check_label(check)
    if reviewer is None:
        return (f"{label} (no vision reviewer configured)", False)
    # Defence-in-depth over the contract's relative-path pattern: resolve symlinks
    # and require the screenshot to stay INSIDE the eval worktree, so a planner- or
    # repo-planted link can never steer codex's `-i` at an off-worktree image.
    cwd_root = cwd.resolve()
    screenshot = (cwd / spec.screenshot).resolve()
    if not screenshot.is_relative_to(cwd_root):
        return (f"{label} (screenshot escapes worktree: {spec.screenshot})", False)
    if not screenshot.is_file():
        return (f"{label} (screenshot missing: {spec.screenshot})", False)
    out_dir = run_dir.root / "vision" / scratch_name / f"v{index}"
    try:
        verdict = reviewer.review(
            worktree=cwd,
            screenshot_rel=spec.screenshot,
            criteria=spec.criteria,
            out_dir=out_dir,
        )
    except VisionReviewError as exc:
        return (f"{label} (review error: {exc})", False)
    if verdict.passed:
        return (label, True)
    reasons = "; ".join(verdict.reasons) or "no reasons given"
    return (f"{label} FAILED: {reasons}", False)


@dataclass(frozen=True)
class CheckResult:
    """One deterministic check's verdict, with infra detail on a cmd FAILURE.

    ``label`` / ``ok`` are the legacy gate tuple; ``cmd`` is the source command
    string for a FAILED ``CmdCheck`` (else ``None``); ``infra`` is the shared
    classifier's verdict for that failure (``None`` when the check passed or is not
    a command). The infra-repair loop reads ``cmd`` + ``infra`` to decide whether
    a failure is environmental; every other caller folds to ``(label, ok)``."""

    label: str
    ok: bool
    cmd: str | None = None
    infra: InfraClassification | None = None


def _evaluate_checks_detailed(
    checks: Sequence[Check],
    *,
    repo: Path | None,
    ref: str | None,
    run_dir: RunDir,
    scratch_name: str,
    vision_reviewer: VisionReviewer | None = None,
    prepare: PrepareConfig | None = None,
    floor: FloorConfig | None = None,
) -> list[CheckResult]:
    """Run the checks and return ``CheckResult`` rows (the infra-aware evaluator).

    Identical worktree/prepare/floor semantics to ``evaluate_checks`` (its thin
    wrapper), but a FAILED ``CmdCheck`` additionally carries the source command and
    the shared infra classification (``infra.classify_check_failure``) so the gate
    can tell an environmental fault from a genuine assertion failure."""

    if floor is not None:
        checks = [*checks, *(CmdCheck(cmd=cmd) for cmd in floor.checks)]
    needs_worktree = any(
        isinstance(c, (CmdCheck, VisionReviewCheck)) for c in checks
    )
    worktree: Path | None = None
    worktree_error: str | None = None
    if needs_worktree and repo is not None:
        worktree = run_dir.root / "worktrees" / scratch_name
        try:
            wt.add_worktree_detached(repo, worktree, ref=ref or "HEAD")
        except wt.GitError:
            # Unborn HEAD (a fresh repo with zero commits) or an unresolvable ref:
            # there is no tip to check out and evaluate against. cmd/vision checks
            # FAIL deterministically (the phase simply hasn't passed) instead of
            # letting the GitError escape evaluate_checks and crash the whole run.
            worktree = None
            worktree_error = f"unresolvable eval ref {ref or 'HEAD'!r}"
        else:
            # Restore the declared (gitignored, uncommittable) dependency dirs into
            # the throwaway worktree before the cmd checks run, otherwise a build
            # gate like `npx tsc` is structurally unpassable (node_modules absent).
            # A failed prepare FAILS the cmd/vision checks with a clear reason
            # rather than silently leaving them unpassable.
            try:
                materialize_env(repo, worktree, prepare)
            except PrepareError as exc:
                worktree_error = str(exc)
    cwd = worktree if worktree is not None else run_dir.root
    try:
        results: list[CheckResult] = []
        for index, check in enumerate(checks):
            if isinstance(check, ArtifactExistsCheck):
                ok = run_dir.find_artifact(check.artifact_exists) is not None
                results.append(CheckResult(_check_label(check), ok))
            elif worktree_error is not None:
                # A cmd/vision check that needs the worktree we could not create.
                results.append(
                    CheckResult(f"{_check_label(check)} [{worktree_error}]", False)
                )
            elif isinstance(check, VisionReviewCheck):
                vlabel, vok = _vision_result(
                    check,
                    cwd=cwd,
                    run_dir=run_dir,
                    scratch_name=scratch_name,
                    index=index,
                    reviewer=vision_reviewer,
                )
                results.append(CheckResult(vlabel, vok))
            else:
                proc = subprocess.run(
                    check.cmd, shell=True, cwd=str(cwd), capture_output=True, text=True
                )
                ok = proc.returncode == check.expect_exit
                label = _check_label(check)
                infra: InfraClassification | None = None
                if not ok:
                    # Gate observability (Part A): a FAILED cmd check surfaces WHY it
                    # failed so the planner can tell env-vs-code (the dogfood spin-loop
                    # blind spot). The FULL output is persisted under the run dir and
                    # delivered BY REFERENCE: the label carries the PATH (the planner
                    # reads the file for the complete failing output), never an embedded
                    # truncated tail.
                    output = _safe_output(proc.stdout, proc.stderr)
                    label = f"{label} (exit {proc.returncode})"
                    if output:
                        out_path = _persist_check_output(
                            run_dir, scratch_name, index, check.cmd, output
                        )
                        if out_path is not None:
                            label += f"\n      output_file: {out_path}"
                    # Classify the failure (G3): environmental vs genuine, so the
                    # gate can route an infra fault to senior repair, not a charge.
                    infra = classify_check_failure(
                        returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr
                    )
                results.append(
                    CheckResult(label, ok, cmd=check.cmd if not ok else None, infra=infra)
                )
        return results
    finally:
        if worktree is not None and repo is not None:
            wt.remove_worktree(repo, worktree)


def evaluate_checks(
    checks: Sequence[Check],
    *,
    repo: Path | None,
    ref: str | None,
    run_dir: RunDir,
    scratch_name: str,
    vision_reviewer: VisionReviewer | None = None,
    prepare: PrepareConfig | None = None,
    floor: FloorConfig | None = None,
) -> list[tuple[str, bool]]:
    """Run a list of deterministic checks; return ``(label, passed)`` IN ORDER.

    Command checks run against a throwaway worktree of ``ref`` (the integration
    tip / final branch) when a repo exists, else the run dir; artifact checks
    resolve against the keyed log; ``vision_review`` (B3 taste gate) renders its
    verdict through ``vision_reviewer`` against a screenshot a PRIOR cmd check
    produced in the same worktree. This is the one evaluator behind both
    ``complete_run`` evidence (ARCHITECTURE.md) and phase exit criteria (S4 ruling 1):
    a deterministic verdict computed in a tip worktree, never a planner claim.

    ``floor`` (the gate-rebalance deterministic floor) appends the repo-owned
    canonical verification commands AFTER the supplied checks, so they run in the
    SAME ``prepare``-materialized worktree with identical cmd-check semantics (exit
    0 == pass, captured output on failure). The floor is core-owned, not authored
    by the planner; a floor-check failure fails the gate exactly like a failed
    ``done_when``. A thin ``(label, ok)`` projection of ``_evaluate_checks_detailed``.
    """

    return [
        (r.label, r.ok)
        for r in _evaluate_checks_detailed(
            checks,
            repo=repo,
            ref=ref,
            run_dir=run_dir,
            scratch_name=scratch_name,
            vision_reviewer=vision_reviewer,
            prepare=prepare,
            floor=floor,
        )
    ]


def recheck_evidence(
    evidence: Sequence[Check],
    *,
    repo: Path | None,
    final_branch: str | None,
    run_dir: RunDir,
    vision_reviewer: VisionReviewer | None = None,
    prepare: PrepareConfig | None = None,
    floor: FloorConfig | None = None,
) -> list[str]:
    """Deterministically re-run a ``complete_run``'s evidence; return the failing
    labels (empty = the run's certificate holds). Command checks run against the
    final branch worktree, artifact checks against the keyed log (ARCHITECTURE.md),
    vision_review checks through ``vision_reviewer``. The deterministic ``floor``
    runs against the final branch too, so a run cannot complete past it."""

    return [
        label
        for label, ok in evaluate_checks(
            evidence,
            repo=repo,
            ref=final_branch,
            run_dir=run_dir,
            scratch_name="_evidence",
            vision_reviewer=vision_reviewer,
            prepare=prepare,
            floor=floor,
        )
        if not ok
    ]


# --- phase machinery: live exit criteria, advancement, budgets (rulings 1-3) ---


@dataclass(frozen=True)
class _PhaseContext:
    """Everything the phase machinery hands the next planner call (rulings 2-3)."""

    info: PhaseTailInfo
    escalation_active: bool


def _phase_index(skeleton: list[Phase], phase_id: str | None) -> int:
    for i, phase in enumerate(skeleton):
        if phase.id == phase_id:
            return i
    # current_phase_id always names a skeleton phase once one exists; default to
    # the first so a corrupt cursor degrades to "re-evaluate phase 1", never crash.
    return 0


def _passed_in_order(skeleton: list[Phase], passed: set[str]) -> list[str]:
    return [p.id for p in skeleton if p.id in passed]


def _eval_ref(store: _RunStateStore, repo: Path | None) -> str | None:
    """The ref phase exit criteria evaluate against: the integration tip, else
    repo HEAD (ruling 1). ``None`` when there is no repo (artifact-only runs)."""

    if repo is None:
        return None
    return store.state.last_integration_branch or "HEAD"


def _tip_listing(store: _RunStateStore, repo: Path | None) -> tuple[list[str], int]:
    """The integration-tip file listing for the tail (ruling 3b): names + total."""

    if repo is None:
        return [], 0
    files = wt.list_tree(repo, store.state.last_integration_branch or "HEAD")
    return files[:TIP_LISTING_CAP], len(files)


def _advance_phases(
    journal: JournalWriter,
    store: _RunStateStore,
    run_dir: RunDir,
    repo: Path | None,
    passed: set[str],
    started: set[str],
    vision_reviewer: VisionReviewer | None,
    prepare: PrepareConfig | None,
    floor: FloorConfig | None,
) -> list[tuple[str, bool]]:
    """Evaluate the current phase and advance through every satisfied one.

    Each loop: evaluate the current phase's exit criterion in a tip worktree
    (ruling 1). All checks pass -> ``phase_passed`` (guarded on recorded state so
    resume never double-emits, ruling 6) and, unless this is the LAST phase,
    advance to the next (``phase_started``, counters reset, ruling 1). The last
    phase passing does NOT auto-complete, the planner still owns ``complete_run``.
    Returns the CURRENT phase's per-check results once advancement settles.
    """

    skeleton = store.state.skeleton
    assert skeleton is not None
    # Advance by POSITION, resolved from the cursor ONCE: re-looking-up the current
    # id each pass would resolve a DUPLICATE phase id backward and hang forever (the
    # validator rejects dup ids, but the loop stays safe regardless). The skeleton
    # is stable here, _advance_phases never revises it.
    idx = _phase_index(skeleton, store.state.current_phase_id)
    while True:
        phase = skeleton[idx]
        results = evaluate_checks(
            phase.exit_criterion,
            repo=repo,
            ref=_eval_ref(store, repo),
            run_dir=run_dir,
            scratch_name="_phase_eval",
            vision_reviewer=vision_reviewer,
            prepare=prepare,
            floor=floor,
        )
        if not all(ok for _, ok in results):
            return results
        # A pending failed epoch (a B6 task/gate failure OR a G4 semantic-verification
        # failure) MUST be disposed of by the planner before the phase advances. The
        # deterministic gate can pass while a semantic criterion is still unmet, so
        # advancing here would CLEAR the pending disposition (counters reset below)
        # and let an incomplete epoch slip the phase. Hold at the current phase, do
        # not emit phase_passed, and let the failed-epoch boundary run first.
        if store.state.pending_failed_epoch is not None:
            return results
        if phase.id not in passed:
            journal.emit(lambda s: PhasePassed(seq=s, ts=_now(), phase_id=phase.id))
            passed.add(phase.id)
            store.update(passed_phase_ids=_passed_in_order(skeleton, passed))
        if idx + 1 >= len(skeleton):
            return results  # last phase passed: no auto-advance, no auto-complete
        idx += 1
        nxt = skeleton[idx]
        store.update(
            current_phase_id=nxt.id,
            phase_epoch_index=0,
            phase_budget_used=0,
            phase_escalation_active=False,
            phase_failed_epochs=0,
            pending_failed_epoch=None,
        )
        if nxt.id not in started:
            journal.emit(lambda s: PhaseStarted(seq=s, ts=_now(), phase_id=nxt.id))
            started.add(nxt.id)


def _phase_preamble(
    journal: JournalWriter,
    store: _RunStateStore,
    run_dir: RunDir,
    repo: Path | None,
    passed: set[str],
    started: set[str],
    vision_reviewer: VisionReviewer | None,
    prepare: PrepareConfig | None,
    floor: FloorConfig | None,
) -> _PhaseContext:
    """Advance phases, fire a one-shot ``phase_escalated`` on budget exhaustion,
    and assemble the cumulative-state tail context (rulings 1-3)."""

    results = _advance_phases(
        journal, store, run_dir, repo, passed, started, vision_reviewer, prepare, floor
    )
    skeleton = store.state.skeleton
    assert skeleton is not None
    phase = skeleton[_phase_index(skeleton, store.state.current_phase_id)]
    current_failed = any(not ok for _, ok in results)
    budget_used = store.state.phase_budget_used
    if (
        current_failed
        and budget_used >= phase.epoch_budget
        and not store.state.phase_escalation_active
    ):
        journal.emit(lambda s: PhaseEscalated(seq=s, ts=_now(), phase_id=phase.id))
        store.update(phase_escalation_active=True)
    escalation_active = store.state.phase_escalation_active
    tip_files, tip_total = _tip_listing(store, repo)
    info = PhaseTailInfo(
        title=phase.title,
        check_results=results,
        budget_used=budget_used,
        budget=phase.epoch_budget,
        passed_ids=_passed_in_order(skeleton, passed),
        escalation_active=escalation_active,
        tip_files=tip_files,
        tip_total=tip_total,
    )
    return _PhaseContext(info, escalation_active)


# --- one planner boundary: transport retries + re-ask ladder -------------------


def _transport_call(
    journal: JournalWriter,
    store: _RunStateStore,
    planner: PlannerTransport,
    sleep_fn: SleepFn,
    prompt: str,
    max_planner_calls: int | None,
) -> str:
    """Call the transport, surviving rate-limit / transient failures (ruling 2).

    rate_limit → injected-``sleep_fn`` exponential backoff (max 6 waits/run,
    durable); transient → immediate retry (max 3 per call site, then hard); hard
    → escalate. Every attempt journals ``planner_call_started`` then, on failure,
    ``planner_call_failed(classification)``. Returns the raw text on success.
    """

    transient = 0
    while True:
        if max_planner_calls is not None and store.state.planner_call_count >= max_planner_calls:
            raise _Valve(f"safety valve: {max_planner_calls} planner calls reached")
        journal.emit(lambda s: PlannerCallStarted(seq=s, ts=_now()))
        store.update(planner_call_count=store.state.planner_call_count + 1)
        try:
            return planner.plan(prompt)
        except BaseException as exc:  # transport boundary, classify then react
            classification = classify_failure(exc)
            journal.emit(
                lambda s: PlannerCallFailed(seq=s, ts=_now(), classification=classification)
            )
            if classification == "rate_limit":
                if store.state.rate_limit_waits >= MAX_RATE_LIMIT_WAITS:
                    raise _Escalate(
                        f"rate limit: {MAX_RATE_LIMIT_WAITS} backoff waits exhausted"
                    ) from exc
                delay = backoff_delay(store.state.rate_limit_waits)
                store.update(rate_limit_waits=store.state.rate_limit_waits + 1)
                sleep_fn(delay)
                continue
            if classification == "transient":
                transient += 1
                if transient >= MAX_TRANSIENT_RETRIES:
                    raise _Escalate(
                        f"transient planner failures exhausted ({MAX_TRANSIENT_RETRIES}), treated as hard"
                    ) from exc
                continue
            raise _Escalate(f"hard planner failure: {type(exc).__name__}: {exc}") from exc


#: The boundary's checked-out tree of the CURRENT integration tip, materialized
#: INSIDE the run dir (hence inside ``$repo``) so the read-capable planner can grep
#: it. A single fixed name per boundary, torn down after; never collides with the
#: eval/verify worktrees (those carry their own scratch names).
_PLANNER_TIP_WORKTREE = "_planner_tip"
#: Stable file (run dir ROOT, outside the ephemeral tip worktree) holding the
#: structural repo-map handed to the planner BY REFERENCE. Rebuilt/overwritten each
#: boundary; it persists through the boundary's tip-worktree teardown so the planner
#: can read it within its single call.
_PLANNER_REPO_MAP_FILE = "planner_repo_map.txt"


def _build_workspace(store: _RunStateStore, run_dir: RunDir, repo: Path | None) -> WorkspaceInfo | None:
    """Materialize the read-capable workspace handles for one planner boundary.

    Checks out the CURRENT integration tip (``last_integration_branch``, else nothing
    when no branch exists yet) into ``<run_dir>/worktrees/_planner_tip`` so the
    read-capable planner can grep the exact code the gate evaluates, and resolves
    every live keyed-log key (handoffs, verdicts, relocated artifacts) PLUS the G9
    captured check-output files to its absolute path. All paths live under the run
    dir, which is inside ``$repo``, so both planner rigs (codex read-only ``-C repo``,
    claude Read+Grep cwd=repo) can read them.

    ``None`` only when there is no repo (artifact-only runs have no tree + the keyed
    log alone is already in ``<state>``). The tip checkout is best-effort: an unborn
    HEAD / unresolvable ref leaves ``integration_tip=None`` and the manifest still
    surfaces. Pure-deterministic from the run-dir layout + the tip; re-derives
    identical content on resume (same tip)."""

    if repo is None:
        return None
    tip_dir: Path | None = None
    ref = store.state.last_integration_branch
    if ref is not None:
        candidate = run_dir.root / "worktrees" / _PLANNER_TIP_WORKTREE
        try:
            wt.add_worktree_detached(repo, candidate, ref=ref)
        except wt.GitError:
            tip_dir = None  # unresolvable tip: the manifest still helps the planner
        else:
            tip_dir = candidate
    manifest: list[tuple[str, Path]] = [
        (key, run_dir.resolve(key)) for key in run_dir.log_index()
    ]
    # G9 captured check-output files live OUTSIDE the P*/ keyed log; surface them too
    # so the planner can read the exact failing-command output that steered the gate.
    check_root = run_dir.root / "check_output"
    if check_root.is_dir():
        for path in sorted(check_root.rglob("*")):
            if path.is_file():
                manifest.append((path.relative_to(run_dir.root).as_posix(), path))
    return WorkspaceInfo(
        integration_tip=tip_dir, keyed_log_root=run_dir.root, manifest=manifest
    )


def _attach_repo_map(
    workspace: WorkspaceInfo | None, run_dir: RunDir, repo_map: str | None
) -> WorkspaceInfo | None:
    """Persist the structural map to a stable file under the run dir ROOT and point
    the workspace at it (delivered to the planner BY REFERENCE, not inlined).

    The file lives at the run dir root, NOT inside the ``_planner_tip`` worktree the
    boundary tears down in its ``finally``, so it survives for the planner's single
    read. Rebuilt/overwritten each boundary (it reflects the current tip). A None map
    (below threshold / no tip / build failure) writes NO file and leaves the
    workspace's ``repo_map_path`` unset (the entry is omitted cleanly). The write is
    best-effort: a write error leaves the map unreferenced but never crashes the run.
    Pure-deterministic from the tip; identical content on resume (same tip)."""

    if workspace is None or repo_map is None:
        return workspace
    path = run_dir.root / _PLANNER_REPO_MAP_FILE
    try:
        path.write_text(repo_map, encoding="utf-8")
    except OSError:
        return workspace
    return replace(workspace, repo_map_path=path)


def _teardown_workspace(workspace: WorkspaceInfo | None, run_dir: RunDir, repo: Path | None) -> None:
    """Tear down the boundary's tip checkout (no leak), idempotently."""

    if workspace is None or workspace.integration_tip is None or repo is None:
        return
    wt.remove_worktree(repo, workspace.integration_tip)


def _plan_boundary(
    journal: JournalWriter,
    store: _RunStateStore,
    run_dir: RunDir,
    planner: PlannerTransport,
    sleep_fn: SleepFn,
    repo: Path | None,
    last_epoch_rows: list[dict[str, Any]] | None,
    max_planner_calls: int | None,
    phase_ctx: _PhaseContext | None,
    completed_phase_ids: frozenset[str],
    vision_reviewer: VisionReviewer | None,
    prepare: PrepareConfig | None,
    floor: FloorConfig | None,
    failed_epoch: FailedEpochInfo | None,
    has_senior: bool,
    local_max_task_files: int,
    senior_max_task_files: int,
) -> _Boundary:
    """Drive planner calls at one epoch boundary to a dispatchable decision.

    The constructed input carries the fresh phase-status tail (``phase_ctx``,
    rulings 1-3); the gate enforces phase-escalation position legality (ruling 2)
    and rejects ``revise_phases`` that reuses an already-passed phase id. Re-asks
    on an invalid decision (journaled ``planner_call_failed("transient")``) and on
    a ``complete_run`` whose evidence fails, both share the ≤2 re-ask budget;
    exhausting it escalates the run.

    When ``failed_epoch`` is set, the boundary is a FOCUSED disposition: the input
    carries the failed-epoch context (Part B) and the gate constrains the decision
    to ``handle_failed_epoch``.
    """

    reasks = 0
    reask_errors: list[str] = []
    # Read-capable workspace: a checked-out tip tree + a resolvable keyed-log
    # manifest, INSIDE the run dir so both planner rigs can grep it. Built once per
    # boundary (the tip is stable across this boundary's re-asks) and torn down after.
    # Built FIRST so the repo-map below can read the tip checkout it materializes.
    workspace = _build_workspace(store, run_dir, repo)
    try:
        # Whole-repo structural map, delivered to the planner BY REFERENCE (a file
        # path in the <workspace> manifest), never inlined into the prompt. Built from
        # the integration-tip CHECKOUT when one exists (the exact code the epochs have
        # built; the operator tree is deliberately never advanced to the tip), else
        # from the operator tree (first epoch / no tip yet). Reuses the workspace's
        # _planner_tip worktree, so no second checkout is created. Built once per
        # boundary (the tip is stable across this boundary's re-asks); None below
        # threshold / on any failure, the run is unaffected either way.
        if repo is None:
            repo_map = None
        elif workspace is not None and workspace.integration_tip is not None:
            repo_map = build_repo_map(workspace.integration_tip)
        else:
            repo_map = build_repo_map(repo)
        # Persist the map under the run dir ROOT (NOT the ephemeral _planner_tip
        # worktree, which the finally below tears down before the planner reads it),
        # then hand the planner the PATH via the workspace. Rebuilt/overwritten each
        # boundary so it reflects the current tip. None map => no file, no entry.
        workspace = _attach_repo_map(workspace, run_dir, repo_map)
        return _plan_boundary_loop(
            journal, store, run_dir, planner, sleep_fn, repo, last_epoch_rows,
            max_planner_calls, phase_ctx, completed_phase_ids, vision_reviewer,
            prepare, floor, failed_epoch, has_senior, local_max_task_files,
            senior_max_task_files, workspace=workspace,
            reasks=reasks, reask_errors=reask_errors,
        )
    finally:
        _teardown_workspace(workspace, run_dir, repo)


def _plan_boundary_loop(
    journal: JournalWriter,
    store: _RunStateStore,
    run_dir: RunDir,
    planner: PlannerTransport,
    sleep_fn: SleepFn,
    repo: Path | None,
    last_epoch_rows: list[dict[str, Any]] | None,
    max_planner_calls: int | None,
    phase_ctx: _PhaseContext | None,
    completed_phase_ids: frozenset[str],
    vision_reviewer: VisionReviewer | None,
    prepare: PrepareConfig | None,
    floor: FloorConfig | None,
    failed_epoch: FailedEpochInfo | None,
    has_senior: bool,
    local_max_task_files: int,
    senior_max_task_files: int,
    *,
    workspace: WorkspaceInfo | None,
    reasks: int,
    reask_errors: list[str],
) -> _Boundary:
    """The re-ask loop body of ``_plan_boundary`` (the workspace tip checkout is
    materialized + torn down by the caller, so this stays a pure control loop)."""

    while True:
        prompt = build_planner_input(
            job=store.state.job_text,
            skeleton=store.state.skeleton,
            phase_id=store.state.current_phase_id,
            epoch_counter=store.state.epoch_counter,
            log_index=run_dir.log_index(),
            last_epoch_rows=last_epoch_rows,
            reask_errors=reask_errors,
            phase=phase_ctx.info if phase_ctx is not None else None,
            repo_memory=store.state.repo_memory,
            failed_epoch=failed_epoch,
            workspace=workspace,
        )
        try:
            raw = _transport_call(journal, store, planner, sleep_fn, prompt, max_planner_calls)
        except _Escalate as exc:
            return _Boundary("escalate", reason=exc.reason)
        except _Valve as exc:
            return _Boundary("valve", reason=exc.reason)

        gate = validate_decision(
            extract_decision_json(raw),
            existing_log_keys=frozenset(run_dir.log_index()),
            completed_phase_ids=completed_phase_ids,
            skeleton_exists=store.state.skeleton is not None,
            phase_escalated=phase_ctx is not None and phase_ctx.escalation_active,
            failed_epoch_active=failed_epoch is not None,
            has_senior=has_senior,
            local_max_task_files=local_max_task_files,
            senior_max_task_files=senior_max_task_files,
        )
        if gate.decision is None:
            journal.emit(
                lambda s: PlannerCallFailed(seq=s, ts=_now(), classification="transient")
            )
            if reasks >= MAX_REASKS:
                return _Boundary(
                    "escalate",
                    reason="invalid decision after 2 re-asks: " + "; ".join(gate.errors),
                )
            reasks += 1
            reask_errors = gate.errors
            continue

        decision = gate.decision
        tool = decision.tool
        journal.emit(lambda s: PlannerCallSucceeded(seq=s, ts=_now(), tool=tool))

        if isinstance(decision, CompleteRunDecision):
            failures = recheck_evidence(
                decision.args.evidence,
                repo=repo,
                final_branch=store.state.last_integration_branch,
                run_dir=run_dir,
                vision_reviewer=vision_reviewer,
                prepare=prepare,
                floor=floor,
            )
            if failures:
                if reasks >= MAX_REASKS:
                    return _Boundary(
                        "escalate",
                        reason="complete_run evidence failed after 2 re-asks: "
                        + "; ".join(failures),
                    )
                reasks += 1
                reask_errors = ["evidence check failed: " + f for f in failures]
                continue
        return _Boundary("decision", decision=decision)


# --- dispatch helpers ----------------------------------------------------------


def _apply_skeleton(
    journal: JournalWriter,
    store: _RunStateStore,
    decision: ProposeSkeletonDecision,
    started_phases: set[str],
) -> None:
    phases = list(decision.args.phases)
    journal.emit(
        lambda s: SkeletonProposed(
            seq=s, ts=_now(), phases=[PhaseRef(id=p.id, title=p.title) for p in phases]
        )
    )
    first = phases[0].id
    if first not in started_phases:
        journal.emit(lambda s: PhaseStarted(seq=s, ts=_now(), phase_id=first))
        started_phases.add(first)
    store.update(
        skeleton=phases,
        current_phase_id=first,
        phase_epoch_index=0,
        phase_budget_used=0,
        phase_escalation_active=False,
        phase_failed_epochs=0,
        pending_failed_epoch=None,
    )


def _apply_revise(
    journal: JournalWriter,
    store: _RunStateStore,
    decision: RevisePhasesDecision,
    passed_phases: set[str],
    started_phases: set[str],
) -> None:
    """Replace the current phase onward (ruling 2). Already-passed phases are
    kept (the validator rejected any reuse of a passed id); the revised list
    becomes the un-entered tail. Replacing the current phase RESETS its budget
    and clears the escalation flag (the planner gets a fresh attempt), while the
    monotonic epoch index is kept so re-scoped epoch dirs never collide.
    """

    revised = list(decision.args.phases)
    old = store.state.skeleton or []
    kept = [p for p in old if p.id in passed_phases]
    new_skeleton = kept + revised
    journal.emit(
        lambda s: PhasesRevised(
            seq=s,
            ts=_now(),
            reason=decision.args.reason,
            phases=[PhaseRef(id=p.id, title=p.title) for p in revised],
        )
    )
    first = revised[0].id
    if first not in started_phases:
        journal.emit(lambda s: PhaseStarted(seq=s, ts=_now(), phase_id=first))
        started_phases.add(first)
    store.update(
        skeleton=new_skeleton,
        current_phase_id=first,
        phase_budget_used=0,
        phase_escalation_active=False,
        phase_failed_epochs=0,
        pending_failed_epoch=None,
    )


def _dispatch_epoch(
    journal: JournalWriter,
    run_dir: RunDir,
    store: _RunStateStore,
    decision: EpochDecision,
    repo: Path | None,
    ladder: Ladder,
    concurrency: int | None,
    tier0_attempts: int,
    prepare: PrepareConfig | None,
    *,
    epoch_hint: str | None = None,
    force_senior: bool = False,
) -> EpochOutcome:
    """Run one epoch from a decision; persist ``running_epoch`` first (resume).

    ``epoch_hint`` / ``force_senior`` carry a ``handle_failed_epoch`` retry's
    corrective guidance + tier bump into the dispatched epoch.
    """

    mode = _EPOCH_MODE[decision.tool]
    is_implement = decision.tool == "implement"
    epoch_id = f"E{store.state.phase_epoch_index + 1}"
    phase_id = store.state.current_phase_id or "P1"
    base: str | None = None
    if is_implement:
        assert repo is not None
        base = (
            wt.resolve_commit(repo, store.state.last_integration_branch)
            if store.state.last_integration_branch is not None
            else wt.head_commit(repo)
        )
    store.update(status="running_epoch", pending_decision=decision.model_dump(mode="json"))
    outcome = run_epoch(
        run_dir,
        journal=journal,
        args=_epoch_args(decision),
        mode=mode,
        ladder=ladder,
        repo=repo,
        phase_id=phase_id,
        epoch_id=epoch_id,
        base=base,
        concurrency=concurrency,
        tier0_attempts=tier0_attempts,
        prepare=prepare,
        epoch_hint=epoch_hint,
        force_senior=force_senior,
    )
    _record_epoch(store, outcome, decision)
    return outcome


def _epoch_args(decision: EpochDecision) -> EpochArgs:
    """The epoch args of an epoch-tool decision (caller guarantees the tool)."""

    args = decision.args
    assert isinstance(args, (ImplementEpochArgs, ArtifactEpochArgs)), (
        f"{decision.tool} is not an epoch tool"
    )
    return args


def _record_epoch(
    store: _RunStateStore, outcome: EpochOutcome, decision: EpochDecision
) -> None:
    """Persist a completed epoch as ``awaiting_planner`` (the resume cursor).

    An epoch that carries ``criteria`` AND did not fail its floor still owes the G4
    agentic verification pass; record that as the DURABLE ``pending_verification``
    marker (the originating decision + epoch id) so a kill in the window between this
    persist and the verifier finishing cannot let the epoch skip its semantic gate.
    The marker is cleared by ``_verify_epoch`` once the pass reaches a terminal
    classification. An epoch with no criteria, or one whose floor failed (the
    failed-epoch path owns it), owes no verification -> the marker is left clear."""

    new_branch = outcome.integration.branch or store.state.last_integration_branch
    owes_verification = bool(_epoch_criteria(decision)) and not _epoch_failed(outcome)
    store.update(
        status="awaiting_planner",
        epoch_counter=store.state.epoch_counter + 1,
        phase_epoch_index=store.state.phase_epoch_index + 1,
        phase_budget_used=store.state.phase_budget_used + 1,
        last_integration_branch=new_branch,
        pending_decision=None,
        pending_verification=(
            {
                "decision": decision.model_dump(mode="json"),
                "phase_id": outcome.phase_id,
                "epoch_id": outcome.epoch_id,
            }
            if owes_verification
            else None
        ),
    )


def _epoch_failed(outcome: EpochOutcome) -> bool:
    """Did this epoch FAIL? Any task that exhausted its retry ladder is FAILED.

    The integration only merges DONE tasks, so a partial epoch still
    ``status=completed`` structurally; the planner-facing notion of a failed
    epoch is one where work did not get done, which is the failed-task set."""

    return any(t.status == "failed" for t in outcome.tasks)


def _build_failed_epoch_info(
    outcome: EpochOutcome,
    run_dir: RunDir,
    *,
    phase_check_results: list[tuple[str, bool]],
    disposed_count: int,
    cap: int,
    verification_gaps: list[str] | None = None,
) -> FailedEpochInfo:
    """Assemble the focused context a ``handle_failed_epoch`` decision needs.

    Failed tasks + their last reason; the phase exit checks that FAIL (label
    carries the captured command output, Part A); and the DONE tasks' handoff
    resulting_state, the workers' honest pass claim the planner weighs against
    the still-failing gate (gate skepticism)."""

    failed_tasks = [
        (t.task_id, t.failure_reason or "no reason recorded")
        for t in outcome.tasks
        if t.status == "failed"
    ]
    failed_checks = [label for label, ok in phase_check_results if not ok]
    passing: list[tuple[str, str]] = []
    for t in outcome.tasks:
        if t.status == "done" and t.handoff_key:
            try:
                payload = json.loads(
                    run_dir.resolve(t.handoff_key).read_text(encoding="utf-8")
                )
                state = str(payload.get("resulting_state", "")).strip()
            except (ValueError, OSError):
                state = ""
            passing.append((t.task_id, state or "(pass, no state recorded)"))
    return FailedEpochInfo(
        epoch_id=outcome.epoch_id,
        failed_tasks=failed_tasks,
        failed_checks=failed_checks,
        passing_handoffs=passing,
        disposed_count=disposed_count,
        cap=cap,
        verification_gaps=list(verification_gaps or []),
    )


def _epoch_criteria(decision: EpochDecision) -> list[str]:
    """Every task's natural-language ``criteria`` for an epoch, aggregated + deduped
    (G4). Empty => the verification pass is SKIPPED entirely."""

    if not isinstance(decision.args, (ImplementEpochArgs, ArtifactEpochArgs)):
        return []
    seen: dict[str, None] = {}
    for task in decision.args.tasks:
        for c in task.criteria:
            seen.setdefault(c, None)
    return list(seen)


def _artifact_out_keys(decision: EpochDecision) -> dict[str, str]:
    """Map each task id to its ``artifact_out`` log key for an artifact-mode epoch.

    Empty for an implement epoch (its deliverable is the committed diff the verifier
    already sees); empty when the decision is not an epoch tool."""

    if not isinstance(decision.args, ArtifactEpochArgs):
        return {}
    return {t.id: t.artifact_out for t in decision.args.tasks}


def _epoch_artifacts_summary(
    outcome: EpochOutcome, run_dir: RunDir, decision: EpochDecision
) -> list[str]:
    """What the epoch produced, fed to the adversarial verifier (its only handle).

    Each DONE task contributes its handoff ``resulting_state`` (a one-line pointer)
    AND, when the task is an artifact-mode task, the relocated ``artifact_out``
    deliverable's PATH (the absolute run-dir location ``artifact_exists`` resolves via
    ``find_artifact``). The verifier runs in a fresh worktree of the committed
    integration tip, so a research/artifact deliverable (which is NOT committed) is
    invisible in the diff there: handing it the PATH (and telling it to READ the file)
    lets it judge the real deliverable WITHOUT the content being byte-capped-and-embedded
    into the prompt (the principle: agent inputs travel by reference, full content stays
    on disk). The verifier worktree is under the run dir, so the absolute path is
    readable from its cwd (same access argument as the failure files). A deliverable
    ``artifact_exists`` just confirmed present resolves here too, so it is never reported
    missing. An implement task's deliverable is the committed diff the verifier already
    reads, so it contributes only the pointer.
    """

    out_keys = _artifact_out_keys(decision)
    out: list[str] = []
    for t in outcome.tasks:
        if t.status != "done" or not t.handoff_key:
            continue
        try:
            payload = json.loads(run_dir.resolve(t.handoff_key).read_text(encoding="utf-8"))
            state = str(payload.get("resulting_state", "")).strip()
        except (ValueError, OSError):
            state = ""
        out.append(f"{t.task_id}: {state or '(no state recorded)'}")
        artifact_out = out_keys.get(t.task_id)
        if artifact_out is None:
            continue
        path = run_dir.find_artifact(artifact_out)
        if path is None:
            out.append(
                f"{t.task_id} artifact `{artifact_out}`: (relocated but unresolvable)"
            )
        else:
            out.append(
                f"{t.task_id} artifact `{artifact_out}` (relocated deliverable, READ "
                f"its full content at this path): {path}"
            )
    return out


#: Bounded verifier re-runs for a STRUCTURALLY-invalid verdict (G6 Part B). A verdict
#: the verifier model cannot emit validly (bad JSON / missing field / still invalid
#: after advisory truncation) is a verification-INFRASTRUCTURE problem, not a worker
#: semantic gap, so the core re-runs the verifier this many times (the model may emit
#: valid output on a retry) before escalating. A genuine well-formed ``pass=false``
#: verdict is NEVER retried here: it is a semantic gap and routes through the
#: failed-epoch machinery on the first verdict, exactly as before.
VERIFIER_MAX_ATTEMPTS = 2


@dataclass(frozen=True)
class _VerificationInfraFailure:
    """A verification-INFRASTRUCTURE failure (G6 Part B): the deterministic floor and
    the worker handoff were honest, but the verification TOOLING could not produce a
    valid verdict after the bounded re-runs, OR the integration tip could not be checked
    out (unborn / unresolvable ref). Distinct from a semantic gap the planner can fix:
    the core escalates the run with this clear message instead of mis-blaming the worker
    via ``handle_failed_epoch``. A prepare/dependency-install failure is NOT this: that is
    the COMMITTED work failing to build, a defect routed to the planner as a gap."""

    message: str


def _verify_epoch(
    journal: JournalWriter,
    store: _RunStateStore,
    run_dir: RunDir,
    repo: Path | None,
    decision: EpochDecision,
    outcome: EpochOutcome,
    *,
    verifier: EpochVerifier | None,
    prepare: PrepareConfig | None,
) -> list[str] | _VerificationInfraFailure | None:
    """Run the end-of-epoch agentic verification pass (G4); classify the outcome.

    SKIPPED (returns ``None``, no event) when the epoch carries no ``criteria`` or
    there is no verifier (no local tier) or no repo. Otherwise: build a worktree of
    the epoch's integration tip with deps materialized, dispatch the adversarial
    verification pass on the local tier, relocate + re-validate ``verdict.json``.

    Three classified returns:
    - ``[]`` on a clean pass (every criterion met).
    - a non-empty gap list (``list[str]``), routed through ``handle_failed_epoch`` to the
      planner. Two sources: a well-formed ``pass=false`` verdict (the verifier found a
      SEMANTIC gap), OR a prepare/dependency-install failure (the epoch's COMMITTED work
      does not build in a clean environment, e.g. a peer-dep conflict). The latter is a
      DEFECT the planner can fix (fix the manifest), not a tooling failure: its FULL
      prepare output is persisted to a keyed-log path and the gap carries that PATH (read
      by reference). Both fall through to the failed-epoch disposition (which climbs the
      cap, so even an unfixable prepare failure terminates deterministically).
    - a ``_VerificationInfraFailure`` ONLY for a genuine verifier-TOOLING failure the
      planner cannot fix: the verifier could not produce a valid verdict after
      ``VERIFIER_MAX_ATTEMPTS`` re-runs (G6 Part B), or the integration tip could not be
      checked out (unborn / unresolvable ref). The floor passed and the worker is honest;
      the caller escalates rather than charging the worker. A prepare failure is NOT this
      (it IS the worker's work, so it routes as a gap above).

    The deterministic floor already cleared (no task failed), so this pass can only
    fail the epoch, never rubber-stamp it."""

    criteria = _epoch_criteria(decision)
    if verifier is None or repo is None or not criteria:
        # No verification ran for this epoch (criteria-less / artifact-only / no local
        # tier): clear the pending-verification marker (this epoch owes no pass) so resume
        # does not try to re-verify a criteria-less epoch. There is no digest to clear:
        # the digest now lives in the persisted verdict.json (read by reference), so an
        # unverified epoch simply produces no verdict file.
        store.update(pending_verification=None)
        return None
    args = _epoch_args(decision)  # non-empty criteria => an epoch tool (narrows type)
    phase_id = store.state.current_phase_id or "P1"
    journal.emit(
        lambda s: EpochVerificationStarted(
            seq=s, ts=_now(), phase_id=phase_id, epoch_id=outcome.epoch_id,
            criteria=len(criteria),
        )
    )
    ref = store.state.last_integration_branch or "HEAD"
    worktree = run_dir.root / "worktrees" / f"_verify_{outcome.epoch_id}"
    brief = VerificationBrief(
        epoch_goal=f"{args.epoch_title}: {args.rationale}".strip(": "),
        criteria=criteria,
        artifacts=_epoch_artifacts_summary(outcome, run_dir, decision),
    )
    try:
        wt.add_worktree_detached(repo, worktree, ref=ref)
    except wt.GitError:
        # No tip to verify against (unborn HEAD / unresolvable ref). The environment
        # could not be set up: an infrastructure failure, not a worker semantic gap.
        return _VerificationInfraFailure(
            f"verification could not check out the integration tip ({ref!r}); "
            f"the deterministic floor passed, the eval environment could not be built"
        )
    try:
        try:
            materialize_env(repo, worktree, prepare)
        except PrepareError as exc:
            # The epoch's COMMITTED manifest does not install in a clean environment
            # (e.g. a package.json peer-dependency conflict makes `npm install` fail
            # ERESOLVE). That is the worker's WORK being unbuildable, a DEFECT the
            # planner can fix, NOT a verifier-tooling failure: route it as a SEMANTIC
            # gap (through handle_failed_epoch), never a human dead-end escalation. The
            # FULL prepare output is persisted to a keyed-log path so it reaches the
            # planner BY REFERENCE (the gap carries the PATH, not the multi-KB body).
            keyed = _persist_prepare_failure(run_dir, phase_id, outcome.epoch_id, str(exc))
            return [
                f"the epoch's committed code does not install in a clean environment: "
                f"the prepare step failed (see {keyed}); fix the dependency manifest so "
                f"it installs cleanly"
            ]
        verdict = _verify_with_retries(
            verifier, worktree, brief, phase_id, outcome, run_dir
        )
    finally:
        wt.remove_worktree(repo, worktree)
    if isinstance(verdict, _VerificationInfraFailure):
        return verdict
    # The full verdict (digest + per-criterion evidence + gaps) was relocated to its
    # stable keyed-log path by the verifier adapter; it reaches the NEXT planner boundary
    # BY REFERENCE via the <workspace> manifest (the planner reads verdict.json), so
    # nothing is held in run state or embedded in a prompt. The relocated file is the
    # durable record (re-read on resume, never regenerated); construction stays pure.
    if verdict.passed:
        journal.emit(
            lambda s: EpochVerificationPassed(
                seq=s, ts=_now(), phase_id=phase_id, epoch_id=outcome.epoch_id
            )
        )
        # The Passed event is the DURABLE terminal record: clear the pending marker so
        # resume reuses this verdict (sees the event) instead of re-verifying.
        store.update(pending_verification=None)
        return []
    # A well-formed passed=false verdict is a SEMANTIC gap; synthesize a reason from
    # the unmet criteria so the planner always sees WHAT was unmet.
    gaps = list(verdict.gaps) or [
        f"criterion not met: {j.criterion}"
        for j in verdict.per_criterion
        if not j.met
    ] or ["the verification pass marked the epoch not done (no specific gap given)"]
    return gaps


def _verdict_log_key(phase_id: str, epoch_id: str) -> str:
    """The keyed-log path the epoch's relocated ``verdict.json`` lands at.

    A ``P*/E*/...`` key so the relocated verdict shows up in ``log_index()`` and thus in
    the ``<workspace>`` manifest, where the planner reads it BY REFERENCE."""

    return f"{phase_id}/{epoch_id}/verdict.json"


def _persist_prepare_failure(
    run_dir: RunDir, phase_id: str, epoch_id: str, output: str
) -> str:
    """Persist a verification prepare/dependency-install failure under a keyed-log path.

    A ``P*/E*/prepare_failure.txt`` key so the relocated record shows up in
    ``log_index()`` and thus in the ``<workspace>`` manifest, where the planner reads the
    FULL failing output BY REFERENCE (the gap string carries this key, never the multi-KB
    body). The full ``output`` (the failing cmd + its captured stdout/stderr, already in
    the PrepareError text) is written verbatim, no byte cap. Returns the log key."""

    key = f"{phase_id}/{epoch_id}/prepare_failure.txt"
    dest = run_dir.resolve(key)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(output + "\n", encoding="utf-8")
    return key


def _verify_with_retries(
    verifier: EpochVerifier,
    worktree: Path,
    brief: VerificationBrief,
    phase_id: str,
    outcome: EpochOutcome,
    run_dir: RunDir,
) -> EpochVerdict | _VerificationInfraFailure:
    """Dispatch the verifier, re-running it on a STRUCTURAL failure (G6 Part B).

    A ``VerificationError`` means the verifier model produced output that cannot be
    parsed into a valid verdict (bad JSON / missing field / over the DoS size guard).
    The model may emit valid output on a retry, so re-run it up to
    ``VERIFIER_MAX_ATTEMPTS`` times. If a valid verdict still cannot be obtained, return
    a ``_VerificationInfraFailure`` (the caller escalates) rather than charging the epoch
    as a semantic gap. A well-formed verdict (pass true OR false) returns on the first
    valid parse, after the adapter relocates the full ``verdict.json`` to its stable
    keyed-log path (so the planner reads it by reference)."""

    verdict_dest = run_dir.resolve(_verdict_log_key(phase_id, outcome.epoch_id))
    last_error = ""
    for _attempt in range(VERIFIER_MAX_ATTEMPTS):
        try:
            return verifier.verify(
                worktree=worktree,
                brief=brief,
                task_id=f"{phase_id}/{outcome.epoch_id}/verify",
                verdict_dest=verdict_dest,
            )
        except VerificationError as exc:
            last_error = str(exc)
    return _VerificationInfraFailure(
        f"the verification pass could not produce a valid verdict after "
        f"{VERIFIER_MAX_ATTEMPTS} attempts ({last_error}); the deterministic floor "
        f"passed and the worker handoff was honest, so this is a verification-tooling "
        f"failure, not a gap in the work"
    )


def _maybe_open_failed_epoch(
    journal: JournalWriter,
    store: _RunStateStore,
    run_dir: RunDir,
    decision: EpochDecision,
    outcome: EpochOutcome,
    phase_ctx: _PhaseContext | None,
    cap: int,
    *,
    verification_gaps: list[str] | None = None,
) -> RunOutcome | None:
    """If the just-run epoch FAILED, open a focused failed-epoch disposition.

    An epoch fails one of two ways: a task exhausted its retry ladder (the B6 path),
    or it cleared its deterministic floor but the G4 agentic verification pass found a
    semantic ``verification_gaps`` (an unmet acceptance criterion). EITHER routes
    through the SAME machinery: increment the per-phase failed-epoch counter and, at
    the deterministic cap, FORCE a halt-to-human regardless of the planner (Part C,
    the spin-loop backstop); below the cap record the originating decision + the
    failure context (incl. the gaps) in run state so the NEXT boundary is constrained
    to handle_failed_epoch (Part B). Returns a terminal RunOutcome when the cap halts,
    else ``None``."""

    gaps = verification_gaps or []
    if not _epoch_failed(outcome) and not gaps:
        return None
    disposed = store.state.phase_failed_epochs + 1
    phase_id = store.state.current_phase_id or "P1"
    if gaps:
        journal.emit(
            lambda s: EpochVerificationFailed(
                seq=s, ts=_now(), phase_id=phase_id, epoch_id=outcome.epoch_id,
                gaps=gaps,
            )
        )
    else:
        failed_ids = [t.task_id for t in outcome.tasks if t.status == "failed"]
        journal.emit(
            lambda s: EpochFailed(
                seq=s, ts=_now(), phase_id=phase_id, epoch_id=outcome.epoch_id,
                failed_tasks=failed_ids,
            )
        )
    phase_checks = phase_ctx.info.check_results if phase_ctx is not None else []
    info = _build_failed_epoch_info(
        outcome, run_dir, phase_check_results=phase_checks,
        disposed_count=disposed, cap=cap, verification_gaps=gaps,
    )
    if disposed >= cap:
        reason = (
            f"failed-epoch cap reached: {disposed}/{cap} epochs failed in phase "
            f"{phase_id}; halting to a human (the gate/environment is likely the "
            f"problem, not the code)"
        )
        journal.emit(
            lambda s: FailedEpochHandled(
                seq=s, ts=_now(), phase_id=phase_id, epoch_id=outcome.epoch_id,
                action="cap_halt", detail=reason,
            )
        )
        store.update(
            phase_failed_epochs=disposed, pending_failed_epoch=None,
            pending_verification=None,
        )
        return _escalate(journal, store, reason)
    store.update(
        phase_failed_epochs=disposed,
        pending_failed_epoch={
            "decision": decision.model_dump(mode="json"),
            "epoch_id": info.epoch_id,
            "failed_tasks": info.failed_tasks,
            "passing_handoffs": info.passing_handoffs,
            "verification_gaps": info.verification_gaps,
        },
        # The EpochVerificationFailed event (above, for the gap path) + the durable
        # pending_failed_epoch are now the terminal record: clear the pending-
        # verification marker so resume does not re-verify an already-disposed gap.
        pending_verification=None,
    )
    return None


def _pending_failed_epoch_info(
    store: _RunStateStore, phase_ctx: _PhaseContext | None, cap: int
) -> FailedEpochInfo | None:
    """Rebuild the pending failed-epoch context from run state (resume-safe).

    The originating decision + the failed-task / passing-handoff snapshot are
    durable in run state; the failing-check labels (with the latest captured
    output, Part A) are refreshed from THIS pass's phase evaluation so the planner
    always sees the current gate output. ``None`` when nothing is pending.

    A#5 fix: ``cap`` is the REAL ``max_failed_epochs_per_phase`` (the remaining-budget
    denominator), threaded through so the planner sees the true budget (e.g.
    ``disposed=1/3``) instead of the old ``disposed=N/N`` that always rendered at-cap
    (a false 'halting now' signal) by passing the disposed count as its own cap."""

    pending = store.state.pending_failed_epoch
    if pending is None:
        return None
    failed_checks = (
        [label for label, ok in phase_ctx.info.check_results if not ok]
        if phase_ctx is not None
        else []
    )
    return FailedEpochInfo(
        epoch_id=str(pending["epoch_id"]),
        failed_tasks=[(t[0], t[1]) for t in pending["failed_tasks"]],
        failed_checks=failed_checks,
        passing_handoffs=[(h[0], h[1]) for h in pending["passing_handoffs"]],
        disposed_count=store.state.phase_failed_epochs,
        cap=cap,
        verification_gaps=[str(g) for g in pending.get("verification_gaps", [])],
    )


def _apply_failed_epoch_decision(
    journal: JournalWriter,
    run_dir: RunDir,
    store: _RunStateStore,
    decision: HandleFailedEpochDecision,
    repo: Path | None,
    ladder: Ladder,
    concurrency: int | None,
    tier0_attempts: int,
    prepare: PrepareConfig | None,
    cap: int,
    verifier: EpochVerifier | None,
) -> tuple[RunOutcome | None, EpochOutcome | None]:
    """Dispatch a handle_failed_epoch decision; clear the pending failure.

    ``halt`` is terminal (escalation). ``retry`` / ``escalate_senior`` re-dispatch
    the SAME originating epoch decision, retry optionally bumping the starting tier
    + threading the planner's hint to the workers, escalate_senior forcing senior.
    Returns ``(RunOutcome, None)`` on a terminal branch, else ``(None, outcome)``
    with the re-dispatched epoch's outcome for the caller to chain.
    """

    pending = store.state.pending_failed_epoch
    assert pending is not None, "handle_failed_epoch without a pending failed epoch"
    phase_id = store.state.current_phase_id or "P1"
    epoch_id = str(pending["epoch_id"])
    args = decision.args

    if isinstance(args, HaltFailedEpochArgs):
        journal.emit(
            lambda s: FailedEpochHandled(
                seq=s, ts=_now(), phase_id=phase_id, epoch_id=epoch_id,
                action="halt", detail=args.reason,
            )
        )
        store.update(pending_failed_epoch=None)
        return _escalate(journal, store, f"planner halted failed epoch: {args.reason}"), None

    if isinstance(args, RetryFailedEpochArgs):
        action, detail = "retry", args.hint
        epoch_hint = args.hint
        force_senior = args.escalate_tier
    else:  # EscalateSeniorFailedEpochArgs
        action, detail = "escalate_senior", args.diagnosis
        epoch_hint = f"senior diagnosis: {args.diagnosis}"
        force_senior = True

    journal.emit(
        lambda s: FailedEpochHandled(
            seq=s, ts=_now(), phase_id=phase_id, epoch_id=epoch_id,
            action=action, detail=detail,
        )
    )
    store.update(pending_failed_epoch=None)
    # The originating decision is always an epoch tool (implement / research /
    # review / artifact); it is what produced the failed epoch we re-dispatch.
    epoch_decision = parse_decision(pending["decision"])
    if epoch_decision.tool == "implement" and repo is None:
        return _escalate(journal, store, "implement epoch requested but no repo configured"), None
    outcome = _dispatch_epoch(
        journal, run_dir, store, epoch_decision, repo, ladder, concurrency,
        tier0_attempts, prepare, epoch_hint=epoch_hint, force_senior=force_senior,
    )
    if outcome.status == "integration_conflict":
        return _escalate(
            journal, store, f"integration conflict in {outcome.epoch_id} (structural bug)"
        ), None
    # A re-dispatched epoch can itself fail; re-open the disposition (counter keeps
    # climbing toward the cap, so a retry-loop terminates deterministically). The
    # re-dispatch is re-verified the same way (a retry that satisfies the floor but
    # still leaves a semantic gap re-opens the disposition with the fresh gaps).
    verification = (
        _verify_epoch(
            journal, store, run_dir, repo, epoch_decision, outcome,
            verifier=verifier, prepare=prepare,
        )
        if not _epoch_failed(outcome)
        else None
    )
    if isinstance(verification, _VerificationInfraFailure):
        # The verification TOOLING failed (not a semantic gap): escalate with the
        # clear message rather than mis-blaming the worker via handle_failed_epoch.
        return _escalate(journal, store, verification.message), None
    result = _maybe_open_failed_epoch(
        journal, store, run_dir, epoch_decision, outcome, None, cap,
        verification_gaps=verification,
    )
    return result, outcome


# --- G3 automatic senior infra-repair: detect -> repair -> re-run -> cap --------

#: The ladder tier name the infra-repair dispatches on (the senior/cloud tier).
_SENIOR_TIER_NAME = "senior"


def _senior_transport(ladder: Ladder) -> WorkerTransport | None:
    """The ladder's ``senior`` tier transport (the infra-repair runs there)."""

    for name, transport in ladder:
        if name == _SENIOR_TIER_NAME:
            return transport
    return None


def _infra_failures(results: Sequence[CheckResult]) -> list[CheckResult]:
    """The cmd-check rows that failed for an ENVIRONMENTAL reason (G3 classifier)."""

    return [r for r in results if not r.ok and r.infra is not None and r.infra.is_infra]


def _run_one_infra_repair(
    journal: JournalWriter,
    store: _RunStateStore,
    run_dir: RunDir,
    repo: Path,
    senior: WorkerTransport,
    *,
    phase_id: str,
    exit_criterion: Sequence[Check],
    failing: Sequence[CheckResult],
    prepare: PrepareConfig | None,
    floor: FloorConfig | None,
    cfg: InfraRepairConfig,
    attempt: int,
) -> bool:
    """Dispatch ONE senior infra-repair against the gate tip; adopt it if it sticks.

    A worktree of the current tip is materialized, the senior is dispatched with a
    focused ``InfraRepairBrief`` (the failing commands + their output + the host
    guard) and edits the environment IN that worktree; the core commits the edits
    (models never run git) and re-runs the FULL phase exit criterion (plus floor)
    against the repair commit. Pass -> the repair commit is adopted as the new
    integration tip (a real branch, force-moved), so the ordinary phase gate that
    follows now passes; ``True``. No commit / any check still failing -> the repair
    branch is discarded, ``False``. We re-run the WHOLE gate, not only the named
    failing commands, so a repair that fixes its target but REGRESSES an unrelated
    check in the same exit criterion is NOT adopted (A#7).

    A senior transport raise, OR a ``PrepareError`` materializing deps into the
    eval/repair worktree (the exact scenario this feature targets: deps that cannot
    install), is a clean ``False`` (the repair simply did not land), never a run
    crash, letting the cap/escalate path fire (A#1).
    """

    tip = store.state.last_integration_branch or "HEAD"
    base_commit = wt.resolve_commit(repo, tip)
    worktree = run_dir.root / "worktrees" / f"_infra_repair_{attempt}"
    branch = f"grind/{run_dir.root.name}/infra-repair-{attempt}"
    failing_cmds = [r.cmd for r in failing if r.cmd is not None]
    output_tail = "\n\n".join(r.label for r in failing)
    journal.emit(
        lambda s: InfraRepairDispatched(
            seq=s, ts=_now(), phase_id=phase_id,
            command=", ".join(failing_cmds), attempt=attempt, cap=cfg.attempts,
        )
    )
    wt.add_worktree(repo, worktree, branch=branch, base=base_commit)
    try:
        try:
            materialize_env(repo, worktree, prepare)
        except PrepareError:
            # Deps cannot materialize in the repair worktree (the very fault this
            # feature targets). Treat as a repair that did not land so the cap /
            # escalate path fires cleanly; never let it crash out of _drive.
            return False
        brief = InfraRepairBrief(
            failing_commands=failing_cmds,
            output_tail=output_tail,
            reason="; ".join(
                r.infra.reason for r in failing if r.infra is not None
            ),
            allow_host_commands=list(cfg.allow_host_commands),
        )
        request = WorkerRequest(
            task=_infra_repair_task(failing_cmds),
            task_id=f"{phase_id}/infra-repair-{attempt}",
            inputs={},
            scratch=worktree,
            attempt=attempt,
            failure_context=[],
            mode="implement",
            infra_repair=brief,
        )
        try:
            senior.run(request)
        except Exception:  # a senior raise is just a repair that did not land
            return False
        # Drop the worker's disk-contract metadata so it cannot enter the commit.
        (worktree / "handoff.json").unlink(missing_ok=True)
        if not wt.commit_all(worktree, f"grindstone: infra repair (attempt {attempt})"):
            return False  # the senior changed nothing repo-local
        repair_commit = wt.resolve_commit(worktree, "HEAD")
    finally:
        wt.remove_worktree(repo, worktree)
    # Re-run the FULL phase exit criterion (plus floor) against the repair commit,
    # the authoritative judge (not the senior's handoff, not just the named failing
    # commands). A repair that fixes its target but regresses ANY other check in the
    # same gate is NOT adopted (A#7). Still failing -> discard the repair branch.
    recheck = _evaluate_checks_detailed(
        exit_criterion,
        repo=repo,
        ref=repair_commit,
        run_dir=run_dir,
        scratch_name=f"_infra_recheck_{attempt}",
        prepare=prepare,
        floor=floor,
    )
    if not all(r.ok for r in recheck):
        wt.delete_branch(repo, branch)
        return False
    # The repair sticks: adopt it as the integration tip so the phase gate passes.
    store.update(last_integration_branch=branch)
    return True


def _infra_repair_task(failing_cmds: Sequence[str]) -> ImplementTask:
    """The minimal placeholder ``ImplementTask`` an infra-repair request carries.

    The real brief rides ``WorkerRequest.infra_repair``; the prompt builder branches
    on it before ever reading this task. It exists only to satisfy the request's
    typed ``task`` slot, repo-wide ownership (the repair may touch any manifest/
    config) and a non-empty done_when."""

    return ImplementTask(
        id="T1",
        goal="infra repair: make the failing gate command(s) satisfiable",
        done_when=[CmdCheck(cmd=c) for c in failing_cmds] or [CmdCheck(cmd="true")],
        file_ownership=["**"],
    )


def _maybe_repair_infra(
    journal: JournalWriter,
    store: _RunStateStore,
    run_dir: RunDir,
    repo: Path | None,
    ladder: Ladder,
    *,
    prepare: PrepareConfig | None,
    floor: FloorConfig | None,
    infra_repair: InfraRepairConfig | None,
) -> RunOutcome | None:
    """Before each boundary: if the phase gate is INFRA-failing, auto-repair it.

    The self-healing loop (G3). Evaluates the current phase's exit criterion (plus
    the floor) infra-aware; if any cmd check failed ENVIRONMENTALLY and a senior
    tier + an ``infra_repair`` policy exist, dispatch up to ``attempts`` senior
    repairs, each making the gate satisfiable and re-running it. A repair that
    sticks emits ``infra_repair_resolved`` and returns ``None`` (the run proceeds,
    the now-passing gate advances normally). Exhausting the cap with the gate STILL
    infra-failing escalates the run for a human, naming the unsatisfiable command
    (``infra_repair_exhausted`` + a clear ``RunOutcome``). Returns a terminal
    ``RunOutcome`` only on that escalation, else ``None`` (the common path)."""

    if infra_repair is None or repo is None or store.state.skeleton is None:
        return None
    senior = _senior_transport(ladder)
    if senior is None:
        return None  # a rig with no senior tier cannot auto-repair
    skeleton = store.state.skeleton
    phase = skeleton[_phase_index(skeleton, store.state.current_phase_id)]
    phase_id = phase.id
    results = _evaluate_checks_detailed(
        phase.exit_criterion,
        repo=repo,
        ref=_eval_ref(store, repo),
        run_dir=run_dir,
        scratch_name="_infra_detect",
        prepare=prepare,
        floor=floor,
    )
    failing = _infra_failures(results)
    if not failing:
        return None
    def _emit_detected(cmd: str, reason: str) -> None:
        journal.emit(
            lambda s: InfraCheckDetected(
                seq=s, ts=_now(), phase_id=phase_id, command=cmd, reason=reason
            )
        )

    for r in failing:
        assert r.infra is not None
        _emit_detected(r.cmd or "(unknown)", r.infra.reason)
    attempt = 0
    while attempt < infra_repair.attempts:
        attempt += 1
        resolved = _run_one_infra_repair(
            journal, store, run_dir, repo, senior,
            phase_id=phase_id, exit_criterion=phase.exit_criterion,
            failing=failing, prepare=prepare, floor=floor,
            cfg=infra_repair, attempt=attempt,
        )
        if resolved:
            done_attempt = attempt
            journal.emit(
                lambda s: InfraRepairResolved(
                    seq=s, ts=_now(), phase_id=phase_id, attempt=done_attempt
                )
            )
            return None
        # Re-evaluate: a partial repair may have changed WHICH commands fail.
        results = _evaluate_checks_detailed(
            phase.exit_criterion,
            repo=repo,
            ref=_eval_ref(store, repo),
            run_dir=run_dir,
            scratch_name="_infra_detect",
            prepare=prepare,
            floor=floor,
        )
        failing = _infra_failures(results)
        if not failing:
            return None  # the gate cleared (e.g. a non-adopted but real fix)
    # Cap reached and the gate is still infra-failing: escalate, naming the tool.
    cmd = failing[0].cmd or "(unknown command)"
    reason = failing[0].infra.reason if failing[0].infra is not None else "infra"
    journal.emit(
        lambda s: InfraRepairExhausted(
            seq=s, ts=_now(), phase_id=phase_id, command=cmd, reason=reason
        )
    )
    return _escalate(
        journal,
        store,
        f"infra-repair exhausted after {infra_repair.attempts} attempt(s): the gate "
        f"command `{cmd}` cannot be satisfied in this environment ({reason}); a human "
        f"must fix the host environment or allowlist the needed command",
    )


# --- B5 final polish: gated, optional, never-fails post-completion pass ---------


def _final_polish(
    journal: JournalWriter,
    store: _RunStateStore,
    run_dir: RunDir,
    repo: Path | None,
    evidence: Sequence[Check],
    final_polish: FinalPolish | None,
    vision_reviewer: VisionReviewer | None,
    prepare: PrepareConfig | None,
    floor: FloorConfig | None,
) -> None:
    """Optionally let codex polish the finished repo inline, model proposes, the
    state machine disposes. Runs only when configured AND a repo + final branch
    exist. ANY error (codex, worktree, evidence) is a CLEAN no-op: the pass can
    never turn a completed run into a failure, so the whole body is guarded.
    """

    if final_polish is None or repo is None:
        return
    branch = store.state.last_integration_branch
    if branch is None:
        return
    # Idempotent on resume (ruling 6): a kill AFTER adopting the polish but BEFORE
    # status=completed would otherwise re-ask the planner -> complete_run ->
    # re-run polish and STACK a second commit. The journal leads, if a polish
    # outcome is already recorded for this run, the pass has run; do not repeat it.
    if _polish_already_ran(run_dir):
        return
    try:
        _run_final_polish(
            journal, store, run_dir, repo, branch, evidence, final_polish,
            vision_reviewer, prepare, floor,
        )
    except Exception as exc:  # broad: polish must never fail a completed run
        journal.emit(
            lambda s: FinalPolishSkipped(
                seq=s, ts=_now(), reason=f"polish error: {type(exc).__name__}: {exc}"
            )
        )


def _polish_already_ran(run_dir: RunDir) -> bool:
    """Has a final-polish outcome already been journaled for this run? (ruling 6).

    The durable journal is the source of truth: an ``applied`` or ``skipped``
    event means the (idempotent) pass already executed, so a re-entered run must
    not run it again. Reads the flushed events file, every ``emit`` fsyncs.
    """

    return any(
        isinstance(e, (FinalPolishApplied, FinalPolishSkipped))
        for e in read_events(run_dir.events_path)
    )


def _run_final_polish(
    journal: JournalWriter,
    store: _RunStateStore,
    run_dir: RunDir,
    repo: Path,
    branch: str,
    evidence: Sequence[Check],
    final_polish: FinalPolish,
    vision_reviewer: VisionReviewer | None,
    prepare: PrepareConfig | None,
    floor: FloorConfig | None,
) -> None:
    """The polish body (caller guards every failure into a no-op).

    Detached worktree at the final branch (committing there moves NO branch, a
    discarded polish leaves nothing behind); codex edits it; a zero-diff pass is a
    no-op; otherwise the edits are committed and the SAME ``complete_run`` evidence
    is re-run against the polish commit. Pass -> the run's integration branch is
    force-moved to the polish commit (a REAL ref, the commit is never left
    dangling on a torn-down worktree) and that BRANCH NAME becomes the final
    branch; fail -> drop it, the original completion stands.
    """

    base_commit = wt.resolve_commit(repo, branch)  # pre-polish tip (for the diff)
    worktree = run_dir.root / "worktrees" / "_polish"
    wt.add_worktree_detached(repo, worktree, ref=branch)
    # Restore declared deps so codex can run build commands while polishing (and so
    # they are present for the evidence re-check below). A prepare failure here is
    # caught by the caller's broad guard -> the polish pass is a clean no-op.
    materialize_env(repo, worktree, prepare)
    try:
        ok = final_polish.polisher.polish(
            worktree=worktree,
            criteria=final_polish.criteria,
            screenshot_rel=final_polish.screenshot_rel,
            out_dir=run_dir.root / "polish",
        )
        if not ok:
            journal.emit(
                lambda s: FinalPolishSkipped(seq=s, ts=_now(), reason="polish script failed")
            )
            return
        if not wt.commit_all(worktree, "grindstone: final polish"):
            journal.emit(
                lambda s: FinalPolishSkipped(seq=s, ts=_now(), reason="codex made no changes")
            )
            return
        polish_commit = wt.resolve_commit(worktree, "HEAD")
        failures = recheck_evidence(
            evidence,
            repo=repo,
            final_branch=polish_commit,
            run_dir=run_dir,
            vision_reviewer=vision_reviewer,
            prepare=prepare,
            floor=floor,
        )
        if failures:
            journal.emit(
                lambda s: FinalPolishSkipped(
                    seq=s, ts=_now(), reason="polish regressed evidence: " + "; ".join(failures)
                )
            )
            return
        # Materialize a real ref BEFORE the worktree (the only thing referencing
        # the polish commit) is torn down, and record the BRANCH NAME, not the
        # bare sha, so the run's final branch resolves to the polished work.
        changed = wt.changed_paths(repo, base_commit, polish_commit)
        wt.force_branch(repo, branch, polish_commit)
        store.update(last_integration_branch=branch)
        journal.emit(
            lambda s: FinalPolishApplied(
                seq=s, ts=_now(), commit=polish_commit, changed_files=changed
            )
        )
    finally:
        wt.remove_worktree(repo, worktree)


# --- terminals -----------------------------------------------------------------


def _complete(
    journal: JournalWriter, store: _RunStateStore, summary: str
) -> RunOutcome:
    store.update(status="completed", terminal_reason=summary)
    journal.emit(lambda s: RunCompleted(seq=s, ts=_now()))
    return _outcome(store, "completed", reason=None, summary=summary)


def _escalate(journal: JournalWriter, store: _RunStateStore, reason: str) -> RunOutcome:
    store.update(status="escalated", terminal_reason=reason)
    journal.emit(lambda s: RunEscalated(seq=s, ts=_now(), reason=reason))
    return _outcome(store, "escalated", reason=reason, summary=None)


def _fail_valve(
    journal: JournalWriter, store: _RunStateStore, reason: str
) -> RunOutcome:
    # The safety valve (production planner-call cap / harness epoch bound) tripped.
    # Terminal but not an escalation. Emit a vocabulary event so the journal is
    # self-describing, a watcher can render + exit instead of hanging on a run
    # whose durable status is "failed" but whose journal never closed.
    store.update(status="failed", terminal_reason=reason)
    journal.emit(lambda s: RunFailed(seq=s, ts=_now(), reason=reason))
    return _outcome(store, "failed", reason=reason, summary=None)


def _outcome(
    store: _RunStateStore,
    status: Literal["completed", "escalated", "failed"],
    *,
    reason: str | None,
    summary: str | None,
) -> RunOutcome:
    st = store.state
    return RunOutcome(
        status=status,
        reason=reason,
        summary=summary,
        planner_calls=st.planner_call_count,
        epochs_run=st.epoch_counter,
        final_branch=st.last_integration_branch,
    )


# --- the loop ------------------------------------------------------------------


def _drive(
    journal: JournalWriter,
    run_dir: RunDir,
    store: _RunStateStore,
    *,
    planner: PlannerTransport,
    ladder: Ladder,
    repo: Path | None,
    sleep_fn: SleepFn,
    max_planner_calls: int | None,
    max_epochs: int | None,
    concurrency: int | None,
    tier0_attempts: int,
    last_epoch: EpochOutcome | None,
    started_phases: set[str],
    passed_phases: set[str],
    vision_reviewer: VisionReviewer | None,
    final_polish: FinalPolish | None,
    prepare: PrepareConfig | None,
    floor: FloorConfig | None,
    max_failed_epochs_per_phase: int,
    verifier: EpochVerifier | None = None,
    infra_repair: InfraRepairConfig | None = None,
    local_max_task_files: int = DEFAULT_LOCAL_MAX_TASK_FILES,
    senior_max_task_files: int = DEFAULT_SENIOR_MAX_TASK_FILES,
) -> RunOutcome:
    last_epoch_rows = (
        flatten_last_epoch(run_dir, last_epoch) if last_epoch is not None else None
    )
    # A ``senior`` tier in the ladder means a visual implement epoch starts on
    # senior, so the size gate applies the senior file-count bound to it.
    has_senior = any(name == "senior" for name, _ in ladder)
    while True:
        if max_epochs is not None and store.state.epoch_counter >= max_epochs:
            return _fail_valve(journal, store, f"safety valve: {max_epochs} epochs reached")

        # G3 self-healing: before the boundary, if the current phase gate is failing
        # for an ENVIRONMENTAL reason (a missing tool/dependency/install), auto-
        # dispatch a bounded senior infra-repair and re-run the gate, instead of
        # charging the worker or opening a semantic failed epoch. A repair that
        # sticks lets the gate below pass normally; cap-exhaustion escalates here.
        #
        # SKIP while a failed epoch is awaiting disposition (A#10): the planner is
        # constrained to handle_failed_epoch this iteration, and an infra-repair
        # would mutate the integration tip BETWEEN the semantic failure and its
        # disposition. The two repair mechanisms are mutually exclusive.
        if store.state.pending_failed_epoch is None:
            infra_result = _maybe_repair_infra(
                journal, store, run_dir, repo, ladder,
                prepare=prepare, floor=floor, infra_repair=infra_repair,
            )
            if infra_result is not None:
                return infra_result

        # Phase machinery (rulings 1-3): once a skeleton exists, every loop pass
        # freshly evaluates the current phase against the integration tip, fires
        # phase_passed/started advancement (idempotent on resume) + a one-shot
        # phase_escalated on budget exhaustion, and assembles the tail context.
        phase_ctx = (
            _phase_preamble(
                journal,
                store,
                run_dir,
                repo,
                passed_phases,
                started_phases,
                vision_reviewer,
                prepare,
                floor,
            )
            if store.state.skeleton is not None
            else None
        )

        # A FAILED epoch awaiting a focused disposition (Part B): rebuild its
        # context (refreshing the phase-check output from this pass, Part A) and
        # constrain the next decision to handle_failed_epoch. Resume-safe: the
        # originating decision + raw context live in run state.
        failed_epoch_info = _pending_failed_epoch_info(
            store, phase_ctx, max_failed_epochs_per_phase
        )

        boundary = _plan_boundary(
            journal,
            store,
            run_dir,
            planner,
            sleep_fn,
            repo,
            last_epoch_rows,
            max_planner_calls,
            phase_ctx,
            frozenset(passed_phases),
            vision_reviewer,
            prepare,
            floor,
            failed_epoch_info,
            has_senior,
            local_max_task_files,
            senior_max_task_files,
        )
        if boundary.kind == "escalate":
            return _escalate(journal, store, boundary.reason or "planner escalated")
        if boundary.kind == "valve":
            return _fail_valve(journal, store, boundary.reason or "safety valve")

        decision = boundary.decision
        assert decision is not None

        if isinstance(decision, ProposeSkeletonDecision):
            _apply_skeleton(journal, store, decision, started_phases)
            continue
        if isinstance(decision, RevisePhasesDecision):
            _apply_revise(journal, store, decision, passed_phases, started_phases)
            continue
        if isinstance(decision, HandleFailedEpochDecision):
            result, outcome = _apply_failed_epoch_decision(
                journal, run_dir, store, decision, repo, ladder, concurrency,
                tier0_attempts, prepare, max_failed_epochs_per_phase, verifier,
            )
            if result is not None:
                return result
            assert outcome is not None
            last_epoch = outcome
            last_epoch_rows = flatten_last_epoch(run_dir, outcome)
            continue
        if isinstance(decision, EscalateRunDecision):
            return _escalate(journal, store, decision.args.reason)
        if isinstance(decision, CompleteRunDecision):
            _final_polish(
                journal, store, run_dir, repo, decision.args.evidence,
                final_polish, vision_reviewer, prepare, floor,
            )
            return _complete(journal, store, decision.args.summary)

        # epoch tool (implement / research / review / artifact)
        if decision.tool == "implement" and repo is None:
            return _escalate(journal, store, "implement epoch requested but no repo configured")
        outcome = _dispatch_epoch(
            journal, run_dir, store, decision, repo, ladder, concurrency,
            tier0_attempts, prepare,
        )
        if outcome.status == "integration_conflict":
            return _escalate(
                journal, store, f"integration conflict in {outcome.epoch_id} (structural bug)"
            )
        # G4: the deterministic floor cleared (no task failed) -> run the agentic
        # verification pass over the epoch's criteria; an unmet criterion fails the
        # epoch through the SAME failed-epoch machinery. Skipped when a task already
        # failed (the floor is what failed, not the semantics).
        verification = (
            _verify_epoch(
                journal, store, run_dir, repo, decision, outcome,
                verifier=verifier, prepare=prepare,
            )
            if not _epoch_failed(outcome)
            else None
        )
        if isinstance(verification, _VerificationInfraFailure):
            # The verification TOOLING could not produce a valid verdict (G6 Part B):
            # the floor passed and the worker is honest, so this is a verification-
            # infrastructure escalation, NOT a worker semantic gap. Do not route it
            # through handle_failed_epoch (that mis-blames the worker).
            return _escalate(journal, store, verification.message)
        result = _maybe_open_failed_epoch(
            journal, store, run_dir, decision, outcome, phase_ctx,
            max_failed_epochs_per_phase, verification_gaps=verification,
        )
        if result is not None:
            return result
        last_epoch = outcome
        last_epoch_rows = flatten_last_epoch(run_dir, outcome)


# --- public entry points -------------------------------------------------------


def run_grind(
    run_dir: RunDir,
    *,
    job_path: str,
    planner: PlannerTransport,
    ladder: Ladder,
    repo: Path | None = None,
    run_id: str = "run",
    sleep_fn: SleepFn = time.sleep,
    max_planner_calls: int | None = None,
    max_epochs: int | None = None,
    concurrency: int | None = None,
    tier0_attempts: int = TIER0_ATTEMPTS,
    vision_reviewer: VisionReviewer | None = None,
    final_polish: FinalPolish | None = None,
    prepare: PrepareConfig | None = None,
    floor: FloorConfig | None = None,
    verifier: EpochVerifier | None = None,
    infra_repair: InfraRepairConfig | None = None,
    max_failed_epochs_per_phase: int = DEFAULT_MAX_FAILED_EPOCHS_PER_PHASE,
    local_max_task_files: int = DEFAULT_LOCAL_MAX_TASK_FILES,
    senior_max_task_files: int = DEFAULT_SENIOR_MAX_TASK_FILES,
) -> RunOutcome:
    """Run a job end-to-end: planner ⇄ core until completion or escalation.

    ``max_planner_calls`` is the production safety valve: the CLI ALWAYS passes
    a cap (config > built-in default) so an unattended revision spin cannot
    drain the planner subscription (gate-5 P0: 34 calls overnight); ``None``
    (= off) remains a test seam only. ``max_epochs`` is a TEST-harness valve
    (off by default, NOT loop policy). When tripped the run stops with a
    ``failed`` outcome and ``run_state.json`` records why.
    """

    if not ladder:
        raise ValueError("ladder must have at least one tier")
    # A new run starts: only the latest run keeps a rendered journal (reap the
    # rest, leaving their durable events.ndjson behind).
    reap_sibling_journals(run_dir)
    job_file = Path(job_path)
    job_text = job_file.read_text(encoding="utf-8") if job_file.is_file() else job_path
    # S5 seam: freeze the repo-memory digest ONCE at run start (byte-stable head).
    repo_memory = load_digest(repo) if repo is not None else None
    with JournalWriter(run_dir.events_path) as journal:
        journal.emit(
            lambda s: RunStarted(
                seq=s, ts=_now(), run_id=run_id, job_path=job_path,
                max_planner_calls=max_planner_calls,
            )
        )
        store = _RunStateStore(
            run_dir,
            RunState(
                run_id=run_id,
                job_path=job_path,
                job_text=job_text,
                status="awaiting_planner",
                skeleton=None,
                current_phase_id=None,
                epoch_counter=0,
                planner_call_count=0,
                rate_limit_waits=0,
                last_integration_branch=None,
                pending_decision=None,
                terminal_reason=None,
                repo_memory=repo_memory,
            ),
        )
        outcome = _drive(
            journal,
            run_dir,
            store,
            planner=planner,
            ladder=ladder,
            repo=repo,
            sleep_fn=sleep_fn,
            max_planner_calls=max_planner_calls,
            max_epochs=max_epochs,
            concurrency=concurrency,
            tier0_attempts=tier0_attempts,
            last_epoch=None,
            started_phases=set(),
            passed_phases=set(),
            vision_reviewer=vision_reviewer,
            final_polish=final_polish,
            prepare=prepare,
            floor=floor,
            max_failed_epochs_per_phase=max_failed_epochs_per_phase,
            verifier=verifier,
            infra_repair=infra_repair,
            local_max_task_files=local_max_task_files,
            senior_max_task_files=senior_max_task_files,
        )
    # Events are fully flushed (writer closed): render the post-mortem journal.
    write_journal(run_dir)
    return outcome


def _resume_pending_verification(
    journal: JournalWriter,
    store: _RunStateStore,
    run_dir: RunDir,
    repo: Path | None,
    journal_events: Sequence[object],
    *,
    verifier: EpochVerifier | None,
    prepare: PrepareConfig | None,
    cap: int,
) -> RunOutcome | None:
    """Re-run the G4 verification pass for a completed epoch a kill left UNVERIFIED.

    Closes the loop-position hole: an epoch that persisted ``awaiting_planner`` (or was
    finished by ``resume_epoch``) carrying ``criteria`` but whose verification did not
    reach a terminal event is treated as verified-pass by loop position alone. Here, on
    resume, the DURABLE ``pending_verification`` marker plus the ABSENCE of a terminal
    verification event (Passed/Failed) for that epoch means the gate never finished -> we
    re-run it before the next planner boundary. A verdict already produced + recorded
    (a terminal event present) clears the marker without a re-run, so a produced verdict
    is reused, never regenerated. Returns a terminal ``RunOutcome`` when the pass fails
    (infra-escalation, or a gap that trips the failed-epoch cap), else ``None``."""

    pending = store.state.pending_verification
    if pending is None or verifier is None or repo is None:
        # Nothing owes verification, or the pass is disabled / has no repo: a clean
        # skip. Clear a stale marker so the next boundary is not blocked by it.
        if pending is not None:
            store.update(pending_verification=None)
        return None
    epoch_id = str(pending["epoch_id"])
    phase_id = str(pending["phase_id"])
    terminal = any(
        isinstance(e, (EpochVerificationPassed, EpochVerificationFailed))
        and getattr(e, "epoch_id", None) == epoch_id
        for e in journal_events
    )
    if terminal:
        # The verdict was already produced + recorded before the kill: reuse it (the
        # failed-epoch context, if any, is durable in pending_failed_epoch). Just clear
        # the now-redundant marker.
        store.update(pending_verification=None)
        return None
    outcome_path = run_dir.resolve(f"{phase_id}/{epoch_id}/outcome.json")
    if not outcome_path.is_file():
        # No persisted outcome to verify against (a partial write before the kill):
        # clear the marker and proceed, the deterministic floor already gated the epoch.
        store.update(pending_verification=None)
        return None
    decision = parse_decision(pending["decision"])
    outcome = EpochOutcome.from_dict(
        cast(dict[str, object], json.loads(outcome_path.read_text(encoding="utf-8")))
    )
    verification = _verify_epoch(
        journal, store, run_dir, repo, decision, outcome, verifier=verifier, prepare=prepare,
    )
    if isinstance(verification, _VerificationInfraFailure):
        return _escalate(journal, store, verification.message)
    return _maybe_open_failed_epoch(
        journal, store, run_dir, decision, outcome, None, cap,
        verification_gaps=verification,
    )


def resume_grind(
    run_dir: RunDir,
    *,
    planner: PlannerTransport,
    ladder: Ladder,
    repo: Path | None = None,
    sleep_fn: SleepFn = time.sleep,
    max_planner_calls: int | None = None,
    max_epochs: int | None = None,
    concurrency: int | None = None,
    tier0_attempts: int = TIER0_ATTEMPTS,
    vision_reviewer: VisionReviewer | None = None,
    final_polish: FinalPolish | None = None,
    prepare: PrepareConfig | None = None,
    floor: FloorConfig | None = None,
    verifier: EpochVerifier | None = None,
    infra_repair: InfraRepairConfig | None = None,
    max_failed_epochs_per_phase: int = DEFAULT_MAX_FAILED_EPOCHS_PER_PHASE,
    local_max_task_files: int = DEFAULT_LOCAL_MAX_TASK_FILES,
    senior_max_task_files: int = DEFAULT_SENIOR_MAX_TASK_FILES,
) -> RunOutcome:
    """Re-enter a killed run from ``run_state.json`` + the journal (ruling 6).

    ``awaiting_planner`` (incl. a kill mid-planner-call): re-issue, nothing
    landed on disk, the call is idempotent by construction, so it is re-asked,
    not burned. ``running_epoch``: finish the in-flight epoch via ``resume_epoch``
    (re-parsing the pending decision), then continue the loop. Already-terminal
    states return their recorded outcome without touching anything.
    """

    if not ladder:
        raise ValueError("ladder must have at least one tier")
    state = RunState.model_validate_json(
        run_dir.run_state_path.read_text(encoding="utf-8")
    )
    if state.status in ("completed", "escalated", "failed"):
        store = _RunStateStore(run_dir, state)
        status: Literal["completed", "escalated", "failed"] = state.status
        return _outcome(
            store,
            status,
            reason=state.terminal_reason if status != "completed" else None,
            summary=state.terminal_reason if status == "completed" else None,
        )

    journal_events = read_events(run_dir.events_path)
    started_phases = {e.phase_id for e in journal_events if isinstance(e, PhaseStarted)}
    # Seed the passed set from the JOURNAL (not just run_state.json) so a kill
    # between emitting phase_passed and persisting the cursor still suppresses a
    # duplicate phase_passed on resume, the journal leads, ruling 6.
    passed_phases = {e.phase_id for e in journal_events if isinstance(e, PhasePassed)}
    with JournalWriter(run_dir.events_path) as journal:
        journal.emit(lambda s: RunResumed(seq=s, ts=_now(), run_id=state.run_id))
        store = _RunStateStore(run_dir, state)
        last_epoch: EpochOutcome | None = None
        result: RunOutcome | None = None

        if state.status == "running_epoch":
            assert state.pending_decision is not None
            decision = parse_decision(state.pending_decision)
            outcome = resume_epoch(
                run_dir,
                journal=journal,
                args=_epoch_args(decision),
                mode=_EPOCH_MODE[decision.tool],
                ladder=ladder,
                repo=repo,
                concurrency=concurrency,
                tier0_attempts=tier0_attempts,
                prepare=prepare,
            )
            _record_epoch(store, outcome, decision)
            if outcome.status == "integration_conflict":
                result = _escalate(
                    journal, store, f"integration conflict in {outcome.epoch_id} (structural bug)"
                )
            else:
                last_epoch = outcome

        # G4 resume guard: a completed epoch (this resume_epoch one, OR one persisted
        # awaiting_planner before a kill) that carries criteria but never reached a
        # terminal verification event is RE-VERIFIED now, so a kill can never let the
        # semantic gate be skipped. A produced verdict (terminal event present) is reused.
        if result is None:
            result = _resume_pending_verification(
                journal, store, run_dir, repo, journal_events,
                verifier=verifier, prepare=prepare, cap=max_failed_epochs_per_phase,
            )
            # A re-verified epoch with a fresh semantic gap may set last_epoch's verdict
            # in run state; the drive loop below picks the pending-failed-epoch up.

        if result is None:
            result = _drive(
                journal,
                run_dir,
                store,
                planner=planner,
                ladder=ladder,
                repo=repo,
                sleep_fn=sleep_fn,
                max_planner_calls=max_planner_calls,
                max_epochs=max_epochs,
                concurrency=concurrency,
                tier0_attempts=tier0_attempts,
                last_epoch=last_epoch,
                started_phases=started_phases,
                passed_phases=passed_phases,
                vision_reviewer=vision_reviewer,
                final_polish=final_polish,
                prepare=prepare,
                floor=floor,
                max_failed_epochs_per_phase=max_failed_epochs_per_phase,
                verifier=verifier,
                infra_repair=infra_repair,
                local_max_task_files=local_max_task_files,
                senior_max_task_files=senior_max_task_files,
            )
    # Events flushed (writer closed): refresh this run's post-mortem journal.
    write_journal(run_dir)
    return result
