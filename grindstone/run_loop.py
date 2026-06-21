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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Sequence

from pydantic import BaseModel, ConfigDict

from grindstone import worktree as wt
from grindstone.config import PrepareConfig
from grindstone.prepare import PrepareError, materialize_env
from grindstone.planner import extract_decision_json
from grindstone.contracts.models import (
    ArtifactEpochArgs,
    ArtifactExistsCheck,
    Check,
    CmdCheck,
    CompleteRunDecision,
    EpochDecision,
    EscalateRunDecision,
    HaltFailedEpochArgs,
    HandleFailedEpochDecision,
    ImplementEpochArgs,
    Phase,
    ProposeSkeletonDecision,
    RetryFailedEpochArgs,
    RevisePhasesDecision,
    VisionReviewCheck,
    parse_decision,
)
from grindstone.script_polish import Polisher
from grindstone.script_vision import VisionReviewError, VisionReviewer
from grindstone.contracts.semantics import HandoffMode
from grindstone.epoch_loop import EpochArgs, EpochOutcome, resume_epoch, run_epoch
from grindstone.journal import reap_sibling_journals, write_journal
from grindstone.memory import load_digest
from grindstone.events import (
    EpochFailed,
    FailedEpochHandled,
    FinalPolishApplied,
    FinalPolishSkipped,
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
    backoff_delay,
    build_planner_input,
    classify_failure,
    flatten_last_epoch,
    validate_decision,
)
from grindstone.repomap import build_repo_map
from grindstone.rundir import RunDir, atomic_write_json
from grindstone.task_loop import TIER0_ATTEMPTS
from grindstone.worker import WorkerTransport

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


#: How much of a failed command's combined stdout/stderr to keep (gate
#: observability, Part A): the planner sees WHY a check failed, env-vs-code,
#: instead of a bare exit code. Bounded so a chatty build cannot bloat the
#: planner input; the TAIL is kept (errors land last).
CHECK_OUTPUT_TAIL_BYTES = 3072


def _check_label(check: Check) -> str:
    if isinstance(check, CmdCheck):
        return f"cmd `{check.cmd}`"
    if isinstance(check, ArtifactExistsCheck):
        return f"artifact_exists:{check.artifact_exists}"
    return f"vision_review:{check.vision_review.screenshot}"


def _safe_output_tail(stdout: str, stderr: str) -> str:
    """The text-safe, length-bounded tail of a failed command's combined output.

    Joins stdout + stderr, keeps the LAST ``CHECK_OUTPUT_TAIL_BYTES`` bytes
    (errors land last), and scrubs the result text-safe: control bytes other than
    tab/newline are stripped so an odd-byte build log can never corrupt the
    planner input or the persisted record. Empty when the command said nothing.
    """

    parts = [p for p in (stdout, stderr) if p]
    combined = "\n".join(parts).strip()
    if not combined:
        return ""
    raw = combined.encode("utf-8", errors="replace")
    if len(raw) > CHECK_OUTPUT_TAIL_BYTES:
        raw = b"...(truncated)...\n" + raw[-CHECK_OUTPUT_TAIL_BYTES:]
    text = raw.decode("utf-8", errors="replace")
    return "".join(c for c in text if c in "\t\n" or c >= " ")


def _indent(text: str, prefix: str = "        ") -> str:
    return "\n".join(prefix + line for line in text.splitlines())


def _persist_check_output(
    run_dir: RunDir, scratch_name: str, index: int, cmd: str, tail: str
) -> None:
    """Durably record a failed check's captured output under the run dir.

    A flat text file per (eval scratch, check index) so the failure trail
    survives the run and is auditable alongside the journal; best-effort, a
    persist failure never breaks check evaluation (the label still carries the
    same tail to the planner)."""

    try:
        out_dir = run_dir.root / "check_output" / scratch_name
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"c{index}.txt").write_text(
            f"$ {cmd}\n{tail}\n", encoding="utf-8"
        )
    except OSError:
        pass


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


def evaluate_checks(
    checks: Sequence[Check],
    *,
    repo: Path | None,
    ref: str | None,
    run_dir: RunDir,
    scratch_name: str,
    vision_reviewer: VisionReviewer | None = None,
    prepare: PrepareConfig | None = None,
) -> list[tuple[str, bool]]:
    """Run a list of deterministic checks; return ``(label, passed)`` IN ORDER.

    Command checks run against a throwaway worktree of ``ref`` (the integration
    tip / final branch) when a repo exists, else the run dir; artifact checks
    resolve against the keyed log; ``vision_review`` (B3 taste gate) renders its
    verdict through ``vision_reviewer`` against a screenshot a PRIOR cmd check
    produced in the same worktree. This is the one evaluator behind both
    ``complete_run`` evidence (ARCHITECTURE.md) and phase exit criteria (S4 ruling 1):
    a deterministic verdict computed in a tip worktree, never a planner claim.
    """

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
        results: list[tuple[str, bool]] = []
        for index, check in enumerate(checks):
            if isinstance(check, ArtifactExistsCheck):
                ok = run_dir.find_artifact(check.artifact_exists) is not None
                results.append((_check_label(check), ok))
            elif worktree_error is not None:
                # A cmd/vision check that needs the worktree we could not create.
                results.append((f"{_check_label(check)} [{worktree_error}]", False))
            elif isinstance(check, VisionReviewCheck):
                results.append(
                    _vision_result(
                        check,
                        cwd=cwd,
                        run_dir=run_dir,
                        scratch_name=scratch_name,
                        index=index,
                        reviewer=vision_reviewer,
                    )
                )
            else:
                proc = subprocess.run(
                    check.cmd, shell=True, cwd=str(cwd), capture_output=True, text=True
                )
                ok = proc.returncode == check.expect_exit
                label = _check_label(check)
                if not ok:
                    # Gate observability (Part A): a FAILED cmd check keeps its
                    # output so the planner can tell env-vs-code, the dogfood
                    # spin-loop blind spot. Persist the full-ish tail under the
                    # run dir AND fold a short tail into the label that flows to
                    # the planner's next decision input.
                    tail = _safe_output_tail(proc.stdout, proc.stderr)
                    label = f"{label} (exit {proc.returncode})"
                    if tail:
                        _persist_check_output(run_dir, scratch_name, index, check.cmd, tail)
                        label += "\n      output:\n" + _indent(tail)
                results.append((label, ok))
        return results
    finally:
        if worktree is not None and repo is not None:
            wt.remove_worktree(repo, worktree)


def recheck_evidence(
    evidence: Sequence[Check],
    *,
    repo: Path | None,
    final_branch: str | None,
    run_dir: RunDir,
    vision_reviewer: VisionReviewer | None = None,
    prepare: PrepareConfig | None = None,
) -> list[str]:
    """Deterministically re-run a ``complete_run``'s evidence; return the failing
    labels (empty = the run's certificate holds). Command checks run against the
    final branch worktree, artifact checks against the keyed log (ARCHITECTURE.md),
    vision_review checks through ``vision_reviewer``."""

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
        )
        if not all(ok for _, ok in results):
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
) -> _PhaseContext:
    """Advance phases, fire a one-shot ``phase_escalated`` on budget exhaustion,
    and assemble the cumulative-state tail context (rulings 1-3)."""

    results = _advance_phases(
        journal, store, run_dir, repo, passed, started, vision_reviewer, prepare
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
    # Whole-repo map of the CURRENT integration tip (volatile-tail context). Built
    # once per boundary (the tip does not move across this boundary's re-asks);
    # None below threshold / on any failure, the run is unaffected either way.
    repo_map = build_repo_map(repo) if repo is not None else None
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
            repo_map=repo_map,
            failed_epoch=failed_epoch,
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
    _record_epoch(store, outcome)
    return outcome


def _epoch_args(decision: EpochDecision) -> EpochArgs:
    """The epoch args of an epoch-tool decision (caller guarantees the tool)."""

    args = decision.args
    assert isinstance(args, (ImplementEpochArgs, ArtifactEpochArgs)), (
        f"{decision.tool} is not an epoch tool"
    )
    return args


def _record_epoch(store: _RunStateStore, outcome: EpochOutcome) -> None:
    new_branch = outcome.integration.branch or store.state.last_integration_branch
    store.update(
        status="awaiting_planner",
        epoch_counter=store.state.epoch_counter + 1,
        phase_epoch_index=store.state.phase_epoch_index + 1,
        phase_budget_used=store.state.phase_budget_used + 1,
        last_integration_branch=new_branch,
        pending_decision=None,
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
    )


def _maybe_open_failed_epoch(
    journal: JournalWriter,
    store: _RunStateStore,
    run_dir: RunDir,
    decision: EpochDecision,
    outcome: EpochOutcome,
    phase_ctx: _PhaseContext | None,
    cap: int,
) -> RunOutcome | None:
    """If the just-run epoch FAILED, open a focused failed-epoch disposition.

    Increments the per-phase failed-epoch counter and, at the deterministic cap,
    FORCES a halt-to-human regardless of the planner (Part C, the spin-loop
    backstop). Below the cap it records the originating decision + the failure
    context in run state so the NEXT boundary is constrained to handle_failed_epoch
    (Part B). Returns a terminal RunOutcome when the cap halts, else ``None``."""

    if not _epoch_failed(outcome):
        return None
    disposed = store.state.phase_failed_epochs + 1
    phase_id = store.state.current_phase_id or "P1"
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
        disposed_count=disposed, cap=cap,
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
        store.update(phase_failed_epochs=disposed, pending_failed_epoch=None)
        return _escalate(journal, store, reason)
    store.update(
        phase_failed_epochs=disposed,
        pending_failed_epoch={
            "decision": decision.model_dump(mode="json"),
            "epoch_id": info.epoch_id,
            "failed_tasks": info.failed_tasks,
            "passing_handoffs": info.passing_handoffs,
        },
    )
    return None


def _pending_failed_epoch_info(
    store: _RunStateStore, phase_ctx: _PhaseContext | None
) -> FailedEpochInfo | None:
    """Rebuild the pending failed-epoch context from run state (resume-safe).

    The originating decision + the failed-task / passing-handoff snapshot are
    durable in run state; the failing-check labels (with the latest captured
    output, Part A) are refreshed from THIS pass's phase evaluation so the planner
    always sees the current gate output. ``None`` when nothing is pending."""

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
        cap=store.state.phase_failed_epochs,  # display only; cap re-applied on action
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
    # climbing toward the cap, so a retry-loop terminates deterministically).
    result = _maybe_open_failed_epoch(
        journal, store, run_dir, epoch_decision, outcome, None, cap
    )
    return result, outcome


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
            vision_reviewer, prepare,
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
    max_failed_epochs_per_phase: int,
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
            )
            if store.state.skeleton is not None
            else None
        )

        # A FAILED epoch awaiting a focused disposition (Part B): rebuild its
        # context (refreshing the phase-check output from this pass, Part A) and
        # constrain the next decision to handle_failed_epoch. Resume-safe: the
        # originating decision + raw context live in run state.
        failed_epoch_info = _pending_failed_epoch_info(store, phase_ctx)

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
                tier0_attempts, prepare, max_failed_epochs_per_phase,
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
                final_polish, vision_reviewer, prepare,
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
        result = _maybe_open_failed_epoch(
            journal, store, run_dir, decision, outcome, phase_ctx,
            max_failed_epochs_per_phase,
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
            max_failed_epochs_per_phase=max_failed_epochs_per_phase,
            local_max_task_files=local_max_task_files,
            senior_max_task_files=senior_max_task_files,
        )
    # Events are fully flushed (writer closed): render the post-mortem journal.
    write_journal(run_dir)
    return outcome


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
            _record_epoch(store, outcome)
            if outcome.status == "integration_conflict":
                result = _escalate(
                    journal, store, f"integration conflict in {outcome.epoch_id} (structural bug)"
                )
            else:
                last_epoch = outcome

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
                max_failed_epochs_per_phase=max_failed_epochs_per_phase,
                local_max_task_files=local_max_task_files,
                senior_max_task_files=senior_max_task_files,
            )
    # Events flushed (writer closed): refresh this run's post-mortem journal.
    write_journal(run_dir)
    return result
