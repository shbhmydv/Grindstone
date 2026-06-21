"""Synchronous, deterministic single-task state machine (ARCHITECTURE.md/§9/§10/§11).

One task, end-to-end, to a terminal outcome. Identity is **parameterized**
(S2 ruling 3): a frozen ``TaskIdentity`` (phase/epoch/task) threads through the
machine, so the same code drives every task in a fan-out, there are no fixed
``P1/E1/T1`` constants. The run/phase/epoch scaffold and the journal lifetime
are owned by ``epoch_loop`` now; this module only grinds one task against a
**shared, thread-safe** journal and reports its terminal state through a cursor
callback so a kill at any point resumes coherently.

Per attempt (ARCHITECTURE.md, implement tasks):

    fresh worktree from the epoch base -> worker runs there (its scratch CWD)
      -> read handoff.json, relocate to the log key, re-validate (schema + typed
         parse + semantic rules + grounding) + re-run done_when
      -> core commits the worktree (models never run git)
      -> ownership scope check: every committed path must fall in file_ownership
      -> DONE (keep branch for integration) | rejected attempt (worktree + branch
         torn down, zero dead artifacts, and re-queued)

Artifact tasks keep the S1 run-dir scratch (no worktree, no git): never hand a
non-write task the live repo as CWD (v7 bugs A/C/H class).

Attempt policy: up to ``tier0_attempts`` on tier 0, then one attempt per higher
rung. Exhausting the ladder is FAILED. No wall-clock timeouts (§10): worker
supervision is the transport's job; the loop only judges what landed on disk.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Literal, Sequence

from pydantic import BaseModel, ConfigDict

from grindstone.check_handoff import CHECK_COMMAND, CHECK_SCRIPT_NAME, generate_check_script
from grindstone.config import PrepareConfig
from grindstone.prepare import materialize_env
from grindstone.contracts.gate import handoff_schema_errors
from grindstone.contracts.models import (
    ArtifactExistsCheck,
    ArtifactTask,
    Check,
    CmdCheck,
    Handoff,
    ImplementTask,
    VisionReviewCheck,
    parse_handoff,
)
from grindstone.contracts.semantics import HandoffMode, handoff_violations
from grindstone.events import (
    HandoffRejected,
    JournalWriter,
    TaskDispatched,
    TaskDone,
    TaskEscalated,
    TaskFailed,
    TaskRetried,
)
from grindstone.repomap import WORKER_SUBTREE_TOKENS, build_repo_map
from grindstone.rundir import RunDir
from grindstone import worktree as wt
from grindstone.worker import (
    PI_SETTINGS_RELPATH,
    REVIEW_CHECK_COMMAND,
    REVIEW_FILENAME,
    Task,
    WorkerRequest,
    WorkerTransport,
)

TIER0_ATTEMPTS = 3


# --- identity ------------------------------------------------------------------


@dataclass(frozen=True)
class TaskIdentity:
    """The task's place in the run tree; replaces S1's module-level constants."""

    phase_id: str
    epoch_id: str
    task_id: str

    @property
    def fq(self) -> str:
        """Fully-qualified log-key prefix ``<phase>/<epoch>/<task>``."""

        return f"{self.phase_id}/{self.epoch_id}/{self.task_id}"

    def attempt_branch(self, attempt: int) -> str:
        """The per-attempt worktree branch (ruling 4)."""

        return f"grind/{self.phase_id}/{self.epoch_id}/{self.task_id}-a{attempt}"


# --- public result + persisted per-task cursor ---------------------------------


@dataclass(frozen=True)
class TaskOutcome:
    """The terminal result of one task's grind."""

    identity: TaskIdentity
    status: Literal["done", "failed"]
    tier: str
    attempts: int
    handoff: Handoff | None
    handoff_key: str | None
    branch: str | None
    reason: str | None


class TaskCursorState(BaseModel):
    """Durable per-task cursor inside the epoch state (rewritten every transition).

    Records exactly what resume needs: terminal/in-flight status, ladder
    position, attempt counts, the in-flight scratch (or, on DONE, the KEPT
    worktree path that integration prunes), the winning branch, and the
    accumulated failure context. ``pending`` is a task not yet dispatched.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    fq_task_id: str
    task_id: str
    mode: HandoffMode
    status: Literal["pending", "running", "done", "failed"]
    tier_index: int
    tier_name: str
    tier_attempt: int
    attempt: int
    scratch: str | None
    branch: str | None
    failure_context: list[str]
    reason: str | None


def pending_cursor(identity: TaskIdentity, mode: HandoffMode) -> TaskCursorState:
    """The initial cursor for a not-yet-dispatched task."""

    return TaskCursorState(
        fq_task_id=identity.fq,
        task_id=identity.task_id,
        mode=mode,
        status="pending",
        tier_index=0,
        tier_name="",
        tier_attempt=0,
        attempt=0,
        scratch=None,
        branch=None,
        failure_context=[],
        reason=None,
    )


CursorSink = Callable[[TaskCursorState], None]


# --- internals -----------------------------------------------------------------


class _AttemptFailed(Exception):
    """One attempt failed validation/checks; ``reason`` becomes failure context."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass
class _Cursor:
    """Mutable ladder position threaded through the grind loop."""

    tier_index: int
    tier_attempt: int
    attempt: int
    failure_context: list[str] = field(default_factory=list)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


#: Modes whose PRIMARY worker tier is ``senior`` (cloud + web tools), not local.
#: research/review need judgment and web search (Exa via the senior script);
#: implement/artifact are production work for the local rig. The planner stays
#: model-agnostic, it emits a MODE; the core maps mode -> the starting tier.
_SENIOR_FIRST_MODES: frozenset[HandoffMode] = frozenset({"research", "review"})
_SENIOR_TIER = "senior"


def starting_tier(
    mode: HandoffMode, tier_names: Sequence[str], *, visual: bool = False
) -> int:
    """Ladder index a task STARTS on (before any escalation).

    research/review begin on the ``senior`` tier (web search + stronger judgment)
    when the ladder has one; implement/artifact begin on tier 0 (the local rig).
    ``visual`` is taste routing (B3 seam): a VISUAL/taste epoch, UI or polish
    output, also begins on ``senior`` whatever its mode, because the senior is
    the stronger TASTE-BUILDER (it builds work judged by how it looks). The senior
    is a text model, not vision-capable: the genuine image-based judgment is the
    B3 codex vision-review gate in the phase exit criterion, not this worker. A
    rig with no senior tier falls every mode back to local, research there is repo
    investigation only, and a visual epoch grinds locally rather than crash.
    Escalation still walks upward from the start tier.
    """

    if visual or mode in _SENIOR_FIRST_MODES:
        try:
            return tier_names.index(_SENIOR_TIER)
        except ValueError:
            return 0
    return 0


def _tier_allowance(tier_index: int, start_tier: int, tier0_attempts: int) -> int:
    """Attempts permitted on a tier: ``tier0_attempts`` on the task's STARTING
    tier (wherever a mode begins), else 1 on each escalation rung above it."""

    return tier0_attempts if tier_index == start_tier else 1


def _strip_pi_settings(scratch: Path) -> None:
    """Remove the worker's per-cwd ``.pi/settings.json`` (and the now-empty
    ``.pi/`` dir) before commit, orchestration metadata never enters the diff.

    Surgical: only the settings file and an *empty* ``.pi/`` are removed. If the
    worker left other content under ``.pi/`` it stays, and the ownership scope
    check rejects it as an out-of-scope write rather than silently discarding it.
    Model/provider knowledge lives in the transport; the loop only knows the
    shared relative path.
    """

    settings = scratch / PI_SETTINGS_RELPATH
    settings.unlink(missing_ok=True)
    pi_dir = settings.parent
    if pi_dir.is_dir() and not any(pi_dir.iterdir()):
        pi_dir.rmdir()


def _grounding_violations(
    handoff: Handoff, scratch: Path, *, repo: Path | None, mode: HandoffMode
) -> list[str]:
    """Deterministic citation spot-check (ARCHITECTURE.md): every cited file must exist
    inside an allowed root.

    Implement scratch IS a repo checkout, so it is the only allowed root there
    (the operator checkout stays out of bounds). research/review/artifact tasks
    investigate the TARGET REPO from a plain scratch dir, so the repo root is a
    second allowed root, scratch-only rejected every legitimate repo citation
    (gate-5 P0: the task could never hand off and the planner spun revisions).
    Citations resolve against scratch first, then the repo root.
    """

    roots = [scratch.resolve()]
    if mode != "implement" and repo is not None:
        roots.append(repo.resolve())
    out: list[str] = []
    for cite in handoff.citations:
        found: Path | None = None
        for root in roots:
            cand = (root / cite.file).resolve()
            if cand.is_file() and any(cand.is_relative_to(r) for r in roots):
                found = cand
                break
        if found is None:
            out.append(f"citation missing or outside allowed roots: {cite.file}")
            continue
        if cite.line is not None:
            lines = found.read_text(encoding="utf-8", errors="replace").splitlines()
            if len(lines) < cite.line:
                out.append(
                    f"citation line {cite.line} beyond {cite.file} ({len(lines)} lines)"
                )
    return out


def _run_one_check(check: Check, scratch: Path, run_dir: RunDir) -> tuple[str, int]:
    """Re-run one done_when check; return (label, exit). Exit 0 == pass."""

    if isinstance(check, CmdCheck):
        proc = subprocess.run(
            check.cmd, shell=True, cwd=str(scratch), capture_output=True, text=True
        )
        passed = proc.returncode == check.expect_exit
        return (check.cmd, 0 if passed else 1)
    if isinstance(check, ArtifactExistsCheck):
        exists = run_dir.find_artifact(check.artifact_exists) is not None
        return (f"artifact_exists:{check.artifact_exists}", 0 if exists else 1)
    # The taste gate (B3) renders + reviews a screenshot in a TIP worktree
    # (run_loop.evaluate_checks); a worker's done_when CWD is a scratch with no
    # reviewer wired, so a vision_review here is a planner mis-placement, fail
    # deterministically (never crash) and steer it to the phase exit criterion.
    assert isinstance(check, VisionReviewCheck)
    return (
        f"vision_review (belongs in a phase exit criterion): "
        f"{check.vision_review.screenshot}",
        1,
    )


def _run_done_when(
    checks: Sequence[Check], scratch: Path, run_dir: RunDir
) -> list[str]:
    """Re-run all checks; return failure labels (empty == all passed)."""

    failures: list[str] = []
    for check in checks:
        label, code = _run_one_check(check, scratch, run_dir)
        if code != 0:
            failures.append(label)
    return failures


def _collect_handoff(
    request: WorkerRequest,
    task: Task,
    mode: HandoffMode,
    run_dir: RunDir,
    identity: TaskIdentity,
    repo: Path | None,
) -> Handoff:
    """Relocate + fully validate one attempt's handoff; raise ``_AttemptFailed``.

    Order (ARCHITECTURE.md): read the scratch handoff, relocate parseable bytes to the
    log key, re-validate the *relocated* record (schema, typed parse, semantic
    rules, grounding, done_when, DONE status). On any failure the relocated
    record is deleted (zero dead artifacts).
    """

    scratch = request.scratch
    src = scratch / "handoff.json"
    if not src.is_file():
        raise _AttemptFailed("no handoff.json written")
    raw = src.read_text(encoding="utf-8")
    try:
        json.loads(raw)
    except json.JSONDecodeError as exc:
        raise _AttemptFailed(f"handoff.json is not valid JSON: {exc}") from exc

    dest = run_dir.resolve(f"{identity.fq}/handoff.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dest)

    def reject(reason: str) -> _AttemptFailed:
        dest.unlink(missing_ok=True)
        return _AttemptFailed(reason)

    payload = json.loads(dest.read_text(encoding="utf-8"))
    schema_errors = handoff_schema_errors(payload)
    if schema_errors:
        raise reject(f"handoff schema invalid: {schema_errors[0]}")
    try:
        handoff = parse_handoff(payload)
    except ValueError as exc:
        raise reject(f"handoff parse failed: {exc}") from exc
    violations = handoff_violations(
        handoff, mode=mode, expected_task_id=request.task_id
    )
    if violations:
        raise reject("; ".join(violations))
    grounding = _grounding_violations(handoff, scratch, repo=repo, mode=mode)
    if grounding:
        raise reject("; ".join(grounding))
    failed_checks = _run_done_when(task.done_when, scratch, run_dir)
    if failed_checks:
        raise reject(f"done_when failed: {', '.join(failed_checks)}")
    if handoff.status != "DONE":
        detail = "; ".join(handoff.not_done) or handoff.resulting_state
        raise reject(f"handoff status {handoff.status}: {detail}")
    return handoff


def _scratch_for(
    identity: TaskIdentity, run_dir: RunDir, attempt: int, *, implement: bool
) -> Path:
    """The attempt's CWD path (NOT yet created for worktrees, git owns that)."""

    if implement:
        return run_dir.root / "worktrees" / identity.task_id / f"attempt-{attempt}"
    return run_dir.artifacts_dir(f"{identity.fq}/attempt-{attempt}")


def _install_attempt_checks(
    task: Task, scratch: Path, mode: HandoffMode, fq_task_id: str, repo: Path | None
) -> Task:
    """Drop the worker-facing validator in the attempt CWD + wire the gates.

    Writes ``check_handoff.py`` (generated from the dispatched fully-qualified
    task_id + mode) into ``scratch`` and returns a runtime copy of ``task`` with
    ``python3 check_handoff.py`` appended to ``done_when``, so it shows in the
    worker prompt's done_when AND in the core's authoritative re-run. Implement
    attempts additionally get the review gate ``test -s review.md`` appended:
    the fresh-context review demanded as a checked artifact (verified to fire;
    prose review instructions don't). The copy is local to this attempt: the
    planner decision and journal are untouched, so resume reconstructs the
    identical augmented task deterministically.
    """

    # Non-implement attempts get the target repo as a second allowed citation
    # root (their scratch is a plain dir; the files they investigate live in
    # the repo). Implement attempts bake None: their CWD IS a repo checkout
    # and the operator checkout must stay out of bounds. Mirrors the core
    # gate's _grounding_violations exactly, worker-vs-core drift on citation
    # semantics was the gate-5 P0 (self-validated green, core rejected).
    (scratch / CHECK_SCRIPT_NAME).write_text(
        generate_check_script(
            task_id=fq_task_id,
            mode=mode,
            repo_root=repo if mode != "implement" else None,
        ),
        encoding="utf-8",
    )
    appended: list[Check] = [CmdCheck(cmd=CHECK_COMMAND)]
    if mode == "implement":
        appended.append(CmdCheck(cmd=REVIEW_CHECK_COMMAND))
    return task.model_copy(update={"done_when": [*task.done_when, *appended]})


def _worker_subtree(repo: Path | None, task: Task) -> str | None:
    """A repo-map subtree seeded on this task's files (a large-repo nav aid).

    Implement tasks seed on ``file_ownership`` (globs/paths), collapsing the map
    toward the neighborhood the task will edit. research/review/artifact tasks
    seed on ``targets`` when present, else get no map (their scratch is not a repo
    checkout and they navigate via resolved inputs). ``None`` whenever no repo is
    configured, the repo is below threshold, or anything fails (never crashes)."""

    if repo is None:
        return None
    if isinstance(task, ImplementTask):
        focus = [Path(g) for g in task.file_ownership]
        return build_repo_map(repo, map_tokens=WORKER_SUBTREE_TOKENS, focus_files=focus)
    if task.targets:
        focus = [Path(t) for t in task.targets]
        return build_repo_map(repo, map_tokens=WORKER_SUBTREE_TOKENS, focus_files=focus)
    return None


def _dispatch_attempt(
    *,
    identity: TaskIdentity,
    task: Task,
    mode: HandoffMode,
    run_dir: RunDir,
    worker: WorkerTransport,
    cursor: _Cursor,
    repo: Path | None,
    base: str | None,
    scratch: Path,
    branch: str | None,
    prepare: PrepareConfig | None,
) -> Handoff:
    """Run one attempt end-to-end; raise ``_AttemptFailed`` on any failure.

    For implement tasks the scratch is a fresh worktree from ``base``; on the
    handoff's success the core commits and scope-checks the diff. Any failure,
    transport, handoff, or out-of-scope write, maps identically to a failed
    attempt (ARCHITECTURE.md: the loop never inspects worker internals).
    """

    implement = isinstance(task, ImplementTask)
    if implement:
        assert repo is not None and base is not None and branch is not None
        wt.add_worktree(repo, scratch, branch=branch, base=base)
        # Restore the declared (gitignored) dependency dirs into the worker
        # worktree so it does not burn turns on a fresh install and shares the
        # same lockfile-hashed cache the eval gate uses. A prepare failure is a
        # failed attempt (same cache the gate will hit, fail loudly here).
        try:
            materialize_env(repo, scratch, prepare)
        except Exception as exc:  # PrepareError (or any IO): a clean failed attempt
            raise _AttemptFailed(f"prepare error: {exc}") from exc
    else:
        scratch.mkdir(parents=True, exist_ok=True)
    task = _install_attempt_checks(task, scratch, mode, identity.fq, repo)

    inputs = {key: run_dir.resolve(key) for key in task.inputs}
    request = WorkerRequest(
        task=task,
        task_id=identity.fq,
        inputs=inputs,
        scratch=scratch,
        attempt=cursor.attempt,
        failure_context=list(cursor.failure_context),
        mode=mode,
        repo_map=_worker_subtree(repo, task),
    )
    try:
        worker.run(request)
    except Exception as exc:  # transport boundary, any raise is a failed attempt
        raise _AttemptFailed(f"transport error: {type(exc).__name__}: {exc}") from exc
    handoff = _collect_handoff(request, task, mode, run_dir, identity, repo)

    if isinstance(task, ArtifactTask):
        # Publish the produced artifact to its log key, the artifact analogue
        # of the handoff relocation. Gate-6 P0: the file stayed in scratch, so
        # artifact_exists checks were structurally unsatisfiable and the
        # planner revised phases until the safety valve. A promised-but-absent
        # artifact is a truthful failed attempt, and the already-relocated
        # handoff is deleted with it (zero dead artifacts).
        # Read the artifact from the FULL relative path the worker was told to
        # write (prompt + done_when both reference task.artifact_out verbatim, as
        # a CWD-relative path). A basename-only lookup rejected an artifact_out
        # carrying a real subdir (e.g. `MIGRATION/inv.md`) even when the worker
        # produced it correctly, dogfood-1's 15 spurious rejections + retries.
        # LogKey is a validated safe relative path (same value resolve() trusts).
        src = scratch / task.artifact_out
        if not src.is_file():
            run_dir.resolve(f"{identity.fq}/handoff.json").unlink(missing_ok=True)
            raise _AttemptFailed(
                f"artifact_out not produced in CWD: {task.artifact_out}"
            )
        published = run_dir.resolve(task.artifact_out)
        published.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, published)

    if implement:
        assert repo is not None and base is not None and isinstance(task, ImplementTask)
        # handoff.json + check_handoff.py + review.md + .pi/settings.json are
        # orchestration metadata (the handoff is already relocated to the log
        # key; the validator is core-written; the review was gated by the
        # done_when re-run above; the .pi settings only pin the worker's
        # subagents), not the task's repo work, drop the scratch copies so they
        # never enter the commit or trip the ownership scope check.
        (scratch / "handoff.json").unlink(missing_ok=True)
        (scratch / CHECK_SCRIPT_NAME).unlink(missing_ok=True)
        (scratch / REVIEW_FILENAME).unlink(missing_ok=True)
        _strip_pi_settings(scratch)
        wt.commit_all(scratch, f"grind({identity.fq}): {task.goal.splitlines()[0][:72]}")
        # Diff in the WORKTREE (its HEAD is the committed task tip; the main
        # checkout still sits at base).
        out_of_scope = wt.scope_violations(
            wt.changed_paths(scratch, base), list(task.file_ownership)
        )
        if out_of_scope:
            raise _AttemptFailed(
                "out-of-scope writes: " + ", ".join(sorted(out_of_scope))
            )
    return handoff


def _grind(
    *,
    identity: TaskIdentity,
    task: Task,
    mode: HandoffMode,
    run_dir: RunDir,
    journal: JournalWriter,
    sink: CursorSink,
    ladder: Sequence[tuple[str, WorkerTransport]],
    cursor: _Cursor,
    repo: Path | None,
    base: str | None,
    tier0_attempts: int,
    first_attempt_already_dispatched: bool,
    visual: bool,
    prepare: PrepareConfig | None,
) -> TaskOutcome:
    """Drive the ladder from ``cursor`` to a terminal outcome.

    Emits task_dispatched (once), task_retried (each later attempt),
    task_escalated (each tier climb), and task_done / task_failed via the SHARED
    thread-safe journal (``emit`` assigns the seq under the lock). Rewrites the
    task's cursor through ``sink`` before every attempt and at the terminal.
    """

    implement = isinstance(task, ImplementTask)
    dispatched = first_attempt_already_dispatched
    start_tier = starting_tier(mode, [name for name, _ in ladder], visual=visual)

    def write_cursor(
        status: Literal["running", "done", "failed"],
        *,
        tier_index: int,
        tier_name: str,
        scratch: str | None,
        branch: str | None,
        reason: str | None,
    ) -> None:
        sink(
            TaskCursorState(
                fq_task_id=identity.fq,
                task_id=identity.task_id,
                mode=mode,
                status=status,
                tier_index=tier_index,
                tier_name=tier_name,
                tier_attempt=cursor.tier_attempt,
                attempt=cursor.attempt,
                scratch=scratch,
                branch=branch,
                failure_context=list(cursor.failure_context),
                reason=reason,
            )
        )

    while cursor.tier_index < len(ladder):
        tier_name, worker = ladder[cursor.tier_index]
        allowance = _tier_allowance(cursor.tier_index, start_tier, tier0_attempts)
        while cursor.tier_attempt < allowance:
            cursor.tier_attempt += 1
            cursor.attempt += 1
            scratch = _scratch_for(identity, run_dir, cursor.attempt, implement=implement)
            branch = identity.attempt_branch(cursor.attempt) if implement else None
            write_cursor(
                "running",
                tier_index=cursor.tier_index,
                tier_name=tier_name,
                scratch=str(scratch),
                branch=branch,
                reason=None,
            )
            if not dispatched:
                journal.emit(
                    lambda s: TaskDispatched(
                        seq=s, ts=_now(), epoch_id=identity.epoch_id, task_id=identity.task_id
                    )
                )
                dispatched = True
            else:
                attempt_n = cursor.attempt
                journal.emit(
                    lambda s: TaskRetried(
                        seq=s,
                        ts=_now(),
                        epoch_id=identity.epoch_id,
                        task_id=identity.task_id,
                        attempt=attempt_n,
                    )
                )
            try:
                handoff = _dispatch_attempt(
                    identity=identity,
                    task=task,
                    mode=mode,
                    run_dir=run_dir,
                    worker=worker,
                    cursor=cursor,
                    repo=repo,
                    base=base,
                    scratch=scratch,
                    branch=branch,
                    prepare=prepare,
                )
            except _AttemptFailed as failure:
                cursor.failure_context.append(failure.reason)
                if implement and repo is not None and branch is not None:
                    wt.discard_attempt(repo, scratch, branch)
                reason = failure.reason
                journal.emit(
                    lambda s: HandoffRejected(
                        seq=s,
                        ts=_now(),
                        epoch_id=identity.epoch_id,
                        task_id=identity.task_id,
                        reason=reason,
                    )
                )
                continue
            journal.emit(
                lambda s: TaskDone(
                    seq=s, ts=_now(), epoch_id=identity.epoch_id, task_id=identity.task_id
                )
            )
            kept_scratch = str(scratch)
            write_cursor(
                "done",
                tier_index=cursor.tier_index,
                tier_name=tier_name,
                scratch=kept_scratch,
                branch=branch,
                reason=None,
            )
            return TaskOutcome(
                identity=identity,
                status="done",
                tier=tier_name,
                attempts=cursor.attempt,
                handoff=handoff,
                handoff_key=f"{identity.fq}/handoff.json",
                branch=branch,
                reason=None,
            )

        # Tier exhausted; climb if a higher rung exists.
        if cursor.tier_index + 1 < len(ladder):
            next_tier = ladder[cursor.tier_index + 1][0]
            cursor.tier_index += 1
            cursor.tier_attempt = 0
            journal.emit(
                lambda s: TaskEscalated(
                    seq=s,
                    ts=_now(),
                    epoch_id=identity.epoch_id,
                    task_id=identity.task_id,
                    tier=next_tier,
                )
            )
            continue
        break

    reason = cursor.failure_context[-1] if cursor.failure_context else "no attempts"
    journal.emit(
        lambda s: TaskFailed(
            seq=s, ts=_now(), epoch_id=identity.epoch_id, task_id=identity.task_id
        )
    )
    failed_tier = ladder[min(cursor.tier_index, len(ladder) - 1)][0]
    write_cursor(
        "failed",
        tier_index=min(cursor.tier_index, len(ladder) - 1),
        tier_name=failed_tier,
        scratch=None,
        branch=None,
        reason=reason,
    )
    return TaskOutcome(
        identity=identity,
        status="failed",
        tier=failed_tier,
        attempts=cursor.attempt,
        handoff=None,
        handoff_key=None,
        branch=None,
        reason=reason,
    )


def run_task(
    *,
    identity: TaskIdentity,
    task: Task,
    mode: HandoffMode,
    run_dir: RunDir,
    journal: JournalWriter,
    sink: CursorSink,
    ladder: Sequence[tuple[str, WorkerTransport]],
    repo: Path | None = None,
    base: str | None = None,
    tier0_attempts: int = TIER0_ATTEMPTS,
    resume_cursor: TaskCursorState | None = None,
    visual: bool = False,
    prepare: PrepareConfig | None = None,
    epoch_hint: str | None = None,
) -> TaskOutcome:
    """Grind ONE task to terminal against a shared journal (fan-out unit).

    Fresh start: dispatch from attempt 0. Resume (``resume_cursor`` with status
    ``running``): the attempt in flight at kill is **burned**, a killed worker
    cannot be trusted, so its worktree + branch are torn down and the attempt is
    NOT re-run (its number is kept); a handoff_rejected is journaled and the
    grind continues from the recorded ladder position. Implement tasks require
    ``repo`` + ``base`` (the epoch base commit).

    ``epoch_hint`` (a planner ``handle_failed_epoch`` retry corrective) seeds the
    FRESH-start cursor's failure context so the first worker attempt already
    carries the guidance; it is ignored on resume (the cursor's own accumulated
    context leads).
    """

    if not ladder:
        raise ValueError("ladder must have at least one tier")
    implement = isinstance(task, ImplementTask)
    if implement and (repo is None or base is None):
        raise ValueError("implement task requires repo + base")

    if resume_cursor is None:
        return _grind(
            identity=identity,
            task=task,
            mode=mode,
            run_dir=run_dir,
            journal=journal,
            sink=sink,
            ladder=ladder,
            cursor=_Cursor(
                tier_index=starting_tier(mode, [n for n, _ in ladder], visual=visual),
                tier_attempt=0,
                attempt=0,
                failure_context=(
                    [f"planner retry hint: {epoch_hint}"] if epoch_hint else []
                ),
            ),
            repo=repo,
            base=base,
            tier0_attempts=tier0_attempts,
            first_attempt_already_dispatched=False,
            visual=visual,
            prepare=prepare,
        )

    # Resume: burn the in-flight attempt.
    burn = f"resumed: in-flight attempt {resume_cursor.attempt} abandoned (worker killed)"
    if implement and repo is not None:
        scratch = _scratch_for(identity, run_dir, resume_cursor.attempt, implement=True)
        wt.discard_attempt(repo, scratch, identity.attempt_branch(resume_cursor.attempt))
    journal.emit(
        lambda s: HandoffRejected(
            seq=s,
            ts=_now(),
            epoch_id=identity.epoch_id,
            task_id=identity.task_id,
            reason=burn,
        )
    )
    return _grind(
        identity=identity,
        task=task,
        mode=mode,
        run_dir=run_dir,
        journal=journal,
        sink=sink,
        ladder=ladder,
        cursor=_Cursor(
            tier_index=resume_cursor.tier_index,
            tier_attempt=resume_cursor.tier_attempt,
            attempt=resume_cursor.attempt,
            failure_context=[*resume_cursor.failure_context, burn],
        ),
        repo=repo,
        base=base,
        tier0_attempts=tier0_attempts,
        first_attempt_already_dispatched=True,
        visual=visual,
        prepare=prepare,
    )
