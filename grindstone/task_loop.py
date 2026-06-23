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
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Literal, Sequence

from pydantic import BaseModel, ConfigDict

from grindstone.check_handoff import CHECK_COMMAND, CHECK_SCRIPT_NAME, generate_check_script
from grindstone.config import PrepareConfig
from grindstone.domain_skills import load_domain_skill
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
from grindstone.infra import classify_check_failure
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
    MAX_SESSION_LIMIT_WAITS,
    PI_SETTINGS_RELPATH,
    REVIEW_CHECK_COMMAND,
    REVIEW_FILENAME,
    SESSION_LIMIT_RETRY_S,
    RateLimited,
    SessionLimited,
    Task,
    WorkerRequest,
    WorkerTransport,
)

TIER0_ATTEMPTS = 3

#: Injected sleep so the session-limit park is testable (fake records the wait, no
#: wall clock); defaults to the real ``time.sleep``. Mirrors run_loop's ``SleepFn``.
SleepFn = Callable[[float], None]

#: DoS sanity backstop on the ``handoff.json`` disk read (the principle's item F): a
#: generous megabyte-scale ceiling that REJECTS (fail-safe, never truncates) an
#: absurd/corrupt handoff before it is read into memory. Distinct from the semantic
#: ``HANDOFF_MAX_BYTES`` (8 KiB) the validator enforces on a VALID handoff: this only
#: guards the read itself against a pathological file, far above any real handoff.
HANDOFF_FILE_MAX_BYTES = 8 * 1024 * 1024


# --- identity ------------------------------------------------------------------


@dataclass(frozen=True)
class TaskIdentity:
    """The task's place in the run tree; replaces S1's module-level constants."""

    run_id: str
    phase_id: str
    epoch_id: str
    task_id: str

    @property
    def fq(self) -> str:
        """Fully-qualified log-key prefix ``<phase>/<epoch>/<task>``.

        Run-agnostic on purpose: log keys are scoped by the run dir, not the
        name, so only branch names carry the run id (see ``attempt_branch``).
        """

        return f"{self.phase_id}/{self.epoch_id}/{self.task_id}"

    def attempt_branch(self, attempt: int) -> str:
        """The per-attempt worktree branch (ruling 4), run-scoped so two runs
        (or a re-run after escalation) can never collide on a leftover branch.
        """

        return f"grind/{self.run_id}/{self.phase_id}/{self.epoch_id}/{self.task_id}-a{attempt}"


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
    """One attempt failed validation/checks; ``reason`` becomes failure context.

    ``reason`` is the SHORT, already-bounded diagnosis (the journal event + the
    one-line summary the next worker sees). ``detail`` is the optional FULL,
    unbounded text behind that reason (e.g. the complete out-of-scope path list):
    it is persisted to a failure-detail file under the run dir and REFERENCED by
    path, never embedded in the next prompt. ``None`` means the reason IS the full
    detail (nothing larger to persist beyond the reason itself).

    ``chainable`` (implement only) says whether the rejected attempt's branch is a
    SOUND base for an incremental same-tier retry. Most rejections are chainable: a
    failed done_when / review gate / non-DONE status leaves real, fixable work in
    the worktree the next attempt should keep. An OUT-OF-SCOPE rejection is NOT
    chainable: the branch committed files the task must not own, so basing the retry
    on it would carry that pollution into the base (where the scope check can no
    longer even see it); such a retry restarts from the clean epoch base instead.
    """

    def __init__(
        self, reason: str, *, detail: str | None = None, chainable: bool = True
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.detail = detail
        self.chainable = chainable


@dataclass
class _Cursor:
    """Mutable ladder position threaded through the grind loop."""

    tier_index: int
    tier_attempt: int
    attempt: int
    failure_context: list[str] = field(default_factory=list)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


#: The senior tier's name in the ladder; a ``senior`` task starts here when present.
_SENIOR_TIER = "senior"

#: Max out-of-scope paths NAMED in a rejection reason before the rest are summed
#: as ``... and <K> more``. A worker that materializes a dependency tree could
#: otherwise produce a list of every file under it (dogfood: a 1.8M-char reason
#: listing every node_modules/.bin/* path), which then gets fed back into the next
#: attempt's prompt as failure context and blows past the CLI argv limit.
_MAX_NAMED_SCOPE_VIOLATIONS = 20

#: Per-failure-context-entry byte cap folded into the worker prompt (the G6/G7
#: bounding discipline: prior-failure text must never explode the prompt). A single
#: rejection reason kept under this stays well clear of any argv/stdin pressure even
#: across several stacked attempts. With the reference-not-embed scheme below this is
#: a BACKSTOP: the entry is already a one-line summary + paths, well under the cap.
_MAX_FAILURE_CONTEXT_BYTES = 2048

#: Hard cap on the one-line failure SUMMARY the next worker sees inline (the
#: reference-not-embed primary mechanism: a worker carries a short category + brief
#: reason, then a PATH to the full detail on disk it MAY read). Anything longer is
#: head-clipped; the full text lives in the referenced failure-detail file.
_MAX_SUMMARY_CHARS = 200

#: Run-dir subdir, per task, holding one full failure-detail file per failed attempt
#: (``<run>/<P/E/T>/failures/attempt-<n>.txt``) plus, when one was written, that
#: attempt's rejected ``handoff.json``. The worker is referenced to these absolute
#: paths; it reads them on demand instead of carrying the bulk inline.
FAILURES_SUBDIR = "failures"


def _bound_reason(text: str, max_bytes: int = _MAX_FAILURE_CONTEXT_BYTES) -> str:
    """Bound one failure reason to ``max_bytes`` (head kept, a marker appended).

    The HEAD is kept (a reason leads with its diagnosis); a truncated tail is
    replaced by an explicit marker so the planner/worker sees the elision rather
    than a silently-clipped string. Text-safe via the same encode/decode discipline
    used for check-output tails.
    """

    raw = text.encode("utf-8", errors="replace")
    if len(raw) <= max_bytes:
        return text
    head = raw[:max_bytes].decode("utf-8", errors="replace")
    return f"{head}\n...[reason truncated, {len(raw)} bytes total]"


def _format_scope_violations(out_of_scope: list[str]) -> str:
    """A BOUNDED out-of-scope-writes rejection reason.

    Names at most ``_MAX_NAMED_SCOPE_VIOLATIONS`` offending paths (sorted, stable),
    then sums the remainder as ``... and <K> more``. A rejection reason must never
    balloon to megabytes, the dogfood trigger that, fed back as prior-failure
    context, overflowed the model CLI's argv limit and killed every later attempt.
    The FULL, un-elided list is persisted separately (``_full_scope_violations``)
    and referenced by path, so the worker can still see every offending path.
    """

    ordered = sorted(out_of_scope)
    shown = ordered[:_MAX_NAMED_SCOPE_VIOLATIONS]
    reason = "out-of-scope writes: " + ", ".join(shown)
    extra = len(ordered) - len(shown)
    if extra > 0:
        reason += f", ... and {extra} more"
    return reason


def _full_scope_violations(out_of_scope: list[str]) -> str:
    """The COMPLETE out-of-scope-writes detail: a count header + every path, one
    per line (sorted, stable). Persisted to the failure-detail file and referenced
    by path, this is the bulk that must NEVER be embedded inline."""

    ordered = sorted(out_of_scope)
    body = "\n".join(ordered)
    return f"out-of-scope writes ({len(ordered)} paths):\n{body}"


def _summarize_reason(reason: str) -> str:
    """A SHORT one-line summary of a rejection reason for the next worker prompt.

    The reference-not-embed primary mechanism: the worker sees the failure category
    plus a brief reason (e.g. ``done_when failed: test -f src/theme.ts`` or
    ``out-of-scope writes: a/x.py, ... and 46 more``), then a PATH to the full detail.
    Reasons already lead with their category; we collapse to the first line and
    head-clip to ``_MAX_SUMMARY_CHARS`` so the inline text stays tiny regardless of
    how large the underlying failure was."""

    first_line = reason.strip().splitlines()[0] if reason.strip() else reason
    if len(first_line) <= _MAX_SUMMARY_CHARS:
        return first_line
    return first_line[: _MAX_SUMMARY_CHARS - 1].rstrip() + "…"


def _persist_failure_detail(
    run_dir: RunDir,
    identity: TaskIdentity,
    attempt: int,
    detail: str,
    *,
    scratch: Path,
) -> tuple[Path, Path | None]:
    """Durably record one failed attempt's FULL detail under the run dir.

    Writes ``<run>/<P/E/T>/failures/attempt-<n>.txt`` (the complete, text-safe
    reason, however large) and, when the worker left a ``handoff.json`` in its
    scratch, copies that rejected handoff alongside as ``attempt-<n>.handoff.json``
    BEFORE the worktree is torn down. Returns ``(detail_path, handoff_path | None)``,
    both absolute, so the next worker prompt can REFERENCE them instead of carrying
    the bulk inline. Best-effort: a persist failure never breaks the grind, the
    detail_path is still returned (the worker simply finds no file there).
    """

    fail_dir = run_dir.resolve(f"{identity.fq}/{FAILURES_SUBDIR}")
    detail_path = fail_dir / f"attempt-{attempt}.txt"
    handoff_copy: Path | None = None
    try:
        fail_dir.mkdir(parents=True, exist_ok=True)
        safe = detail.encode("utf-8", errors="replace").decode("utf-8", errors="replace")
        detail_path.write_text(safe, encoding="utf-8")
        rejected = scratch / "handoff.json"
        if rejected.is_file():
            handoff_copy = fail_dir / f"attempt-{attempt}.handoff.json"
            shutil.copyfile(rejected, handoff_copy)
    except OSError:
        pass
    return detail_path.resolve(), handoff_copy.resolve() if handoff_copy else None


def _failure_context_entry(
    run_dir: RunDir,
    identity: TaskIdentity,
    attempt: int,
    failure: "_AttemptFailed",
    *,
    scratch: Path,
) -> str:
    """Build ONE ``<prior_failures>`` entry: a short summary + PATHS, not bulk.

    Persists the full detail (``failure.detail`` when the reason had larger text
    behind it, else the reason itself) to disk, then returns a one-line entry of
    the form ``attempt <n>: <summary> [full detail: <path>; rejected handoff:
    <path>]``. The summary is genuinely short; the full text lives only at the
    referenced path the worker may read. ``_bound_reason`` still wraps the result
    as a backstop, but the entry is already tiny."""

    full = failure.detail if failure.detail is not None else failure.reason
    detail_path, handoff_path = _persist_failure_detail(
        run_dir, identity, attempt, full, scratch=scratch
    )
    summary = _summarize_reason(failure.reason)
    refs = [f"full detail: {detail_path}"]
    if handoff_path is not None:
        refs.append(f"rejected handoff: {handoff_path}")
    entry = f"attempt {attempt}: {summary} [{'; '.join(refs)}]"
    return _bound_reason(entry)


def starting_tier(
    task: Task, tier_names: Sequence[str], *, force_senior: bool = False
) -> int:
    """Ladder index a task STARTS on (before any escalation).

    Routing is PER TASK, by whether the task needs JUDGMENT/TASTE, for ALL modes
    uniformly. A task with ``senior=True`` (taste composition / layout / polish,
    approach synthesis, a design-quality verdict) starts on the ``senior`` tier
    when the ladder has one; every other task (mechanical scaffolding, factual
    research incl. web search, a structural review) starts on tier 0 (the local
    rig). This replaces the old wholesale "research/review always senior" + epoch
    ``visual`` routing: research fact-gathering is local-capable (the local rig has
    web search + fetch), only the judgment slice goes senior.

    ``force_senior`` is the planner's ``handle_failed_epoch`` tier bump / escalate:
    it starts EVERY task of the retried epoch on the senior tier regardless of its
    own flag. A rig with no senior tier falls everything back to local rather than
    crash. Escalation still walks upward from the start tier.
    """

    if force_senior or task.senior:
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


def _strip_orchestration_files(scratch: Path) -> None:
    """Drop the orchestration metadata an implement worker leaves in its scratch
    (handoff.json + check_handoff.py + review.md + ``.pi/settings.json``) so they
    never enter a commit or trip the ownership scope check. The handoff is already
    relocated to the log key, the validator is core-written, the review was gated
    by the done_when re-run, and the ``.pi`` settings only pin the worker's
    subagents, none of it is the task's repo work.
    """

    (scratch / "handoff.json").unlink(missing_ok=True)
    (scratch / CHECK_SCRIPT_NAME).unlink(missing_ok=True)
    (scratch / REVIEW_FILENAME).unlink(missing_ok=True)
    _strip_pi_settings(scratch)


def _commit_partial_for_chain(
    repo: Path, scratch: Path, branch: str, identity: TaskIdentity
) -> None:
    """Commit a CHAINABLE-failed attempt's working-tree work to its branch, so the
    next incremental same-tier retry (based on this branch) inherits it as already-
    present files. Best-effort: a worker writes its edits to the worktree but the
    core only commits on success, so a failed attempt's work lives only in the
    working tree, which the subsequent ``remove_worktree`` would destroy. Committing
    it here persists it on the branch (the chain base) WITHOUT integrating it, the
    final successful attempt is still scope-checked against the EPOCH base, so nothing
    bypasses the gate. Orchestration metadata is stripped first (same as success).
    """

    _strip_orchestration_files(scratch)
    wt.commit_all(scratch, f"grind-wip({identity.fq}): partial attempt kept for retry")


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
    """Re-run one done_when check; return (label, exit). Exit 0 == pass.

    Orthogonal robustness (gate-rebalance G3): a FAILED cmd check is classified
    (the shared ``infra.classify_check_failure``) so an ENVIRONMENTAL fault (exit
    127, a missing tool/dependency) is SURFACED in the label as ``[infra: ...]``,
    distinct from a genuine assertion failure. The worker's failure context then
    tells the planner (and the human reading the journal) that a missing gate tool,
    not the code, broke the check, instead of looping a blind repair on it."""

    if isinstance(check, CmdCheck):
        proc = subprocess.run(
            check.cmd, shell=True, cwd=str(scratch), capture_output=True, text=True
        )
        passed = proc.returncode == check.expect_exit
        if passed:
            return (check.cmd, 0)
        infra = classify_check_failure(
            returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr
        )
        label = f"{check.cmd} [infra: {infra.reason}]" if infra.is_infra else check.cmd
        return (label, 1)
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
    # DoS sanity backstop (the principle's item F): reject (never truncate) an absurdly
    # large handoff before reading it into memory. A valid handoff serializes far under
    # HANDOFF_MAX_BYTES; this generous megabyte-scale guard only ever fires on a
    # pathological/corrupt file (fail-safe -> a failed attempt), never on real content.
    try:
        if src.stat().st_size > HANDOFF_FILE_MAX_BYTES:
            raise _AttemptFailed(
                f"handoff.json is over the {HANDOFF_FILE_MAX_BYTES}-byte DoS guard; "
                f"rejecting (fail-safe), not truncating"
            )
    except OSError as exc:
        raise _AttemptFailed(f"handoff.json unreadable: {exc}") from exc
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


def _load_domain_skills(repo: Path | None, task: Task) -> dict[str, str]:
    """The SELECTED domain skills' text for this task (name -> ``.md`` body).

    Retrieve-not-concatenate: load ONLY the skills the planner named in
    ``task.skills`` from the target repo's ``.grindstone/skills/`` catalogue. Empty
    when the task selected none, or when there is no repo (an artifact task with no
    target repo cannot resolve a catalogue). A named-but-missing skill is a clean
    FAILED attempt (``_AttemptFailed``), not a crash: the gate already rejects a name
    absent from the catalogue index, so this only fires on an index/file mismatch.
    """

    if repo is None or not task.skills:
        return {}
    out: dict[str, str] = {}
    for name in task.skills:
        try:
            out[name] = load_domain_skill(repo, name)
        except (FileNotFoundError, ValueError) as exc:
            raise _AttemptFailed(f"domain skill {name!r} could not be loaded: {exc}") from exc
    return out


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
    epoch_base: str | None = None,
    prior_work_present: bool = False,
) -> Handoff:
    """Run one attempt end-to-end; raise ``_AttemptFailed`` on any failure.

    For implement tasks the scratch is a worktree from ``base`` (the WORKTREE base:
    the epoch base for a first/escalated attempt, the prior attempt's branch for an
    incremental same-tier retry). On the handoff's success the core commits and
    scope-checks the diff. The scope check runs against ``epoch_base`` (defaulting to
    ``base``), NOT the worktree base, so an out-of-scope write committed by ANY
    attempt in the incremental chain is still caught (the chain base would otherwise
    hide an earlier attempt's pollution inside the base). Any failure, transport,
    handoff, or out-of-scope write, maps identically to a failed attempt
    (ARCHITECTURE.md: the loop never inspects worker internals). ``prior_work_present``
    (an incremental same-tier retry: ``base`` is the prior attempt's branch) is passed
    to the worker so its prompt says the prior work is already in the CWD.
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
        domain_skills=_load_domain_skills(repo, task),
        prior_work_present=prior_work_present,
    )
    try:
        worker.run(request)
    except RateLimited:
        # A rate/session limit is NOT a burned attempt: it is the SAME work
        # blocked on a quota window, not a failure of the work. Let it ESCAPE the
        # transport boundary so the ladder driver can park-and-retry the same
        # attempt without charging the attempt/tier ladder (a SessionLimited is a
        # RateLimited subclass, so this catches both; the driver parks the hourly
        # session limit and surfaces any other RateLimited as a failed attempt).
        raise
    except Exception as exc:  # transport boundary, any other raise is a failed attempt
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
        _strip_orchestration_files(scratch)
        wt.commit_all(
            scratch, f"grind({identity.fq}): {task.goal.splitlines()[0][:72]}"
        )
        # Floor core invariant (gate-rebalance): an implement task claiming DONE
        # must actually LAND committed work on its branch, measured against the EPOCH
        # base (not the chained worktree base, which may already carry a prior
        # attempt's commits). An empty epoch_base..HEAD diff means the whole chain
        # changed nothing in the owned files (only metadata, or a no-op) yet handed
        # off DONE, the handoff re-validation + a trivially-true done_when would
        # otherwise pass it while integration merges nothing. Reject it.
        landed_base = epoch_base if epoch_base is not None else base
        assert landed_base is not None
        if not wt.changed_paths(scratch, landed_base):
            raise _AttemptFailed(
                "no committed work: the implement task handed off DONE but left a "
                "zero-diff branch (nothing changed in its file_ownership)"
            )
        # Diff in the WORKTREE (its HEAD is the committed task tip; the main
        # checkout still sits at base).
        # A worker that materializes deps (e.g. `npm install`) populates the
        # declared env_dirs; with no effective .gitignore in the fresh worktree the
        # core force-adds + commits them, so they would read as a wall of
        # out-of-scope writes (dogfood: 1.8M-char rejection over node_modules/.bin/*).
        # They are NOT authored work, so the declared dep dirs are exempt from the
        # scope check; an undeclared write outside ownership is still a violation.
        dep_dirs = list(prepare.env_dirs) if prepare is not None else None
        # Scope-check against the EPOCH base, not the (possibly chained) worktree
        # base: a chained retry's base is the prior attempt's branch, so diffing
        # against it would hide any out-of-scope file an earlier attempt committed.
        scope_base = epoch_base if epoch_base is not None else base
        out_of_scope = wt.scope_violations(
            wt.changed_paths(scratch, scope_base), list(task.file_ownership), dep_dirs
        )
        if out_of_scope:
            raise _AttemptFailed(
                _format_scope_violations(out_of_scope),
                detail=_full_scope_violations(out_of_scope),
                chainable=False,
            )
    return handoff


def _dispatch_session_aware(
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
    implement: bool,
    sleep_fn: SleepFn,
    epoch_base: str | None = None,
    prior_work_present: bool = False,
) -> Handoff:
    """Run one attempt, PARKING (not burning) on a long session limit.

    A ``SessionLimited`` (the claude/codex/opencode-go quota-window limit, which
    resets in HOURS) raised by the worker transport is NOT a failed attempt: it is
    the SAME work blocked on a window, so we sleep ``SESSION_LIMIT_RETRY_S`` via the
    injected ``sleep_fn`` and re-run the SAME attempt, WITHOUT charging the
    attempt/tier ladder, bounded by ``MAX_SESSION_LIMIT_WAITS`` (then it surfaces as
    a failed attempt rather than hanging forever). For an implement task the partial
    worktree must be discarded between tries because ``_dispatch_attempt`` re-adds it.

    Any OTHER ``RateLimited`` (a transient 429 the worker transport surfaced) is
    converted to a normal failed attempt here, EXACTLY preserving the pre-fix
    behavior (the worker path has no short-backoff concept); only the long session
    limit gets the hourly park. Every non-limit failure keeps its existing
    ``_AttemptFailed`` mapping (raised inside ``_dispatch_attempt``)."""

    waits = 0
    while True:
        try:
            return _dispatch_attempt(
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
                epoch_base=epoch_base,
                prior_work_present=prior_work_present,
            )
        except SessionLimited as exc:
            if waits >= MAX_SESSION_LIMIT_WAITS:
                raise _AttemptFailed(
                    f"session limit: {MAX_SESSION_LIMIT_WAITS} hourly waits exhausted: {exc}"
                ) from exc
            waits += 1
            # Discard the partial worktree the failed try created so the next
            # try's add_worktree is clean (no-op for non-implement scratch).
            if implement and repo is not None and branch is not None:
                wt.discard_attempt(repo, scratch, branch)
            sleep_fn(SESSION_LIMIT_RETRY_S)
            continue
        except RateLimited as exc:
            # A transient 429 the worker surfaced: no short-backoff concept on the
            # worker path, so it stays a failed attempt (the prior behavior).
            raise _AttemptFailed(f"transport error: {type(exc).__name__}: {exc}") from exc


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
    force_senior: bool,
    prepare: PrepareConfig | None,
    sleep_fn: SleepFn,
) -> TaskOutcome:
    """Drive the ladder from ``cursor`` to a terminal outcome.

    Emits task_dispatched (once), task_retried (each later attempt),
    task_escalated (each tier climb), and task_done / task_failed via the SHARED
    thread-safe journal (``emit`` assigns the seq under the lock). Rewrites the
    task's cursor through ``sink`` before every attempt and at the terminal.
    """

    implement = isinstance(task, ImplementTask)
    dispatched = first_attempt_already_dispatched
    start_tier = starting_tier(
        task, [name for name, _ in ladder], force_senior=force_senior
    )

    # Incremental-retry chain (implement only). A same-tier, non-escalation RETRY
    # bases its worktree on the PRIOR attempt's branch so the prior attempt's work
    # is PRESENT in the new CWD (the worker may fix it incrementally or reset and
    # redo). ``chain_base`` holds that branch (None -> base the next attempt on the
    # fresh epoch base). The failed branch is kept (worktree torn down, branch left)
    # until the next attempt has been based on it, then it is deleted as superseded;
    # a tier escalation starts clean from the epoch base (a higher tier should not
    # inherit a lower tier's partial work). ``stale_branch`` is the kept failed
    # branch awaiting deletion once it is superseded (or at the terminal).
    chain_base: str | None = None
    stale_branch: str | None = None

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
            # Incremental retry: a same-tier retry (chain_base set) bases on the
            # prior attempt's branch so its work is present; otherwise (first
            # attempt or a freshly-escalated tier) base on the epoch base.
            attempt_base = chain_base if chain_base is not None else base
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
                handoff = _dispatch_session_aware(
                    identity=identity,
                    task=task,
                    mode=mode,
                    run_dir=run_dir,
                    worker=worker,
                    cursor=cursor,
                    repo=repo,
                    base=attempt_base,
                    scratch=scratch,
                    branch=branch,
                    prepare=prepare,
                    implement=implement,
                    sleep_fn=sleep_fn,
                    epoch_base=base,
                    prior_work_present=implement and chain_base is not None,
                )
            except _AttemptFailed as failure:
                # Reference-not-embed: the FULL failure detail is persisted to a
                # file under the run dir and the NEXT attempt's worker prompt
                # (build_worker_prompt's <prior_failures>) carries only a one-line
                # summary + the PATH to that file (which the worker may read). This
                # keeps the prompt tiny no matter how large the failure was (the
                # dogfood 1.8M-char scope list that overflowed the model CLI). Persist
                # BEFORE discarding the worktree so the rejected handoff can be copied
                # out of the scratch first. _bound_reason still backstops the entry.
                cursor.failure_context.append(
                    _failure_context_entry(
                        run_dir, identity, cursor.attempt, failure, scratch=scratch
                    )
                )
                if implement and repo is not None and branch is not None:
                    if failure.chainable:
                        # Incremental retry: COMMIT this attempt's partial work to its
                        # branch (the core only commits on success, so otherwise the
                        # work lives only in the worktree we are about to remove), then
                        # tear down the worktree checkout but KEEP the branch so the next
                        # same-tier retry bases on it and inherits that work. The branch
                        # this attempt was based on (``stale_branch``) is now its
                        # ancestor, fully superseded, so delete it. The kept branch
                        # becomes the new chain base + stale branch (deleted once the
                        # NEXT attempt supersedes it, on escalation, or at the terminal).
                        _commit_partial_for_chain(repo, scratch, branch, identity)
                        wt.remove_worktree(repo, scratch)
                        if stale_branch is not None and stale_branch != branch:
                            wt.delete_branch(repo, stale_branch)
                        chain_base = branch
                        stale_branch = branch
                    else:
                        # A non-chainable rejection (out-of-scope writes): the branch
                        # is poisoned, so discard it entirely and drop the chain, the
                        # next retry restarts from the clean epoch base. Any kept prior
                        # branch is dropped too (the worker is starting over).
                        wt.discard_attempt(repo, scratch, branch)
                        if stale_branch is not None and stale_branch != branch:
                            wt.delete_branch(repo, stale_branch)
                        chain_base = None
                        stale_branch = None
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
            # Success: the winning branch is based on the incremental chain, so any
            # kept prior-attempt branch is now its ancestor and fully superseded.
            # Delete it so a rejected attempt leaves nothing (ruling 4); the winning
            # branch is kept for integration.
            if implement and repo is not None and stale_branch is not None and stale_branch != branch:
                wt.delete_branch(repo, stale_branch)
            stale_branch = None
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

        # Tier exhausted; climb if a higher rung exists. A higher tier starts CLEAN
        # from the epoch base, never inheriting the lower tier's partial work, so the
        # incremental chain is reset and the kept lower-tier branch is deleted.
        if implement and repo is not None and stale_branch is not None:
            wt.delete_branch(repo, stale_branch)
        chain_base = None
        stale_branch = None
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
    force_senior: bool = False,
    prepare: PrepareConfig | None = None,
    epoch_hint: str | None = None,
    sleep_fn: SleepFn = time.sleep,
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
                tier_index=starting_tier(
                    task, [n for n, _ in ladder], force_senior=force_senior
                ),
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
            force_senior=force_senior,
            prepare=prepare,
            sleep_fn=sleep_fn,
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
        force_senior=force_senior,
        prepare=prepare,
        sleep_fn=sleep_fn,
    )
