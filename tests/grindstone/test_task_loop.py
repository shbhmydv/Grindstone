"""Per-task state-machine invariants, exercised through 1-task epochs (the
per-task machine no longer has a standalone entry point, ruling 3). Covers
scripted failures, retry/escalate, the disk contract, grounding, done_when
re-run, status mapping, the ownership scope check, and the in-flight snapshot.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from grindstone.contracts.models import (
    ArtifactEpochArgs,
    ArtifactExistsCheck,
    ArtifactTask,
    CmdCheck,
    ImplementEpochArgs,
    ImplementTask,
    parse_handoff,
)
from grindstone.config import PrepareConfig
from grindstone.epoch_loop import EpochOutcome, EpochState
from grindstone.events import RunStarted, read_events, replay
from grindstone.mock_worker import MockWorker
from grindstone.rundir import RunDir, create_run_dir
from grindstone.worker import WorkerRequest, WorkerTransport

from grindstone.worker import PI_SETTINGS_RELPATH

from tests.grindstone.conftest import (
    OUT_CONTENT,
    OUT_FILE,
    HandoffWorker,
    artifact_epoch,
    git,
    handoff_payload,
    implement_epoch,
    init_git_repo,
    make_ok_worker,
    make_toy_task,
    run_one_epoch,
    tracked_files,
)

FQ = "P1/E1/T1"


def _ladder(*workers: WorkerTransport) -> list[tuple[str, WorkerTransport]]:
    names = ["local", "cloud", "senior"]
    return [(names[i], w) for i, w in enumerate(workers)]


def _run_impl(
    repo: Path,
    run_dir: RunDir,
    task: ImplementTask,
    ladder: list[tuple[str, WorkerTransport]],
    **kw: object,
) -> EpochOutcome:
    return run_one_epoch(
        run_dir,
        args=implement_epoch(task),
        mode="implement",
        ladder=ladder,
        repo=repo,
        **kw,
    )


def _epoch_state(run_dir: RunDir) -> EpochState:
    return EpochState.model_validate_json(run_dir.state_path.read_text())


# --- happy path ----------------------------------------------------------------


def test_first_attempt_success(git_repo: Path, run_dir: RunDir, toy_task: ImplementTask) -> None:
    outcome = _run_impl(git_repo, run_dir, toy_task, _ladder(make_ok_worker()))
    [task] = outcome.tasks
    assert (task.status, task.attempts, task.tier) == ("done", 1, "local")
    record = run_dir.resolve(f"{FQ}/handoff.json")
    assert record.is_file()
    handoff = parse_handoff(json.loads(record.read_text()))
    assert handoff.status == "DONE"
    assert handoff.task_id == FQ
    assert _epoch_state(run_dir).tasks["T1"].status == "done"


class _EnvSpyWorker:
    """An ok worker that records whether its scratch carries the materialized
    env_dir, so the worker-path materialization can be asserted."""

    def __init__(self) -> None:
        self.saw_env_dir = False

    def run(self, request: WorkerRequest) -> None:
        self.saw_env_dir = (request.scratch / "node_modules" / "marker").is_file()
        (request.scratch / "review.md").write_text("reviewed\n", encoding="utf-8")
        (request.scratch / OUT_FILE).write_text(OUT_CONTENT, encoding="utf-8")
        (request.scratch / "handoff.json").write_text(
            json.dumps(handoff_payload(FQ)), encoding="utf-8"
        )


def test_worker_worktree_gets_declared_deps_materialized(tmp_path: Path) -> None:
    """The worker worktree is seeded with the declared (gitignored) deps before
    the agent runs, so it does not burn turns on a fresh install and shares the
    eval gate's cache. node_modules is gitignored so it never enters the commit."""

    repo = init_git_repo(tmp_path / "repo")
    (repo / ".gitignore").write_text(
        ".grindstone/\n__pycache__/\nnode_modules/\n", encoding="utf-8"
    )
    (repo / "package-lock.json").write_text("v1", encoding="utf-8")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "lockfile")
    run = create_run_dir(repo, "run-worker-prep")

    prepare = PrepareConfig(
        cmd="mkdir -p node_modules && echo ok > node_modules/marker",
        env_dirs=["node_modules"],
        cache_key_files=["package-lock.json"],
    )
    spy = _EnvSpyWorker()
    outcome = _run_impl(
        repo, run, make_toy_task(), _ladder(spy), prepare=prepare
    )
    [task] = outcome.tasks
    assert task.status == "done"
    assert spy.saw_env_dir is True  # deps were present when the worker ran
    # node_modules is gitignored -> the commit carries only the owned file.
    assert "node_modules/marker" not in tracked_files(repo, outcome.integration.branch or "HEAD")


def test_journal_replays_into_coherent_tree(
    git_repo: Path, run_dir: RunDir, toy_task: ImplementTask
) -> None:
    _run_impl(git_repo, run_dir, toy_task, _ladder(make_ok_worker()))
    events = read_events(run_dir.events_path)
    assert sum(isinstance(e, RunStarted) for e in events) == 1
    tree = replay(events)
    assert tree.status == "completed"
    node = tree.phases[0].epochs[0].tasks[0]
    assert (node.id, node.status) == ("T1", "done")


# --- scripted retries ----------------------------------------------------------


def test_retry_then_succeed(git_repo: Path, run_dir: RunDir, toy_task: ImplementTask) -> None:
    worker = MockWorker(script=["rate_limit", "bad_json", "ok"], artifacts={OUT_FILE: OUT_CONTENT})
    outcome = _run_impl(git_repo, run_dir, toy_task, _ladder(worker))
    assert outcome.tasks[0].status == "done"
    assert outcome.tasks[0].attempts == 3
    kinds = [e.event for e in read_events(run_dir.events_path)]
    assert kinds.count("task_dispatched") == 1
    assert kinds.count("task_retried") == 2
    assert kinds.count("handoff_rejected") == 2
    assert kinds.count("task_done") == 1


@pytest.mark.parametrize("behavior", ["rate_limit", "bad_json", "empty", "timeout"])
def test_each_failure_behavior_is_a_rejected_attempt(
    git_repo: Path, run_dir: RunDir, toy_task: ImplementTask, behavior: str
) -> None:
    worker = MockWorker(script=[behavior, "ok"], artifacts={OUT_FILE: OUT_CONTENT})
    outcome = _run_impl(git_repo, run_dir, toy_task, _ladder(worker))
    assert outcome.tasks[0].status == "done"
    assert outcome.tasks[0].attempts == 2
    rejected = [e for e in read_events(run_dir.events_path) if e.event == "handoff_rejected"]
    assert len(rejected) == 1


# --- ladder escalation ---------------------------------------------------------


def test_escalates_after_three_tier0_attempts(
    git_repo: Path, run_dir: RunDir, toy_task: ImplementTask
) -> None:
    tier0 = MockWorker(script=["empty", "empty", "empty"])
    tier1 = make_ok_worker()
    outcome = _run_impl(git_repo, run_dir, toy_task, _ladder(tier0, tier1))
    assert outcome.tasks[0].status == "done"
    assert outcome.tasks[0].tier == "cloud"
    assert outcome.tasks[0].attempts == 4  # 3 on tier0 + 1 on tier1
    escalations = [e for e in read_events(run_dir.events_path) if e.event == "task_escalated"]
    assert [e.tier for e in escalations] == ["cloud"]


def test_exhausting_ladder_is_failed(
    git_repo: Path, run_dir: RunDir, toy_task: ImplementTask
) -> None:
    tier0 = MockWorker(script=["empty", "empty", "empty"])
    tier1 = MockWorker(script=["rate_limit"])
    outcome = _run_impl(git_repo, run_dir, toy_task, _ladder(tier0, tier1))
    assert outcome.tasks[0].status == "failed"
    assert outcome.tasks[0].attempts == 4
    assert outcome.tasks[0].failure_reason is not None
    kinds = [e.event for e in read_events(run_dir.events_path)]
    assert kinds.count("task_failed") == 1
    assert kinds.count("task_done") == 0
    assert _epoch_state(run_dir).tasks["T1"].status == "failed"


# --- disk contract: zero dead artifacts ----------------------------------------


def test_unparseable_handoff_is_not_relocated(
    git_repo: Path, run_dir: RunDir, toy_task: ImplementTask
) -> None:
    outcome = _run_impl(git_repo, run_dir, toy_task, _ladder(MockWorker(script=["bad_json"])), tier0_attempts=1)
    assert outcome.tasks[0].status == "failed"
    assert not run_dir.resolve(f"{FQ}/handoff.json").exists()


def test_invalid_relocated_handoff_is_deleted(
    git_repo: Path, run_dir: RunDir, toy_task: ImplementTask
) -> None:
    bad = handoff_payload(task_id="P1/E1/T2")  # parseable JSON, wrong task_id
    outcome = _run_impl(git_repo, run_dir, toy_task, _ladder(HandoffWorker(bad)), tier0_attempts=1)
    assert outcome.tasks[0].status == "failed"
    assert "T2" in (outcome.tasks[0].failure_reason or "")
    assert not run_dir.resolve(f"{FQ}/handoff.json").exists()


class _OversizeHandoffWorker:
    """Writes an absurdly large handoff.json (over the DoS guard) plus the deliverable.

    Models a pathological/corrupt handoff: the size backstop must REJECT it (fail-safe)
    before reading it, while never truncating real content."""

    def run(self, request: WorkerRequest) -> None:
        from grindstone.task_loop import HANDOFF_FILE_MAX_BYTES

        (request.scratch / "review.md").write_text("reviewed\n", encoding="utf-8")
        (request.scratch / OUT_FILE).write_text(OUT_CONTENT, encoding="utf-8")
        # A syntactically valid JSON object whose size blows past the guard.
        blob = "z" * (HANDOFF_FILE_MAX_BYTES + 4096)
        (request.scratch / "handoff.json").write_text(
            json.dumps({"junk": blob}), encoding="utf-8"
        )


def test_oversized_handoff_is_rejected_by_dos_guard(
    git_repo: Path, run_dir: RunDir, toy_task: ImplementTask
) -> None:
    """Item F: a handoff.json above the megabyte-scale DoS guard is REJECTED (fail-safe,
    a failed attempt), not truncated, and never relocated."""

    outcome = _run_impl(
        git_repo, run_dir, toy_task, _ladder(_OversizeHandoffWorker()), tier0_attempts=1
    )
    assert outcome.tasks[0].status == "failed"
    assert "DoS guard" in (outcome.tasks[0].failure_reason or "")
    assert not run_dir.resolve(f"{FQ}/handoff.json").exists()


def test_normal_large_handoff_reads_fine(
    git_repo: Path, run_dir: RunDir, toy_task: ImplementTask
) -> None:
    """The guard only fires on absurd files: a normal handoff (well under the guard) is
    read in full and accepted (no truncation of real content)."""

    outcome = _run_impl(git_repo, run_dir, toy_task, _ladder(make_ok_worker()), tier0_attempts=1)
    assert outcome.tasks[0].status == "done"
    assert run_dir.resolve(f"{FQ}/handoff.json").is_file()


# --- grounding spot-check ------------------------------------------------------


def test_hallucinated_citation_rejected(
    git_repo: Path, run_dir: RunDir, toy_task: ImplementTask
) -> None:
    payload = handoff_payload(citations=[{"file": "nonexistent.py"}])
    worker = HandoffWorker(payload, files={OUT_FILE: OUT_CONTENT})
    outcome = _run_impl(git_repo, run_dir, toy_task, _ladder(worker), tier0_attempts=1)
    assert outcome.tasks[0].status == "failed"
    assert "citation" in (outcome.tasks[0].failure_reason or "")


def test_citation_line_beyond_file_rejected(
    git_repo: Path, run_dir: RunDir, toy_task: ImplementTask
) -> None:
    payload = handoff_payload(citations=[{"file": OUT_FILE, "line": 99}])
    worker = HandoffWorker(payload, files={OUT_FILE: OUT_CONTENT})
    outcome = _run_impl(git_repo, run_dir, toy_task, _ladder(worker), tier0_attempts=1)
    assert outcome.tasks[0].status == "failed"
    assert "line" in (outcome.tasks[0].failure_reason or "")


# --- done_when re-run gate -----------------------------------------------------


def test_done_when_re_run_overrides_worker_claim(
    git_repo: Path, run_dir: RunDir
) -> None:
    # The worker claims "ok" but writes the file to the WRONG path; the structural
    # done_when (the expected file must exist) is RE-RUN on return and fails,
    # overriding the lying claim. (Checks are structural facts only; content
    # acceptance is `criteria`, judged by the agentic pass, not a done_when grep.)
    task = ImplementTask(
        id="T1",
        goal=f"create {OUT_FILE}",
        done_when=[CmdCheck(cmd=f"test -f {OUT_FILE}")],
        file_ownership=[OUT_FILE, "decoy.txt"],
    )
    worker = MockWorker(script=["ok"], artifacts={"decoy.txt": OUT_CONTENT})
    outcome = _run_impl(git_repo, run_dir, task, _ladder(worker), tier0_attempts=1)
    assert outcome.tasks[0].status == "failed"
    assert "done_when" in (outcome.tasks[0].failure_reason or "")


def test_artifact_exists_check_resolves_against_run_dir(
    git_repo: Path, run_dir: RunDir
) -> None:
    key = "shared/data.txt"
    target = run_dir.resolve(key)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("present", encoding="utf-8")
    task = ImplementTask(
        id="T1",
        goal="rely on a pre-existing artifact",
        done_when=[ArtifactExistsCheck(artifact_exists=key)],
        file_ownership=[OUT_FILE],
    )
    worker = MockWorker(script=["ok"], artifacts={OUT_FILE: OUT_CONTENT})
    outcome = _run_impl(git_repo, run_dir, task, _ladder(worker))
    assert outcome.tasks[0].status == "done"


# --- ownership scope check (S2) ------------------------------------------------


def test_out_of_scope_write_is_rejected_and_retried(
    git_repo: Path, run_dir: RunDir, toy_task: ImplementTask
) -> None:
    # Worker satisfies done_when + grounding but ALSO writes outside ownership;
    # the scope check (post-commit) rejects the attempt. Then it behaves.
    bad = HandoffWorker(handoff_payload(), files={OUT_FILE: OUT_CONTENT, "evil.txt": "x\n"})
    good = make_ok_worker()

    class _Switch:
        def __init__(self) -> None:
            self.calls = 0

        def run(self, request: object) -> None:
            self.calls += 1
            (bad if self.calls == 1 else good).run(request)  # type: ignore[arg-type]

    outcome = _run_impl(git_repo, run_dir, toy_task, _ladder(_Switch()))
    assert outcome.tasks[0].status == "done"
    assert outcome.tasks[0].attempts == 2
    rejected = [e for e in read_events(run_dir.events_path) if e.event == "handoff_rejected"]
    assert any("out-of-scope" in e.reason for e in rejected)


# --- floor core invariant: work must actually be committed (gate-rebalance G2) --


def test_zero_diff_done_handoff_is_rejected(git_repo: Path, run_dir: RunDir) -> None:
    """An implement task that hands off DONE but lands a ZERO-DIFF branch (the
    worker wrote nothing to its owned files) is rejected: a DONE claim that
    committed no work is a structural gap the floor catches, the handoff
    re-validation + a trivially-true done_when would otherwise pass it."""

    # done_when is trivially true and grounding cites only base files, so the
    # ONLY thing that can reject this attempt is the committed-diff invariant.
    task = ImplementTask(
        id="T1",
        goal="claim done while writing nothing",
        done_when=[CmdCheck(cmd="true")],
        file_ownership=[OUT_FILE],
    )
    # The worker writes ONLY metadata (review.md + handoff), no owned file -> the
    # commit is zero-diff after the metadata is stripped pre-commit.
    worker = HandoffWorker(
        handoff_payload(citations=[{"file": "README.md"}]), files={}
    )
    outcome = _run_impl(git_repo, run_dir, task, _ladder(worker), tier0_attempts=1)
    assert outcome.tasks[0].status == "failed"
    assert "no committed work" in (outcome.tasks[0].failure_reason or "")


def test_real_committed_work_passes_the_invariant(
    git_repo: Path, run_dir: RunDir, toy_task: ImplementTask
) -> None:
    """The good case: a task that actually writes its owned file lands a
    non-empty diff and passes the committed-diff invariant cleanly."""

    outcome = _run_impl(git_repo, run_dir, toy_task, _ladder(make_ok_worker()))
    assert outcome.tasks[0].status == "done"
    assert OUT_FILE in tracked_files(git_repo, outcome.integration.branch or "HEAD")


# --- status mapping ------------------------------------------------------------


@pytest.mark.parametrize("status", ["FAILED", "PARTIAL"])
def test_non_done_status_is_failed_attempt(
    git_repo: Path, run_dir: RunDir, toy_task: ImplementTask, status: str
) -> None:
    worker = HandoffWorker(handoff_payload(status=status))
    outcome = _run_impl(git_repo, run_dir, toy_task, _ladder(worker), tier0_attempts=1)
    assert outcome.tasks[0].status == "failed"
    assert status in (outcome.tasks[0].failure_reason or "")


# --- research mode citation requirement (artifact epoch, no worktree) ----------


def test_research_handoff_requires_citation(run_dir: RunDir) -> None:
    task = ArtifactTask(
        id="T1",
        goal="write research notes",
        done_when=[CmdCheck(cmd="test -f notes.md")],
        artifact_out="notes.md",
    )
    payload = handoff_payload(citations=[])
    worker = HandoffWorker(payload, files={"notes.md": "findings"})
    outcome = run_one_epoch(
        run_dir,
        args=artifact_epoch(task),
        mode="research",
        ladder=_ladder(worker),
        tier0_attempts=1,
    )
    assert outcome.tasks[0].status == "failed"
    assert "citation" in (outcome.tasks[0].failure_reason or "")
    assert outcome.integration.status == "skipped"


def test_artifact_out_with_subdir_published_from_full_path(
    git_repo: Path, run_dir: RunDir
) -> None:
    """Regression (dogfood-1): an ``artifact_out`` carrying a real subdirectory
    (``MIGRATION/inv.md``) is what a live planner emits when the job names a path.
    The worker prompt + ``done_when`` both reference that FULL path, so an obedient
    worker writes ``<scratch>/MIGRATION/inv.md``, and the relocation must read it
    from the full path. The old basename-only lookup searched ``<scratch>/inv.md``,
    rejected the correctly-produced artifact ("artifact_out not produced in CWD"),
    and forced needless retries until a worker happened to drop the subdir."""

    task = ArtifactTask(
        id="T1",
        goal="produce the inventory report",
        done_when=[CmdCheck(cmd="test -s MIGRATION/inv.md")],
        artifact_out="MIGRATION/inv.md",
    )
    payload = handoff_payload(citations=[{"file": "README.md"}])
    worker = HandoffWorker(payload, files={"MIGRATION/inv.md": "# inventory\n"})
    outcome = run_one_epoch(
        run_dir,
        args=artifact_epoch(task),
        mode="artifact",
        ladder=_ladder(worker),
        repo=git_repo,
        tier0_attempts=1,
    )
    assert outcome.tasks[0].status == "done", outcome.tasks[0].failure_reason
    # Published to the keyed log at the full artifact_out path.
    assert run_dir.resolve("MIGRATION/inv.md").is_file()


# --- mode -> tier routing (research/review start on senior; web search) --------


def _local_senior(
    local: WorkerTransport, senior: WorkerTransport
) -> list[tuple[str, WorkerTransport]]:
    return [("local", local), ("senior", senior)]


def test_research_starts_on_senior_tier(git_repo: Path, run_dir: RunDir) -> None:
    # research routes to senior (web search). BOTH tiers can succeed, so the
    # winning tier reveals where the task STARTED, proving local was skipped.
    task = ArtifactTask(
        id="T1", goal="investigate online",
        done_when=[CmdCheck(cmd="test -f notes.md")],
        artifact_out="notes.md",
    )
    payload = handoff_payload(citations=[{"file": "README.md"}])
    local = HandoffWorker(payload, files={"notes.md": "x"})
    senior = HandoffWorker(payload, files={"notes.md": "x"})
    outcome = run_one_epoch(
        run_dir, args=artifact_epoch(task), mode="research",
        ladder=_local_senior(local, senior), repo=git_repo, tier0_attempts=2,
    )
    assert outcome.tasks[0].status == "done"
    assert outcome.tasks[0].tier == "senior"


def test_review_starts_on_senior_tier(git_repo: Path, run_dir: RunDir) -> None:
    task = ArtifactTask(
        id="T1", goal="review the module",
        done_when=[CmdCheck(cmd="test -f verdict.md")],
        artifact_out="verdict.md",
    )
    payload = handoff_payload(citations=[{"file": "README.md"}])
    local = HandoffWorker(payload, files={"verdict.md": "x"})
    senior = HandoffWorker(payload, files={"verdict.md": "x"})
    outcome = run_one_epoch(
        run_dir, args=artifact_epoch(task), mode="review",
        ladder=_local_senior(local, senior), repo=git_repo, tier0_attempts=2,
    )
    assert outcome.tasks[0].tier == "senior"


def test_artifact_mode_starts_on_local_tier(git_repo: Path, run_dir: RunDir) -> None:
    # artifact is production work (no web needed), starts on local, like implement.
    task = ArtifactTask(
        id="T1", goal="produce findings",
        done_when=[CmdCheck(cmd="test -f findings.md")],
        artifact_out="findings.md",
    )
    payload = handoff_payload(citations=[{"file": "README.md"}])
    local = HandoffWorker(payload, files={"findings.md": "x"})
    senior = HandoffWorker(payload, files={"findings.md": "x"})
    outcome = run_one_epoch(
        run_dir, args=artifact_epoch(task), mode="artifact",
        ladder=_local_senior(local, senior), repo=git_repo, tier0_attempts=2,
    )
    assert outcome.tasks[0].tier == "local"


def test_research_falls_back_to_local_without_senior(
    git_repo: Path, run_dir: RunDir
) -> None:
    # A rig with no senior tier runs research on local (repo investigation only).
    task = ArtifactTask(
        id="T1", goal="investigate",
        done_when=[CmdCheck(cmd="test -f notes.md")],
        artifact_out="notes.md",
    )
    payload = handoff_payload(citations=[{"file": "README.md"}])
    local = HandoffWorker(payload, files={"notes.md": "x"})
    outcome = run_one_epoch(
        run_dir, args=artifact_epoch(task), mode="research",
        ladder=[("local", local)], repo=git_repo, tier0_attempts=2,
    )
    assert outcome.tasks[0].status == "done"
    assert outcome.tasks[0].tier == "local"


# --- taste routing: a VISUAL implement/review epoch builds on the senior tier --


def _visual_implement(task: ImplementTask) -> ImplementEpochArgs:
    return ImplementEpochArgs(
        epoch_title="polish the UI", rationale="visual output", tasks=[task], visual=True
    )


def test_visual_implement_routes_to_senior_tier(
    git_repo: Path, run_dir: RunDir, toy_task: ImplementTask
) -> None:
    # Taste routing: a visual implement epoch is built on the vision-capable
    # senior tier, not the local mode default. BOTH tiers succeed, so the
    # winning tier reveals where the task STARTED, proving local was skipped.
    outcome = run_one_epoch(
        run_dir, args=_visual_implement(toy_task), mode="implement",
        ladder=_local_senior(make_ok_worker(), make_ok_worker()),
        repo=git_repo, tier0_attempts=2,
    )
    assert outcome.tasks[0].status == "done"
    assert outcome.tasks[0].tier == "senior"


def test_non_visual_implement_routes_to_local_tier(
    git_repo: Path, run_dir: RunDir, toy_task: ImplementTask
) -> None:
    # Without the flag, implement keeps its mode default (local).
    outcome = run_one_epoch(
        run_dir, args=implement_epoch(toy_task), mode="implement",
        ladder=_local_senior(make_ok_worker(), make_ok_worker()),
        repo=git_repo, tier0_attempts=2,
    )
    assert outcome.tasks[0].tier == "local"


def test_visual_review_still_routes_to_senior_tier(
    git_repo: Path, run_dir: RunDir
) -> None:
    # review already starts on senior; the visual flag must not break that.
    task = ArtifactTask(
        id="T1", goal="taste-review the UI",
        done_when=[CmdCheck(cmd="test -f verdict.md")],
        artifact_out="P1/E1/T1/verdict.md", targets=["ui/app.tsx"],
    )
    payload = handoff_payload(citations=[{"file": "README.md"}])
    args = ArtifactEpochArgs(
        epoch_title="review the UI", rationale="taste", tasks=[task], visual=True
    )
    outcome = run_one_epoch(
        run_dir, args=args, mode="review",
        ladder=_local_senior(
            HandoffWorker(payload, files={"verdict.md": "x"}),
            HandoffWorker(payload, files={"verdict.md": "x"}),
        ),
        repo=git_repo, tier0_attempts=2,
    )
    assert outcome.tasks[0].tier == "senior"


def test_visual_implement_falls_back_to_local_without_senior(
    git_repo: Path, run_dir: RunDir, toy_task: ImplementTask
) -> None:
    # No senior tier in the ladder: a visual epoch falls back to local without
    # crashing (mirrors research's senior-less fallback).
    outcome = run_one_epoch(
        run_dir, args=_visual_implement(toy_task), mode="implement",
        ladder=[("local", make_ok_worker())], repo=git_repo, tier0_attempts=2,
    )
    assert outcome.tasks[0].status == "done"
    assert outcome.tasks[0].tier == "local"


def test_artifact_out_is_published_to_the_log_key(
    git_repo: Path, run_dir: RunDir
) -> None:
    """Gate-6 P0: the worker's artifact_out file stayed in scratch, nothing
    published it to the run-dir log key (handoff.json IS relocated; the
    artifact was not), so artifact_exists checks were structurally
    unsatisfiable and the planner revised phases until the safety valve."""

    task = ArtifactTask(
        id="T1",
        goal="produce findings",
        done_when=[CmdCheck(cmd="test -f P1/E1/T1/findings.md")],
        artifact_out="P1/E1/T1/findings.md",
    )
    payload = handoff_payload(citations=[{"file": "README.md"}])
    # Faithful to the real prompt: the worker writes the FULL artifact_out path.
    worker = HandoffWorker(payload, files={"P1/E1/T1/findings.md": "three defects"})
    outcome = run_one_epoch(
        run_dir,
        args=artifact_epoch(task),
        mode="artifact",
        ladder=_ladder(worker),
        repo=git_repo,
        tier0_attempts=1,
    )
    assert outcome.tasks[0].status == "done"
    published = run_dir.resolve("P1/E1/T1/findings.md")
    assert published.is_file()
    assert published.read_text(encoding="utf-8") == "three defects"


def test_missing_artifact_out_fails_the_attempt(
    git_repo: Path, run_dir: RunDir
) -> None:
    # Truthful failure: an accepted handoff whose promised artifact_out file
    # was never produced is a failed attempt, and the already-relocated
    # handoff.json is deleted (zero dead artifacts).
    task = ArtifactTask(
        id="T1",
        goal="produce findings",
        done_when=[CmdCheck(cmd="true")],
        artifact_out="P1/E1/T1/findings.md",
    )
    payload = handoff_payload(citations=[{"file": "README.md"}])
    worker = HandoffWorker(payload, files={})  # no findings.md written
    outcome = run_one_epoch(
        run_dir,
        args=artifact_epoch(task),
        mode="artifact",
        ladder=_ladder(worker),
        repo=git_repo,
        tier0_attempts=1,
    )
    assert outcome.tasks[0].status == "failed"
    assert "artifact_out" in (outcome.tasks[0].failure_reason or "")
    assert not run_dir.resolve("P1/E1/T1/handoff.json").exists()


def test_research_citation_of_repo_file_passes(git_repo: Path, run_dir: RunDir) -> None:
    """E2E gate-5 P0: research/review/artifact tasks investigate the TARGET
    REPO, so a repo-root-relative citation must pass grounding. The old gate
    required every citation INSIDE the scratch dir, structurally impossible
    for repo files, so legitimate handoffs were rejected every attempt and
    the planner spun phase revisions unbounded (34 codex calls)."""

    task = ArtifactTask(
        id="T1",
        goal="investigate the README",
        done_when=[CmdCheck(cmd="test -f notes.md")],
        artifact_out="notes.md",
    )
    payload = handoff_payload(citations=[{"file": "README.md", "line": 1}])
    worker = HandoffWorker(payload, files={"notes.md": "findings"})
    outcome = run_one_epoch(
        run_dir,
        args=artifact_epoch(task),
        mode="research",
        ladder=_ladder(worker),
        repo=git_repo,
        tier0_attempts=1,
    )
    assert outcome.tasks[0].status == "done"


def test_research_citation_outside_scratch_and_repo_rejected(
    git_repo: Path, run_dir: RunDir
) -> None:
    # Containment half: a file that EXISTS but lives outside both allowed
    # roots (scratch + target repo) is still a grounding violation.
    task = ArtifactTask(
        id="T1",
        goal="investigate",
        done_when=[CmdCheck(cmd="test -f notes.md")],
        artifact_out="P1/E1/T1/notes.md",
    )
    payload = handoff_payload(citations=[{"file": "/etc/hostname"}])
    worker = HandoffWorker(payload, files={"notes.md": "findings"})
    outcome = run_one_epoch(
        run_dir,
        args=artifact_epoch(task),
        mode="research",
        ladder=_ladder(worker),
        repo=git_repo,
        tier0_attempts=1,
    )
    assert outcome.tasks[0].status == "failed"
    assert "citation" in (outcome.tasks[0].failure_reason or "")


def test_implement_citation_stays_scratch_contained(
    git_repo: Path, run_dir: RunDir, toy_task: ImplementTask
) -> None:
    # Implement scratch IS a repo checkout, the repo root is NOT an extra
    # allowed citation root there; citing the operator checkout absolutely
    # stays rejected.
    payload = handoff_payload(citations=[{"file": str(git_repo / "README.md")}])
    worker = HandoffWorker(payload, files={OUT_FILE: OUT_CONTENT})
    outcome = _run_impl(git_repo, run_dir, toy_task, _ladder(worker), tier0_attempts=1)
    assert outcome.tasks[0].status == "failed"
    assert "citation" in (outcome.tasks[0].failure_reason or "")


# --- worker-facing handoff validator (check_handoff.py injection) --------------


def test_check_handoff_script_injected_and_scope_exempt(
    git_repo: Path, run_dir: RunDir, toy_task: ImplementTask
) -> None:
    # The core writes check_handoff.py into the attempt CWD before the worker
    # runs and appends `python3 check_handoff.py` to the task's runtime
    # done_when (so it reaches the prompt AND the core re-run). The toy task
    # owns only out.txt, so if the script were committed the scope check would
    # reject it, a clean DONE in one attempt proves it is exempt (dropped
    # pre-commit exactly like handoff.json).
    seen: dict[str, object] = {}

    class _Probe:
        def run(self, request: object) -> None:
            req = request  # type: ignore[assignment]
            seen["present"] = (req.scratch / "check_handoff.py").is_file()  # type: ignore[attr-defined]
            seen["done_when"] = [
                c.cmd
                for c in req.task.done_when  # type: ignore[attr-defined]
                if isinstance(c, CmdCheck)
            ]
            make_ok_worker().run(request)  # type: ignore[arg-type]

    outcome = _run_impl(git_repo, run_dir, toy_task, _ladder(_Probe()))
    assert outcome.tasks[0].status == "done"
    assert outcome.tasks[0].attempts == 1
    assert seen["present"] is True
    assert "python3 check_handoff.py" in seen["done_when"]


# --- implement-mode review gate (test -s review.md injection) -------------------


def test_review_gate_appended_for_implement_and_scope_exempt(
    git_repo: Path, run_dir: RunDir, toy_task: ImplementTask
) -> None:
    # Implement attempts get `test -s review.md` appended to the runtime
    # done_when (the verified checked-review mechanism: a review demanded as a
    # gated artifact fires 6/6, prose instructions 0/9). The toy task owns only
    # out.txt, so a clean DONE proves review.md is dropped pre-commit exactly
    # like handoff.json and check_handoff.py.
    seen: dict[str, object] = {}

    class _Probe:
        def run(self, request: object) -> None:
            seen["done_when"] = [
                c.cmd
                for c in request.task.done_when  # type: ignore[attr-defined]
                if isinstance(c, CmdCheck)
            ]
            make_ok_worker().run(request)  # type: ignore[arg-type]

    outcome = _run_impl(git_repo, run_dir, toy_task, _ladder(_Probe()))
    assert outcome.tasks[0].status == "done"
    assert outcome.tasks[0].attempts == 1
    # The clean one-attempt DONE above is the scope-exemption proof: the toy
    # task owns only out.txt, so a committed review.md would have been rejected
    # as an out-of-scope write.
    assert "test -s review.md" in seen["done_when"]


def test_review_gate_not_appended_for_artifact_modes(run_dir: RunDir) -> None:
    task = ArtifactTask(
        id="T1",
        goal="write research notes",
        done_when=[CmdCheck(cmd="test -f notes.md")],
        artifact_out="notes.md",
    )
    seen: dict[str, object] = {}

    class _Probe:
        def run(self, request: object) -> None:
            seen["done_when"] = [
                c.cmd
                for c in request.task.done_when  # type: ignore[attr-defined]
                if isinstance(c, CmdCheck)
            ]
            make_ok_worker(out_file="notes.md").run(request)  # type: ignore[arg-type]

    outcome = run_one_epoch(
        run_dir, args=artifact_epoch(task), mode="research", ladder=_ladder(_Probe())
    )
    assert outcome.tasks[0].status == "done"
    assert "test -s review.md" not in seen["done_when"]


# --- per-cwd .pi/settings.json (subagent model pin) is orchestration metadata --


def test_pi_settings_dropped_pre_commit_and_scope_exempt(
    git_repo: Path, run_dir: RunDir, toy_task: ImplementTask
) -> None:
    # The worker script (local_request.sh) drops a per-cwd .pi/settings.json pinning spawned
    # subagents to the parent model. Like handoff.json/check_handoff.py/review.md
    # it is orchestration metadata: the core strips it (and the now-empty .pi/
    # dir) before commit, so it never enters the diff nor trips the ownership
    # scope check. The toy task owns only out.txt, so a clean one-attempt DONE
    # plus a committed tree free of .pi/ proves both.
    class _DropsPiSettings:
        def run(self, request: object) -> None:
            settings = request.scratch / PI_SETTINGS_RELPATH  # type: ignore[attr-defined]
            settings.parent.mkdir(parents=True, exist_ok=True)
            settings.write_text('{"subagents": {}}', encoding="utf-8")
            make_ok_worker().run(request)  # type: ignore[arg-type]

    outcome = run_one_epoch(
        run_dir,
        args=implement_epoch(toy_task),
        mode="implement",
        ladder=[("local", _DropsPiSettings())],
        repo=git_repo,
    )
    assert outcome.tasks[0].status == "done"
    assert outcome.tasks[0].attempts == 1
    branch = outcome.integration.branch
    assert branch is not None
    committed = tracked_files(git_repo, branch)
    assert not any(p.startswith(".pi/") or p == ".pi" for p in committed)


def test_pi_dir_kept_when_worker_leaves_other_content(
    git_repo: Path, run_dir: RunDir, toy_task: ImplementTask
) -> None:
    # Cleanup is surgical: remove settings.json and the .pi/ dir ONLY if empty.
    # If the worker left other content under .pi/, that content stays, the
    # ownership scope check then (correctly) rejects the attempt as out-of-scope
    # rather than the orchestrator silently rm -r'ing worker output.
    class _LeavesExtraPiContent:
        def run(self, request: object) -> None:
            settings = request.scratch / PI_SETTINGS_RELPATH  # type: ignore[attr-defined]
            settings.parent.mkdir(parents=True, exist_ok=True)
            settings.write_text('{"subagents": {}}', encoding="utf-8")
            (settings.parent / "stray.txt").write_text("x\n", encoding="utf-8")
            make_ok_worker().run(request)  # type: ignore[arg-type]

    outcome = run_one_epoch(
        run_dir,
        args=implement_epoch(toy_task),
        mode="implement",
        ladder=[("local", _LeavesExtraPiContent())],
        repo=git_repo,
        tier0_attempts=1,
    )
    assert outcome.tasks[0].status == "failed"
    assert "out-of-scope" in (outcome.tasks[0].failure_reason or "")


def test_missing_review_is_caught_by_core_re_run(
    git_repo: Path, run_dir: RunDir, toy_task: ImplementTask
) -> None:
    # A worker that does the work but skips the review (deletes review.md after
    # the mock wrote it) must be rejected by the CORE re-run, the gate is
    # authoritative, not advisory.
    class _SkipsReview:
        def run(self, request: object) -> None:
            make_ok_worker().run(request)  # type: ignore[arg-type]
            (request.scratch / "review.md").unlink()  # type: ignore[attr-defined]

    outcome = _run_impl(
        git_repo, run_dir, toy_task, _ladder(_SkipsReview()), tier0_attempts=1
    )
    assert outcome.tasks[0].status == "failed"
    reason = outcome.tasks[0].failure_reason or ""
    assert "done_when" in reason and "review.md" in reason


def test_injected_check_is_part_of_core_re_run(
    git_repo: Path, run_dir: RunDir, toy_task: ImplementTask
) -> None:
    # If a worker hands back a valid handoff but its CWD no longer satisfies
    # `python3 check_handoff.py`, the CORE re-run of done_when must catch it.
    # Proves the synthetic check is genuinely appended to the authoritative
    # re-run, not merely shown to the worker.
    class _DeletesScript:
        def run(self, request: object) -> None:
            make_ok_worker().run(request)  # type: ignore[arg-type]
            (request.scratch / "check_handoff.py").unlink()  # type: ignore[attr-defined]

    outcome = _run_impl(
        git_repo, run_dir, toy_task, _ladder(_DeletesScript()), tier0_attempts=1
    )
    assert outcome.tasks[0].status == "failed"
    reason = outcome.tasks[0].failure_reason or ""
    assert "done_when" in reason and "check_handoff.py" in reason


# --- in-flight snapshot (resume substrate) -------------------------------------


def test_running_state_snapshot_during_attempt(
    git_repo: Path, run_dir: RunDir, toy_task: ImplementTask
) -> None:
    seen: dict[str, object] = {}

    class _Peeking:
        def run(self, request: object) -> None:
            st = _epoch_state(run_dir).tasks["T1"]
            seen["status"] = st.status
            seen["attempt"] = st.attempt
            seen["scratch"] = st.scratch
            make_ok_worker().run(request)  # type: ignore[arg-type]

    outcome = _run_impl(git_repo, run_dir, toy_task, _ladder(_Peeking()))
    assert outcome.tasks[0].status == "done"
    assert seen["status"] == "running"
    assert seen["attempt"] == 1
    assert seen["scratch"] is not None


# --- PART C: bounded rejection reasons (no list/reason can balloon a prompt) ---
# RCA: a worker materialized node_modules, the scope check rejected every path,
# producing a ~1.8M-char reason. That reason was replayed into the next attempt's
# worker prompt and, passed as an argv string by the request scripts, overflowed
# the kernel argv limit so the CLI never launched. The reason and the failure
# context fed into the prompt must both be bounded.

from grindstone.task_loop import (  # noqa: E402
    _MAX_NAMED_SCOPE_VIOLATIONS,
    _MAX_FAILURE_CONTEXT_BYTES,
    _bound_reason,
    _format_scope_violations,
)


def test_format_scope_violations_caps_named_paths() -> None:
    paths = [f"node_modules/.bin/tool-{i:04d}" for i in range(5000)]
    reason = _format_scope_violations(paths)
    # At most N paths are named, the rest summed as "... and K more".
    named = reason.split("out-of-scope writes: ", 1)[1].split(", ... and ", 1)[0]
    assert len(named.split(", ")) == _MAX_NAMED_SCOPE_VIOLATIONS
    assert f"... and {5000 - _MAX_NAMED_SCOPE_VIOLATIONS} more" in reason
    # The whole reason is small (kilobytes), never megabytes.
    assert len(reason) < 4096


def test_format_scope_violations_small_list_has_no_more_suffix() -> None:
    reason = _format_scope_violations(["b/z.py", "a/x.py"])
    # Sorted, fully named, no truncation marker.
    assert reason == "out-of-scope writes: a/x.py, b/z.py"
    assert "more" not in reason


def test_bound_reason_truncates_oversized_with_marker() -> None:
    huge = "x" * (2 * _MAX_FAILURE_CONTEXT_BYTES)
    bounded = _bound_reason(huge)
    assert len(bounded.encode("utf-8")) <= _MAX_FAILURE_CONTEXT_BYTES + 64
    assert "reason truncated" in bounded
    assert str(len(huge.encode("utf-8"))) in bounded


def test_bound_reason_passes_through_small_text() -> None:
    small = "out-of-scope writes: a/x.py"
    assert _bound_reason(small) == small


# --- PART D: reference-not-embed prior-failure feedback ------------------------
# The worker's prior-failure feedback is a SHORT summary + PATHS to the full detail
# on disk, never the embedded bulk. The full detail must EXIST at the referenced
# path; the inline summary stays tiny regardless of how large the failure was.

from grindstone.task_loop import (  # noqa: E402
    _MAX_SUMMARY_CHARS,
    _full_scope_violations,
    _summarize_reason,
)


def test_summarize_reason_keeps_first_line_short() -> None:
    reason = "done_when failed: test -f src/design-system/theme.ts"
    assert _summarize_reason(reason) == reason


def test_summarize_reason_clips_oversized_to_one_short_line() -> None:
    huge = "out-of-scope writes: " + ", ".join(f"node_modules/p{i}" for i in range(500))
    summary = _summarize_reason(huge)
    assert len(summary) <= _MAX_SUMMARY_CHARS
    assert "\n" not in summary
    assert summary.startswith("out-of-scope writes:")


def test_full_scope_violations_lists_every_path() -> None:
    paths = [f"node_modules/.bin/tool-{i:04d}" for i in range(5000)]
    full = _full_scope_violations(paths)
    assert "5000 paths" in full
    assert full.count("\n") == 5000  # header line + one line per path


def test_scope_rejection_feedback_is_summary_plus_path_not_bulk(
    git_repo: Path, run_dir: RunDir, toy_task: ImplementTask
) -> None:
    """A worker that writes a HUGE out-of-scope set is rejected; the NEXT attempt's
    failure_context carries a short summary + a PATH, the full path list lives at
    that path on disk, and the inline entry stays tiny (never the bulk)."""

    evil = {f"junk/file-{i:04d}.txt": "x\n" for i in range(300)}
    bad = HandoffWorker(handoff_payload(), files={OUT_FILE: OUT_CONTENT, **evil})

    seen: list[list[str]] = []

    class _Probe:
        """Second attempt: record the failure_context it received, then succeed."""

        def run(self, request: WorkerRequest) -> None:
            seen.append(list(request.failure_context))
            make_ok_worker().run(request)

    class _Switch:
        def __init__(self) -> None:
            self.calls = 0

        def run(self, request: WorkerRequest) -> None:
            self.calls += 1
            (bad if self.calls == 1 else _Probe()).run(request)

    outcome = _run_impl(git_repo, run_dir, toy_task, _ladder(_Switch()))
    assert outcome.tasks[0].status == "done"
    assert outcome.tasks[0].attempts == 2

    # The second attempt saw exactly one prior-failure entry.
    assert len(seen) == 1 and len(seen[0]) == 1
    entry = seen[0][0]
    # SHORT: the inline entry is a summary + path, never the 300-path bulk.
    assert len(entry) < 2048
    assert "out-of-scope writes" in entry
    assert "full detail:" in entry
    # The 300 junk paths are NOT all embedded inline.
    assert entry.count("junk/file-") < 300
    # The referenced detail PATH exists on disk and contains the FULL list.
    path_str = entry.split("full detail: ", 1)[1].split("]", 1)[0].split(";", 1)[0].strip()
    detail = Path(path_str)
    assert detail.is_file()
    body = detail.read_text(encoding="utf-8")
    assert "300 paths" in body
    assert body.count("junk/file-") == 300


def test_done_when_failure_feedback_references_persisted_detail(
    git_repo: Path, run_dir: RunDir
) -> None:
    """A done_when rejection records a summary + path; the rejected handoff.json is
    copied alongside the detail and referenced, and the worker still learns what
    failed (the failure category survives in the summary)."""

    task = ImplementTask(
        id="T1",
        goal="create out.txt but fail the gate",
        done_when=[CmdCheck(cmd="test -f never_made.txt")],
        file_ownership=[OUT_FILE, "never_made.txt"],
    )
    # Worker writes the handoff + a real owned file (so it is not a zero-diff/empty
    # attempt) but never makes never_made.txt, so done_when fails.
    bad = HandoffWorker(
        handoff_payload(out_file=OUT_FILE), files={OUT_FILE: OUT_CONTENT}
    )

    seen: list[list[str]] = []

    class _Probe:
        def run(self, request: WorkerRequest) -> None:
            seen.append(list(request.failure_context))
            # second attempt also fails the same way -> drives to FAILED quickly
            bad.run(request)

    class _Switch:
        def __init__(self) -> None:
            self.calls = 0

        def run(self, request: WorkerRequest) -> None:
            self.calls += 1
            (bad if self.calls == 1 else _Probe()).run(request)

    _run_impl(git_repo, run_dir, task, _ladder(_Switch()), tier0_attempts=2)

    assert seen and seen[0]
    entry = seen[0][0]
    assert "done_when failed" in entry  # worker still learns WHAT failed
    assert "full detail:" in entry
    assert "rejected handoff:" in entry  # the handoff was copied + referenced
    handoff_str = entry.split("rejected handoff: ", 1)[1].split("]", 1)[0].strip()
    assert Path(handoff_str).is_file()  # the rejected handoff exists on disk
