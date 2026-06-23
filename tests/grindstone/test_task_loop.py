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
from grindstone.worker import SessionLimited, WorkerRequest, WorkerTransport

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
    names = ["worker", "cloud", "senior"]
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
    assert (task.status, task.attempts, task.tier) == ("done", 1, "worker")
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


# --- session-limit hourly park (not a burned attempt) --------------------------


def test_session_limit_parks_then_retries_same_attempt(
    git_repo: Path, run_dir: RunDir, toy_task: ImplementTask
) -> None:
    # A long session limit on the worker path must PARK (sleep 3600s) and retry
    # the SAME attempt, NOT burn the attempt/tier ladder: the worker hits a session
    # limit twice, then succeeds, and the task is done on attempt 1 with zero
    # rejected attempts (the park is not a handoff_rejected).
    recorded: list[float] = []
    worker = MockWorker(
        script=["session_limit", "session_limit", "ok"],
        artifacts={OUT_FILE: OUT_CONTENT},
    )
    outcome = _run_impl(
        git_repo, run_dir, toy_task, _ladder(worker), sleep_fn=recorded.append
    )
    assert outcome.tasks[0].status == "done"
    assert outcome.tasks[0].attempts == 1  # the ladder was NOT charged
    assert recorded == [3600.0, 3600.0]  # two hourly parks, no wall clock
    kinds = [e.event for e in read_events(run_dir.events_path)]
    assert kinds.count("handoff_rejected") == 0  # a park is never a rejected attempt
    assert kinds.count("task_retried") == 0
    assert kinds.count("task_done") == 1


def test_non_limit_failure_path_is_unchanged(
    git_repo: Path, run_dir: RunDir, toy_task: ImplementTask
) -> None:
    # Regression guard: a NON-session failure (a transient 429 the worker surfaces
    # as RateLimited, or a bad handoff) keeps its existing burned-attempt behavior,
    # the park only catches the long session limit.
    worker = MockWorker(script=["rate_limit", "ok"], artifacts={OUT_FILE: OUT_CONTENT})
    outcome = _run_impl(git_repo, run_dir, toy_task, _ladder(worker))
    assert outcome.tasks[0].status == "done"
    assert outcome.tasks[0].attempts == 2  # the 429 burned one attempt (unchanged)
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


# --- starting_tier: pure per-task routing unit ---------------------------------


def test_starting_tier_routes_on_per_task_senior_flag() -> None:
    from grindstone.task_loop import starting_tier

    tiers = ["worker", "senior"]

    def impl(senior: bool) -> ImplementTask:
        return ImplementTask(
            id="T1", goal="g", done_when=[CmdCheck(cmd="true")],
            file_ownership=["f.txt"], senior=senior,
        )

    def art(senior: bool) -> ArtifactTask:
        return ArtifactTask(
            id="T1", goal="g", done_when=[CmdCheck(cmd="true")],
            artifact_out="o.md", senior=senior,
        )

    # senior:true -> senior (index 1) for every shape; senior:false -> local (0).
    for make in (impl, art):
        assert starting_tier(make(True), tiers) == 1
        assert starting_tier(make(False), tiers) == 0
    # research/review tasks are NOT wholesale senior anymore: a plain artifact task
    # (the shape research/review dispatch) defaults local.
    assert starting_tier(art(False), tiers) == 0
    # force_senior overrides a local flag (the handle_failed_epoch tier bump).
    assert starting_tier(impl(False), tiers, force_senior=True) == 1
    # No senior tier in the ladder -> everything falls back to local (no crash).
    assert starting_tier(impl(True), ["worker"]) == 0
    assert starting_tier(art(True), ["worker"], force_senior=True) == 0


# --- per-task -> tier routing (senior flag, uniform across all modes) ----------


def _local_senior(
    local: WorkerTransport, senior: WorkerTransport
) -> list[tuple[str, WorkerTransport]]:
    return [("worker", local), ("senior", senior)]


def _artifact_task(tid: str, out: str, *, senior: bool = False) -> ArtifactTask:
    return ArtifactTask(
        id=tid, goal="do the thing",
        done_when=[CmdCheck(cmd=f"test -f {out}")],
        artifact_out=out, senior=senior,
    )


def test_research_defaults_to_local_tier(git_repo: Path, run_dir: RunDir) -> None:
    # NEW routing: research is no longer wholesale-senior. A plain research task
    # (fact gathering, web search, which the local rig CAN do) starts on local.
    task = _artifact_task("T1", "notes.md")
    payload = handoff_payload(citations=[{"file": "README.md"}])
    outcome = run_one_epoch(
        run_dir, args=artifact_epoch(task), mode="research",
        ladder=_local_senior(
            HandoffWorker(payload, files={"notes.md": "x"}),
            HandoffWorker(payload, files={"notes.md": "x"}),
        ),
        repo=git_repo, tier0_attempts=2,
    )
    assert outcome.tasks[0].status == "done"
    assert outcome.tasks[0].tier == "worker"


def test_review_defaults_to_local_tier(git_repo: Path, run_dir: RunDir) -> None:
    # NEW routing: a structural/mechanical review defaults local too.
    task = _artifact_task("T1", "verdict.md")
    payload = handoff_payload(citations=[{"file": "README.md"}])
    outcome = run_one_epoch(
        run_dir, args=artifact_epoch(task), mode="review",
        ladder=_local_senior(
            HandoffWorker(payload, files={"verdict.md": "x"}),
            HandoffWorker(payload, files={"verdict.md": "x"}),
        ),
        repo=git_repo, tier0_attempts=2,
    )
    assert outcome.tasks[0].tier == "worker"


@pytest.mark.parametrize("mode", ["research", "review", "artifact"])
def test_senior_task_routes_to_senior_tier_across_modes(
    git_repo: Path, run_dir: RunDir, mode: str
) -> None:
    # A senior:true task (judgment/taste/synthesis) starts on the senior tier for
    # EVERY non-write mode. BOTH tiers succeed, so the winning tier reveals where
    # the task STARTED, proving local was skipped.
    task = _artifact_task("T1", "out.md", senior=True)
    payload = handoff_payload(citations=[{"file": "README.md"}])
    outcome = run_one_epoch(
        run_dir, args=artifact_epoch(task), mode=mode,
        ladder=_local_senior(
            HandoffWorker(payload, files={"out.md": "x"}),
            HandoffWorker(payload, files={"out.md": "x"}),
        ),
        repo=git_repo, tier0_attempts=2,
    )
    assert outcome.tasks[0].status == "done"
    assert outcome.tasks[0].tier == "senior"


def test_senior_implement_task_routes_to_senior_tier(
    git_repo: Path, run_dir: RunDir
) -> None:
    # A senior:true implement task (taste/layout/polish) builds on the senior tier.
    task = ImplementTask(
        id="T1", goal="create out.txt containing GRINDSTONE",
        done_when=[CmdCheck(cmd="test -f out.txt")],
        file_ownership=["out.txt"], senior=True,
    )
    outcome = run_one_epoch(
        run_dir, args=implement_epoch(task), mode="implement",
        ladder=_local_senior(make_ok_worker(), make_ok_worker()),
        repo=git_repo, tier0_attempts=2,
    )
    assert outcome.tasks[0].status == "done"
    assert outcome.tasks[0].tier == "senior"


def test_non_senior_implement_task_routes_to_local_tier(
    git_repo: Path, run_dir: RunDir, toy_task: ImplementTask
) -> None:
    # Without the flag, an implement task runs local (the default).
    outcome = run_one_epoch(
        run_dir, args=implement_epoch(toy_task), mode="implement",
        ladder=_local_senior(make_ok_worker(), make_ok_worker()),
        repo=git_repo, tier0_attempts=2,
    )
    assert outcome.tasks[0].tier == "worker"


def test_senior_task_falls_back_to_local_without_senior_tier(
    git_repo: Path, run_dir: RunDir
) -> None:
    # No senior tier in the ladder: a senior:true task falls back to local without
    # crashing.
    task = _artifact_task("T1", "notes.md", senior=True)
    payload = handoff_payload(citations=[{"file": "README.md"}])
    outcome = run_one_epoch(
        run_dir, args=artifact_epoch(task), mode="research",
        ladder=[("worker", HandoffWorker(payload, files={"notes.md": "x"}))],
        repo=git_repo, tier0_attempts=2,
    )
    assert outcome.tasks[0].status == "done"
    assert outcome.tasks[0].tier == "worker"


def test_force_senior_overrides_local_task_flag(
    git_repo: Path, run_dir: RunDir, toy_task: ImplementTask
) -> None:
    # The handle_failed_epoch tier bump: force_senior starts EVERY task on senior,
    # even a task whose own senior flag is False.
    from grindstone.epoch_loop import run_epoch
    from grindstone.events import JournalWriter

    with JournalWriter(run_dir.events_path) as journal:
        from tests.grindstone.conftest import _emit_run_frame, _close_run_frame

        args = implement_epoch(toy_task)
        _emit_run_frame(journal, args)
        outcome = run_epoch(
            run_dir, journal=journal, args=args, mode="implement",
            ladder=_local_senior(make_ok_worker(), make_ok_worker()),
            repo=git_repo, tier0_attempts=2, force_senior=True,
        )
        _close_run_frame(journal, outcome)
    assert outcome.tasks[0].tier == "senior"


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
    # The worker script (worker_request.sh) drops a per-cwd .pi/settings.json pinning spawned
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
        ladder=[("worker", _DropsPiSettings())],
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
        ladder=[("worker", _LeavesExtraPiContent())],
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


# --- domain skills: task_loop loads + delivers the selected skills --------------


class _SkillSpyWorker:
    """An ok worker that records the domain_skills delivered on its request."""

    def __init__(self) -> None:
        self.seen: dict[str, str] = {}

    def run(self, request: WorkerRequest) -> None:
        self.seen = dict(request.domain_skills)
        (request.scratch / "review.md").write_text("reviewed\n", encoding="utf-8")
        (request.scratch / OUT_FILE).write_text(OUT_CONTENT, encoding="utf-8")
        (request.scratch / "handoff.json").write_text(
            json.dumps(handoff_payload(FQ)), encoding="utf-8"
        )


def _catalogue(repo: Path, *, index: str, skills: dict[str, str]) -> None:
    skills_dir = repo / ".grindstone" / "skills"
    skills_dir.mkdir(parents=True, exist_ok=True)
    (skills_dir / "index.md").write_text(index, encoding="utf-8")
    for name, body in skills.items():
        (skills_dir / f"{name}.md").write_text(body, encoding="utf-8")


def test_task_loop_delivers_only_selected_domain_skills(tmp_path: Path) -> None:
    repo = init_git_repo(tmp_path / "repo")
    _catalogue(
        repo,
        index="- rn-nav: navigation\n- rn-a11y: accessibility\n",
        skills={"rn-nav": "NAV BODY", "rn-a11y": "A11Y BODY"},
    )
    run = create_run_dir(repo, "run-skills")
    task = make_toy_task().model_copy(update={"skills": ["rn-nav"]})
    spy = _SkillSpyWorker()
    outcome = _run_impl(repo, run, task, _ladder(spy))
    [t] = outcome.tasks
    assert t.status == "done"
    # ONLY the selected skill rode the request (retrieve, not concatenate).
    assert spy.seen == {"rn-nav": "NAV BODY"}


def test_task_loop_no_catalogue_is_noop(tmp_path: Path) -> None:
    repo = init_git_repo(tmp_path / "repo")  # no .grindstone/skills/
    run = create_run_dir(repo, "run-no-skills")
    spy = _SkillSpyWorker()
    outcome = _run_impl(repo, run, make_toy_task(), _ladder(spy))
    [t] = outcome.tasks
    assert t.status == "done"
    assert spy.seen == {}


def test_task_loop_missing_skill_file_fails_attempt(tmp_path: Path) -> None:
    # The index advertises rn-nav but ships no rn-nav.md: the loader raises, which
    # the dispatch maps to a clean failed attempt (never a crash).
    repo = init_git_repo(tmp_path / "repo")
    _catalogue(repo, index="- rn-nav: navigation\n", skills={})
    run = create_run_dir(repo, "run-missing-skill")
    task = make_toy_task().model_copy(update={"skills": ["rn-nav"]})
    outcome = _run_impl(repo, run, task, _ladder(make_ok_worker()))
    [t] = outcome.tasks
    assert t.status == "failed"
    assert t.failure_reason is not None and "domain skill" in t.failure_reason


# --- incremental retry: a same-tier retry keeps the prior attempt's work -------


class _RecordingRetryWorker:
    """A stateful worker for the incremental-retry tests.

    Attempt 1: write a PARTIAL (in-scope) version of the owned file + a
    chainable-FAILED handoff (status FAILED is a chainable rejection: it does NOT
    poison the branch the way an out-of-scope write does, so the next same-tier
    retry bases on attempt 1's branch and inherits its COMMITTED partial work).
    Later attempts: record whether ``prior_work_present`` was set and whether
    attempt 1's partial content is on disk in the CWD, then finish the owned file
    + a DONE handoff."""

    def __init__(self) -> None:
        self.calls = 0
        self.prior_flags: list[bool] = []
        self.saw_partial_content: list[bool] = []

    def run(self, request: WorkerRequest) -> None:
        self.calls += 1
        (request.scratch / "review.md").write_text("reviewed\n", encoding="utf-8")
        if self.calls == 1:
            # A partial, in-scope edit of the owned file (kept on the chain branch).
            (request.scratch / OUT_FILE).write_text("WIP\n", encoding="utf-8")
            payload = handoff_payload(FQ, status="FAILED", citations=[{"file": OUT_FILE}])
            (request.scratch / "handoff.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )
            return
        # Retry: observe the inherited state, then finish cleanly.
        self.prior_flags.append(request.prior_work_present)
        prior = request.scratch / OUT_FILE
        self.saw_partial_content.append(
            prior.is_file() and prior.read_text() == "WIP\n"
        )
        prior.write_text(OUT_CONTENT, encoding="utf-8")
        (request.scratch / "handoff.json").write_text(
            json.dumps(handoff_payload(FQ, citations=[{"file": OUT_FILE}])),
            encoding="utf-8",
        )


def test_same_tier_retry_bases_on_prior_attempt_work(
    git_repo: Path, run_dir: RunDir, toy_task: ImplementTask
) -> None:
    # A chainable rejection (status FAILED) commits attempt 1's partial work to its
    # branch + keeps it; attempt 2 is based on it, so attempt 1's WIP out.txt is
    # present in attempt 2's CWD and the worker is told prior_work_present.
    w = _RecordingRetryWorker()
    outcome = _run_impl(git_repo, run_dir, toy_task, _ladder(w))
    assert outcome.tasks[0].status == "done"
    assert outcome.tasks[0].attempts == 2
    assert w.prior_flags == [True]
    assert w.saw_partial_content == [True]


def test_first_attempt_starts_fresh_from_base(
    git_repo: Path, run_dir: RunDir, toy_task: ImplementTask
) -> None:
    # The FIRST attempt is never told prior work is present (there is none yet):
    # a one-shot success carries prior_work_present False.
    class _Probe:
        def __init__(self) -> None:
            self.flags: list[bool] = []

        def run(self, request: WorkerRequest) -> None:
            self.flags.append(request.prior_work_present)
            make_ok_worker().run(request)

    probe = _Probe()
    outcome = _run_impl(git_repo, run_dir, toy_task, _ladder(probe))
    assert outcome.tasks[0].status == "done"
    assert probe.flags == [False]


def test_tier_escalation_starts_clean_from_base(
    git_repo: Path, run_dir: RunDir, toy_task: ImplementTask
) -> None:
    # A higher tier must NOT inherit the lower tier's partial work: the escalated
    # attempt starts fresh from the epoch base, so prior_work_present is False and
    # the lower tier's partial.txt is absent.
    class _SeniorProbe:
        def __init__(self) -> None:
            self.flags: list[bool] = []
            self.saw_partial: list[bool] = []

        def run(self, request: WorkerRequest) -> None:
            self.flags.append(request.prior_work_present)
            self.saw_partial.append((request.scratch / "partial.txt").is_file())
            make_ok_worker().run(request)

    # tier0 always fails (chainable FAILED status, writing partial.txt each time);
    # exhausting it escalates to the senior probe.
    class _AlwaysPartial:
        def run(self, request: WorkerRequest) -> None:
            (request.scratch / "review.md").write_text("r\n", encoding="utf-8")
            (request.scratch / "partial.txt").write_text("wip\n", encoding="utf-8")
            (request.scratch / OUT_FILE).write_text(OUT_CONTENT, encoding="utf-8")
            (request.scratch / "handoff.json").write_text(
                json.dumps(handoff_payload(FQ, status="FAILED",
                                           citations=[{"file": OUT_FILE}])),
                encoding="utf-8",
            )

    senior = _SeniorProbe()
    outcome = _run_impl(
        git_repo, run_dir, toy_task, _ladder(_AlwaysPartial(), senior),
        tier0_attempts=2,
    )
    assert outcome.tasks[0].status == "done"
    assert outcome.tasks[0].tier == "cloud"
    # The senior's first (and only) attempt started clean: no inherited partial.
    assert senior.flags == [False]
    assert senior.saw_partial == [False]


def test_out_of_scope_retry_does_not_inherit_poisoned_branch(
    git_repo: Path, run_dir: RunDir, toy_task: ImplementTask
) -> None:
    # An out-of-scope rejection is NON-chainable: the next attempt restarts from the
    # clean epoch base (the poisoned evil.txt is NOT inherited) and prior_work_present
    # is False.
    bad = HandoffWorker(handoff_payload(), files={OUT_FILE: OUT_CONTENT, "evil.txt": "x\n"})

    class _Probe:
        def __init__(self) -> None:
            self.flags: list[bool] = []
            self.saw_evil: list[bool] = []

        def run(self, request: WorkerRequest) -> None:
            self.flags.append(request.prior_work_present)
            self.saw_evil.append((request.scratch / "evil.txt").is_file())
            make_ok_worker().run(request)

    probe = _Probe()

    class _Switch:
        def __init__(self) -> None:
            self.calls = 0

        def run(self, request: WorkerRequest) -> None:
            self.calls += 1
            (bad if self.calls == 1 else probe).run(request)

    outcome = _run_impl(git_repo, run_dir, toy_task, _ladder(_Switch()))
    assert outcome.tasks[0].status == "done"
    assert probe.flags == [False]
    assert probe.saw_evil == [False]


def test_session_limit_park_still_works_with_incremental_retry(
    git_repo: Path, run_dir: RunDir, toy_task: ImplementTask
) -> None:
    # The session-limit park (re-run the SAME attempt) must still work alongside the
    # incremental-retry chain: a session limit on attempt 1 parks + retries the same
    # attempt, which then succeeds in ONE charged attempt.
    waits: list[float] = []

    class _LimitOnce:
        def __init__(self) -> None:
            self.calls = 0

        def run(self, request: WorkerRequest) -> None:
            self.calls += 1
            if self.calls == 1:
                raise SessionLimited("quota window")
            make_ok_worker().run(request)

    outcome = _run_impl(
        git_repo, run_dir, toy_task, _ladder(_LimitOnce()),
        sleep_fn=waits.append,
    )
    assert outcome.tasks[0].status == "done"
    assert outcome.tasks[0].attempts == 1  # the park did not charge the ladder
    assert len(waits) == 1
