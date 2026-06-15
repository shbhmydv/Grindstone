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
from grindstone.epoch_loop import EpochOutcome, EpochState
from grindstone.events import RunStarted, read_events, replay
from grindstone.mock_worker import MockWorker
from grindstone.rundir import RunDir
from grindstone.worker import WorkerTransport

from grindstone.worker import PI_SETTINGS_RELPATH

from tests.grindstone.conftest import (
    OUT_CONTENT,
    OUT_FILE,
    HandoffWorker,
    artifact_epoch,
    handoff_payload,
    implement_epoch,
    make_ok_worker,
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
    git_repo: Path, run_dir: RunDir, toy_task: ImplementTask
) -> None:
    worker = MockWorker(script=["ok"], artifacts={OUT_FILE: "WRONG\n"})
    outcome = _run_impl(git_repo, run_dir, toy_task, _ladder(worker), tier0_attempts=1)
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
