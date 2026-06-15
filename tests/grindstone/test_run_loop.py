"""The multi-epoch run loop (S3): scripted planner runs end-to-end.

Skeleton → epochs → complete; revise_phases; escalate_run; complete_run with
failing then passing evidence; the invalid-decision re-ask ladder; rate-limit
backoff (recorded sleeps, no wall clock); transient retries + exhaustion; hard
failure; epoch chaining; stable-head byte-identity across a run; safety valves.
"""

from __future__ import annotations

from pathlib import Path

import subprocess

from grindstone.contracts.models import ArtifactExistsCheck, CmdCheck
from grindstone.events import read_events, replay
from grindstone.mock_planner import MockPlanner
from grindstone.planner import PlannerTransport
from grindstone.rundir import RunDir, create_run_dir
from grindstone.run_loop import RunState, evaluate_checks, run_grind

from tests.grindstone.conftest import (
    OwnershipWorker,
    check_cmd,
    complete_decision,
    escalate_decision,
    impl_task,
    implement_decision,
    phase_dict,
    revise_decision,
    skeleton_decision,
    tracked_files,
    two_phase_skeleton,
)


def _ladder() -> list[tuple[str, OwnershipWorker]]:
    return [("local", OwnershipWorker())]


def test_evaluate_checks_artifact_bare_filename(tmp_path: Path) -> None:
    """Gate-6 RCA: a skeleton-time exit criterion can only name the FILE, the
    P*/E*/T*/ placement is chosen epochs later by the producing task. A bare
    filename passes iff exactly ONE logged artifact carries that name; exact
    keys keep working; ambiguity and absence stay False."""

    run = create_run_dir(tmp_path, "run-1")
    target = run.resolve("P2/E3/T1/findings.md")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("x", encoding="utf-8")
    results = evaluate_checks(
        [
            ArtifactExistsCheck(artifact_exists="findings.md"),
            ArtifactExistsCheck(artifact_exists="P2/E3/T1/findings.md"),
            ArtifactExistsCheck(artifact_exists="ghost.md"),
        ],
        repo=None,
        ref=None,
        run_dir=run,
        scratch_name="eval",
    )
    assert [ok for _, ok in results] == [True, True, False]


def test_evaluate_checks_fresh_repo_unborn_head_fails_not_crashes(tmp_path: Path) -> None:
    """A repo with ZERO commits (unborn HEAD) has no tip to check out. A cmd
    exit-criterion must FAIL deterministically, not let GitError escape and crash
    the whole run (the most likely first-contact failure on a fresh project)."""

    repo = tmp_path / "fresh"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)  # no commits → unborn HEAD
    run = create_run_dir(tmp_path, "run-fresh")
    results = evaluate_checks(
        [CmdCheck(cmd="true")], repo=repo, ref=None, run_dir=run, scratch_name="eval"
    )
    assert len(results) == 1
    label, ok = results[0]
    assert ok is False and "unresolvable" in label


def _run_state(run_dir: RunDir) -> RunState:
    return RunState.model_validate_json(run_dir.run_state_path.read_text())


class _Recording:
    """Wrap a planner, capturing every prompt it is handed."""

    def __init__(self, inner: PlannerTransport) -> None:
        self.inner = inner
        self.prompts: list[str] = []

    def plan(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.inner.plan(prompt)


# --- S5 repo-memory seam: frozen at run start, fed to every planner call ------


def test_repo_memory_frozen_into_state_and_planner_input(
    git_repo: Path, run_dir: RunDir
) -> None:
    digest = git_repo / ".grindstone" / "memory" / "digest.md"
    digest.parent.mkdir(parents=True)
    digest.write_text("repo fact: ship small epochs", encoding="utf-8")
    planner = _Recording(
        MockPlanner(script=[two_phase_skeleton(), complete_decision(check_cmd("true"))])
    )
    outcome = run_grind(
        run_dir, job_path="job.md", planner=planner, ladder=_ladder(), repo=git_repo
    )
    assert outcome.status == "completed"
    # Frozen into durable run state...
    assert _run_state(run_dir).repo_memory == "repo fact: ship small epochs"
    # ...and rendered into the <repo_memory> slot of every constructed call.
    assert planner.prompts
    for prompt in planner.prompts:
        assert "<repo_memory>\nrepo fact: ship small epochs\n</repo_memory>" in prompt


def test_no_repo_memory_leaves_slot_empty(git_repo: Path, run_dir: RunDir) -> None:
    planner = _Recording(
        MockPlanner(script=[two_phase_skeleton(), complete_decision(check_cmd("true"))])
    )
    run_grind(run_dir, job_path="job.md", planner=planner, ladder=_ladder(), repo=git_repo)
    assert _run_state(run_dir).repo_memory is None
    assert all("<repo_memory>\n</repo_memory>" in p for p in planner.prompts)


# --- happy path: skeleton -> 2 implement epochs -> complete --------------------


def test_two_epoch_run_completes_and_chains(git_repo: Path, run_dir: RunDir) -> None:
    planner = MockPlanner(
        script=[
            two_phase_skeleton(),
            implement_decision(impl_task("T1", "f1.txt")),
            implement_decision(impl_task("T1", "f2.txt")),
            complete_decision(check_cmd("test -f f1.txt"), check_cmd("test -f f2.txt")),
        ]
    )
    outcome = run_grind(
        run_dir, job_path="job.md", planner=planner, ladder=_ladder(), repo=git_repo
    )
    assert outcome.status == "completed"
    assert outcome.epochs_run == 2
    assert outcome.planner_calls == 4
    # Epoch chaining: the final branch carries BOTH files (E2 base = E1 tip).
    assert outcome.final_branch is not None
    files = set(tracked_files(git_repo, outcome.final_branch))
    assert {"f1.txt", "f2.txt"} <= files
    # Journal replays coherently and the run-state is terminal.
    tree = replay(read_events(run_dir.events_path))
    assert tree.status == "completed"
    assert tree.planner_calls == 4
    assert _run_state(run_dir).status == "completed"


def test_planner_calls_match_journal(git_repo: Path, run_dir: RunDir) -> None:
    planner = MockPlanner(
        script=[
            two_phase_skeleton(),
            implement_decision(impl_task("T1", "f1.txt")),
            complete_decision(check_cmd("test -f f1.txt")),
        ]
    )
    outcome = run_grind(run_dir, job_path="job.md", planner=planner, ladder=_ladder(), repo=git_repo)
    started = sum(1 for e in read_events(run_dir.events_path) if e.event == "planner_call_started")
    assert outcome.planner_calls == started == 3


# --- revise_phases -------------------------------------------------------------


def test_revise_phases_replaces_skeleton_then_proceeds(git_repo: Path, run_dir: RunDir) -> None:
    # The original P1 exit criterion is NOT yet satisfied (f1.txt absent), so the
    # phase is un-passed and revise_phases is legal (S4: revise may not touch a
    # passed phase). The revised P1 still gates on f1.txt; P2/P3 pass trivially.
    pending = [check_cmd("test -f f1.txt")]
    planner = MockPlanner(
        script=[
            skeleton_decision(
                phase_dict("P1", title="build", exit_criterion=pending),
                phase_dict("P2", exit_criterion=pending),
            ),
            revise_decision(
                phase_dict("P1", title="rebuilt", exit_criterion=pending),
                phase_dict("P2"),
                phase_dict("P3"),
            ),
            implement_decision(impl_task("T1", "f1.txt")),
            complete_decision(check_cmd("test -f f1.txt")),
        ]
    )
    outcome = run_grind(run_dir, job_path="job.md", planner=planner, ladder=_ladder(), repo=git_repo)
    assert outcome.status == "completed"
    kinds = [e.event for e in read_events(run_dir.events_path)]
    assert "phases_revised" in kinds
    state = _run_state(run_dir)
    assert state.skeleton is not None and [p.id for p in state.skeleton] == ["P1", "P2", "P3"]
    assert state.skeleton[0].title == "rebuilt"


# --- escalate_run --------------------------------------------------------------


def test_escalate_run_is_terminal(git_repo: Path, run_dir: RunDir) -> None:
    planner = MockPlanner(script=[two_phase_skeleton(), escalate_decision("spec is ambiguous")])
    outcome = run_grind(run_dir, job_path="job.md", planner=planner, ladder=_ladder(), repo=git_repo)
    assert outcome.status == "escalated"
    assert outcome.reason is not None and "ambiguous" in outcome.reason
    kinds = [e.event for e in read_events(run_dir.events_path)]
    assert "run_escalated" in kinds and "run_completed" not in kinds


# --- complete_run evidence: fail -> re-ask -> pass ------------------------------


def test_complete_run_failing_evidence_triggers_reask(git_repo: Path, run_dir: RunDir) -> None:
    planner = MockPlanner(
        script=[
            two_phase_skeleton(),
            implement_decision(impl_task("T1", "f1.txt")),
            complete_decision(check_cmd("test -f ghost.txt")),  # evidence fails
            complete_decision(check_cmd("test -f f1.txt")),  # evidence passes
        ]
    )
    outcome = run_grind(run_dir, job_path="job.md", planner=planner, ladder=_ladder(), repo=git_repo)
    assert outcome.status == "completed"
    assert outcome.planner_calls == 4
    # The rejected complete still VALIDATED, so two planner_call_succeeded(complete_run).
    completes = [
        e for e in read_events(run_dir.events_path)
        if e.event == "planner_call_succeeded" and getattr(e, "tool", "") == "complete_run"
    ]
    assert len(completes) == 2


# --- invalid-decision re-ask ladder -> escalate --------------------------------


def test_invalid_decisions_reask_then_escalate(git_repo: Path, run_dir: RunDir) -> None:
    planner = MockPlanner(script=[two_phase_skeleton(), "invalid", "invalid", "invalid"])
    outcome = run_grind(run_dir, job_path="job.md", planner=planner, ladder=_ladder(), repo=git_repo)
    assert outcome.status == "escalated"
    assert outcome.reason is not None and "2 re-asks" in outcome.reason
    # Three invalid attempts, each journaled planner_call_failed(transient).
    transient_fails = [
        e for e in read_events(run_dir.events_path)
        if e.event == "planner_call_failed" and getattr(e, "classification", "") == "transient"
    ]
    assert len(transient_fails) == 3


# --- rate-limit backoff (injected sleep, recorded) -----------------------------


def test_rate_limit_backoff_records_injected_sleeps(git_repo: Path, run_dir: RunDir) -> None:
    recorded: list[float] = []
    planner = MockPlanner(
        script=[
            two_phase_skeleton(),
            "rate_limit",
            "rate_limit",
            implement_decision(impl_task("T1", "f1.txt")),
            complete_decision(check_cmd("test -f f1.txt")),
        ]
    )
    outcome = run_grind(
        run_dir, job_path="job.md", planner=planner, ladder=_ladder(), repo=git_repo,
        sleep_fn=recorded.append,
    )
    assert outcome.status == "completed"
    assert recorded == [30.0, 60.0]  # exponential, no wall clock
    assert _run_state(run_dir).rate_limit_waits == 2


def test_rate_limit_exhaustion_escalates(git_repo: Path, run_dir: RunDir) -> None:
    recorded: list[float] = []
    planner = MockPlanner(script=[two_phase_skeleton()] + ["rate_limit"] * 7)
    outcome = run_grind(
        run_dir, job_path="job.md", planner=planner, ladder=_ladder(), repo=git_repo,
        sleep_fn=recorded.append,
    )
    assert outcome.status == "escalated"
    assert recorded == [30.0, 60.0, 120.0, 240.0, 480.0, 600.0]  # 6 waits then stop
    assert outcome.reason is not None and "rate limit" in outcome.reason


# --- transient retries ---------------------------------------------------------


def test_transient_retries_then_succeeds(git_repo: Path, run_dir: RunDir) -> None:
    planner = MockPlanner(
        script=[
            two_phase_skeleton(),
            "transient",
            "transient",
            implement_decision(impl_task("T1", "f1.txt")),
            complete_decision(check_cmd("test -f f1.txt")),
        ]
    )
    outcome = run_grind(run_dir, job_path="job.md", planner=planner, ladder=_ladder(), repo=git_repo)
    assert outcome.status == "completed"


def test_transient_exhaustion_escalates(git_repo: Path, run_dir: RunDir) -> None:
    planner = MockPlanner(script=[two_phase_skeleton(), "transient", "transient", "transient"])
    outcome = run_grind(run_dir, job_path="job.md", planner=planner, ladder=_ladder(), repo=git_repo)
    assert outcome.status == "escalated"
    assert outcome.reason is not None and "transient" in outcome.reason


def test_hard_failure_escalates(git_repo: Path, run_dir: RunDir) -> None:
    planner = MockPlanner(script=[two_phase_skeleton(), "hard"])
    outcome = run_grind(run_dir, job_path="job.md", planner=planner, ladder=_ladder(), repo=git_repo)
    assert outcome.status == "escalated"
    assert outcome.reason is not None and "hard" in outcome.reason
    fails = [
        e for e in read_events(run_dir.events_path)
        if e.event == "planner_call_failed" and getattr(e, "classification", "") == "hard"
    ]
    assert len(fails) == 1


# --- stable-head byte-identity across the run ----------------------------------


def test_stable_head_identical_across_calls_in_a_run(git_repo: Path, run_dir: RunDir) -> None:
    rec = _Recording(
        MockPlanner(
            script=[
                two_phase_skeleton(),
                implement_decision(impl_task("T1", "f1.txt")),
                implement_decision(impl_task("T1", "f2.txt")),
                complete_decision(check_cmd("test -f f1.txt")),
            ]
        )
    )
    run_grind(run_dir, job_path="job.md", planner=rec, ladder=_ladder(), repo=git_repo)
    heads = [p.split("<state>", 1)[0] for p in rec.prompts]
    # Call 0 has no skeleton; calls 1..3 share one byte-identical head.
    assert heads[0] != heads[1]
    assert heads[1] == heads[2] == heads[3]


def test_prose_wrapped_planner_output_is_extracted(git_repo: Path, run_dir: RunDir) -> None:
    # Real codex wraps the decision in reasoning/fences; the loop's extractor
    # must survive it end-to-end (not just in extractor unit fixtures).
    planner = MockPlanner(
        script=[
            two_phase_skeleton(),
            implement_decision(impl_task("T1", "f1.txt")),
            complete_decision(check_cmd("test -f f1.txt")),
        ],
        wrap="prose",
    )
    outcome = run_grind(run_dir, job_path="job.md", planner=planner, ladder=_ladder(), repo=git_repo)
    assert outcome.status == "completed"


# --- safety valves (TEST-only bounds) ------------------------------------------


def test_planner_call_valve_stops_run(git_repo: Path, run_dir: RunDir) -> None:
    planner = MockPlanner(
        script=[two_phase_skeleton(), implement_decision(impl_task("T1", "f1.txt"))]
    )
    outcome = run_grind(
        run_dir, job_path="job.md", planner=planner, ladder=_ladder(), repo=git_repo,
        max_planner_calls=1,
    )
    assert outcome.status == "failed"
    assert outcome.reason is not None and "safety valve" in outcome.reason
    assert _run_state(run_dir).status == "failed"


def test_epoch_valve_stops_run(git_repo: Path, run_dir: RunDir) -> None:
    planner = MockPlanner(
        script=[
            two_phase_skeleton(),
            implement_decision(impl_task("T1", "f1.txt")),
            implement_decision(impl_task("T1", "f2.txt")),
        ]
    )
    outcome = run_grind(
        run_dir, job_path="job.md", planner=planner, ladder=_ladder(), repo=git_repo,
        max_epochs=1,
    )
    assert outcome.status == "failed"
    assert outcome.epochs_run == 1
    assert outcome.reason is not None and "epochs reached" in outcome.reason
