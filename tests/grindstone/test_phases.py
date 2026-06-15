"""S4 phase machinery: live exit-criteria evaluation, multi-phase advancement,
per-phase epoch-index reset, budgets + phase escalation, escalation position
legality, last-phase no-auto-complete, and cumulative-state tail surfacing.

All deterministic: scripted mock planner + the stateless OwnershipWorker; no
wall clock, no randomness (that lives only in the fuzz test).
"""

from __future__ import annotations

from pathlib import Path

from grindstone.events import read_events
from grindstone.mock_planner import MockPlanner
from grindstone.planner import PlannerTransport
from grindstone.rundir import RunDir
from grindstone.run_loop import RunState, run_grind

from tests.grindstone.conftest import (
    OwnershipWorker,
    check_cmd,
    complete_decision,
    impl_task,
    implement_decision,
    phase_dict,
    revise_decision,
    skeleton_decision,
    two_phase_skeleton,
)


def _ladder() -> list[tuple[str, OwnershipWorker]]:
    return [("local", OwnershipWorker())]


def _run_state(run_dir: RunDir) -> RunState:
    return RunState.model_validate_json(run_dir.run_state_path.read_text())


class _Recording:
    """Capture every prompt handed to the wrapped planner."""

    def __init__(self, inner: PlannerTransport) -> None:
        self.inner = inner
        self.prompts: list[str] = []

    def plan(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.inner.plan(prompt)


# --- criteria-driven advancement + per-phase epoch-index reset (rulings 1) -----


def test_phases_advance_on_criteria_and_reset_epoch_index(
    git_repo: Path, run_dir: RunDir
) -> None:
    g1 = [check_cmd("test -f f1.txt")]
    g2 = [check_cmd("test -f f2.txt")]
    planner = MockPlanner(
        script=[
            skeleton_decision(
                phase_dict("P1", title="build", exit_criterion=g1),
                phase_dict("P2", title="verify", exit_criterion=g2),
            ),
            implement_decision(impl_task("T1", "f1.txt")),
            implement_decision(impl_task("T1", "f2.txt")),
            complete_decision(check_cmd("test -f f1.txt"), check_cmd("test -f f2.txt")),
        ]
    )
    outcome = run_grind(
        run_dir, job_path="job.md", planner=planner, ladder=_ladder(), repo=git_repo
    )
    assert outcome.status == "completed"
    events = read_events(run_dir.events_path)
    assert [e.phase_id for e in events if e.event == "phase_passed"] == ["P1", "P2"]
    assert [e.phase_id for e in events if e.event == "phase_started"] == ["P1", "P2"]
    # Each phase's epoch counter resets to E1 (E-id is per-phase, ruling 1).
    epochs = [(e.phase_id, e.epoch_id) for e in events if e.event == "epoch_started"]
    assert epochs == [("P1", "E1"), ("P2", "E1")]


def test_phase_passes_only_when_criterion_met(git_repo: Path, run_dir: RunDir) -> None:
    # P1 gates on a file the single epoch DOES create -> it passes; the run then
    # advances to the trivially-true P2 in the same preamble pass (multi-advance).
    planner = MockPlanner(
        script=[
            skeleton_decision(
                phase_dict("P1", exit_criterion=[check_cmd("test -f f1.txt")]),
                phase_dict("P2", exit_criterion=[check_cmd("true")]),
            ),
            implement_decision(impl_task("T1", "f1.txt")),
            complete_decision(check_cmd("test -f f1.txt")),
        ]
    )
    outcome = run_grind(
        run_dir, job_path="job.md", planner=planner, ladder=_ladder(), repo=git_repo
    )
    assert outcome.status == "completed"
    # After the single implement epoch, P1 passes and P2 (true) passes too.
    assert _run_state(run_dir).passed_phase_ids == ["P1", "P2"]


# --- last phase passing does NOT auto-complete (ruling 1) ----------------------


def test_last_phase_pass_does_not_auto_complete(git_repo: Path, run_dir: RunDir) -> None:
    # Both phases' criteria are already satisfied at run start, so BOTH pass in
    # the first preamble, yet the run only completes once the planner emits
    # complete_run (the planner stays in the loop; evidence is the certificate).
    planner = MockPlanner(
        script=[two_phase_skeleton(), complete_decision(check_cmd("true"))]
    )
    outcome = run_grind(
        run_dir, job_path="job.md", planner=planner, ladder=_ladder(), repo=git_repo
    )
    assert outcome.status == "completed"
    events = read_events(run_dir.events_path)
    assert [e.phase_id for e in events if e.event == "phase_passed"] == ["P1", "P2"]
    # skeleton + complete only: no epoch, no auto-complete on the last pass.
    assert outcome.planner_calls == 2
    assert outcome.epochs_run == 0


# --- budgets + phase escalation + revise recovery (ruling 2) -------------------


def test_budget_exhaustion_escalates_then_revise_recovers(
    git_repo: Path, run_dir: RunDir
) -> None:
    goal = [check_cmd("test -f goal.txt")]
    rec = _Recording(
        MockPlanner(
            script=[
                skeleton_decision(
                    phase_dict("P1", exit_criterion=goal, budget=1),
                    phase_dict("P2", exit_criterion=goal, budget=1),
                ),
                # Creates other.txt, P1's goal.txt criterion never passes; the
                # one-epoch budget is then spent -> phase escalation.
                implement_decision(impl_task("T1", "other.txt")),
                # Re-scope P1 onto what actually exists; budget + flag reset.
                revise_decision(
                    phase_dict("P1", title="rescoped", exit_criterion=[check_cmd("test -f other.txt")]),
                    phase_dict("P2"),
                ),
                complete_decision(check_cmd("test -f other.txt")),
            ]
        )
    )
    outcome = run_grind(
        run_dir, job_path="job.md", planner=rec, ladder=_ladder(), repo=git_repo
    )
    assert outcome.status == "completed"
    events = read_events(run_dir.events_path)
    escalated = [e.phase_id for e in events if e.event == "phase_escalated"]
    assert escalated == ["P1"]  # fired once on budget exhaustion
    assert "phases_revised" in [e.event for e in events]
    # The revise call's prompt carried the escalation demand (call index 2).
    assert "<escalation>" in rec.prompts[2]
    # Recovery cleared the flag; the run completed.
    assert _run_state(run_dir).phase_escalation_active is False


def test_escalation_demand_rejects_work_epoch(git_repo: Path, run_dir: RunDir) -> None:
    goal = [check_cmd("test -f goal.txt")]
    planner = MockPlanner(
        script=[
            skeleton_decision(
                phase_dict("P1", exit_criterion=goal, budget=1),
                phase_dict("P2", exit_criterion=goal),
            ),
            implement_decision(impl_task("T1", "other.txt")),  # exhausts P1's budget
            implement_decision(impl_task("T2", "more.txt")),  # illegal under escalation
            implement_decision(impl_task("T2", "more.txt")),  # re-ask 1
            implement_decision(impl_task("T2", "more.txt")),  # re-ask 2 -> escalate run
        ]
    )
    outcome = run_grind(
        run_dir, job_path="job.md", planner=planner, ladder=_ladder(), repo=git_repo
    )
    assert outcome.status == "escalated"
    assert outcome.reason is not None and "phase escalation in force" in outcome.reason
    # Three rejected work epochs, each journaled planner_call_failed(transient).
    transient = [
        e for e in read_events(run_dir.events_path)
        if e.event == "planner_call_failed" and getattr(e, "classification", "") == "transient"
    ]
    assert len(transient) == 3


# --- cumulative-state tail surfacing (ruling 3) --------------------------------


def test_tail_surfaces_phase_status_and_integration_tip(
    git_repo: Path, run_dir: RunDir
) -> None:
    rec = _Recording(
        MockPlanner(
            script=[
                skeleton_decision(
                    phase_dict("P1", exit_criterion=[check_cmd("test -f f1.txt")]),
                    phase_dict("P2", exit_criterion=[check_cmd("test -f f2.txt")]),
                ),
                implement_decision(impl_task("T1", "f1.txt")),
                complete_decision(check_cmd("test -f f1.txt")),
            ]
        )
    )
    run_grind(run_dir, job_path="job.md", planner=rec, ladder=_ladder(), repo=git_repo)

    # The first call (skeleton) carries no phase status (no skeleton yet).
    assert "<phase_status>" not in rec.prompts[0]
    # Second call: P1 active, its criterion freshly evaluated as FAIL.
    p1 = rec.prompts[1]
    assert "<phase_status>" in p1
    assert "current_phase: P1" in p1
    assert "[FAIL] cmd `test -f f1.txt`" in p1
    assert "<integration_tip" in p1
    # Third call: P1 passed + advanced to P2; the tip listing references f1.txt
    # (a name, never its body).
    p2 = rec.prompts[2]
    assert "current_phase: P2" in p2
    assert "passed_phases: P1" in p2
    tip = p2.split("<integration_tip", 1)[1].split("</integration_tip>", 1)[0]
    assert "f1.txt" in tip
