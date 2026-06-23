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
    phase_complete_decision,
    phase_dict,
    revise_decision,
    skeleton_decision,
    two_phase_skeleton,
)


def _ladder() -> list[tuple[str, OwnershipWorker]]:
    return [("worker", OwnershipWorker())]


def _run_state(run_dir: RunDir) -> RunState:
    return RunState.model_validate_json(run_dir.run_state_path.read_text())


class _Recording:
    """Capture every prompt handed to the wrapped planner."""

    def __init__(self, inner: PlannerTransport) -> None:
        self.inner = inner
        self.prompts: list[str] = []

    def plan(self, prompt: str, *, workdir: Path | None = None) -> str:
        self.prompts.append(prompt)
        return self.inner.plan(prompt, workdir=workdir)


# --- criteria-driven advancement + per-phase epoch-index reset (rulings 1) -----


def test_phases_advance_on_phase_complete_and_reset_epoch_index(
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
            phase_complete_decision("f1.txt"),  # planner ends P1 (deliverable exists)
            implement_decision(impl_task("T1", "f2.txt")),
            phase_complete_decision("f2.txt"),  # planner ends P2
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


def test_green_floor_does_not_auto_pass_phase_on_entry(
    git_repo: Path, run_dir: RunDir
) -> None:
    """The P3 regression: a phase whose build-health floor is ALREADY green on
    entry must NOT auto-pass with zero epochs. The planner is still consulted and
    plans an epoch; the phase ends only when the planner emits phase_complete."""

    # P1's floor (`true`) is green from the very first preamble, yet the phase must
    # not skip: the scripted planner gets a real boundary, plans an epoch, and only
    # then completes the phase by citing the deliverable it built.
    planner = MockPlanner(
        script=[
            skeleton_decision(
                phase_dict("P1", exit_criterion=[check_cmd("true")]),
                phase_dict("P2", exit_criterion=[check_cmd("true")]),
            ),
            implement_decision(impl_task("T1", "f1.txt")),  # P1 is NOT skipped
            phase_complete_decision("f1.txt"),
            implement_decision(impl_task("T1", "f2.txt")),
            phase_complete_decision("f2.txt"),
            complete_decision(check_cmd("true")),
        ]
    )
    outcome = run_grind(
        run_dir, job_path="job.md", planner=planner, ladder=_ladder(), repo=git_repo
    )
    assert outcome.status == "completed"
    events = read_events(run_dir.events_path)
    # Both phases actually RAN an epoch (zero-epoch auto-pass is impossible now).
    epochs = [(e.phase_id, e.epoch_id) for e in events if e.event == "epoch_started"]
    assert epochs == [("P1", "E1"), ("P2", "E1")]
    assert [e.phase_id for e in events if e.event == "phase_passed"] == ["P1", "P2"]


# --- last phase passing does NOT auto-complete (ruling 1) ----------------------


def test_last_phase_pass_does_not_auto_complete(git_repo: Path, run_dir: RunDir) -> None:
    # Even with both phases' floors already green at run start, NOTHING passes on
    # entry: the planner must phase_complete each, and the run completes only on its
    # explicit complete_run (the planner stays in the loop; evidence is the cert).
    planner = MockPlanner(
        script=[
            two_phase_skeleton(),
            phase_complete_decision("README.md"),  # ends P1 (README.md exists at tip)
            phase_complete_decision("README.md"),  # ends P2
            complete_decision(check_cmd("true")),
        ]
    )
    outcome = run_grind(
        run_dir, job_path="job.md", planner=planner, ladder=_ladder(), repo=git_repo
    )
    assert outcome.status == "completed"
    events = read_events(run_dir.events_path)
    assert [e.phase_id for e in events if e.event == "phase_passed"] == ["P1", "P2"]
    # No epoch ran; the planner drove every phase boundary explicitly.
    assert outcome.epochs_run == 0


# --- phase_complete grounding: missing deliverables bounce, do NOT halt --------


def test_phase_complete_missing_deliverable_bounces_back_to_planner(
    git_repo: Path, run_dir: RunDir
) -> None:
    """A phase_complete that cites a deliverable which does NOT exist at the tip is
    REJECTED and bounced back into planning (the same self-correction shape as a
    failed complete_run), NOT a halt. The planner re-emits with a real path and the
    phase ends."""

    rec = _Recording(
        MockPlanner(
            script=[
                two_phase_skeleton(),
                implement_decision(impl_task("T1", "f1.txt")),
                # First completion cites a path that was never built -> rejected,
                # re-ask. The run does not halt; the planner gets feedback and fixes it.
                phase_complete_decision("ghost.txt"),
                phase_complete_decision("f1.txt"),  # the real deliverable -> ends P1
                phase_complete_decision("f1.txt"),  # ends P2 (f1.txt still at tip)
                complete_decision(check_cmd("test -f f1.txt")),
            ]
        )
    )
    outcome = run_grind(
        run_dir, job_path="job.md", planner=rec, ladder=_ladder(), repo=git_repo
    )
    assert outcome.status == "completed"  # never halted on the missing citation
    events = read_events(run_dir.events_path)
    assert [e.phase_id for e in events if e.event == "phase_passed"] == ["P1", "P2"]
    # The rejected completion was re-asked WITHIN the same boundary (the bounce-back,
    # the same self-correction shape as a failed complete_run, never a terminal halt):
    # the very next prompt carried the missing-path feedback in an <errors> block.
    assert any(
        "<errors>" in p and "ghost.txt" in p and "does not exist" in p
        for p in rec.prompts
    )
    # No escalation/halt was emitted; the run reached its own complete_run.
    assert "run_escalated" not in [e.event for e in events]


def test_red_floor_blocks_completion_drives_fix_epoch(
    git_repo: Path, run_dir: RunDir
) -> None:
    """A RED build-health floor does not prevent the planner from being consulted,
    but the planner (seeing the red floor) plans a FIX epoch rather than completing;
    once the fix turns the floor green it phase_complete's. The phase ends only after
    the floor is healthy AND the planner judges it done."""

    # P1's floor needs gate.txt. The first epoch creates the WRONG file (floor still
    # red); the planner reads the FAIL, plans a fix epoch that creates gate.txt
    # (floor green), then completes the phase.
    rec = _Recording(
        MockPlanner(
            script=[
                skeleton_decision(
                    phase_dict("P1", exit_criterion=[check_cmd("test -f gate.txt")]),
                    phase_dict("P2", exit_criterion=[check_cmd("true")]),
                ),
                implement_decision(impl_task("T1", "wrong.txt")),  # floor stays RED
                implement_decision(impl_task("T2", "gate.txt")),   # FIX epoch -> floor GREEN
                phase_complete_decision("gate.txt"),               # now legal -> ends P1
                phase_complete_decision("gate.txt"),               # ends P2
                complete_decision(check_cmd("test -f gate.txt")),
            ]
        )
    )
    outcome = run_grind(
        run_dir, job_path="job.md", planner=rec, ladder=_ladder(), repo=git_repo
    )
    assert outcome.status == "completed"
    events = read_events(run_dir.events_path)
    # Two P1 epochs ran (the fix was needed); the floor moved RED -> GREEN across them.
    p1_epochs = [e.epoch_id for e in events if e.event == "epoch_started" and e.phase_id == "P1"]
    assert p1_epochs == ["E1", "E2"]
    # The boundary after the first (wrong) epoch showed the planner a RED floor.
    assert "[FAIL] cmd `test -f gate.txt`" in rec.prompts[2]
    # The boundary after the fix epoch showed a GREEN floor (completion now grounded).
    assert "[PASS] cmd `test -f gate.txt`" in rec.prompts[3]
    assert [e.phase_id for e in events if e.event == "phase_passed"] == ["P1", "P2"]


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
                phase_complete_decision("f1.txt"),  # planner ends P1 -> advance to P2
                complete_decision(check_cmd("test -f f1.txt")),
            ]
        )
    )
    run_grind(run_dir, job_path="job.md", planner=rec, ladder=_ladder(), repo=git_repo)

    # The first call (skeleton) carries no phase status (no skeleton yet).
    assert "<phase_status>" not in rec.prompts[0]
    # Second call: P1 active, its build-health floor freshly evaluated as FAIL.
    p1 = rec.prompts[1]
    assert "<phase_status>" in p1
    assert "current_phase: P1" in p1
    assert "build_health_floor" in p1
    assert "[FAIL] cmd `test -f f1.txt`" in p1
    assert "<integration_tip" in p1
    # Third call: still P1 (the floor passed after the epoch built f1.txt, but the
    # phase has NOT auto-advanced); the planner sees the green floor and decides.
    p1_again = rec.prompts[2]
    assert "current_phase: P1" in p1_again
    assert "[PASS] cmd `test -f f1.txt`" in p1_again
    # Fourth call: P1 phase_complete'd -> advanced to P2; the tip listing references
    # f1.txt (a name, never its body).
    p2 = rec.prompts[3]
    assert "current_phase: P2" in p2
    assert "passed_phases: P1" in p2
    tip = p2.split("<integration_tip", 1)[1].split("</integration_tip>", 1)[0]
    assert "f1.txt" in tip
