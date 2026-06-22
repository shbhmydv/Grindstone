"""Failed-epoch disposition: gate observability + handle_failed_epoch + the cap.

The RCA-driven Part 2 fix. A failed epoch must trigger a FOCUSED planner
decision (retry / escalate_senior / halt), not a blind revise_phases spin; the
planner sees WHY the gate failed (captured command output); and a deterministic
per-phase cap forces a halt-to-human so the 15-epoch dogfood loop can never recur.
"""

from __future__ import annotations

from pathlib import Path

from grindstone.contracts.models import (
    CmdCheck,
    EscalateSeniorFailedEpochArgs,
    HaltFailedEpochArgs,
    HandleFailedEpochDecision,
    RetryFailedEpochArgs,
    parse_decision,
)
from grindstone.contracts.gate import decision_schema_errors
from grindstone.events import EpochFailed, FailedEpochHandled, read_events, replay
from grindstone.mock_planner import MockPlanner
from grindstone.planner import FailedEpochInfo, validate_decision, volatile_tail
from grindstone.rundir import RunDir, create_run_dir
from grindstone.run_loop import RunState, evaluate_checks, run_grind

from tests.grindstone.conftest import (
    FailingWorker,
    OwnershipWorker,
    artifact_decision,
    artifact_task,
    check_cmd,
    complete_decision,
    handle_failed_epoch_escalate,
    handle_failed_epoch_halt,
    handle_failed_epoch_retry,
    impl_task,
    implement_decision,
    phase_dict,
    revise_decision,
    skeleton_decision,
    two_phase_skeleton,
)


def _run_state(run_dir: RunDir) -> RunState:
    return RunState.model_validate_json(run_dir.run_state_path.read_text())


# A skeleton whose P1 exit criterion can NEVER pass (a missing file) so the phase
# gate keeps failing the same way, the dogfood structural-failure shape.
def _unpassable_skeleton(cmd: str = "test -f never_exists.txt") -> dict[str, object]:
    pending = [check_cmd(cmd)]
    return skeleton_decision(
        phase_dict("P1", title="build", exit_criterion=pending, budget=20),
        phase_dict("P2", title="verify", exit_criterion=pending, budget=20),
    )


def _ladder() -> list[tuple[str, OwnershipWorker]]:
    return [("worker", OwnershipWorker())]


# --- Part A: gate observability -----------------------------------------------


def test_evaluate_checks_captures_failed_cmd_output(tmp_path: Path) -> None:
    """A FAILED cmd check keeps the command's stdout/stderr (truncated, text-safe)
    in its label AND persists it under the run dir, so the planner learns WHY."""

    run = create_run_dir(tmp_path, "run-A")
    results = evaluate_checks(
        [CmdCheck(cmd="echo build-broke-here >&2; exit 1")],
        repo=None,
        ref=None,
        run_dir=run,
        scratch_name="eval",
    )
    label, ok = results[0]
    assert ok is False
    assert "build-broke-here" in label
    assert "exit 1" in label
    # Persisted durably under the run dir.
    persisted = run.root / "check_output" / "eval" / "c0.txt"
    assert persisted.is_file()
    assert "build-broke-here" in persisted.read_text()


def test_evaluate_checks_passing_cmd_keeps_no_output(tmp_path: Path) -> None:
    """A PASSING check carries no captured output (only failures are surfaced)."""

    run = create_run_dir(tmp_path, "run-A2")
    results = evaluate_checks(
        [CmdCheck(cmd="echo emitted-noise; true")],
        repo=None,
        ref=None,
        run_dir=run,
        scratch_name="eval",
    )
    label, ok = results[0]
    assert ok is True
    assert label == "cmd `echo emitted-noise; true`"  # bare label, no output section
    assert "output:" not in label
    assert not (run.root / "check_output").exists()


def test_failed_check_output_flows_to_planner_via_tail() -> None:
    """The captured failing-check label rides the <failed_epoch> block, so the
    planner's next decision input carries the env-vs-code evidence."""

    info = FailedEpochInfo(
        epoch_id="E1",
        failed_tasks=[("T1", "no handoff.json written")],
        failed_checks=["cmd `npx tsc` (exit 1)\n      output:\n        not found: tsc"],
        passing_handoffs=[("T1", "implemented as asked, tsc passed locally")],
        disposed_count=1,
        cap=3,
    )
    tail = volatile_tail(
        phase_id="P1",
        epoch_counter=1,
        log_index=[],
        last_epoch_rows=None,
        reask_errors=[],
        failed_epoch=info,
    )
    assert "<failed_epoch" in tail
    assert "not found: tsc" in tail
    assert "tsc passed locally" in tail  # the honest-pass evidence
    assert "handle_failed_epoch" in tail


# --- Part B: handle_failed_epoch parse + validate ------------------------------


def test_handle_failed_epoch_parses_each_action() -> None:
    retry = parse_decision(handle_failed_epoch_retry("fix the path", escalate_tier=True))
    assert isinstance(retry, HandleFailedEpochDecision)
    assert isinstance(retry.args, RetryFailedEpochArgs)
    assert retry.args.hint == "fix the path" and retry.args.escalate_tier is True

    esc = parse_decision(handle_failed_epoch_escalate("local cannot do this"))
    assert isinstance(esc.args, EscalateSeniorFailedEpochArgs)
    assert esc.args.diagnosis == "local cannot do this"

    halt = parse_decision(handle_failed_epoch_halt("env is broken"))
    assert isinstance(halt.args, HaltFailedEpochArgs)
    assert halt.args.reason == "env is broken"


def test_handle_failed_epoch_schema_round_trips() -> None:
    for payload in (
        handle_failed_epoch_retry("h"),
        handle_failed_epoch_escalate("d"),
        handle_failed_epoch_halt("r"),
    ):
        assert decision_schema_errors(payload) == []


def test_handle_failed_epoch_only_legal_when_failure_pending() -> None:
    import json

    text = json.dumps(handle_failed_epoch_halt("stop"))
    # Illegal with no failed epoch pending.
    gate = validate_decision(
        text,
        existing_log_keys=frozenset(),
        completed_phase_ids=frozenset(),
        skeleton_exists=True,
        failed_epoch_active=False,
    )
    assert gate.decision is None
    assert any("only legal when an epoch has failed" in e for e in gate.errors)
    # Legal once one is awaiting disposition.
    gate2 = validate_decision(
        text,
        existing_log_keys=frozenset(),
        completed_phase_ids=frozenset(),
        skeleton_exists=True,
        failed_epoch_active=True,
    )
    assert gate2.decision is not None


def test_failed_epoch_active_forbids_other_tools() -> None:
    import json

    text = json.dumps(complete_decision(check_cmd("true")))
    gate = validate_decision(
        text,
        existing_log_keys=frozenset(),
        completed_phase_ids=frozenset(),
        skeleton_exists=True,
        failed_epoch_active=True,
    )
    assert gate.decision is None
    assert any("only handle_failed_epoch is" in e for e in gate.errors)


# --- Part B: end-to-end state-machine branches --------------------------------


def _failing_ladder() -> list[tuple[str, FailingWorker]]:
    return [("worker", FailingWorker())]


def test_failed_epoch_drives_retry_then_completes(git_repo: Path, run_dir: RunDir) -> None:
    """A failed epoch constrains the next call to handle_failed_epoch; a retry
    re-dispatches the SAME epoch (now with a worker that succeeds) and the run
    proceeds. The retry hint reaches the retried worker."""

    # First epoch fails (no handoff), the retry succeeds (writes the file).
    swap = _SwapWorker()
    planner = MockPlanner(
        script=[
            two_phase_skeleton(),
            implement_decision(impl_task("T1", "f1.txt")),  # fails
            handle_failed_epoch_retry("create the file at repo root"),  # retry -> succeeds
            complete_decision(check_cmd("test -f f1.txt")),
        ]
    )
    outcome = run_grind(
        run_dir, job_path="job.md", planner=planner, ladder=[("worker", swap)],
        repo=git_repo, tier0_attempts=1,
    )
    assert outcome.status == "completed"
    kinds = [e.event for e in read_events(run_dir.events_path)]
    assert "epoch_failed" in kinds
    handled = [e for e in read_events(run_dir.events_path) if isinstance(e, FailedEpochHandled)]
    assert handled and handled[0].action == "retry"
    # The hint reached the retried worker's failure context.
    assert any(
        any("create the file at repo root" in c for c in ctx)
        for ctx in swap.seen_failure_contexts
    )


def test_failed_epoch_halt_is_terminal(git_repo: Path, run_dir: RunDir) -> None:
    """A halt disposition stops the run for a human (escalated, terminal)."""

    planner = MockPlanner(
        script=[
            two_phase_skeleton(),
            implement_decision(impl_task("T1", "f1.txt")),  # fails
            handle_failed_epoch_halt("the gate cannot pass in this env"),
        ]
    )
    outcome = run_grind(
        run_dir, job_path="job.md", planner=planner, ladder=_failing_ladder(),
        repo=git_repo, tier0_attempts=1,
    )
    assert outcome.status == "escalated"
    assert outcome.reason is not None and "halted failed epoch" in outcome.reason
    handled = [e for e in read_events(run_dir.events_path) if isinstance(e, FailedEpochHandled)]
    assert handled and handled[-1].action == "halt"


def test_failed_epoch_escalate_senior_routes_to_senior(git_repo: Path, run_dir: RunDir) -> None:
    """escalate_senior re-dispatches the epoch forced onto the senior tier."""

    local = FailingWorker()
    # Senior fails its FIRST attempt (so the first epoch fails on both tiers),
    # then succeeds, the forced-senior retry lands on a now-working senior.
    senior = _SwapWorker(fail_calls=1)
    planner = MockPlanner(
        script=[
            two_phase_skeleton(),
            implement_decision(impl_task("T1", "f1.txt")),  # fails on local AND senior
            handle_failed_epoch_escalate("local keeps missing it; senior please"),
            complete_decision(check_cmd("test -f f1.txt")),
        ]
    )
    outcome = run_grind(
        run_dir, job_path="job.md",
        planner=planner, ladder=[("worker", local), ("senior", senior)],
        repo=git_repo, tier0_attempts=1,
    )
    assert outcome.status == "completed"
    handled = [e for e in read_events(run_dir.events_path) if isinstance(e, FailedEpochHandled)]
    assert handled and handled[0].action == "escalate_senior"
    # The forced-senior retry started on the senior tier (the swap's 2nd call).
    assert senior.seen_failure_contexts  # senior was reached on the retry


def test_revise_phases_still_works_for_structural_replan(git_repo: Path, run_dir: RunDir) -> None:
    """revise_phases is unchanged for a genuine structural replan (no failed
    epoch in play): it replaces the skeleton tail and the run proceeds."""

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
            ),
            implement_decision(impl_task("T1", "f1.txt")),
            complete_decision(check_cmd("test -f f1.txt")),
        ]
    )
    outcome = run_grind(
        run_dir, job_path="job.md", planner=planner, ladder=_ladder(), repo=git_repo
    )
    assert outcome.status == "completed"
    assert "phases_revised" in [e.event for e in read_events(run_dir.events_path)]


# --- Part B: budget-exhausted-BY-a-failed-epoch precedence (regression) --------
#
# The live integration bug: a phase whose epoch_budget is exhausted by a FAILED
# epoch fires BOTH demands at the next boundary, phase_escalation_active (budget)
# AND pending_failed_epoch. The old legality let handle_failed_epoch past the
# failed-epoch gate then REJECTED it for not being an escalation tool, deadlocking
# the run into "invalid decision after 2 re-asks". The failed-epoch disposition
# must TAKE PRECEDENCE. These drive the real state machine (not _position_legality
# in isolation), the gap that let it ship.


def _budget1_skeleton() -> dict[str, object]:
    """P1 with epoch_budget=1, so ONE failed epoch exhausts the budget while its
    gate (a missing file) still fails, firing budget-escalation + a pending failed
    epoch at the SAME boundary. P2 closes the run once P1 passes."""

    return skeleton_decision(
        phase_dict("P1", title="build", exit_criterion=[check_cmd("test -f f1.txt")], budget=1),
        phase_dict("P2", title="verify", exit_criterion=[check_cmd("test -f f1.txt")]),
    )


def test_budget_exhausted_by_failed_epoch_retry_progresses(
    git_repo: Path, run_dir: RunDir
) -> None:
    """epoch_budget=1 + a failed epoch fires budget-escalation AND a pending failed
    epoch together. handle_failed_epoch must be LEGAL (not deadlocked); a retry
    re-dispatches and the run PROGRESSES, never escalating with 'invalid decision'."""

    swap = _SwapWorker()  # fails first epoch, succeeds on the retry -> writes f1.txt
    planner = MockPlanner(
        script=[
            _budget1_skeleton(),
            implement_decision(impl_task("T1", "f1.txt")),  # fails, budget(1) now spent
            handle_failed_epoch_retry("create f1.txt at repo root"),  # MUST be legal
            complete_decision(check_cmd("test -f f1.txt")),
        ]
    )
    outcome = run_grind(
        run_dir, job_path="job.md", planner=planner, ladder=[("worker", swap)],
        repo=git_repo, tier0_attempts=1,
    )
    assert outcome.status == "completed"
    reason = outcome.reason or ""
    assert "invalid decision" not in reason and "phase escalation in force" not in reason
    handled = [e for e in read_events(run_dir.events_path) if isinstance(e, FailedEpochHandled)]
    assert handled and handled[0].action == "retry"
    # The budget-escalation flag fired but did NOT linger to block the later
    # complete_run, it reset when P1 advanced to P2 after the successful retry.
    assert _run_state(run_dir).phase_escalation_active is False


def test_budget_exhausted_by_failed_epoch_escalate_senior_progresses(
    git_repo: Path, run_dir: RunDir
) -> None:
    """Same collision, escalate_senior action: it re-dispatches forced onto senior
    and the run progresses, the budget-escalation rule never blocks it."""

    local = FailingWorker()
    senior = _SwapWorker(fail_calls=1)  # fails the first (shared) epoch, then succeeds
    planner = MockPlanner(
        script=[
            _budget1_skeleton(),
            implement_decision(impl_task("T1", "f1.txt")),  # fails on local AND senior
            handle_failed_epoch_escalate("local cannot; senior please"),  # MUST be legal
            complete_decision(check_cmd("test -f f1.txt")),
        ]
    )
    outcome = run_grind(
        run_dir, job_path="job.md",
        planner=planner, ladder=[("worker", local), ("senior", senior)],
        repo=git_repo, tier0_attempts=1,
    )
    assert outcome.status == "completed"
    assert "invalid decision" not in (outcome.reason or "")
    handled = [e for e in read_events(run_dir.events_path) if isinstance(e, FailedEpochHandled)]
    assert handled and handled[0].action == "escalate_senior"


def test_budget_exhausted_by_failed_epoch_halt_terminates(
    git_repo: Path, run_dir: RunDir
) -> None:
    """Same collision, halt action: handle_failed_epoch is legal and its halt still
    terminates (escalates for a human), NOT via the 'invalid decision' deadlock."""

    planner = MockPlanner(
        script=[
            _budget1_skeleton(),
            implement_decision(impl_task("T1", "f1.txt")),  # fails, budget spent
            handle_failed_epoch_halt("the gate cannot pass in this env"),  # MUST be legal
        ]
    )
    outcome = run_grind(
        run_dir, job_path="job.md", planner=planner, ladder=_failing_ladder(),
        repo=git_repo, tier0_attempts=1,
    )
    assert outcome.status == "escalated"
    reason = outcome.reason or ""
    assert "halted failed epoch" in reason
    assert "invalid decision" not in reason
    handled = [e for e in read_events(run_dir.events_path) if isinstance(e, FailedEpochHandled)]
    assert handled and handled[-1].action == "halt"


def test_budget_escalation_without_failed_epoch_still_constrains(
    git_repo: Path, run_dir: RunDir
) -> None:
    """The genuine budget-escalation path (NO failed epoch) is unchanged: a phase
    whose gate never passes while its epochs all COMPLETE structurally exhausts the
    budget with no pending failed epoch, and the next decision is constrained to
    revise_phases / escalate_run. Here an artifact epoch completes (no task fails)
    yet the phase gate (a missing file) stays red, so budget-escalation fires alone
    and a stray implement is rejected, forcing revise_phases, which then resolves."""

    pending = [check_cmd("test -f f1.txt")]
    planner = MockPlanner(
        script=[
            skeleton_decision(
                # P1 gate needs f1.txt but the epoch only writes an artifact note,
                # so the epoch COMPLETES (no failed task) while the gate stays red.
                phase_dict("P1", title="build", exit_criterion=pending, budget=1),
                phase_dict("P2", title="verify", exit_criterion=pending),
            ),
            artifact_decision(artifact_task("T1")),  # completes; gate still red; budget(1) spent
            implement_decision(impl_task("T1", "f1.txt")),  # REJECTED: escalation in force
            revise_decision(  # the legal escalation response
                phase_dict("P1", title="rebuilt", exit_criterion=pending),
                phase_dict("P2", exit_criterion=pending),
            ),
            implement_decision(impl_task("T1", "f1.txt")),  # now writes f1.txt
            complete_decision(check_cmd("test -f f1.txt")),
        ]
    )
    outcome = run_grind(
        run_dir, job_path="job.md", planner=planner, ladder=_ladder(), repo=git_repo,
    )
    assert outcome.status == "completed"
    kinds = [e.event for e in read_events(run_dir.events_path)]
    assert "phase_escalated" in kinds
    assert "phases_revised" in kinds
    # No failed epoch was ever opened, this was a pure budget escalation.
    assert "epoch_failed" not in kinds


# --- Part C: deterministic per-phase failure cap ------------------------------


def test_failed_epoch_cap_forces_halt(git_repo: Path, run_dir: RunDir) -> None:
    """After max_failed_epochs_per_phase failed epochs in one phase, the state
    machine FORCES a halt-to-human regardless of the planner ordering retries."""

    planner = MockPlanner(
        script=[
            two_phase_skeleton(),
            implement_decision(impl_task("T1", "f1.txt")),  # fail #1
            handle_failed_epoch_retry("retry 1"),
            implement_decision(impl_task("T1", "f1.txt")),  # fail #2
            handle_failed_epoch_retry("retry 2"),
            # The re-dispatched epoch from "retry 2" is fail #3 -> cap forces halt;
            # the planner is never asked again.
        ]
    )
    outcome = run_grind(
        run_dir, job_path="job.md", planner=planner, ladder=_failing_ladder(),
        repo=git_repo, tier0_attempts=1, max_failed_epochs_per_phase=3,
    )
    assert outcome.status == "escalated"
    assert outcome.reason is not None and "failed-epoch cap reached" in outcome.reason
    handled = [e for e in read_events(run_dir.events_path) if isinstance(e, FailedEpochHandled)]
    assert handled[-1].action == "cap_halt"


def test_failed_epoch_cap_of_one_halts_immediately(git_repo: Path, run_dir: RunDir) -> None:
    """A cap of 1: the FIRST failed epoch halts, the planner is never asked to
    dispose of it (the hardest backstop)."""

    planner = MockPlanner(
        script=[
            two_phase_skeleton(),
            implement_decision(impl_task("T1", "f1.txt")),  # fails -> cap=1 halts
        ]
    )
    outcome = run_grind(
        run_dir, job_path="job.md", planner=planner, ladder=_failing_ladder(),
        repo=git_repo, tier0_attempts=1, max_failed_epochs_per_phase=1,
    )
    assert outcome.status == "escalated"
    assert "failed-epoch cap reached" in (outcome.reason or "")


def test_epoch_failed_replays_coherently(git_repo: Path, run_dir: RunDir) -> None:
    planner = MockPlanner(
        script=[
            two_phase_skeleton(),
            implement_decision(impl_task("T1", "f1.txt")),
        ]
    )
    run_grind(
        run_dir, job_path="job.md", planner=planner, ladder=_failing_ladder(),
        repo=git_repo, tier0_attempts=1, max_failed_epochs_per_phase=1,
    )
    tree = replay(read_events(run_dir.events_path))
    assert tree.status == "escalated"
    failed = [e for e in read_events(run_dir.events_path) if isinstance(e, EpochFailed)]
    assert failed and "T1" in failed[0].failed_tasks


# --- a stateful swap worker (fail first epoch, succeed on the retry) ----------


class _SwapWorker:
    """Fail the first call (no handoff), succeed on every later call. Models 'the
    retry fixes it'. Records every call's failure context (so a test can assert
    the planner's retry hint reached the worker)."""

    def __init__(self, fail_calls: int = 1) -> None:
        self._fail_calls = fail_calls
        self._calls = 0
        self._ok = OwnershipWorker()
        self.seen_failure_contexts: list[list[str]] = []

    def run(self, request) -> None:  # type: ignore[no-untyped-def]
        self.seen_failure_contexts.append(list(request.failure_context))
        self._calls += 1
        if self._calls <= self._fail_calls:
            return  # no handoff.json -> failed attempt
        self._ok.run(request)
