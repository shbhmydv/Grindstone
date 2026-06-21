"""End-to-end coverage for the gate rebalance (G5): the two automatic responses
to a failing gate, driven through the REAL state machine.

The rebalance splits verification into a deterministic floor, planner-authored
structural ``checks``, and natural-language ``criteria`` judged by an agentic pass,
and classifies a failing gate before charging anyone:

  - INFRA: a gate check that fails for an ENVIRONMENTAL reason (exit 127 /
    command-not-found) auto-dispatches a bounded SENIOR infra-repair, which makes
    the environment satisfiable; the gate is re-run, passes, and the run PROCEEDS
    with NO worker charged and NO semantic failed epoch.
  - SEMANTIC: an epoch whose deterministic floor passes but whose agentic verdict
    is ``pass=false`` with gaps opens a failed epoch routed through
    ``handle_failed_epoch``; the planner retries with the gaps as feedback; a
    second verdict passes and the run completes.

Both run as full ``run_grind`` E2Es through the existing double harness (a scripted
``MockPlanner`` + a tiered ladder of fake transports), asserting the terminal run
state AND the event sequence. The narrow per-piece behaviors (the classifier, the
verdict contract, the host guard, the failed-epoch block) are unit-tested in
``test_infra_repair.py`` / ``test_epoch_verification.py``; this file proves the two
responses compose end to end through the loop.
"""

from __future__ import annotations

import json

from grindstone.config import InfraRepairConfig
from grindstone.events import (
    EpochVerificationFailed,
    FailedEpochHandled,
    InfraRepairDispatched,
    read_events,
)
from grindstone.mock_planner import MockPlanner
from grindstone.rundir import RunDir
from grindstone.run_loop import RunState, run_grind
from grindstone.verify import WorkerEpochVerifier
from grindstone.worker import VERDICT_FILENAME, WorkerRequest

from pathlib import Path

from tests.grindstone.conftest import (
    FailingWorker,
    OwnershipWorker,
    check_cmd,
    complete_decision,
    handle_failed_epoch_retry,
    implement_decision,
    phase_dict,
    skeleton_decision,
)


def _run_state(run_dir: RunDir) -> RunState:
    return RunState.model_validate_json(run_dir.run_state_path.read_text())


# --- (a) INFRA: classify -> senior repair -> gate re-run -> proceed -------------
#
# A gate command that fails with the canonical command-not-found signature (exit
# 127 from a missing binary) UNTIL a marker file `tool_installed` exists in the
# worktree. A senior repair that writes the marker (a repo-local, committed fix)
# flips the gate green on the re-run, exactly as `npm install` would land a dep.

_INFRA_GATE = check_cmd("test -f tool_installed || notarealbinary --check")


def _infra_skeleton() -> dict[str, object]:
    return skeleton_decision(
        phase_dict("P1", title="build", exit_criterion=[_INFRA_GATE], budget=20),
        phase_dict("P2", title="verify", exit_criterion=[_INFRA_GATE], budget=20),
    )


class _SeniorInfraRepair:
    """A fake senior infra-repair transport that 'installs the tool' (writes the
    marker) so the failing command can run. Only acts on an infra-repair request,
    and writes the disk-contract handoff the core expects (the core re-runs the
    gate to judge, not this handoff)."""

    def __init__(self) -> None:
        self.repairs = 0

    def run(self, request: WorkerRequest) -> None:
        assert request.infra_repair is not None, "senior got a non-repair request"
        self.repairs += 1
        (request.scratch / "tool_installed").write_text("ok\n", encoding="utf-8")
        (request.scratch / "handoff.json").write_text(
            json.dumps(
                {
                    "schema_version": "1",
                    "task_id": request.task_id,
                    "status": "DONE",
                    "what_changed": [],
                    "resulting_state": "installed the missing tool",
                    "downstream_needs": [],
                    "not_done": [],
                    "citations": [],
                    "checks": [],
                    "occupancy": {"compacted": False, "subagent_splits": 0},
                }
            ),
            encoding="utf-8",
        )


def test_infra_gate_failure_auto_repairs_and_run_proceeds(
    git_repo: Path, run_dir: RunDir
) -> None:
    """A command-not-found gate failure auto-dispatches a senior infra-repair that
    fixes it; the gate re-runs green and the run PROCEEDS to completion, with the
    local worker NEVER charged and NO semantic failed epoch ever opened."""

    senior = _SeniorInfraRepair()
    local = FailingWorker()  # the infra failure must never reach the local worker
    planner = MockPlanner(
        script=[
            _infra_skeleton(),
            # The planner is only reached AFTER the gate is repaired green; it then
            # completes the run (the same now-passing gate is the evidence).
            complete_decision(_INFRA_GATE),
        ]
    )
    outcome = run_grind(
        run_dir,
        job_path="job.md",
        planner=planner,
        ladder=[("local", local), ("senior", senior)],
        repo=git_repo,
        infra_repair=InfraRepairConfig(attempts=2),
    )
    assert outcome.status == "completed", outcome
    kinds = [e.event for e in read_events(run_dir.events_path)]
    # The classify -> dispatch -> resolve -> proceed sequence fired in order.
    assert "infra_check_detected" in kinds
    assert "infra_repair_dispatched" in kinds
    assert "infra_repair_resolved" in kinds
    assert kinds.index("infra_repair_resolved") < kinds.index("run_completed")
    # NOT a worker charge: the local worker was never dispatched, no failed epoch.
    assert local.seen_failure_contexts == []
    assert "epoch_failed" not in kinds
    assert "epoch_verification_failed" not in kinds
    # Exactly one repair landed it (the cap bounded the dispatches; not unbounded).
    dispatched = [e for e in read_events(run_dir.events_path) if isinstance(e, InfraRepairDispatched)]
    assert len(dispatched) == 1
    assert senior.repairs == 1


# --- (b) SEMANTIC: floor passes, verdict fails -> failed epoch -> retry -> pass --
#
# The deterministic floor clears (the file exists), but the agentic verdict is
# pass=false on the FIRST verification and pass=true on the SECOND, modelling a
# retry that closed the gap. The planner disposes via handle_failed_epoch retry,
# threading the gaps as a corrective hint, and the run completes.

_SEMANTIC_FAIL_GAP = "the Lesson screen never maps the Pink ramp"


def _impl_with_criteria(tid: str, fname: str, criteria: list[str]) -> dict[str, object]:
    return {
        "id": tid,
        "goal": f"create {fname}",
        "done_when": [check_cmd(f"test -f {fname}")],
        "criteria": criteria,
        "file_ownership": [fname],
    }


def _semantic_skeleton() -> dict[str, object]:
    """A floor that passes once f1.txt exists, so the deterministic gate clears and
    only the agentic pass can fail the epoch."""

    gate = [check_cmd("test -f f1.txt")]
    return skeleton_decision(
        phase_dict("P1", title="build", exit_criterion=gate, budget=20),
        phase_dict("P2", title="verify", exit_criterion=gate, budget=20),
    )


class _SwapVerifier:
    """A local-tier verification transport: fails the FIRST verdict (with a concrete
    gap), passes every later one, modelling 'the retry closed the gap'. The verdict
    is the only output channel (a re-read disk contract); records its briefs so the
    test can assert the criteria reached the pass."""

    def __init__(self) -> None:
        self.calls = 0
        self.seen_criteria: list[list[str]] = []

    def run(self, request: WorkerRequest) -> None:
        assert request.verification is not None, "verifier got a non-verification request"
        self.calls += 1
        self.seen_criteria.append(list(request.verification.criteria))
        passed = self.calls > 1
        payload = {
            "pass": passed,
            "per_criterion": [
                {"criterion": c, "met": passed, "evidence": "checked the produced files"}
                for c in request.verification.criteria
            ],
            "gaps": [] if passed else [_SEMANTIC_FAIL_GAP],
        }
        (request.scratch / VERDICT_FILENAME).write_text(json.dumps(payload), encoding="utf-8")


class _UniqueContentWorker:
    """An OwnershipWorker whose content changes each call, so the re-dispatched epoch
    produces a NON-zero diff (the retry path's commit gate rejects a zero-diff
    rewrite of identical bytes; that is unrelated to the semantic routing)."""

    def __init__(self) -> None:
        self._calls = 0

    def run(self, request: WorkerRequest) -> None:
        self._calls += 1
        OwnershipWorker(content=f"v{self._calls}\n").run(request)


def test_semantic_gap_opens_failed_epoch_retry_closes_it(
    git_repo: Path, run_dir: RunDir
) -> None:
    """The floor passes but the agentic verdict marks a criterion unmet: a failed
    epoch opens and routes to handle_failed_epoch with the gap as feedback; the
    planner retries; the re-verification passes and the run completes. The gap
    reached the planner input (it rode the failed-epoch block as the retry hint),
    and the same criteria reached the verifier on both passes."""

    verifier = WorkerEpochVerifier(_SwapVerifier())
    worker = _UniqueContentWorker()
    planner = MockPlanner(
        script=[
            _semantic_skeleton(),
            implement_decision(_impl_with_criteria("T1", "f1.txt", ["map the Pink ramp"])),
            # The planner disposes the semantic failure by retrying, threading the
            # verdict's gap back as the corrective hint (the gaps reached its input).
            handle_failed_epoch_retry(f"close the gap: {_SEMANTIC_FAIL_GAP}"),
            complete_decision(check_cmd("test -f f1.txt")),
        ]
    )
    outcome = run_grind(
        run_dir,
        job_path="job.md",
        planner=planner,
        ladder=[("local", worker)],
        repo=git_repo,
        verifier=verifier,
    )
    assert outcome.status == "completed", outcome
    kinds = [e.event for e in read_events(run_dir.events_path)]
    # The semantic failure (NOT a task/gate failure) opened, was handled by retry,
    # and the re-verification passed before completion.
    assert "epoch_verification_failed" in kinds
    assert "epoch_failed" not in kinds  # never charged as a task/gate failure
    assert kinds.index("epoch_verification_failed") < kinds.index("run_completed")
    # The verdict's concrete gap was surfaced on the failed-epoch event.
    vf = [e for e in read_events(run_dir.events_path) if isinstance(e, EpochVerificationFailed)]
    assert vf and vf[0].gaps == [_SEMANTIC_FAIL_GAP]
    # The planner disposed it with a retry (not a halt / escalate).
    handled = [e for e in read_events(run_dir.events_path) if isinstance(e, FailedEpochHandled)]
    assert handled and handled[0].action == "retry"
    # The gap reached the planner: it appears verbatim in the retry hint the planner
    # chose (the failed-epoch block carried it into the planner's decision input).
    assert any(_SEMANTIC_FAIL_GAP in (e.detail or "") for e in handled)
    # The criteria reached the verifier on the first (fail) and second (pass) runs.
    fake = verifier._transport  # type: ignore[attr-defined]
    assert isinstance(fake, _SwapVerifier)
    assert fake.calls == 2
    assert fake.seen_criteria == [["map the Pink ramp"], ["map the Pink ramp"]]
    # No pending failed epoch lingers; the run finished clean.
    assert _run_state(run_dir).pending_failed_epoch is None
