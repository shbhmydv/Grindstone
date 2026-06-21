"""G3 automatic senior infra-repair: classify, auto-repair, cap, host guard.

The gate-rebalance G3 batch. A gate check that fails for an ENVIRONMENTAL reason
(exit 127, a missing tool / dependency, an install failure) must NOT charge the
worker or open a semantic failed epoch: the core auto-dispatches a SENIOR
infra-repair against the gate tip, told to make the environment satisfiable
WITHOUT rewriting app logic, then re-runs the gate. Bounded by a per-gate cap; on
exhaustion the run escalates for a human naming the unsatisfiable command. A
host-command guard keeps host-level fixes deny-by-default (carried into the
dispatch + surfaced in the prompt).
"""

from __future__ import annotations

import json
from pathlib import Path

from grindstone.config import InfraRepairConfig
from grindstone.events import (
    InfraCheckDetected,
    InfraRepairDispatched,
    InfraRepairExhausted,
    InfraRepairResolved,
    read_events,
)
from grindstone.infra import classify_check_failure
from grindstone.mock_planner import MockPlanner
from grindstone.rundir import RunDir
from grindstone.run_loop import run_grind
from grindstone.worker import (
    InfraRepairBrief,
    WorkerRequest,
    build_infra_repair_prompt,
    build_worker_prompt,
)

from tests.grindstone.conftest import (
    FailingWorker,
    check_cmd,
    complete_decision,
    phase_dict,
    skeleton_decision,
)


# --- the classifier (hard unit coverage: positive, negative, boundary) ---------


def test_classifier_exit_127_is_infra() -> None:
    c = classify_check_failure(returncode=127, stdout="", stderr="rg: not found")
    assert c.is_infra and "127" in c.reason


def test_classifier_command_not_found_is_infra() -> None:
    c = classify_check_failure(
        returncode=1, stdout="", stderr="bash: tsc: command not found"
    )
    assert c.is_infra and "not found" in c.reason.lower()


def test_classifier_node_cannot_find_module_is_NOT_infra() -> None:
    # A#2: a bare "Cannot find module" is the dominant signature of a genuine code
    # bug (a renamed/deleted import), NOT an environment fault. It must NOT be
    # auto-handed to the senior to "repair the environment" (that masks the defect).
    c = classify_check_failure(
        returncode=1, stdout="", stderr="Error: Cannot find module './deleted-file'"
    )
    assert c.is_infra is False


def test_classifier_python_module_not_found_traceback_is_NOT_infra() -> None:
    # A#2 (the key negative): pytest dumps a worker-deleted module's import failure
    # as a ModuleNotFoundError traceback on STDOUT, which the classifier scans. That
    # is a code bug, not infra: it must be charged to the worker, not auto-repaired.
    c = classify_check_failure(
        returncode=1,
        stdout=(
            "============================= test session starts ===========\n"
            "ImportError while importing test module 'tests/test_foo.py'.\n"
            "tests/test_foo.py:3: in <module>\n"
            "    from app.widgets import Button\n"
            "E   ModuleNotFoundError: No module named 'app.widgets'\n"
            "1 error in 0.12s\n"
        ),
        stderr="",
    )
    assert c.is_infra is False


def test_classifier_bare_import_error_is_NOT_infra() -> None:
    # A#2: a bare ImportError (e.g. a circular import a worker introduced) is a code
    # bug, never an environment fault.
    c = classify_check_failure(returncode=1, stdout="", stderr="ImportError: bad")
    assert c.is_infra is False


def test_classifier_npm_install_error_is_infra() -> None:
    c = classify_check_failure(
        returncode=1, stdout="", stderr="npm ERR! code ENOENT\nnpm ERR! missing"
    )
    assert c.is_infra


def test_classifier_pip_no_distribution_is_infra() -> None:
    c = classify_check_failure(
        returncode=1, stdout="", stderr="No matching distribution found for foo"
    )
    assert c.is_infra


def test_classifier_missing_interpreter_is_infra() -> None:
    c = classify_check_failure(
        returncode=126, stdout="", stderr="/usr/bin/env: 'node': No such file or directory"
    )
    assert c.is_infra


def test_classifier_plain_test_failure_is_NOT_infra() -> None:
    # The conservative core: a real assertion failure (ordinary test output, exit 1)
    # must NOT be mistaken for infra, else the senior spins on genuine bugs.
    c = classify_check_failure(
        returncode=1,
        stdout="FAIL src/foo.test.ts\n  expected 2 to equal 3\n1 failed",
        stderr="",
    )
    assert c.is_infra is False


def test_classifier_clean_exit_is_NOT_infra() -> None:
    # Totality boundary: a 0 return code still reports not-infra.
    c = classify_check_failure(returncode=0, stdout="", stderr="")
    assert c.is_infra is False


def test_classifier_assertion_mentioning_import_word_loosely() -> None:
    # "import" in prose (not an ImportError) must not trip the signature.
    c = classify_check_failure(
        returncode=1, stdout="please import the data first", stderr=""
    )
    assert c.is_infra is False


# --- the infra-repair prompt + host guard --------------------------------------


def _brief(allow: list[str]) -> InfraRepairBrief:
    return InfraRepairBrief(
        failing_commands=["npx tsc --noEmit"],
        output_tail="Cannot find module 'typescript'",
        reason="missing node dependency (Cannot find module)",
        allow_host_commands=allow,
    )


def _infra_request(brief: InfraRepairBrief, scratch: Path) -> WorkerRequest:
    from grindstone.contracts.models import CmdCheck, ImplementTask

    task = ImplementTask(
        id="T1", goal="infra repair", done_when=[CmdCheck(cmd="true")],
        file_ownership=["**"],
    )
    return WorkerRequest(
        task=task, task_id="P1/infra-repair-1", inputs={}, scratch=scratch,
        attempt=1, failure_context=[], mode="implement", infra_repair=brief,
    )


def test_infra_repair_prompt_states_repo_local_and_empty_allowlist(tmp_path: Path) -> None:
    prompt = build_infra_repair_prompt(_infra_request(_brief([]), tmp_path), _brief([]))
    assert "INFRA REPAIR" in prompt
    assert "npx tsc --noEmit" in prompt
    assert "Cannot find module" in prompt
    assert "repo-local" in prompt.lower()
    # Deny-by-default: an empty allowlist forbids host-level commands explicitly.
    assert "NO host-level commands are allowlisted" in prompt
    assert "needs host command" in prompt.lower()


def test_infra_repair_prompt_surfaces_allowlist(tmp_path: Path) -> None:
    brief = _brief(["apt-get", "brew"])
    prompt = build_infra_repair_prompt(_infra_request(brief, tmp_path), brief)
    assert "`apt-get`" in prompt and "`brew`" in prompt
    assert "ALLOWLISTED" in prompt


def test_build_worker_prompt_routes_infra_requests(tmp_path: Path) -> None:
    # The shared dispatcher branches to the infra prompt when the brief is set.
    brief = _brief([])
    prompt = build_worker_prompt(_infra_request(brief, tmp_path))
    assert "INFRA REPAIR" in prompt


# --- end-to-end: detect -> senior repair -> re-run gate -> proceed -------------
#
# A gate command that fails infra-style (a genuine LAUNCH fault: "command not
# found") UNTIL a marker file `deps_ok` exists in the worktree. A repair that
# writes the marker (a repo-local fix, committed) flips the gate green on the
# re-run. We use command-not-found (not ModuleNotFoundError, which post-A#2 is
# correctly a code bug, not infra) so the classifier flags it environmental.

_GATE = check_cmd(
    'test -f deps_ok || { echo "build-tool: command not found" >&2; exit 1; }'
)


def _gate_skeleton() -> dict[str, object]:
    return skeleton_decision(
        phase_dict("P1", title="build", exit_criterion=[_GATE], budget=20),
        phase_dict("P2", title="verify", exit_criterion=[_GATE], budget=20),
    )


class _SeniorRepairWorker:
    """A fake senior infra-repair transport that 'installs deps' (writes deps_ok).

    Only acts when handed an infra-repair brief (asserts it carries the failing
    command + the empty/loaded allowlist a test set). Records the briefs it saw so
    a test can assert the host-guard policy reached the dispatch."""

    def __init__(self, *, fix: bool) -> None:
        self._fix = fix
        self.seen_briefs: list[InfraRepairBrief] = []

    def run(self, request: WorkerRequest) -> None:
        assert request.infra_repair is not None, "senior got a non-repair request"
        self.seen_briefs.append(request.infra_repair)
        if self._fix:
            (request.scratch / "deps_ok").write_text("installed\n", encoding="utf-8")
        # The disk contract: a handoff (the core re-runs the gate to judge, not this).
        (request.scratch / "handoff.json").write_text(
            json.dumps(
                {
                    "schema_version": "1",
                    "task_id": request.task_id,
                    "status": "DONE" if self._fix else "FAILED",
                    "what_changed": [],
                    "resulting_state": "infra repair",
                    "downstream_needs": [],
                    "not_done": [] if self._fix else ["needs host command: apt install"],
                    "citations": [],
                    "checks": [],
                    "occupancy": {"compacted": False, "subagent_splits": 0},
                }
            ),
            encoding="utf-8",
        )


def test_infra_fail_auto_dispatches_senior_repair_and_proceeds(
    git_repo: Path, run_dir: RunDir
) -> None:
    """An infra-classified gate failure dispatches a senior repair (NOT a worker
    charge / NOT a semantic failed epoch); the repair fixes it, the gate goes green
    and the run completes. No epoch_failed is ever opened."""

    senior = _SeniorRepairWorker(fix=True)
    local = FailingWorker()  # must never be charged for the infra failure
    planner = MockPlanner(
        script=[
            _gate_skeleton(),
            complete_decision(_GATE),  # after repair the gate (and evidence) pass
        ]
    )
    outcome = run_grind(
        run_dir, job_path="job.md", planner=planner,
        ladder=[("local", local), ("senior", senior)], repo=git_repo,
        infra_repair=InfraRepairConfig(attempts=2),
    )
    assert outcome.status == "completed"
    kinds = [e.event for e in read_events(run_dir.events_path)]
    assert "infra_check_detected" in kinds
    assert "infra_repair_dispatched" in kinds
    assert "infra_repair_resolved" in kinds
    # The infra failure was NEVER charged to the local worker as a failed epoch.
    assert "epoch_failed" not in kinds
    assert local.seen_failure_contexts == []  # local was never dispatched
    # The senior repair ran with the host-guard allowlist carried (empty by default).
    assert senior.seen_briefs and senior.seen_briefs[0].allow_host_commands == []


def test_infra_repair_exhausts_cap_and_escalates_with_clear_message(
    git_repo: Path, run_dir: RunDir
) -> None:
    """A repair that never fixes the env exhausts the cap and escalates the run for
    a human, naming the unsatisfiable command (not a vague worker failure)."""

    senior = _SeniorRepairWorker(fix=False)  # never writes deps_ok
    planner = MockPlanner(script=[_gate_skeleton()])  # never reached past the gate
    outcome = run_grind(
        run_dir, job_path="job.md", planner=planner,
        ladder=[("local", FailingWorker()), ("senior", senior)], repo=git_repo,
        infra_repair=InfraRepairConfig(attempts=2),
    )
    assert outcome.status == "escalated"
    reason = outcome.reason or ""
    assert "infra-repair exhausted" in reason
    assert "deps_ok" in reason  # names the unsatisfiable command
    kinds = [e.event for e in read_events(run_dir.events_path)]
    assert "infra_repair_exhausted" in kinds
    # The cap bounded the dispatches (exactly `attempts` repairs, never unbounded).
    dispatched = [e for e in read_events(run_dir.events_path) if isinstance(e, InfraRepairDispatched)]
    assert len(dispatched) == 2


def test_infra_repair_disabled_when_no_config(git_repo: Path, run_dir: RunDir) -> None:
    """With no infra_repair policy the auto-repair never fires; the infra-failing
    gate is left to the ordinary path (here the planner just escalates the run)."""

    senior = _SeniorRepairWorker(fix=True)
    planner = MockPlanner(
        script=[_gate_skeleton(), {"schema_version": "1", "tool": "escalate_run",
                                   "args": {"reason": "gate cannot pass"}}]
    )
    outcome = run_grind(
        run_dir, job_path="job.md", planner=planner,
        ladder=[("local", FailingWorker()), ("senior", senior)], repo=git_repo,
        infra_repair=None,
    )
    assert outcome.status == "escalated"
    kinds = [e.event for e in read_events(run_dir.events_path)]
    assert "infra_check_detected" not in kinds
    assert senior.seen_briefs == []  # senior never dispatched for repair


def test_infra_repair_host_allowlist_carried_into_dispatch(
    git_repo: Path, run_dir: RunDir
) -> None:
    """The host-command guard's allowlist is carried into the repair dispatch (and
    thus the prompt), proving the policy reaches the senior, not just the config."""

    senior = _SeniorRepairWorker(fix=True)
    planner = MockPlanner(script=[_gate_skeleton(), complete_decision(_GATE)])
    run_grind(
        run_dir, job_path="job.md", planner=planner,
        ladder=[("local", FailingWorker()), ("senior", senior)], repo=git_repo,
        infra_repair=InfraRepairConfig(attempts=1, allow_host_commands=["apt-get"]),
    )
    assert senior.seen_briefs
    assert senior.seen_briefs[0].allow_host_commands == ["apt-get"]


# --- A#1: a PrepareError materializing the repair worktree does not crash -------


def test_infra_repair_prepare_error_does_not_crash_escalates(
    git_repo: Path, run_dir: RunDir, monkeypatch: object
) -> None:
    """A#1: when deps cannot materialize in the repair eval-worktree (the exact
    scenario this feature targets), ``materialize_env`` raises ``PrepareError``;
    the run does NOT crash. The repair counts as not-landed, the cap exhausts, and
    the run escalates cleanly (no stack trace out of ``_drive``)."""

    import grindstone.run_loop as run_loop
    from grindstone.prepare import PrepareError

    real_materialize = run_loop.materialize_env

    def fake_materialize(repo: Path, worktree: Path, prepare: object) -> None:
        # Detection worktrees succeed (so the infra failure IS detected); the
        # repair/recheck worktrees raise (deps cannot install there).
        if worktree.name.startswith("_infra_repair") or worktree.name.startswith(
            "_infra_recheck"
        ):
            raise PrepareError("prepare failed: deps cannot install in eval worktree")
        real_materialize(repo, worktree, prepare)  # type: ignore[arg-type]

    monkeypatch.setattr(run_loop, "materialize_env", fake_materialize)  # type: ignore[attr-defined]

    senior = _SeniorRepairWorker(fix=True)  # would fix, but prepare never lands it
    planner = MockPlanner(script=[_gate_skeleton()])  # never reached past the gate
    outcome = run_grind(
        run_dir, job_path="job.md", planner=planner,
        ladder=[("local", FailingWorker()), ("senior", senior)], repo=git_repo,
        infra_repair=InfraRepairConfig(attempts=2),
    )
    # No crash: a clean terminal escalation, the cap/escalate path fired.
    assert outcome.status == "escalated"
    assert "infra-repair exhausted" in (outcome.reason or "")
    kinds = [e.event for e in read_events(run_dir.events_path)]
    assert "infra_repair_exhausted" in kinds
    # Repairs were dispatched (and each did not land because prepare raised).
    dispatched = [
        e for e in read_events(run_dir.events_path) if isinstance(e, InfraRepairDispatched)
    ]
    assert len(dispatched) == 2
    assert "infra_repair_resolved" not in kinds


# --- attempts=0: escalate immediately, ZERO dispatches (distinct from no-config) -


def test_infra_repair_attempts_zero_escalates_with_no_dispatch(
    git_repo: Path, run_dir: RunDir
) -> None:
    """``InfraRepairConfig(attempts=0)`` detects the infra failure but dispatches NO
    repair: it escalates immediately with zero ``infra_repair_dispatched`` events.
    Distinct from the no-config path (which does not even detect)."""

    senior = _SeniorRepairWorker(fix=True)
    planner = MockPlanner(script=[_gate_skeleton()])
    outcome = run_grind(
        run_dir, job_path="job.md", planner=planner,
        ladder=[("local", FailingWorker()), ("senior", senior)], repo=git_repo,
        infra_repair=InfraRepairConfig(attempts=0),
    )
    assert outcome.status == "escalated"
    assert "infra-repair exhausted" in (outcome.reason or "")
    kinds = [e.event for e in read_events(run_dir.events_path)]
    # Detected (config present) but never dispatched (cap is 0) and never resolved.
    assert "infra_check_detected" in kinds
    assert "infra_repair_exhausted" in kinds
    assert "infra_repair_dispatched" not in kinds
    assert senior.seen_briefs == []  # the senior was never asked to repair


# --- FLOOR-driven repair: the real dogfood shape (a build floor needing deps) ---

# A floor check (the repo/core-owned canonical verification command) that fails
# infra-style until the marker exists. The phase exit criterion itself is trivially
# green, so ONLY the floor surfaces the infra failure (today's tests only triggered
# via exit_criterion). This is the real dogfood shape: a `floor` build check needs
# deps that are missing in the fresh eval worktree.
_FLOOR_GATE = 'test -f deps_ok || { echo "tsc: command not found" >&2; exit 1; }'


def test_infra_repair_triggers_from_floor_check(
    git_repo: Path, run_dir: RunDir
) -> None:
    """An infra failure surfaced by a ``FloorConfig`` check (not the exit criterion)
    still dispatches the senior repair; the repair writes the marker, the floor
    clears, and the run completes."""

    from grindstone.config import FloorConfig

    senior = _SeniorRepairWorker(fix=True)
    skeleton = skeleton_decision(
        phase_dict("P1", title="build", exit_criterion=[check_cmd("true")], budget=20),
        phase_dict("P2", title="verify", exit_criterion=[check_cmd("true")], budget=20),
    )
    planner = MockPlanner(script=[skeleton, complete_decision(check_cmd("true"))])
    outcome = run_grind(
        run_dir, job_path="job.md", planner=planner,
        ladder=[("local", FailingWorker()), ("senior", senior)], repo=git_repo,
        infra_repair=InfraRepairConfig(attempts=2),
        floor=FloorConfig(checks=[_FLOOR_GATE]),
    )
    assert outcome.status == "completed"
    kinds = [e.event for e in read_events(run_dir.events_path)]
    assert "infra_check_detected" in kinds
    assert "infra_repair_dispatched" in kinds
    assert "infra_repair_resolved" in kinds
    assert "epoch_failed" not in kinds
    assert senior.seen_briefs  # the floor's command rode the brief
    assert any("tsc" in b.failing_commands[0] for b in senior.seen_briefs)


# --- A#7: a repair that regresses an unrelated check in the gate is NOT adopted --


class _RegressingRepairWorker:
    """A senior that 'fixes' the named infra command (writes deps_ok) but REGRESSES
    an unrelated sibling check in the same exit criterion (deletes other_ok)."""

    def __init__(self) -> None:
        self.seen_briefs: list[InfraRepairBrief] = []

    def run(self, request: WorkerRequest) -> None:
        assert request.infra_repair is not None
        self.seen_briefs.append(request.infra_repair)
        (request.scratch / "deps_ok").write_text("installed\n", encoding="utf-8")
        # Regress the sibling check: remove the file its command depends on.
        (request.scratch / "other_ok").unlink(missing_ok=True)
        (request.scratch / "handoff.json").write_text(
            json.dumps(
                {
                    "schema_version": "1", "task_id": request.task_id, "status": "DONE",
                    "what_changed": [], "resulting_state": "infra repair",
                    "downstream_needs": [], "not_done": [], "citations": [], "checks": [],
                    "occupancy": {"compacted": False, "subagent_splits": 0},
                }
            ),
            encoding="utf-8",
        )


def test_infra_repair_regressing_sibling_check_not_adopted(
    git_repo: Path, run_dir: RunDir
) -> None:
    """A#7: the FULL exit criterion is re-run after a repair. A repair that fixes
    the named infra command but regresses a SIBLING check in the same criterion is
    NOT adopted as the tip; the gate never goes green, so the cap exhausts and the
    run escalates (the regression was caught, not silently tipped in)."""

    # other_ok is seeded committed at repo root, so the sibling check passes at the
    # tip; the repair worker deletes it, regressing that check on the repair commit.
    (git_repo / "other_ok").write_text("seed\n", encoding="utf-8")
    import subprocess as _sp
    _sp.run(["git", "add", "other_ok"], cwd=git_repo, check=True)
    _sp.run(["git", "commit", "-m", "seed other_ok"], cwd=git_repo, check=True)

    infra_check = check_cmd(
        'test -f deps_ok || { echo "tool: command not found" >&2; exit 1; }'
    )
    sibling_check = check_cmd("test -f other_ok")
    skeleton = skeleton_decision(
        phase_dict(
            "P1", title="build", exit_criterion=[infra_check, sibling_check], budget=20
        ),
        phase_dict("P2", title="verify", exit_criterion=[check_cmd("true")], budget=20),
    )
    senior = _RegressingRepairWorker()
    planner = MockPlanner(script=[skeleton])  # never reached past the gate
    outcome = run_grind(
        run_dir, job_path="job.md", planner=planner,
        ladder=[("local", FailingWorker()), ("senior", senior)], repo=git_repo,
        infra_repair=InfraRepairConfig(attempts=2),
    )
    assert outcome.status == "escalated"
    kinds = [e.event for e in read_events(run_dir.events_path)]
    assert "infra_repair_exhausted" in kinds
    # The regressing repair was dispatched but never adopted (no resolved event).
    assert "infra_repair_dispatched" in kinds
    assert "infra_repair_resolved" not in kinds
    # The tip was never moved to a repair branch (regression caught by full recheck).
    state = json.loads((run_dir.root / "run_state.json").read_text(encoding="utf-8"))
    assert state.get("last_integration_branch") in (None, "")


# --- A#10: infra-repair is skipped while a failed epoch awaits disposition -------


def test_infra_repair_skipped_while_failed_epoch_pending(
    git_repo: Path, run_dir: RunDir, monkeypatch: object
) -> None:
    """A#10: ``_maybe_repair_infra`` must not run on the iteration where a
    ``pending_failed_epoch`` is set (the planner is constrained to
    handle_failed_epoch). The two repair mechanisms are mutually exclusive, so an
    infra-repair can never mutate the integration tip between a semantic failure and
    its disposition. We spy on the entry point and assert it is NEVER entered while
    a failed epoch is pending."""

    import grindstone.run_loop as run_loop

    real_maybe = run_loop._maybe_repair_infra
    pending_at_call: list[bool] = []

    def spy(journal: object, store: object, *args: object, **kwargs: object) -> object:
        pending_at_call.append(store.state.pending_failed_epoch is not None)  # type: ignore[attr-defined]
        return real_maybe(journal, store, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(run_loop, "_maybe_repair_infra", spy)  # type: ignore[attr-defined]

    from tests.grindstone.conftest import (
        OwnershipWorker,
        impl_task,
        implement_decision,
        handle_failed_epoch_retry,
        two_phase_skeleton,
    )

    class _SwapWorker:
        """Fails the first dispatch (no handoff), succeeds the retry (writes file)."""

        def __init__(self) -> None:
            self._calls = 0
            self._ok = OwnershipWorker()

        def run(self, request: WorkerRequest) -> None:
            self._calls += 1
            if self._calls <= 1:
                return  # no handoff.json -> failed attempt -> pending epoch
            self._ok.run(request)

    planner = MockPlanner(
        script=[
            two_phase_skeleton(),
            implement_decision(impl_task("T1", "f1.txt")),  # fails -> pending epoch
            handle_failed_epoch_retry("create the file at repo root"),  # retry -> ok
            complete_decision(check_cmd("test -f f1.txt")),
        ]
    )
    outcome = run_grind(
        run_dir, job_path="job.md", planner=planner, ladder=[("local", _SwapWorker())],
        repo=git_repo, tier0_attempts=1, infra_repair=InfraRepairConfig(attempts=2),
    )
    assert outcome.status == "completed"
    # The disposition really happened (a failed epoch was opened then handled).
    kinds = [e.event for e in read_events(run_dir.events_path)]
    assert "epoch_failed" in kinds
    # The spy was entered at least once, and NEVER while a failed epoch was pending.
    assert pending_at_call  # it did run on the non-pending iterations
    assert not any(pending_at_call)  # but never with pending set (A#10 skip)
