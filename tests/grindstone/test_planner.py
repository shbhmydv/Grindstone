"""The real planner: pure input construction + the self-validated ``decide`` loop.

Drives ``ScriptPlanner.decide`` through a mock transport (no real rig): the read
priority (``decision.json`` > ``--out`` > stdout), the re-ask loop (invalid then
valid), the two-node failure mapping (``RateLimited`` / ``PlannerError``), and that a
valid epoch and a valid end each parse to the right typed ``Decision``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from grindstone import worktree as wt
from grindstone.contracts.models import EndDecision, EpochDecision
from grindstone.loop import PlannerContext
from grindstone.mock_planner import MockPlannerTransport, MockRig
from grindstone.planner import (
    PlannerError,
    RateLimited,
    ScriptPlanner,
    build_planner_input,
)
from grindstone.rundir import RunDir

# --- decision payloads ---------------------------------------------------------

_EPOCH: dict[str, object] = {
    "kind": "epoch",
    "epoch": {
        "title": "groundwork",
        "tasks": [
            {"id": "T1", "mode": "implement", "goal": "build a.py", "file_ownership": ["a.py"]}
        ],
    },
}
_END: dict[str, object] = {"kind": "end", "summary": "job complete"}
#: A decision-shaped object that fails the core gate (an epoch with no title/tasks).
_INVALID: dict[str, object] = {"kind": "epoch", "epoch": {}}


def _rig(decision: dict[str, object], *, channel: str = "decision_json") -> MockRig:
    return MockRig.from_decision(decision, channel=channel)


# --- pure input construction ---------------------------------------------------


def _raw_context(run_dir: RunDir) -> PlannerContext:
    """A hand-built context for the PURE input test (no repo, no I/O)."""

    return PlannerContext(
        job="# job\nbuild the widget\n",
        repo=None,
        run_dir=run_dir,
        run_branch=None,
        tip_ref="deadbeef",
        tip_files=("README.md", "src/app.py"),
        log_index=("P1/E1/T1/handoff.json", "P1/E1/T2/report.md"),
        carried=("T2 BLOCKED: missing dep foo",),
        epoch_index=3,
        max_epochs=40,
    )


def test_build_planner_input_renders_every_context_field(run_dir: RunDir) -> None:
    ctx = _raw_context(run_dir)
    prompt = build_planner_input(
        ctx,
        domain_skill_index={"rn-taste": "tasteful RN component conventions"},
        reask_errors=("epoch must declare at least one task",),
    )
    # byte-stable CORE present
    assert "You are the planner for Grindstone" in prompt
    assert "Model proposes, state machine disposes." in prompt
    # the job (volatile tail)
    assert "build the widget" in prompt
    # the running state + epoch counter
    assert "epoch 3 of at most 40" in prompt
    # the keyed-log index a task may name as inputs
    assert "P1/E1/T1/handoff.json" in prompt
    assert "P1/E1/T2/report.md" in prompt
    # the integration-tip file list
    assert "README.md" in prompt and "src/app.py" in prompt
    # the prior epoch's carried failures to steer around
    assert "T2 BLOCKED: missing dep foo" in prompt
    # the domain-skill catalogue INDEX to select from
    assert "rn-taste" in prompt and "tasteful RN component conventions" in prompt
    # the repo-map + grep read-tools note
    assert "repomap.py" in prompt
    # the re-ask feedback
    assert "epoch must declare at least one task" in prompt


def test_planner_core_reframes_setup_as_host_global(run_dir: RunDir) -> None:
    # FIX 3: setup is HOST-GLOBAL prep only; project-local dep installs must NOT be
    # declared as setup (they would not reach the isolated task worktrees).
    prompt = build_planner_input(_raw_context(run_dir), domain_skill_index={})
    low = prompt.lower()
    assert "host-global" in low
    assert "npm ci" in low and "pip install" in low  # named as what NOT to put here
    assert "would not reach" in low or "do not put" in low


def test_build_planner_input_empty_run_is_clean(run_dir: RunDir) -> None:
    ctx = PlannerContext(
        job="x", repo=None, run_dir=run_dir, run_branch=None, tip_ref=None,
        tip_files=(), log_index=(), carried=(), epoch_index=1, max_epochs=5,
    )
    prompt = build_planner_input(ctx, domain_skill_index={})
    assert "(none, this is the first epoch)" in prompt  # no carried
    # no catalogue -> the rendered selection block (not the CORE's prose mention)
    assert "Domain skills this target repo provides" not in prompt
    assert "<errors>" not in prompt  # no re-ask


# --- decide: the self-validated loop -------------------------------------------


def _context(git_repo: Path, run_dir: RunDir, *, carried: tuple[str, ...] = ()) -> PlannerContext:
    tip_ref = wt.head_commit(git_repo)
    return PlannerContext(
        job="# job\nbuild it\n",
        repo=git_repo,
        run_dir=run_dir,
        run_branch="grind/run-1",
        tip_ref=tip_ref,
        tip_files=tuple(wt.list_tree(git_repo, tip_ref)),
        log_index=tuple(run_dir.log_index()),
        carried=carried,
        epoch_index=1,
        max_epochs=5,
    )


def test_decide_accepts_a_valid_epoch_on_disk(git_repo: Path, run_dir: RunDir) -> None:
    planner = ScriptPlanner(MockPlannerTransport([_rig(_EPOCH)]))
    decision = planner.decide(_context(git_repo, run_dir))
    assert isinstance(decision, EpochDecision)
    assert decision.epoch.title == "groundwork"
    assert decision.epoch.tasks[0].id == "T1"
    # the on-disk validator was armed in the in-repo planner tip (self-validation).
    assert (run_dir.root / "_planner_tip" / "check_decision.py").is_file()


def test_decide_accepts_a_valid_end(git_repo: Path, run_dir: RunDir) -> None:
    planner = ScriptPlanner(MockPlannerTransport([_rig(_END)]))
    decision = planner.decide(_context(git_repo, run_dir))
    assert isinstance(decision, EndDecision)
    assert decision.summary == "job complete"


def test_decide_re_asks_on_invalid_then_accepts(git_repo: Path, run_dir: RunDir) -> None:
    # first dispatch lands a gate-invalid decision.json; the loop re-asks and the
    # second dispatch lands a valid one.
    planner = ScriptPlanner(MockPlannerTransport([_rig(_INVALID), _rig(_END)]))
    decision = planner.decide(_context(git_repo, run_dir))
    assert isinstance(decision, EndDecision)


def test_decide_read_priority_decision_json_beats_out_and_stdout(
    git_repo: Path, run_dir: RunDir
) -> None:
    # decision.json (the real self-validate proof) wins over a different --out/stdout.
    rig = MockRig(
        decision_json=json.dumps(_EPOCH),
        out=json.dumps(_END),
        stdout=json.dumps(_END),
    )
    planner = ScriptPlanner(MockPlannerTransport([rig]))
    decision = planner.decide(_context(git_repo, run_dir))
    assert isinstance(decision, EpochDecision)  # not the end on --out/stdout


def test_decide_read_priority_out_beats_stdout(git_repo: Path, run_dir: RunDir) -> None:
    rig = MockRig(decision_json=None, out=json.dumps(_END), stdout="garbage, no object")
    planner = ScriptPlanner(MockPlannerTransport([rig]))
    decision = planner.decide(_context(git_repo, run_dir))
    assert isinstance(decision, EndDecision)


def test_decide_falls_back_to_stdout(git_repo: Path, run_dir: RunDir) -> None:
    rig = MockRig(stdout=f"reasoning... {json.dumps(_END)} done")
    planner = ScriptPlanner(MockPlannerTransport([rig]))
    decision = planner.decide(_context(git_repo, run_dir))
    assert isinstance(decision, EndDecision)  # extracted from a prose-wrapped stdout


# --- decide: the two-node failure mapping --------------------------------------


def test_decide_rate_limit_maps_to_rate_limited(git_repo: Path, run_dir: RunDir) -> None:
    planner = ScriptPlanner(MockPlannerTransport(["rate_limit"]))
    with pytest.raises(RateLimited):
        planner.decide(_context(git_repo, run_dir))


def test_decide_transport_error_maps_to_planner_error(git_repo: Path, run_dir: RunDir) -> None:
    planner = ScriptPlanner(MockPlannerTransport(["error"]))
    with pytest.raises(PlannerError):
        planner.decide(_context(git_repo, run_dir))


def test_decide_exhausted_reasks_map_to_planner_error(git_repo: Path, run_dir: RunDir) -> None:
    # every attempt lands an invalid decision; the budget exhausts to PlannerError
    # (NOT RateLimited: a bad decision is not a rate limit).
    transport = MockPlannerTransport([_rig(_INVALID), _rig(_INVALID), _rig(_INVALID)])
    planner = ScriptPlanner(transport, max_reasks=2)
    with pytest.raises(PlannerError) as excinfo:
        planner.decide(_context(git_repo, run_dir))
    assert not isinstance(excinfo.value, RateLimited)


def test_decide_reuses_the_planner_tip_across_boundaries(git_repo: Path, run_dir: RunDir) -> None:
    # two boundaries at the SAME tip reuse the one in-repo worktree (refreshed only
    # when the tip moves).
    planner = ScriptPlanner(MockPlannerTransport([_rig(_EPOCH), _rig(_END)]))
    ctx = _context(git_repo, run_dir)
    assert isinstance(planner.decide(ctx), EpochDecision)
    assert isinstance(planner.decide(ctx), EndDecision)
    assert (run_dir.root / "_planner_tip").is_dir()
