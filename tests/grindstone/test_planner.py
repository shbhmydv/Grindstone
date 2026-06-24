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
from grindstone.loop import CloseoutContext, PlannerContext, TaskOutcome
from tests.grindstone.mock_planner import MockPlannerTransport, MockRig
from grindstone.planner import (
    PlannerError,
    RateLimited,
    ScriptPlanner,
    build_closeout_input,
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


def _raw_context(run_dir: RunDir, *, baton: str = "") -> PlannerContext:
    """A hand-built context for the PURE input test (no repo, no I/O)."""

    return PlannerContext(
        job="# job\nbuild the widget\n",
        repo=None,
        run_dir=run_dir,
        run_branch=None,
        tip_ref="deadbeef",
        log_index=("E1/T1/handoff.md", "E1/T2/report.md"),
        baton=baton,
        epoch_index=3,
        max_epochs=40,
    )


def test_build_planner_input_renders_every_context_field(run_dir: RunDir) -> None:
    ctx = _raw_context(
        run_dir, baton="## Tasks pending\nT2 escalated: missing dep foo\n"
    )
    prompt = build_planner_input(
        ctx,
        domain_skill_index={"rn-taste": "tasteful RN component conventions"},
        reask_errors=("epoch must declare at least one task",),
    )
    # byte-stable PLAN preamble present
    assert "You are the planner for Grindstone" in prompt
    assert "proposes, state machine disposes." in prompt
    # the job (volatile tail)
    assert "build the widget" in prompt
    # the running state + epoch counter
    assert "epoch 3 of at most 40" in prompt
    # the keyed-log index a task may name as inputs
    assert "E1/T1/handoff.md" in prompt
    assert "E1/T2/report.md" in prompt
    # the prior epoch's BATON (the planner's living memory) is rendered verbatim
    assert "<baton>" in prompt
    assert "T2 escalated: missing dep foo" in prompt
    assert "your memory across this run" in prompt
    # the domain-skill catalogue INDEX to select from
    assert "rn-taste" in prompt and "tasteful RN component conventions" in prompt
    # the grep + read read-tools note (the planner pulls; nothing is pushed)
    assert "GREP and READ it to ground your plan" in prompt
    # the re-ask feedback
    assert "epoch must declare at least one task" in prompt


def test_planner_core_reframes_setup_as_host_global(run_dir: RunDir) -> None:
    # setup is HOST-GLOBAL prep only; project-local dep installs must NOT be declared as
    # setup (they would not reach the isolated task worktrees).
    prompt = build_planner_input(_raw_context(run_dir), domain_skill_index={})
    low = prompt.lower()
    assert "host-global" in low
    assert "installs in setup" in low  # named as what NOT to put here
    assert "would not reach" in low


def test_build_planner_input_empty_run_is_clean(run_dir: RunDir) -> None:
    ctx = PlannerContext(
        job="x", repo=None, run_dir=run_dir, run_branch=None, tip_ref=None,
        log_index=(), baton="", epoch_index=1, max_epochs=5,
    )
    prompt = build_planner_input(ctx, domain_skill_index={})
    assert "(none yet, this is the first epoch)" in prompt  # no baton
    # no catalogue -> the rendered selection block (not the preamble's prose mention)
    assert "Domain skills this target repo provides" not in prompt
    assert "<errors>" not in prompt  # no re-ask


# --- decide: the self-validated loop -------------------------------------------


def _context(git_repo: Path, run_dir: RunDir, *, baton: str = "") -> PlannerContext:
    tip_ref = wt.head_commit(git_repo)
    return PlannerContext(
        job="# job\nbuild it\n",
        repo=git_repo,
        run_dir=run_dir,
        run_branch="grind/run-1",
        tip_ref=tip_ref,
        log_index=tuple(run_dir.log_index()),
        baton=baton,
        epoch_index=1,
        max_epochs=5,
    )


# --- pure close-out input construction -----------------------------------------


def _closeout_context(run_dir: RunDir) -> CloseoutContext:
    return CloseoutContext(
        job="# job\nbuild the widget\n",
        repo=None,
        run_dir=run_dir,
        staging_ref="cafef00d",
        prior_baton="## Project summary\nthe widget is half built\n",
        epoch_index=3,
        epoch_id="E3",
        title="finish the widget",
        task_outcomes=(
            TaskOutcome(
                task_id="E3/T1", mode="implement", outcome="passed",
                handoff_key="E3/T1/handoff.md", verdict_key="E3/T1/verdict.json",
                reason="good enough to build on",
            ),
            TaskOutcome(
                task_id="E3/T2", mode="implement", outcome="escalated",
                handoff_key="E3/T2/handoff.md", verdict_key="E3/T2/verdict.json",
                reason="missing dependency the worker cannot install",
            ),
        ),
        setup_error=None,
        integration_conflict=None,
    )


def test_build_closeout_input_renders_report_and_skeleton(run_dir: RunDir) -> None:
    prompt = build_closeout_input(
        _closeout_context(run_dir),
        domain_skill_index={"rn-taste": "tasteful RN component conventions"},
    )
    # the close-out preamble + the four-section baton skeleton it carries
    assert "closing out the epoch you just ran" in prompt
    for section in ("## Project summary", "## Tasks done", "## Tasks pending",
                    "## Current status"):
        assert section in prompt
    # the job + the prior baton (the planner's memory to reconcile against)
    assert "build the widget" in prompt
    assert "the widget is half built" in prompt
    # the per-task epoch report: ids, deterministic outcomes, keyed-log pointers, reason
    assert "E3/T1 (implement): passed" in prompt
    assert "E3/T2 (implement): escalated" in prompt
    assert "E3/T1/handoff.md" in prompt and "E3/T2/verdict.json" in prompt
    assert "missing dependency the worker cannot install" in prompt
    # the request lands the baton on disk, free-form (NEVER parsed)
    assert "./baton.md" in prompt
    # the domain catalogue is offered (the pending list can name a skill to select next)
    assert "rn-taste" in prompt


def test_build_closeout_input_surfaces_setup_and_conflict(run_dir: RunDir) -> None:
    ctx = _closeout_context(run_dir)
    ctx = CloseoutContext(
        job=ctx.job, repo=ctx.repo, run_dir=ctx.run_dir, staging_ref=ctx.staging_ref,
        prior_baton="", epoch_index=ctx.epoch_index, epoch_id=ctx.epoch_id,
        title=ctx.title, task_outcomes=(), setup_error="`npm ci`: exit 1: ENOENT",
        integration_conflict="ownership overlap: T1 and T2 both own shared.py",
    )
    prompt = build_closeout_input(ctx)
    assert "(none, first epoch)" in prompt  # empty prior baton
    assert "setup_error: `npm ci`: exit 1: ENOENT" in prompt
    assert "ownership overlap: T1 and T2 both own shared.py" in prompt
    assert "(no tasks ran this epoch)" in prompt


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


# --- close_out: the free-form baton, read back by channel priority --------------


def _live_closeout_context(git_repo: Path, run_dir: RunDir) -> CloseoutContext:
    return CloseoutContext(
        job="# job\nbuild it\n",
        repo=git_repo,
        run_dir=run_dir,
        staging_ref=wt.head_commit(git_repo),
        prior_baton="",
        epoch_index=1,
        epoch_id="E1",
        title="build",
        task_outcomes=(
            TaskOutcome(
                task_id="E1/T1", mode="implement", outcome="passed",
                handoff_key="E1/T1/handoff.md", verdict_key="E1/T1/verdict.json",
                reason="ok",
            ),
        ),
        setup_error=None,
        integration_conflict=None,
    )


def test_close_out_returns_baton_free_form(git_repo: Path, run_dir: RunDir) -> None:
    # The baton is FREE-FORM (never parsed): close_out checks out the staging tree, runs
    # the rig, and returns whatever prose it produced (here via the --out channel).
    baton = "## Project summary\nbuilt the thing\n## Tasks done\n- E1/T1 passed\n"
    planner = ScriptPlanner(MockPlannerTransport([MockRig(out=baton)]))
    result = planner.close_out(_live_closeout_context(git_repo, run_dir))
    assert result == baton
    # close_out grounds in the SAME in-repo planner-tip worktree decide uses.
    assert (run_dir.root / "_planner_tip").is_dir()


def test_close_out_rate_limit_propagates(git_repo: Path, run_dir: RunDir) -> None:
    # A rate limit propagates (node #1): the loop razes + restarts the epoch.
    planner = ScriptPlanner(MockPlannerTransport(["rate_limit"]))
    with pytest.raises(RateLimited):
        planner.close_out(_live_closeout_context(git_repo, run_dir))
