"""Stochastic convergence + invariant E2E, and the rubber-stamp safety net.

BONES "stochastic-first testing": the bugs that cost us three days were EMERGENT
(a worker that stripped its CWD into the live repo, two local tasks contending one
GPU slot, a critic fumbling a rigid verdict), and a wall of unit tests caught none.
The only thing that catches a rubber-stamp critic or a flaky worker is to RUN a job
many times and assert it CONVERGES to a clean terminal while the invariants hold
EVERY time. So:

PART A (deterministic via SEEDED randomness, no real rigs): a goal-driven planner
plans a realistic multi-epoch job (research -> implement fan-out with disjoint
ownership -> review -> end) and a seeded stochastic worker injects per-task
PASS / RETRY-then-pass / FAILED / BLOCKED / rate-limit outcomes. Over many seeds we
assert the invariants: the run TERMINATES within the backstop (never hangs); the
durable run branch only ever FAST-FORWARDS (its tip is a clean checkpoint at every
boundary); NO worktree leaks into the operator checkout; the keyed log + events stay
consistent; the same seed REPRODUCES (no forbidden global randomness); and a KILL
injected at a random boundary RESUMES and still converges.

PART B (the safety net): a run where the worker's output FAILS the job-level
done_when AND the critic RUBBER-STAMPS every task. The one deterministic
final-acceptance invariant must catch it: the run ENDS, not completes. Plus: an
honest critic's RETRY re-runs the worker and ESCALATE surfaces to the planner as
carried context.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest

from grindstone import worktree as wt
from grindstone.contracts.models import (
    EndDecision,
    Epoch,
    EpochDecision,
    Task,
    parse_handoff,
)
from grindstone.loop import (
    AcceptanceCheck,
    PlannerContext,
    RunResult,
    make_acceptance,
    resume_run,
    start_run,
)
from grindstone.mock_planner import GoalPlanner, MockDecisionPlanner
from grindstone.mock_worker import (
    CrashingWorker,
    LoopWorker,
    SimulatedKill,
    StochasticWorker,
)
from grindstone.rundir import RunDir, create_run_dir
from grindstone.worker import Backends, WorkerRequest

# The goal the GoalPlanner steers toward: three modules built via implement fan-out.
IMPL_FILES: tuple[str, ...] = ("mod_a.py", "mod_b.py", "mod_c.py")
MAX_EPOCHS = 25
#: A short task-id, the form every journal task event carries (P*/E*/T* is reduced
#: to its trailing T-segment for the per-epoch grouping).
_SHORT_TASK_ID = re.compile(r"^T[1-8]$")


@pytest.fixture
def job_path(tmp_path: Path) -> Path:
    p = tmp_path / "job.md"
    p.write_text(
        "# job\nresearch, build three small modules, review.\n", encoding="utf-8"
    )
    return p


def _all_files_built(impl_files: tuple[str, ...]) -> AcceptanceCheck:
    """The job's own final acceptance (invariant #2): the run is COMPLETED only if
    every target module landed on the integration tip, else the planner's END is a
    clean partial-end. A pure tip-tree check (no shell), so the stochastic sweep
    stays fast + deterministic."""

    def _check(context: PlannerContext) -> bool:
        if context.repo is None or context.tip_ref is None:
            return False
        tree = set(wt.list_tree(context.repo, context.tip_ref))
        return all(f in tree for f in impl_files)

    return _check


def _no_sleep(_seconds: float) -> None:
    """An injected ``sleep_fn`` so a rate-limit park never burns real wall-clock."""


# --- the invariant battery (asserted after EVERY stochastic run) ----------------


def _assert_terminates(result: RunResult) -> None:
    """Invariant: the run reached a clean terminal within the epoch backstop."""

    assert result.status in ("completed", "ended")
    assert 0 <= result.epochs <= MAX_EPOCHS


def _assert_run_branch_fast_forwards(
    git_repo: Path, run_branch: str, tip_history: list[str | None]
) -> None:
    """Invariant: the durable run branch ONLY ever fast-forwards, so its tip is a
    clean checkpoint at every boundary.

    The planner records the integration tip it saw at each boundary; each must be an
    ancestor of the next (a forward move or a hold, NEVER a rewind or a sideways
    replace). A non-implement epoch holds the tip; a passing implement epoch advances
    it; nothing ever moves it backward."""

    seen = [t for t in tip_history if t is not None]
    for prev, cur in zip(seen, seen[1:]):
        assert wt.is_ancestor(git_repo, prev, cur), (
            f"run branch tip moved non-forward: {prev[:8]} is not an ancestor of "
            f"{cur[:8]}"
        )
    if wt.branch_exists(git_repo, run_branch) and seen:
        # The live branch tip is the last boundary tip or a descendant of it.
        assert wt.is_ancestor(git_repo, seen[-1], wt.resolve_commit(git_repo, run_branch))


def _registered_worktrees(git_repo: Path) -> list[Path]:
    out = subprocess.run(
        ["git", "worktree", "list", "--porcelain"],
        cwd=str(git_repo), capture_output=True, text=True, check=True,
    ).stdout
    return [
        Path(line[len("worktree ") :])
        for line in out.splitlines()
        if line.startswith("worktree ")
    ]


def _assert_no_worktree_leak(
    git_repo: Path, run_dir: RunDir, impl_files: tuple[str, ...]
) -> None:
    """Invariant: nothing is written into the operator checkout, and the run's
    throwaway worktrees are razed.

    The escape-proofing lesson (run 124321Z): a worker must never reach the operator
    repo. So no built module appears in the operator working tree (the work lives
    ONLY on the run branch / external worktrees), the operator tree is clean, and no
    task worktree remains registered under the external base."""

    for f in impl_files:
        assert not (git_repo / f).exists(), f"built module {f} leaked into the operator checkout"
    porcelain = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=str(git_repo), capture_output=True, text=True, check=True,
    ).stdout
    assert porcelain.strip() == "", f"operator working tree is dirty:\n{porcelain}"
    base = run_dir.worktrees_root
    leaked = [w for w in _registered_worktrees(git_repo) if base in w.parents or w == base]
    assert leaked == [], f"throwaway worktrees survived: {leaked}"


def _assert_log_and_events_consistent(run_dir: RunDir, result: RunResult) -> None:
    """Invariant: the keyed log + the events journal stay consistent, with no
    half-relocated artifact and only well-formed task ids."""

    from grindstone.events import read_events, replay  # local: avoid a heavy top import

    tree = replay(read_events(run_dir.events_path))
    assert tree.status == result.status
    for event in read_events(run_dir.events_path):
        tid = getattr(event, "task_id", None)
        if tid is not None:
            assert _SHORT_TASK_ID.match(tid), f"malformed task id in journal: {tid!r}"
    # Every relocated handoff parses (no torn / half-written record), and every
    # published artifact is a non-empty file (no half-relocated deliverable).
    for handoff in run_dir.root.rglob("P*/E*/T*/handoff.json"):
        parse_handoff(json.loads(handoff.read_text(encoding="utf-8")))
    for key in run_dir.log_index():
        if key.endswith((".md",)):
            assert run_dir.resolve(key).stat().st_size > 0, f"empty artifact {key}"


def _assert_invariants(
    *,
    result: RunResult,
    git_repo: Path,
    run_dir: RunDir,
    planner: GoalPlanner,
    impl_files: tuple[str, ...] = IMPL_FILES,
) -> None:
    run_branch = f"grind/{run_dir.root.name}"
    _assert_terminates(result)
    _assert_run_branch_fast_forwards(git_repo, run_branch, planner.tip_history)
    _assert_no_worktree_leak(git_repo, run_dir, impl_files)
    _assert_log_and_events_consistent(run_dir, result)


# --- A. seeded stochastic convergence ------------------------------------------


@pytest.mark.parametrize("seed", range(30))
def test_stochastic_run_converges_and_holds_invariants(
    seed: int, git_repo: Path, run_dir: RunDir, job_path: Path
) -> None:
    """A full multi-epoch job under a seeded stochastic worker terminates cleanly,
    and every invariant holds, for each of many seeds. ``rate_limit_once`` parks the
    very first dispatch (the node-#1 restart path), then the run proceeds."""

    planner = GoalPlanner(impl_files=IMPL_FILES)
    worker = StochasticWorker(seed=seed, rate_limit_once=(seed % 3 == 0))
    result = start_run(
        job_path=job_path, run_dir=run_dir, repo=git_repo, planner=planner,
        backends=Backends.single(worker, slots=4), max_epochs=MAX_EPOCHS,
        acceptance=_all_files_built(IMPL_FILES), sleep_fn=_no_sleep, backoff_s=1.0,
    )
    _assert_invariants(result=result, git_repo=git_repo, run_dir=run_dir, planner=planner)
    # A completed run means acceptance saw all modules on the tip; an ended run is a
    # clean partial-end. Either is correct; the invariants above hold for both.
    if result.status == "completed":
        tip = set(wt.list_tree(git_repo, f"grind/{run_dir.root.name}"))
        assert all(f in tip for f in IMPL_FILES)


@pytest.mark.parametrize("seed", [1, 7, 13, 21])
def test_same_seed_reproduces(
    seed: int, tmp_path: Path, job_path: Path
) -> None:
    """The same seed yields the same terminal twice (proof the worker seeds a LOCAL
    ``random.Random``, never the forbidden global generator). Two independent repos /
    run dirs, identical seed -> identical status + epoch count."""

    def _once(tag: str) -> RunResult:
        from tests.grindstone.conftest import init_git_repo

        repo = init_git_repo(tmp_path / f"repo-{tag}")
        rd = create_run_dir(repo, f"run-{tag}")
        return start_run(
            job_path=job_path, run_dir=rd, repo=repo,
            planner=GoalPlanner(impl_files=IMPL_FILES),
            backends=Backends.single(StochasticWorker(seed=seed), slots=4),
            max_epochs=MAX_EPOCHS, acceptance=_all_files_built(IMPL_FILES),
            sleep_fn=_no_sleep,
        )

    first, second = _once("a"), _once("b")
    assert (first.status, first.epochs) == (second.status, second.epochs)


# --- A. resume from a KILL at a random boundary ---------------------------------


@pytest.mark.parametrize("seed", range(12))
def test_resume_from_random_kill_converges(
    seed: int, git_repo: Path, run_dir: RunDir, job_path: Path
) -> None:
    """A hard crash injected mid-run (a simulated kill) propagates out of
    ``start_run``; ``resume_run`` razes the in-flight epoch and re-plans from the last
    clean boundary, and the resumed run still converges with every invariant intact.

    The crash lands on an early worker dispatch (1..3), so it fires before the small
    job can finish, exercising both a kill inside the research epoch and a kill inside
    the implement fan-out across the seed sweep."""

    crash_on = 1 + (seed % 3)
    crashing = CrashingWorker(inner=StochasticWorker(seed=seed), crash_on=crash_on)
    with pytest.raises(SimulatedKill, match="simulated kill"):
        start_run(
            job_path=job_path, run_dir=run_dir, repo=git_repo,
            planner=GoalPlanner(impl_files=IMPL_FILES),
            backends=Backends.single(crashing, slots=4), max_epochs=MAX_EPOCHS,
            acceptance=_all_files_built(IMPL_FILES), sleep_fn=_no_sleep,
        )

    # Resume with a FRESH planner + a non-crashing worker: it re-derives its position
    # from disk (the durable keyed log + the run-branch tip) and finishes the job.
    resumed = GoalPlanner(impl_files=IMPL_FILES)
    result = resume_run(
        run_dir=run_dir, repo=git_repo, planner=resumed,
        backends=Backends.single(StochasticWorker(seed=seed), slots=4),
        max_epochs=MAX_EPOCHS, acceptance=_all_files_built(IMPL_FILES),
        sleep_fn=_no_sleep,
    )
    _assert_invariants(result=result, git_repo=git_repo, run_dir=run_dir, planner=resumed)


# --- B. the rubber-stamp / safety-net test --------------------------------------


def test_rubber_stamp_critic_is_caught_by_final_acceptance(
    git_repo: Path, run_dir: RunDir, job_path: Path
) -> None:
    """The critic cannot be fully trusted, so the ONE deterministic gate is the real
    safety net. A worker produces work that does NOT satisfy the job-level done_when
    and a RUBBER-STAMP critic passes every task; the run must END (not complete),
    because the single final acceptance (invariant #2) re-checks done_when once.

    The worker happily builds ``built.py`` and the critic always PASSes, so the epoch
    integrates and the planner ENDs claiming success, but the job's done_when demands
    a DIFFERENT file that was never produced. Only the deterministic gate stands
    between a lying pipeline and a false 'completed'."""

    planner = MockDecisionPlanner(
        [
            _epoch_impl("built.py"),
            _end("everything looks done to me"),
        ]
    )
    rubber_stamp = LoopWorker(critic_outcome="PASS")
    result = start_run(
        job_path=job_path, run_dir=run_dir, repo=git_repo, planner=planner,
        backends=Backends.single(rubber_stamp, slots=2),
        # The job's own acceptance: it requires a file the worker never builds.
        acceptance=make_acceptance("test -f required_other.py"),
    )
    assert result.status == "ended", "a rubber-stamp critic must not yield a 'completed' run"
    # The work the critic waved through DID integrate (the gate is the FINAL acceptance,
    # not a per-epoch veto), proving the safety net is the deterministic done_when alone.
    assert "built.py" in wt.list_tree(git_repo, f"grind/{run_dir.root.name}")


# --- B. an honest critic's RETRY / ESCALATE actually routes ---------------------


class _ScriptedCriticWorker:
    """A loop worker that writes valid implement work every attempt but whose CRITIC
    follows a scripted outcome sequence (consumed once per critic dispatch). Models
    an HONEST critic: a RETRY then a PASS proves a retry RE-RUNS the worker; a lone
    ESCALATE proves the route to the planner. Single-task use, so no locking."""

    def __init__(self, critic_outcomes: list[str]) -> None:
        self._critic_outcomes = critic_outcomes
        self._critic_calls = 0

    def run(self, request: WorkerRequest) -> None:
        if request.critic is not None:
            outcome = self._critic_outcomes[
                min(self._critic_calls, len(self._critic_outcomes) - 1)
            ]
            self._critic_calls += 1
            (request.scratch / "verdict.json").write_text(
                json.dumps({"outcome": outcome, "reason": f"honest critic {outcome}"}),
                encoding="utf-8",
            )
            return
        # Always produce valid implement work so the gate passes and the critic runs.
        for rel in request.task.file_ownership:
            (request.scratch / rel).write_text(
                f"# {request.task_id}\nvalue = 1\n", encoding="utf-8"
            )
        (request.scratch / "review.md").write_text("ok\n", encoding="utf-8")
        (request.scratch / "handoff.json").write_text(
            json.dumps(_handoff_for(request)), encoding="utf-8"
        )


def test_honest_critic_retry_re_runs_then_passes(
    git_repo: Path, run_dir: RunDir, job_path: Path
) -> None:
    """An honest critic RETRY re-runs the SAME worker (the bounded same-tier self
    heal), and a subsequent PASS merges the work: the task converges in 2 attempts
    and the run completes with the file integrated."""

    planner = MockDecisionPlanner([_epoch_impl("a.py"), _end("done")])
    worker = _ScriptedCriticWorker(critic_outcomes=["RETRY", "PASS"])
    result = start_run(
        job_path=job_path, run_dir=run_dir, repo=git_repo, planner=planner,
        backends=Backends.single(worker, slots=2),
    )
    assert result.status == "completed"
    assert worker._critic_calls == 2  # the RETRY forced a second worker+critic attempt
    assert "a.py" in wt.list_tree(git_repo, f"grind/{run_dir.root.name}")


def test_honest_critic_escalate_surfaces_to_planner(
    git_repo: Path, run_dir: RunDir, job_path: Path
) -> None:
    """An honest critic ESCALATE routes the task to the planner: it is NOT merged,
    and the NEXT boundary sees it as carried context the planner steers on."""

    planner = MockDecisionPlanner([_epoch_impl("a.py"), _end("steered around it")])
    worker = _ScriptedCriticWorker(critic_outcomes=["ESCALATE"])
    result = start_run(
        job_path=job_path, run_dir=run_dir, repo=git_repo, planner=planner,
        backends=Backends.single(worker, slots=2),
    )
    assert result.status == "completed"  # the planner ended cleanly after the escalate
    # No fast-forward: the escalated work never reached the run branch.
    assert not wt.branch_exists(git_repo, f"grind/{run_dir.root.name}")
    # The escalation surfaced to the planner's next boundary as carried context.
    second = planner.contexts[1]
    assert any("escalated" in c for c in second.carried)


# --- small builders -------------------------------------------------------------


def _epoch_impl(owned: str) -> EpochDecision:
    return EpochDecision(
        kind="epoch",
        epoch=Epoch(
            title="build",
            tasks=[
                Task(
                    id="T1", mode="implement", goal=f"build {owned}",
                    file_ownership=[owned],
                )
            ],
        ),
    )


def _end(summary: str) -> EndDecision:
    return EndDecision(kind="end", summary=summary)


def _handoff_for(request: WorkerRequest) -> dict[str, object]:
    return {
        "schema_version": "1",
        "task_id": request.task_id,
        "status": "DONE",
        "what_changed": [
            {"kind": "file", "ref": f} for f in request.task.file_ownership
        ],
        "resulting_state": "work complete",
        "citations": [{"file": f} for f in request.task.file_ownership],
        "checks": [{"check": "self-check", "exit_code": 0}],
        "occupancy": {"compacted": False, "subagent_splits": 0},
    }
