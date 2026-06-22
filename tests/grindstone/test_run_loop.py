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

    def plan(self, prompt: str, *, workdir: Path | None = None) -> str:
        self.prompts.append(prompt)
        return self.inner.plan(prompt, workdir=workdir)


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


# --- read-capable <workspace>: tip tree + keyed-log manifest -------------------


def test_planner_workspace_exposes_tip_tree_and_resolvable_manifest(
    git_repo: Path, run_dir: RunDir
) -> None:
    """After an implement epoch the planner boundary must surface a `<workspace>`
    block: a checked-out integration-tip tree (INSIDE the run dir, hence inside the
    repo so both planner rigs can read it) and a manifest resolving each live log
    key to a real file. The tip worktree is materialized for the boundary and torn
    down after (no leak)."""

    planner = _Recording(
        MockPlanner(
            script=[
                two_phase_skeleton(),
                implement_decision(impl_task("T1", "f1.txt")),
                complete_decision(check_cmd("test -f f1.txt")),
            ]
        )
    )
    outcome = run_grind(
        run_dir, job_path="job.md", planner=planner, ladder=_ladder(), repo=git_repo
    )
    assert outcome.status == "completed"
    # The complete_run boundary (the 3rd call) ran AFTER an epoch produced an
    # integration tip + a handoff log key, so its prompt carries the full workspace.
    final_prompt = planner.prompts[-1]
    assert "<workspace>" in final_prompt
    assert "keyed_log_root" in final_prompt
    # The keyed-log root is the run dir, INSIDE the target repo (codex -C repo and
    # claude cwd=repo both reach it); never a /tmp path outside the sandbox.
    run_root = run_dir.root.resolve()
    assert str(run_root) in final_prompt
    assert str(run_root).startswith(str(git_repo.resolve()))
    # The integration-tip tree is a real checked-out dir under the run dir.
    assert "integration_tip" in final_prompt
    import re

    m = re.search(r"integration_tip[^:]*: (\S+)", final_prompt)
    assert m is not None
    tip_path = Path(m.group(1))
    assert str(tip_path).startswith(str(run_root))  # inside the run dir / repo
    # The manifest resolves the handoff log key to an absolute path that EXISTS.
    handoff_keys = [k for k in run_dir.log_index() if k.endswith("handoff.json")]
    assert handoff_keys, "the implement epoch must have produced a handoff log key"
    handoff_key = handoff_keys[0]
    assert handoff_key in final_prompt
    assert str(run_dir.resolve(handoff_key)) in final_prompt
    assert run_dir.resolve(handoff_key).is_file()
    # No tip-tree worktree leaked after the run (lifecycle cleaned up).
    leftover = list((run_dir.root / "worktrees").glob("_planner_tip*"))
    assert leftover == []


def test_first_planner_call_grinds_in_head_worktree_for_self_validation(
    git_repo: Path, run_dir: RunDir
) -> None:
    """The FIRST epoch boundary has no integration branch yet, but the planner
    still needs a writable worktree to self-validate its decision in (a real
    dogfood halt was on that first research epoch). The workspace falls back to a detached
    HEAD checkout under ``_planner_tip``; the prompt exposes it and the keyed-log
    root, never a literal None."""

    planner = _Recording(
        MockPlanner(script=[two_phase_skeleton(), complete_decision(check_cmd("true"))])
    )
    run_grind(run_dir, job_path="job.md", planner=planner, ladder=_ladder(), repo=git_repo)
    first = planner.prompts[0]
    assert "<workspace>" in first
    assert "_planner_tip" in first  # the writable HEAD worktree is exposed
    assert "integration_tip: None" not in first
    assert "none yet" not in first
    # The lifecycle still tears the worktree down (no leak after the run).
    assert list((run_dir.root / "worktrees").glob("_planner_tip*")) == []


# --- repo-map source: the integration TIP, not the stale operator tree ---------


def test_planner_repo_map_reads_integration_tip_not_operator_tree(
    git_repo: Path, run_dir: RunDir, monkeypatch
) -> None:
    """The planner's structural repo-map must reflect what the epochs have BUILT,
    not the repo as it stood at run start. The operator working tree is never
    advanced to the integration tip (it stays out of bounds); the tip lives only as
    a branch ref. So the boundary must build the map from the tip CHECKOUT.

    Drives one implement epoch that integrates ``f1.txt`` onto the tip (a file that
    never appears in the operator tree), then asserts the post-epoch boundary builds
    its map from a checkout that CONTAINS ``f1.txt`` (the tip), and that that source
    is NOT the operator tree (which still lacks ``f1.txt``)."""

    import grindstone.run_loop as rl

    seen: list[Path] = []
    real = rl.build_repo_map

    def _spy(repo_root, **kw):
        seen.append(Path(repo_root))
        return real(repo_root, **kw)

    monkeypatch.setattr(rl, "build_repo_map", _spy)

    planner = MockPlanner(
        script=[
            two_phase_skeleton(),
            implement_decision(impl_task("T1", "f1.txt")),
            complete_decision(check_cmd("test -f f1.txt")),
        ]
    )
    outcome = run_grind(
        run_dir, job_path="job.md", planner=planner, ladder=_ladder(), repo=git_repo
    )
    assert outcome.status == "completed"

    # The operator tree was never advanced: f1.txt lives only on the tip branch.
    assert not (git_repo / "f1.txt").exists()
    assert outcome.final_branch is not None
    assert "f1.txt" in tracked_files(git_repo, outcome.final_branch)

    # Every boundary builds its map from a dedicated _planner_tip checkout under the
    # run dir, NEVER the operator tree (which still lacks f1.txt): the first boundary
    # from a detached HEAD, each later one from the integration tip that CONTAINS
    # f1.txt. The checkout is torn down after the boundary, so we assert via the
    # recorded path's relation, not its survival.
    assert seen, "build_repo_map must be called at every boundary"
    for src in seen:
        assert "_planner_tip" in str(src), src
        assert str(src).startswith(str(run_dir.root.resolve())), src
        assert src != git_repo and src != git_repo.resolve()


def test_repo_map_written_to_run_dir_and_referenced_in_workspace(
    git_repo: Path, run_dir: RunDir, monkeypatch
) -> None:
    """The structural map is delivered BY REFERENCE: when a map is built (>=
    MIN_FILES_FOR_MAP on the tip), the boundary WRITES it to a stable file under the
    run dir ROOT (it must survive the _planner_tip worktree teardown) and the
    <workspace> block points the planner at that path, never inlining the map text.

    The map source is stubbed to a fixed non-None string so the test does not need a
    50-file fixture; the file-write + manifest-reference wiring is what is asserted."""

    import grindstone.run_loop as rl

    monkeypatch.setattr(rl, "build_repo_map", lambda *a, **k: "util.py:\n  def helper():")

    planner = _Recording(
        MockPlanner(
            script=[
                two_phase_skeleton(),
                implement_decision(impl_task("T1", "f1.txt")),
                complete_decision(check_cmd("test -f f1.txt")),
            ]
        )
    )
    outcome = run_grind(
        run_dir, job_path="job.md", planner=planner, ladder=_ladder(), repo=git_repo
    )
    assert outcome.status == "completed"

    map_file = run_dir.root / "planner_repo_map.txt"
    assert map_file.is_file(), "the map must be written under the run dir root"
    assert "def helper():" in map_file.read_text()
    # The map file lives at the run dir root, NOT inside the torn-down tip worktree.
    assert "_planner_tip" not in str(map_file)
    assert not list((run_dir.root / "worktrees").glob("_planner_tip*"))

    final_prompt = planner.prompts[-1]
    assert "<workspace>" in final_prompt
    # The labeled workspace entry + the PATH to the on-disk map are surfaced; the map
    # TEXT is not inlined and no inline <repo_map> block exists.
    assert "repo_map (" in final_prompt
    assert str(map_file.resolve()) in final_prompt
    assert "<repo_map>" not in final_prompt
    assert "def helper():" not in final_prompt


def test_repo_map_below_threshold_writes_no_file_and_omits_entry(
    git_repo: Path, run_dir: RunDir, monkeypatch
) -> None:
    """Below threshold / first epoch the map is None -> no file is written and the
    <workspace> omits the repo_map entry cleanly (no stale file, no dangling path)."""

    import grindstone.run_loop as rl

    monkeypatch.setattr(rl, "build_repo_map", lambda *a, **k: None)

    planner = _Recording(
        MockPlanner(script=[two_phase_skeleton(), complete_decision(check_cmd("true"))])
    )
    run_grind(run_dir, job_path="job.md", planner=planner, ladder=_ladder(), repo=git_repo)

    assert not (run_dir.root / "planner_repo_map.txt").exists()
    # No repo_map manifest entry and no inline map block in any prompt. (Guard against
    # the run-dir path itself containing "repo_map" via the tmp test-name directory:
    # assert on the labeled line / the path to the map FILE, not a bare substring.)
    for prompt in planner.prompts:
        assert "planner_repo_map.txt" not in prompt
        assert "<repo_map>" not in prompt


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
