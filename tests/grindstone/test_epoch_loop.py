"""One epoch end-to-end: fan-out, the done-predicate, ownership-scoped
fast-forward integration, the conflict-aborts-epoch structural guard, partial
epochs, per-task event causality under concurrency, and the EpochOutcome /
outcome.json record (ARCHITECTURE.md / S2 rulings 1, 7, 8, 10).
"""

from __future__ import annotations

import json
from pathlib import Path

from grindstone.contracts.models import CmdCheck, ImplementTask
from grindstone.epoch_loop import (
    EpochState,
    IntegrationState,
    _EpochStateStore,
    _integrate,
    epoch_done_predicate,
)
from grindstone.events import Event, read_events
from grindstone.mock_worker import MockWorker
from grindstone.rundir import RunDir, create_run_dir
from grindstone.task_loop import TaskIdentity, TaskOutcome, pending_cursor
from grindstone import worktree as wt
from grindstone.worker import WorkerTransport

from tests.grindstone.conftest import (
    OUT_CONTENT,
    HandoffWorker,
    RoutingWorker,
    handoff_payload,
    implement_epoch,
    make_toy_task,
    run_one_epoch,
    tracked_files,
)


def _disjoint_tasks(n: int) -> list[ImplementTask]:
    return [
        make_toy_task(task_id=f"T{i}", out_file=f"f{i}.txt", owned=[f"f{i}.txt"])
        for i in range(1, n + 1)
    ]


def _file_workers(n: int) -> RoutingWorker:
    return RoutingWorker(
        {f"T{i}": MockWorker(script=["ok"], artifacts={f"f{i}.txt": OUT_CONTENT}) for i in range(1, n + 1)}
    )


def _run(run_dir: RunDir, repo: Path, tasks: list[ImplementTask], worker: WorkerTransport, **kw: object):
    return run_one_epoch(
        run_dir,
        args=implement_epoch(*tasks),
        mode="implement",
        ladder=[("worker", worker)],
        repo=repo,
        **kw,
    )


# --- fan-out + integration (the happy epoch) -----------------------------------


def test_three_task_epoch_integrates_all(git_repo: Path, run_dir: RunDir) -> None:
    tasks = _disjoint_tasks(3)
    outcome = _run(run_dir, git_repo, tasks, _file_workers(3), concurrency=2)
    assert outcome.status == "completed"
    assert [t.status for t in outcome.tasks] == ["done", "done", "done"]
    assert outcome.integration.status == "completed"
    assert outcome.integration.merged == ["T1", "T2", "T3"]
    branch = outcome.integration.branch
    assert branch is not None
    assert tracked_files(git_repo, branch) == [
        ".gitignore",
        "README.md",
        "f1.txt",
        "f2.txt",
        "f3.txt",
    ]
    # The run branch is the surviving tip; transient grind-wip branches + worktrees
    # are pruned after integration (ruling 7, "once it passes it cleans up + falls").
    assert branch == f"grind/{run_dir.root.name}"
    import subprocess

    refs = subprocess.run(
        ["git", "branch", "--list", f"grind-wip/{run_dir.root.name}/*"],
        cwd=str(git_repo), capture_output=True, text=True,
    ).stdout
    assert refs.strip() == ""
    assert not (run_dir.root / "worktrees" / "T1").exists()


def test_outcome_json_is_written_and_flat(git_repo: Path, run_dir: RunDir) -> None:
    tasks = _disjoint_tasks(2)
    outcome = _run(run_dir, git_repo, tasks, _file_workers(2))
    path = run_dir.resolve("P1/E1/outcome.json")
    assert path.is_file()
    data = json.loads(path.read_text())
    assert data["status"] == "completed"
    assert {t["task_id"] for t in data["tasks"]} == {"T1", "T2"}
    assert data["integration"]["merged"] == ["T1", "T2"]
    assert data["tasks"][0]["handoff_key"] == "P1/E1/T1/handoff.json"


# --- done-predicate ------------------------------------------------------------


def test_done_predicate_holds_at_terminal(git_repo: Path, run_dir: RunDir) -> None:
    _run(run_dir, git_repo, _disjoint_tasks(2), _file_workers(2))
    state = EpochState.model_validate_json(run_dir.state_path.read_text())
    assert epoch_done_predicate(state) is True


def test_done_predicate_false_with_inflight() -> None:
    ident = TaskIdentity("run-1", "P1", "E1", "T1")
    cur = pending_cursor(ident, "implement")
    state = EpochState(
        phase_id="P1",
        epoch_id="E1",
        title="t",
        mode="implement",
        is_implement=True,
        base="abc",
        integration=IntegrationState(branch="b", status="pending", merged=[], conflict=None),
        tasks={"T1": cur},  # status pending -> queue not empty
    )
    assert epoch_done_predicate(state) is False
    running = state.model_copy(
        update={"tasks": {"T1": cur.model_copy(update={"status": "running"})}}
    )
    assert epoch_done_predicate(running) is False
    done = state.model_copy(
        update={"tasks": {"T1": cur.model_copy(update={"status": "done"})}}
    )
    assert epoch_done_predicate(done) is True


# --- run-scoped branch names (cross-run collision is structurally impossible) ---


def test_attempt_branch_is_run_scoped() -> None:
    ident = TaskIdentity(run_id="20260101T000000Z", phase_id="P1", epoch_id="E1", task_id="T1")
    assert ident.attempt_branch(1) == "grind-wip/20260101T000000Z/P1/E1/T1-a1"
    # The log-key prefix stays run-agnostic (only branch names carry the run id).
    assert ident.fq == "P1/E1/T1"


def test_run_scoping_isolates_leftover_branch_from_prior_run(git_repo: Path) -> None:
    """The real bug: a leftover ``grind/...`` branch from a prior run must not
    collide when a fresh run (different run id) creates the same phase/epoch/task
    worktree branch. Run-scoping the name makes the create path crash-free.
    """

    base = wt.head_commit(git_repo)
    # A leftover task branch from a PRIOR run id, in the SAME repo.
    old = TaskIdentity(run_id="OLDRUN", phase_id="P1", epoch_id="E1", task_id="T1")
    wt.add_worktree(git_repo, git_repo.parent / "wt-old" / "a1", branch=old.attempt_branch(1), base=base)
    assert wt.branch_exists(git_repo, old.attempt_branch(1))

    # A fresh run with a different run id creates the same phase/epoch/task branch.
    new = TaskIdentity(run_id="NEWRUN", phase_id="P1", epoch_id="E1", task_id="T1")
    assert new.attempt_branch(1) != old.attempt_branch(1)
    # No "branch already exists" crash, because the name is run-scoped.
    wt.add_worktree(git_repo, git_repo.parent / "wt-new" / "a1", branch=new.attempt_branch(1), base=base)
    assert wt.branch_exists(git_repo, new.attempt_branch(1))


def test_run_epoch_branch_names_carry_run_id(git_repo: Path, run_dir: RunDir) -> None:
    """A successful epoch leaves ONLY the persistent run branch ``grind/{run_id}``;
    the transient ``grind-wip/{run_id}/...`` staging + attempt branches are gone.

    A second run (distinct run dir / run id) can never share the run branch with
    this one, and after the epoch nothing under ``grind-wip/`` survives.
    """

    import subprocess

    outcome = _run(run_dir, git_repo, _disjoint_tasks(2), _file_workers(2))
    assert outcome.status == "completed"
    run_id = run_dir.root.name
    # The integration tip is now the persistent RUN branch (the surviving ref).
    assert outcome.integration.branch == f"grind/{run_id}"
    assert wt.branch_exists(git_repo, f"grind/{run_id}")
    # Every transient branch under grind-wip/{run_id}/ is cleaned up after success.
    refs = subprocess.run(
        ["git", "branch", "--list", f"grind-wip/{run_id}/*"],
        cwd=str(git_repo), capture_output=True, text=True,
    ).stdout
    assert refs.strip() == ""


# --- partial epoch (some tasks fail; DONE ones still integrate) -----------------


def test_partial_epoch_integrates_done_tasks(git_repo: Path, run_dir: RunDir) -> None:
    tasks = _disjoint_tasks(3)
    worker = RoutingWorker(
        {
            "T1": MockWorker(script=["ok"], artifacts={"f1.txt": OUT_CONTENT}),
            "T2": MockWorker(script=["empty"]),  # always fails
            "T3": MockWorker(script=["ok"], artifacts={"f3.txt": OUT_CONTENT}),
        }
    )
    outcome = _run(run_dir, git_repo, tasks, worker, concurrency=3, tier0_attempts=1)
    assert outcome.status == "completed"
    by_id = {t.task_id: t.status for t in outcome.tasks}
    assert by_id == {"T1": "done", "T2": "failed", "T3": "done"}
    assert outcome.integration.merged == ["T1", "T3"]
    branch = outcome.integration.branch
    assert branch is not None
    assert tracked_files(git_repo, branch) == [".gitignore", "README.md", "f1.txt", "f3.txt"]


# --- conflict aborts the epoch (structural bug, not retried) --------------------


def test_overlapping_ownership_conflict_aborts_epoch(git_repo: Path, run_dir: RunDir) -> None:
    # Two tasks deliberately share ownership of shared.txt (this would be
    # rejected by the planner-time disjointness validator; here we bypass it to
    # prove the integration conflict path aborts structurally).
    def shared_task(tid: str) -> ImplementTask:
        return ImplementTask(
            id=tid,
            goal=f"{tid} writes shared.txt",
            done_when=[CmdCheck(cmd="test -f shared.txt")],
            file_ownership=["shared.txt"],
        )

    tasks = [shared_task("T1"), shared_task("T2")]
    worker = RoutingWorker(
        {
            "T1": HandoffWorker(
                handoff_payload(task_id="P1/E1/T1", out_file="shared.txt"),
                files={"shared.txt": "from-T1\n"},
            ),
            "T2": HandoffWorker(
                handoff_payload(task_id="P1/E1/T2", out_file="shared.txt"),
                files={"shared.txt": "from-T2\n"},
            ),
        }
    )
    outcome = _run(run_dir, git_repo, tasks, worker, concurrency=1)
    assert outcome.status == "integration_conflict"
    assert outcome.integration.status == "conflict"
    assert outcome.integration.conflict
    kinds = [e.event for e in read_events(run_dir.events_path)]
    assert kinds.count("epoch_completed") == 1
    assert "run_escalated" in kinds
    assert "run_completed" not in kinds


# --- concurrency: per-task event causality, no cross-task ordering asserted -----


def _task_events(events: list[Event], task_id: str) -> list[str]:
    bearing = {
        "task_dispatched",
        "task_retried",
        "task_escalated",
        "task_done",
        "task_failed",
        "handoff_rejected",
    }
    return [e.event for e in events if e.event in bearing and getattr(e, "task_id", None) == task_id]


def test_concurrent_fanout_per_task_causality(git_repo: Path, run_dir: RunDir) -> None:
    tasks = _disjoint_tasks(4)
    worker = RoutingWorker(
        {
            "T1": MockWorker(script=["ok"], artifacts={"f1.txt": OUT_CONTENT}),
            "T2": MockWorker(script=["bad_json", "ok"], artifacts={"f2.txt": OUT_CONTENT}),
            "T3": MockWorker(script=["ok"], artifacts={"f3.txt": OUT_CONTENT}),
            "T4": MockWorker(script=["rate_limit", "empty", "ok"], artifacts={"f4.txt": OUT_CONTENT}),
        }
    )
    outcome = _run(run_dir, git_repo, tasks, worker, concurrency=2)
    assert [t.status for t in outcome.tasks] == ["done"] * 4
    events = read_events(run_dir.events_path)
    # Global seq strictly increasing + unique (the writer lock under concurrency).
    seqs = [e.seq for e in events]
    assert seqs == sorted(seqs)
    assert len(seqs) == len(set(seqs))
    # Per-task ORDER is causal: dispatched first, terminal last, retries climb.
    for tid in ("T1", "T2", "T3", "T4"):
        seq = _task_events(events, tid)
        assert seq[0] == "task_dispatched"
        assert seq.count("task_dispatched") == 1
        assert seq[-1] == "task_done"
        retried = [
            e.attempt
            for e in events
            if e.event == "task_retried" and getattr(e, "task_id", None) == tid
        ]
        assert retried == sorted(retried)


def test_default_concurrency_caps_at_four(git_repo: Path, run_dir: RunDir) -> None:
    # 6 tasks, default concurrency = min(4, n), just assert it runs to completion.
    tasks = _disjoint_tasks(6)
    outcome = _run(run_dir, git_repo, tasks, _file_workers(6))
    assert outcome.status == "completed"
    assert len(outcome.integration.merged) == 6


# --- artifact epoch: no worktree, no integration -------------------------------


def test_artifact_epoch_skips_integration(tmp_path: Path) -> None:
    from grindstone.contracts.models import ArtifactTask
    from tests.grindstone.conftest import artifact_epoch

    # No git repo needed for an artifact epoch.
    run_dir = create_run_dir(tmp_path, "run-art")
    task = ArtifactTask(
        id="T1",
        goal="produce a note",
        done_when=[CmdCheck(cmd="test -f note.md")],
        artifact_out="P1/E1/T1/note.md",
    )
    worker = HandoffWorker(handoff_payload(out_file="note.md"), files={"note.md": "hi"})
    outcome = run_one_epoch(
        run_dir, args=artifact_epoch(task), mode="artifact", ladder=[("worker", worker)]
    )
    assert outcome.status == "completed"
    assert outcome.integration.status == "skipped"
    assert not (run_dir.root / "worktrees").exists()


# --- stale integration branch must not poison a fresh integration ---------------


def _commit_branch_adding(repo: Path, branch: str, base: str, files: dict[str, str]) -> str:
    """Create ``branch`` at ``base``, add ``files``, commit, return the tip sha.

    A throwaway DONE-task branch built without running a worker: a worktree on a
    new branch off ``base``, the files written + committed, the worktree removed.
    """

    import subprocess

    wt_path = repo / ".tmp-build" / branch.replace("/", "_")
    wt.add_worktree(repo, wt_path, branch=branch, base=base)
    for rel, content in files.items():
        (wt_path / rel).write_text(content, encoding="utf-8")
    wt.commit_all(wt_path, f"build {branch}")
    tip = wt.resolve_commit(repo, branch)
    wt.remove_worktree(repo, wt_path)
    subprocess.run(["git", "worktree", "prune"], cwd=str(repo), check=True)
    return tip


def _store_for(run_dir: RunDir, branch: str, base: str, merged: list[str]) -> _EpochStateStore:
    state = EpochState(
        phase_id="P1",
        epoch_id="E1",
        title="t",
        mode="implement",
        is_implement=True,
        base=base,
        integration=IntegrationState(branch=branch, status="pending", merged=merged, conflict=None),
        tasks={},
    )
    return _EpochStateStore(run_dir, state)


def _done_outcome(task_id: str, branch: str) -> TaskOutcome:
    return TaskOutcome(
        identity=TaskIdentity("run-1", "P1", "E1", task_id),
        status="done",
        tier="worker",
        attempts=1,
        handoff=None,
        handoff_key=None,
        branch=branch,
        reason=None,
    )


def test_fresh_integration_drops_stale_same_named_branch(git_repo: Path, run_dir: RunDir) -> None:
    # A crash left a STAGING branch of the SAME name carrying its own package.json.
    # Without the drop, merging this run's fresh task branches into that stale branch
    # yields a phantom both-added conflict on package.json (a file genuinely owned by
    # exactly one task this run). On success the run branch fast-forwards to the
    # staging tip and the staging branch is deleted, so we assert the RUN branch.
    base = wt.head_commit(git_repo)
    staging = "grind-wip/run-1/P1/E1/_staging"
    run_branch = "grind/run-1"
    _commit_branch_adding(git_repo, staging, base, {"package.json": "STALE"})

    t1 = _commit_branch_adding(git_repo, "grind-wip/run-1/P1/E1/T1-a1", base, {"package.json": "FRESH"})
    t2 = _commit_branch_adding(git_repo, "grind-wip/run-1/P1/E1/T2-a1", base, {"app.json": "APP"})
    assert t1 and t2

    store = _store_for(run_dir, staging, base, merged=[])
    outcome = _integrate(
        repo=git_repo,
        run_dir=run_dir,
        store=store,
        branch=staging,
        base=base,
        done_in_order=[
            _done_outcome("T1", "grind-wip/run-1/P1/E1/T1-a1"),
            _done_outcome("T2", "grind-wip/run-1/P1/E1/T2-a1"),
        ],
    )

    assert outcome.status == "completed"
    assert outcome.conflict is None
    assert outcome.merged == ["T1", "T2"]
    # The integration tip is the persistent run branch; the staging branch is gone.
    assert outcome.branch == run_branch
    assert not wt.branch_exists(git_repo, staging)
    # The stale staging branch was dropped and recreated at base: the FRESH
    # package.json wins, app.json is present, the STALE content is gone, all carried
    # onto the run branch.
    import subprocess

    pkg = subprocess.run(
        ["git", "show", f"{run_branch}:package.json"], cwd=str(git_repo), capture_output=True, text=True
    ).stdout
    assert pkg == "FRESH"
    assert tracked_files(git_repo, run_branch) == [".gitignore", "README.md", "app.json", "package.json"]


def test_resume_mid_integration_keeps_existing_branch(git_repo: Path, run_dir: RunDir) -> None:
    # Resume mid-integration: the staging branch already carries T1 (merged
    # non-empty), so _integrate must NOT drop it (that would discard real progress);
    # it merges only the remaining task, keeps T1's content, then fast-forwards the
    # run branch to the staging tip. Idempotent: re-running does not double-merge.
    base = wt.head_commit(git_repo)
    staging = "grind-wip/run-1/P1/E1/_staging"
    run_branch = "grind/run-1"
    t1 = _commit_branch_adding(git_repo, "grind-wip/run-1/P1/E1/T1-a1", base, {"f1.txt": "ONE"})
    t2 = _commit_branch_adding(git_repo, "grind-wip/run-1/P1/E1/T2-a1", base, {"f2.txt": "TWO"})
    assert t1 and t2

    # Materialize the staging branch with T1 already merged into it (the durable
    # state records merged == ["T1"]).
    import subprocess

    int_wt = run_dir.root / "worktrees" / "_seed"
    wt.add_worktree(git_repo, int_wt, branch=staging, base=base)
    merge = wt.merge_into(int_wt, "grind-wip/run-1/P1/E1/T1-a1")
    assert merge.ok
    wt.remove_worktree(git_repo, int_wt)
    subprocess.run(["git", "worktree", "prune"], cwd=str(git_repo), check=True)
    staging_tip_before = wt.resolve_commit(git_repo, staging)

    store = _store_for(run_dir, staging, base, merged=["T1"])
    outcome = _integrate(
        repo=git_repo,
        run_dir=run_dir,
        store=store,
        branch=staging,
        base=base,
        done_in_order=[
            _done_outcome("T1", "grind-wip/run-1/P1/E1/T1-a1"),
            _done_outcome("T2", "grind-wip/run-1/P1/E1/T2-a1"),
        ],
    )

    assert outcome.status == "completed"
    assert outcome.merged == ["T1", "T2"]
    assert outcome.branch == run_branch
    # T1's content survived (the staging branch was NOT reset to base, T1 only merged
    # once) and T2 was added; the run branch carries both. No double-merge of T1.
    assert tracked_files(git_repo, run_branch) == [".gitignore", "README.md", "f1.txt", "f2.txt"]
    # The run tip descends from the seeded T1 staging tip (T2 merged on top), so the
    # prior progress is an ancestor, not discarded and not moved backward.
    assert wt.is_ancestor(git_repo, staging_tip_before, run_branch)


# --- the persistent run branch: fast-forward across epochs, conflict isolation ---


def _run_branch_name(run_dir: RunDir) -> str:
    return f"grind/{run_dir.root.name}"


def test_run_branch_persists_and_fast_forwards_across_epochs(
    git_repo: Path, run_dir: RunDir
) -> None:
    """Across two epochs ONLY ``grind/{run_id}`` survives, and it advances by
    fast-forward: epoch 2 starts its staging branch off the epoch-1 run tip, so the
    epoch-1 tip is an ancestor of the epoch-2 run tip and the run tip equals the
    latest staging tip. No other ``grind*`` ref is left behind."""

    import subprocess

    run_branch = _run_branch_name(run_dir)
    # Epoch 1 (P1/E1): build f1.txt off the operator base.
    base0 = wt.head_commit(git_repo)
    out1 = run_one_epoch(
        run_dir,
        args=implement_epoch(make_toy_task("T1", "f1.txt", ["f1.txt"])),
        mode="implement",
        ladder=[("worker", _file_workers(1))],
        repo=git_repo,
        phase_id="P1",
        epoch_id="E1",
        base=base0,
    )
    assert out1.status == "completed"
    assert out1.integration.branch == run_branch
    tip1 = wt.resolve_commit(git_repo, run_branch)

    # Epoch 2 (P1/E2): base = the CURRENT run tip (what the run loop threads), build
    # f2.txt. The staging branch is created off the run tip, then the run branch FFs.
    worker2 = RoutingWorker({"T2": MockWorker(script=["ok"], artifacts={"f2.txt": OUT_CONTENT})})
    out2 = run_one_epoch(
        run_dir,
        args=implement_epoch(make_toy_task("T2", "f2.txt", ["f2.txt"])),
        mode="implement",
        ladder=[("worker", worker2)],
        repo=git_repo,
        phase_id="P1",
        epoch_id="E2",
        base=tip1,
    )
    assert out2.status == "completed"
    assert out2.integration.branch == run_branch
    tip2 = wt.resolve_commit(git_repo, run_branch)

    # The run branch advanced by a true fast-forward (epoch-1 tip is an ancestor).
    assert tip1 != tip2
    assert wt.is_ancestor(git_repo, tip1, run_branch)
    # The run tip carries BOTH epochs' work (an ancestor chain, nothing dropped).
    assert tracked_files(git_repo, run_branch) == [
        ".gitignore", "README.md", "f1.txt", "f2.txt",
    ]
    # ONLY the run branch survives: no grind-wip/* and no other grind/* refs.
    all_grind = subprocess.run(
        ["git", "branch", "--list", "grind*"], cwd=str(git_repo), capture_output=True, text=True
    ).stdout
    names = sorted(line.strip().lstrip("* ").strip() for line in all_grind.splitlines() if line.strip())
    assert names == [run_branch]


def test_integration_conflict_leaves_run_branch_untouched(
    git_repo: Path, run_dir: RunDir
) -> None:
    """A merge conflict during staging aborts the epoch and leaves the persistent
    run branch UNCHANGED (the conflict never reaches the run branch). Reported as a
    conflict; the run branch tip is identical before and after."""

    run_branch = _run_branch_name(run_dir)
    base = wt.head_commit(git_repo)
    # Seed a run branch at base (as if a prior epoch had run): the conflict must not
    # move it. We materialize it at base; later we assert it never advanced.
    wt.fast_forward_branch(git_repo, run_branch, base)
    run_tip_before = wt.resolve_commit(git_repo, run_branch)

    # Two task branches that both add shared.txt -> a both-added merge conflict.
    staging = f"grind-wip/{run_dir.root.name}/P1/E1/_staging"
    b1 = _commit_branch_adding(
        git_repo, f"grind-wip/{run_dir.root.name}/P1/E1/T1-a1", base, {"shared.txt": "from-T1\n"}
    )
    b2 = _commit_branch_adding(
        git_repo, f"grind-wip/{run_dir.root.name}/P1/E1/T2-a1", base, {"shared.txt": "from-T2\n"}
    )
    assert b1 and b2

    store = _store_for(run_dir, staging, base, merged=[])
    outcome = _integrate(
        repo=git_repo,
        run_dir=run_dir,
        store=store,
        branch=staging,
        base=base,
        done_in_order=[
            _done_outcome("T1", f"grind-wip/{run_dir.root.name}/P1/E1/T1-a1"),
            _done_outcome("T2", f"grind-wip/{run_dir.root.name}/P1/E1/T2-a1"),
        ],
    )

    assert outcome.status == "conflict"
    assert outcome.conflict
    # The run branch is untouched: same tip, no advance, no FF on a conflict.
    assert wt.resolve_commit(git_repo, run_branch) == run_tip_before
