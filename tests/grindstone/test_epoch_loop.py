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
    epoch_done_predicate,
)
from grindstone.events import Event, read_events
from grindstone.mock_worker import MockWorker
from grindstone.rundir import RunDir, create_run_dir
from grindstone.task_loop import TaskIdentity, pending_cursor
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
        ladder=[("local", worker)],
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
    # Task branches + worktrees are pruned after integration (ruling 7).
    import subprocess

    refs = subprocess.run(
        ["git", "branch", "--list", "grind/P1/E1/T*"], cwd=str(git_repo), capture_output=True, text=True
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
    ident = TaskIdentity("P1", "E1", "T1")
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
    # 6 tasks, default concurrency = min(4, n) — just assert it runs to completion.
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
        run_dir, args=artifact_epoch(task), mode="artifact", ladder=[("local", worker)]
    )
    assert outcome.status == "completed"
    assert outcome.integration.status == "skipped"
    assert not (run_dir.root / "worktrees").exists()
