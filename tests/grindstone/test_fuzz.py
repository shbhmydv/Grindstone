"""Seeded fuzz: an unscripted 3-task epoch must always terminate with no orphans.

The only sanctioned randomness in the suite (ARCHITECTURE.md). For each seed every task
gets a random behavior script on every ladder tier; we assert the epoch reaches
a terminal EpochOutcome and EVERY dispatched task reaches a terminal event
(done | failed) — never hanging, never leaving a task un-terminated.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from grindstone.events import read_events, replay
from grindstone.mock_worker import MockWorker, fuzz_script
from grindstone.rundir import create_run_dir

from tests.grindstone.conftest import (
    OUT_CONTENT,
    RoutingWorker,
    implement_epoch,
    init_git_repo,
    make_toy_task,
    run_one_epoch,
)

N_TASKS = 3
MAX_ATTEMPTS = 4  # 3 (tier0) + 1 (tier1)


def _tier(seed: int, salt: int, length: int) -> RoutingWorker:
    return RoutingWorker(
        {
            f"T{i}": MockWorker(
                script=fuzz_script(seed * 100 + salt + i, length),
                artifacts={f"f{i}.txt": OUT_CONTENT},
            )
            for i in range(1, N_TASKS + 1)
        }
    )


@pytest.mark.parametrize("seed", range(30))
def test_fuzz_epoch_always_terminates(tmp_path: Path, seed: int) -> None:
    repo = init_git_repo(tmp_path / "repo")
    run_dir = create_run_dir(repo, f"run-{seed}")
    tasks = [
        make_toy_task(task_id=f"T{i}", out_file=f"f{i}.txt", owned=[f"f{i}.txt"])
        for i in range(1, N_TASKS + 1)
    ]
    outcome = run_one_epoch(
        run_dir,
        args=implement_epoch(*tasks),
        mode="implement",
        ladder=[("local", _tier(seed, 0, 3)), ("cloud", _tier(seed, 50, 1))],
        repo=repo,
        concurrency=2,
    )
    # Disjoint ownership => integration can never conflict.
    assert outcome.status == "completed"
    assert len(outcome.tasks) == N_TASKS
    for t in outcome.tasks:
        assert t.status in {"done", "failed"}
        assert 1 <= t.attempts <= MAX_ATTEMPTS
    # No orphans: every task node in the replay tree is terminal.
    tree = replay(read_events(run_dir.events_path))
    for node in tree.phases[0].epochs[0].tasks:
        assert node.status in {"done", "failed"}
