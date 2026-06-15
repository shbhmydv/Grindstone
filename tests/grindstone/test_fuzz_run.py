"""Seeded fuzz over the WHOLE run loop (S3): scripted planner + worker mixes.

Extends the S2 epoch fuzz upward. For each seed a planner emits an unscripted
mix of decisions and failures, workers fail unpredictably, and the run must
always TERMINATE (the test returning is the proof) with planner_calls bounded by
the safety valve. The only sanctioned randomness in the suite; no wall clock
(``sleep_fn`` is a no-op recorder).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from grindstone.events import read_events, replay
from grindstone.planner import RateLimited, TransportError
from grindstone.rundir import create_run_dir
from grindstone.run_loop import run_grind
from grindstone.worker import WorkerRequest

from tests.grindstone.conftest import (
    OwnershipWorker,
    artifact_decision,
    artifact_task,
    check_cmd,
    complete_decision,
    escalate_decision,
    phase_dict,
    revise_decision,
    skeleton_decision,
)

MAX_PLANNER_CALLS = 20
MAX_EPOCHS = 10


class _FuzzPlanner:
    """Emits a seeded mix of valid decisions and failures (no repo: artifact only)."""

    _CHOICES = ["complete", "escalate", "artifact", "revise", "rate_limit", "transient", "bad_json", "empty", "invalid"]
    _WEIGHTS = [3, 1, 4, 1, 1, 1, 1, 1, 1]

    def __init__(self, seed: int) -> None:
        self.rng = random.Random(seed)
        self.proposed = False

    def plan(self, prompt: str) -> str:
        if not self.proposed:
            self.proposed = True
            return json.dumps(skeleton_decision(phase_dict("P1"), phase_dict("P2")))
        choice = self.rng.choices(self._CHOICES, weights=self._WEIGHTS)[0]
        if choice == "rate_limit":
            raise RateLimited("fuzz rate limit")
        if choice == "transient":
            raise TransportError("fuzz 5xx")
        if choice == "bad_json":
            return "reasoning... { broken"
        if choice == "empty":
            return ""
        if choice == "invalid":
            return json.dumps({"schema_version": "1", "tool": "artifact", "args": {}})
        if choice == "complete":
            return json.dumps(complete_decision(check_cmd("true")))
        if choice == "escalate":
            return json.dumps(escalate_decision("fuzz escalate"))
        if choice == "revise":
            return json.dumps(revise_decision(phase_dict("P1"), phase_dict("P2")))
        n = self.rng.randint(1, 2)
        return json.dumps(artifact_decision(*[artifact_task(f"T{i}") for i in range(1, n + 1)]))


class _SeededWorker:
    """An artifact worker that fails unpredictably (a task may FAIL; epoch still completes)."""

    def __init__(self, seed: int) -> None:
        self.rng = random.Random(seed * 7 + 1)
        self.ok = OwnershipWorker()

    def run(self, request: WorkerRequest) -> None:
        behavior = self.rng.choices(["ok", "empty", "bad_json", "rate_limit"], weights=[5, 1, 1, 1])[0]
        if behavior == "ok":
            self.ok.run(request)
            return
        if behavior == "empty":
            return
        if behavior == "bad_json":
            (request.scratch / "handoff.json").write_text("{bad", encoding="utf-8")
            return
        raise RateLimited("fuzz worker 429")


@pytest.mark.parametrize("seed", range(20))
def test_fuzz_run_always_terminates_bounded(tmp_path: Path, seed: int) -> None:
    run_dir = create_run_dir(tmp_path, f"run-{seed}")
    outcome = run_grind(
        run_dir,
        job_path="job.md",
        planner=_FuzzPlanner(seed),
        ladder=[("local", _SeededWorker(seed))],
        repo=None,  # artifact-only fuzz: no git, no integration
        sleep_fn=lambda _delay: None,  # never touch the wall clock
        max_planner_calls=MAX_PLANNER_CALLS,
        max_epochs=MAX_EPOCHS,
        tier0_attempts=1,
    )
    assert outcome.status in {"completed", "escalated", "failed"}
    assert outcome.planner_calls <= MAX_PLANNER_CALLS
    # The journal always replays into a terminal tree.
    tree = replay(read_events(run_dir.events_path))
    assert tree.status in {"completed", "escalated", "running"}
    assert tree.planner_calls == outcome.planner_calls


class _MultiPhasePlanner:
    """A fuzz planner whose skeleton has 3 phases with mixed criteria — one
    always-failing phase drives budgets -> phase escalations -> revise/escalate.

    Exercises the S4 phase machinery (advancement, budgets, escalation legality)
    under unscripted decision sequences; the invariant is still termination.
    """

    _CHOICES = ["artifact", "revise", "escalate", "complete", "rate_limit", "bad_json", "invalid"]
    _WEIGHTS = [5, 2, 1, 1, 1, 1, 1]

    def __init__(self, seed: int) -> None:
        self.rng = random.Random(seed * 13 + 5)
        self.proposed = False

    def plan(self, prompt: str) -> str:
        if not self.proposed:
            self.proposed = True
            return json.dumps(
                skeleton_decision(
                    phase_dict("P1", exit_criterion=[check_cmd("true")], budget=2),
                    phase_dict("P2", exit_criterion=[check_cmd("test -f __never__")], budget=2),
                    phase_dict("P3", exit_criterion=[check_cmd("true")], budget=2),
                )
            )
        choice = self.rng.choices(self._CHOICES, weights=self._WEIGHTS)[0]
        if choice == "rate_limit":
            raise RateLimited("fuzz rate limit")
        if choice == "bad_json":
            return "reasoning... { broken"
        if choice == "invalid":
            return json.dumps({"schema_version": "1", "tool": "artifact", "args": {}})
        if choice == "complete":
            return json.dumps(complete_decision(check_cmd("true")))
        if choice == "escalate":
            return json.dumps(escalate_decision("fuzz escalate"))
        if choice == "revise":
            return json.dumps(
                revise_decision(phase_dict("P2", exit_criterion=[check_cmd("true")]), phase_dict("P3"))
            )
        n = self.rng.randint(1, 2)
        return json.dumps(artifact_decision(*[artifact_task(f"T{i}") for i in range(1, n + 1)]))


@pytest.mark.parametrize("seed", range(20))
def test_fuzz_run_multiphase_terminates_bounded(tmp_path: Path, seed: int) -> None:
    run_dir = create_run_dir(tmp_path, f"mp-{seed}")
    outcome = run_grind(
        run_dir,
        job_path="job.md",
        planner=_MultiPhasePlanner(seed),
        ladder=[("local", _SeededWorker(seed))],
        repo=None,
        sleep_fn=lambda _delay: None,
        max_planner_calls=MAX_PLANNER_CALLS,
        max_epochs=MAX_EPOCHS,
        tier0_attempts=1,
    )
    assert outcome.status in {"completed", "escalated", "failed"}
    assert outcome.planner_calls <= MAX_PLANNER_CALLS
    tree = replay(read_events(run_dir.events_path))
    assert tree.status in {"completed", "escalated", "running"}
    assert tree.planner_calls == outcome.planner_calls
