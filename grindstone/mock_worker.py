"""Deterministic mock worker: scripted behaviors for the loop tests.

A test double for the loop's worker seam. The script is a list of behaviors
consumed one entry per ``run()`` call, so a test pins the exact failure sequence
with zero randomness: ``["rate_limit", "bad_json", "ok"]`` raises, then writes
garbage, then behaves. ``fuzz_script`` generates seeded unscripted sequences for
the one fuzz test that proves the loop always terminates (the only sanctioned
randomness).
"""

from __future__ import annotations

import json
import random
import threading
from dataclasses import dataclass, field

from grindstone.worker import (
    CRITIC_VERDICT_FILENAME,
    REVIEW_FILENAME,
    RateLimited,
    TransportError,
    WorkerRequest,
    WorkerTransport,
)

#: The behavior vocabulary. Order is the scripting order. ``session_limit`` is a
#: scriptable behavior too (a quota window: the driver parks and retries rather
#: than burning the attempt) but is kept OUT of ``BEHAVIORS`` so the fuzz generator
#: never emits it (a real park would make the termination-fuzz test sleep).
BEHAVIORS: tuple[str, ...] = ("ok", "rate_limit", "bad_json", "empty", "timeout")

#: The critic-dispatch vocabulary (consumed when a run carries a ``CriticBrief``):
#: the lenient triage outcome the mock writes to ``verdict.json``, plus ``no_verdict``
#: (writes nothing, modelling a critic that produced no parseable verdict).
CRITIC_OUTCOMES: tuple[str, ...] = ("PASS", "RETRY", "ESCALATE", "no_verdict")


def _valid_handoff(
    request: WorkerRequest, cited: list[str], *, status: str = "DONE"
) -> dict[str, object]:
    """A schema- and semantic-valid handoff for the dispatched task."""

    return {
        "schema_version": "1",
        "task_id": request.task_id,
        "status": status,
        "what_changed": [{"kind": "file", "ref": f} for f in cited],
        "resulting_state": "toy work complete"
        if status == "DONE"
        else "toy worker reported a blocker",
        "downstream_needs": [],
        "not_done": [] if status == "DONE" else ["a host dependency the worker may not install"],
        "citations": [{"file": f} for f in cited],
        "checks": [{"check": "self-check", "exit_code": 0}],
        "occupancy": {"compacted": False, "subagent_splits": 0},
    }


@dataclass
class MockWorker:
    """A worker whose every ``run()`` follows the next scripted behavior.

    ``artifacts`` maps a scratch-relative path to its content; on ``ok`` the worker
    writes them (the files the handoff citations point at) and emits a valid
    handoff. The script must be long enough for the calls the loop will make;
    running past its end raises.
    """

    script: list[str]
    artifacts: dict[str, str] = field(default_factory=dict)
    _calls: int = 0

    def run(self, request: WorkerRequest) -> None:
        if self._calls >= len(self.script):
            raise AssertionError("mock worker script exhausted")
        behavior = self.script[self._calls]
        self._calls += 1
        if request.critic is not None:
            self._critic(request, behavior)
            return

        handoff = request.scratch / "handoff.json"

        if behavior in ("rate_limit", "session_limit"):
            raise RateLimited(f"mock {behavior}")
        if behavior == "bad_json":
            handoff.write_text("{ this is not valid json", encoding="utf-8")
            return
        if behavior == "empty":
            return
        if behavior == "timeout":
            # Hung-then-killed: a partial file lands, then the supervisor kills the
            # worker. No real sleep, the kill is modelled as a raise.
            handoff.write_text('{"schema_version": "1"', encoding="utf-8")
            raise TransportError("mock hang killed")
        if behavior in ("ok", "blocked"):
            cited = self._write_artifacts(request)
            if request.mode == "implement" and behavior == "ok":
                # A compliant implement worker self-reviews before handing off.
                (request.scratch / REVIEW_FILENAME).write_text(
                    "mock review: no findings\n", encoding="utf-8"
                )
            status = "DONE" if behavior == "ok" else "BLOCKED"
            handoff.write_text(
                json.dumps(_valid_handoff(request, cited, status=status)),
                encoding="utf-8",
            )
            return
        raise ValueError(f"unknown mock behavior: {behavior!r}")

    def _write_artifacts(self, request: WorkerRequest) -> list[str]:
        cited: list[str] = []
        for rel, content in self.artifacts.items():
            path = request.scratch / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            cited.append(rel)
        return cited

    def _critic(self, request: WorkerRequest, outcome: str) -> None:
        """Write the scripted critic ``verdict.json`` (or, for ``no_verdict``,
        nothing, modelling a critic that produced no parseable verdict)."""

        if outcome == "no_verdict":
            return
        if outcome not in CRITIC_OUTCOMES:
            raise ValueError(f"unknown mock critic outcome: {outcome!r}")
        (request.scratch / CRITIC_VERDICT_FILENAME).write_text(
            json.dumps({"outcome": outcome, "reason": f"mock critic {outcome}"}),
            encoding="utf-8",
        )


@dataclass
class LoopWorker:
    """A concurrency-safe loop test double (the flat-script ``MockWorker`` is
    positional, so it cannot drive an epoch's CONCURRENT fan-out). Every worker
    dispatch writes exactly the files the task CLAIMS (each ``file_ownership`` path
    for implement, the ``artifact_out`` for non-write), so the scope + citation gates
    pass by construction, and emits a DONE handoff; every critic dispatch writes the
    same ``critic_outcome`` (default ``PASS``). No positional cursor, so concurrent
    tasks never race.

    Knobs: ``read_cite`` makes a non-write task READ a file from the integration-tip
    ``read_root`` and fold it into its artifact + citations (the integration-tip-read
    proof); ``rate_limit_once`` makes the FIRST worker dispatch raise ``RateLimited``
    exactly once (the node-#1 park + epoch-restart proof), then behave.
    """

    critic_outcome: str = "PASS"
    read_cite: str | None = None
    rate_limit_once: bool = False
    _rl_fired: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def run(self, request: WorkerRequest) -> None:
        if request.critic is not None:
            (request.scratch / CRITIC_VERDICT_FILENAME).write_text(
                json.dumps(
                    {"outcome": self.critic_outcome, "reason": f"loop {self.critic_outcome}"}
                ),
                encoding="utf-8",
            )
            return

        if self.rate_limit_once:
            with self._lock:
                if not self._rl_fired:
                    self._rl_fired = True
                    raise RateLimited("loop mock first-dispatch rate limit")

        task = request.task
        cited: list[str] = []
        if request.mode == "implement":
            for rel in task.file_ownership:
                path = request.scratch / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                # Content keyed on the task id so each epoch's commit is a real diff.
                path.write_text(f"# {request.task_id}\nvalue = {request.task_id!r}\n",
                                encoding="utf-8")
                cited.append(rel)
            (request.scratch / REVIEW_FILENAME).write_text(
                "loop review: no findings\n", encoding="utf-8"
            )
        else:
            assert task.artifact_out is not None
            body = f"# artifact {request.task_id}\n"
            if self.read_cite is not None:
                assert request.read_root is not None, "non-write task needs a read_root"
                tip = (request.read_root / self.read_cite).read_text(encoding="utf-8")
                body += f"reviewed integration-tip file {self.read_cite}:\n{tip}"
                cited.append(self.read_cite)  # resolves against read_root (the tip)
            out = request.scratch / task.artifact_out
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(body, encoding="utf-8")
            cited.append(task.artifact_out)

        (request.scratch / "handoff.json").write_text(
            json.dumps(_valid_handoff(request, cited)), encoding="utf-8"
        )


def fuzz_script(seed: int, length: int) -> list[str]:
    """Generate a deterministic-from-seed random behavior script (fuzz only).

    The only sanctioned randomness in the test suite: a seeded generator that
    explores unscripted failure sequences. The invariant the fuzz test asserts is
    termination, not a specific outcome.
    """

    rng = random.Random(seed)
    return [rng.choice(BEHAVIORS) for _ in range(length)]


#: The per-task outcome vocabulary the seeded stochastic worker draws from. Each
#: task's outcome is a PURE function of ``(seed, task_id)`` (a per-task
#: ``random.Random``), so the result is deterministic regardless of the order
#: concurrent fan-out threads dispatch in, AND stable across a rate-limit
#: epoch-restart (which re-enters ``run_task`` fresh). The four map onto the loop's
#: real routes: ``pass`` -> merged; ``retry_pass`` -> a FAILED first attempt then a
#: DONE retry (the bounded same-tier self-heal); ``failed`` -> FAILED every attempt
#: (retries exhaust -> escalated to the planner); ``blocked`` -> a worker-reported
#: BLOCKED (straight to the planner, critic skipped).
STOCHASTIC_OUTCOMES: tuple[str, ...] = ("pass", "retry_pass", "failed", "blocked")

#: Default outcome weights: mostly-pass with a tail of self-heals + hard failures,
#: so a multi-epoch run converges most of the time while every failure route is
#: exercised across a seed sweep.
DEFAULT_OUTCOME_WEIGHTS: tuple[float, float, float, float] = (0.6, 0.2, 0.1, 0.1)


@dataclass
class StochasticWorker:
    """A SEEDED stochastic loop worker for the convergence / invariant E2E.

    Concurrency-safe by construction: a task's outcome is drawn from a per-task
    ``random.Random(f"{seed}|{task_id}")``, never a shared global generator and
    never in dispatch order, so two threads fanning out cannot race the draw and a
    re-run (rate-limit restart, resume) reproduces the SAME per-task outcome. The
    only shared mutable state is the one-shot rate-limit flag, guarded by a lock.

    Retry is detected statelessly from ``request.failure_context`` (a retry attempt
    carries the prior rejection), so no per-task counter can drift across a restart.
    Every DONE handoff writes exactly what the task CLAIMS (each ``file_ownership``
    path + ``review.md`` for implement, the ``artifact_out`` for a non-write task),
    so the scope + citation gates pass by construction; the critic dispatch always
    rubber-stamps PASS (a DONE handoff here is honest work, the rubber-stamp SAFETY
    net is tested separately against a critic that passes BROKEN work).
    """

    seed: int
    weights: tuple[float, float, float, float] = DEFAULT_OUTCOME_WEIGHTS
    rate_limit_once: bool = False
    _rl_fired: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def _outcome(self, task_id: str) -> str:
        rng = random.Random(f"{self.seed}|{task_id}")
        return rng.choices(STOCHASTIC_OUTCOMES, weights=list(self.weights), k=1)[0]

    def run(self, request: WorkerRequest) -> None:
        if request.critic is not None:
            (request.scratch / CRITIC_VERDICT_FILENAME).write_text(
                json.dumps({"outcome": "PASS", "reason": "stochastic critic PASS"}),
                encoding="utf-8",
            )
            return

        with self._lock:
            if self.rate_limit_once and not self._rl_fired:
                self._rl_fired = True
                raise RateLimited("stochastic worker one-shot rate limit")

        outcome = self._outcome(request.task_id)
        is_retry = bool(request.failure_context)
        if outcome == "blocked":
            status = "BLOCKED"
        elif outcome == "failed":
            status = "FAILED"
        elif outcome == "retry_pass":
            status = "DONE" if is_retry else "FAILED"
        else:
            status = "DONE"

        cited = self._write_work(request) if status == "DONE" else []
        (request.scratch / "handoff.json").write_text(
            json.dumps(_valid_handoff(request, cited, status=status)), encoding="utf-8"
        )

    def _write_work(self, request: WorkerRequest) -> list[str]:
        """Write exactly the claimed deliverables for a DONE handoff (mirrors the
        ``LoopWorker`` happy path) and return the citation paths."""

        task = request.task
        cited: list[str] = []
        if request.mode == "implement":
            for rel in task.file_ownership:
                path = request.scratch / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    f"# {request.task_id}\nvalue = {request.task_id!r}\n",
                    encoding="utf-8",
                )
                cited.append(rel)
            (request.scratch / REVIEW_FILENAME).write_text(
                "stochastic review: no findings\n", encoding="utf-8"
            )
        else:
            assert task.artifact_out is not None
            out = request.scratch / task.artifact_out
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(f"# artifact {request.task_id}\n", encoding="utf-8")
            cited.append(task.artifact_out)
        return cited


class SimulatedKill(BaseException):
    """A modelled host SIGKILL: a ``BaseException`` (NOT an ``Exception``) so it
    escapes the loop's transport boundary (which catches ``Exception`` and demotes it
    to a retryable task failure) and propagates clean out of ``start_run``, exactly
    like the host process dying mid-epoch. Subclasses ``BaseException`` rather than
    reusing ``KeyboardInterrupt`` so it never trips pytest's interrupt handling."""


@dataclass
class CrashingWorker:
    """Wraps an inner worker and raises a HARD kill on the ``crash_on``-th worker
    dispatch (a simulated SIGKILL mid-run).

    The kill (a ``SimulatedKill`` ``BaseException``) is NOT caught by the loop's
    transport boundary, so it propagates out of ``start_run`` exactly like the host
    process dying: the journal is left at a mid-epoch boundary (an ``epoch_started``
    with no ``epoch_completed``) and the in-flight epoch's throwaway worktrees + wip
    branches survive on disk, for ``resume_run`` to raze + re-plan. Critic dispatches
    are passed through (the kill models a death DURING a worker grind). If the run
    finishes before the ``crash_on``-th dispatch, no kill fires (a clean run)."""

    inner: WorkerTransport
    crash_on: int
    _n: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def run(self, request: WorkerRequest) -> None:
        if request.critic is None:
            with self._lock:
                self._n += 1
                if self._n == self.crash_on:
                    raise SimulatedKill("simulated kill (mid-epoch crash)")
        self.inner.run(request)
