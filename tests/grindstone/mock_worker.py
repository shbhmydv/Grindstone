"""Deterministic mock worker: scripted behaviors for the loop tests.

A test double for the loop's worker seam. The script is a list of behaviors
consumed one entry per ``run()`` call, so a test pins the exact failure sequence
with zero randomness: ``["rate_limit", "empty", "ok"]`` raises, then leaves no
work, then behaves. ``fuzz_script`` generates seeded unscripted sequences for the
one fuzz test that proves the loop always terminates (the only sanctioned
randomness).

The worker writes a FREE-FORM ``handoff.md`` report (never a JSON schema): the
state machine gates the deterministic facts (the committed diff / the produced
artifact), and the critic reads the report as prose.
"""

from __future__ import annotations

import json
import random
import threading
from dataclasses import dataclass, field
from pathlib import Path

from grindstone.worker import (
    CRITIC_VERDICT_FILENAME,
    HANDOFF_FILENAME,
    RateLimited,
    TransportError,
    WorkerRequest,
    WorkerTransport,
)

#: The behavior vocabulary. Order is the scripting order. ``session_limit`` is a
#: scriptable behavior too (a quota window: the driver parks and retries rather
#: than burning the attempt) but is kept OUT of ``BEHAVIORS`` so the fuzz generator
#: never emits it (a real park would make the termination-fuzz test sleep).
BEHAVIORS: tuple[str, ...] = ("ok", "rate_limit", "empty", "timeout")

#: The critic-dispatch vocabulary (consumed when a run carries a ``CriticBrief``):
#: the lenient triage outcome the mock writes to ``verdict.json``, plus ``no_verdict``
#: (writes nothing, modelling a critic that produced no parseable verdict).
CRITIC_OUTCOMES: tuple[str, ...] = ("PASS", "RETRY", "ESCALATE", "no_verdict")


def _handoff_md(request: WorkerRequest, *, blocked: bool = False) -> str:
    """A short free-form worker report. ``blocked`` flavors it as an environmental
    blocker the worker could not resolve (the critic reads this prose and ESCALATEs)."""

    if blocked:
        return (
            f"# handoff {request.task_id}\n\n"
            "BLOCKED: a host dependency I may not install stopped me; this needs a "
            "human decision. I could not finish the work.\n"
        )
    return (
        f"# handoff {request.task_id}\n\n"
        "Did the work; everything is DONE. Files touched are listed in the diff / "
        "artifact. Grounded in the real files.\n"
    )


@dataclass
class MockWorker:
    """A worker whose every ``run()`` follows the next scripted behavior.

    ``artifacts`` maps a scratch-relative path to its content; on ``ok`` / ``blocked``
    the worker writes them (the implement files or the non-write artifact_out) and a
    free-form ``handoff.md`` report. The script must be long enough for the calls the
    loop will make; running past its end raises.
    """

    script: list[str]
    artifacts: dict[str, str] = field(default_factory=dict)
    stdout: str = ""
    _calls: int = 0

    def run(self, request: WorkerRequest) -> str:
        if self._calls >= len(self.script):
            raise AssertionError("mock worker script exhausted")
        behavior = self.script[self._calls]
        self._calls += 1
        if request.critic is not None:
            self._critic(request, behavior)
            return ""

        handoff = request.scratch / HANDOFF_FILENAME

        if behavior in ("rate_limit", "session_limit"):
            raise RateLimited(f"mock {behavior}")
        if behavior == "empty":
            # No work and no report: a zero-diff / missing-artifact gate failure. The
            # real ScriptWorker.run still RETURNS its stdout, which the failure-debug
            # tail captures, so emit ``stdout`` here too.
            return self.stdout
        if behavior == "report_only":
            # A free-form report but no committable diff (no artifacts): a no-op-WITH-
            # report zero-diff fail, the failure whose VERSIONED handoff must survive.
            handoff.write_text(_handoff_md(request), encoding="utf-8")
            return self.stdout
        if behavior == "timeout":
            # Hung-then-killed: a partial report lands, then the supervisor kills the
            # worker. No real sleep, the kill is modelled as a raise.
            handoff.write_text("# handoff (partial", encoding="utf-8")
            raise TransportError("mock hang killed")
        if behavior in ("ok", "blocked"):
            self._write_artifacts(request)
            handoff.write_text(
                _handoff_md(request, blocked=behavior == "blocked"), encoding="utf-8"
            )
            return self.stdout
        raise ValueError(f"unknown mock behavior: {behavior!r}")

    def _write_artifacts(self, request: WorkerRequest) -> None:
        for rel, content in self.artifacts.items():
            path = request.scratch / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")

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
    for implement, the ``artifact_out`` for non-write), so the deterministic gate
    passes by construction, plus a free-form ``handoff.md`` report; every critic
    dispatch writes the same ``critic_outcome`` (default ``PASS``). No positional
    cursor, so concurrent tasks never race.

    Knobs: ``read_cite`` makes a non-write task READ a file from the integration-tip
    ``read_root`` and fold it into its artifact (the integration-tip-read proof);
    ``rate_limit_once`` makes the FIRST worker dispatch raise ``RateLimited`` exactly
    once (the node-#1 park + epoch-restart proof), then behave.
    """

    critic_outcome: str = "PASS"
    read_cite: str | None = None
    rate_limit_once: bool = False
    _rl_fired: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def run(self, request: WorkerRequest) -> str:
        if request.critic is not None:
            (request.scratch / CRITIC_VERDICT_FILENAME).write_text(
                json.dumps(
                    {"outcome": self.critic_outcome, "reason": f"loop {self.critic_outcome}"}
                ),
                encoding="utf-8",
            )
            return ""

        if self.rate_limit_once:
            with self._lock:
                if not self._rl_fired:
                    self._rl_fired = True
                    raise RateLimited("loop mock first-dispatch rate limit")

        task = request.task
        if request.mode == "implement":
            for rel in task.file_ownership:
                path = request.scratch / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                # Content keyed on the task id so each epoch's commit is a real diff.
                path.write_text(f"# {request.task_id}\nvalue = {request.task_id!r}\n",
                                encoding="utf-8")
        else:
            assert task.artifact_out is not None
            body = f"# artifact {request.task_id}\n"
            if self.read_cite is not None:
                assert request.read_root is not None, "non-write task needs a read_root"
                tip = (request.read_root / self.read_cite).read_text(encoding="utf-8")
                body += f"reviewed integration-tip file {self.read_cite}:\n{tip}"
            # The worker writes the BASENAME in its CWD; Python owns the publish key.
            out = request.scratch / Path(task.artifact_out).name
            out.write_text(body, encoding="utf-8")

        (request.scratch / HANDOFF_FILENAME).write_text(
            _handoff_md(request), encoding="utf-8"
        )
        return ""


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
#: epoch-restart (which re-enters ``run_task`` fresh). The three map onto the loop's
#: real routes: ``pass`` -> merged; ``retry_pass`` -> a failed first attempt then a
#: clean retry (the bounded same-tier self-heal); ``failed`` -> no work every attempt
#: (retries exhaust -> escalated to the planner).
STOCHASTIC_OUTCOMES: tuple[str, ...] = ("pass", "retry_pass", "failed")

#: Default outcome weights: mostly-pass with a tail of self-heals + hard failures,
#: so a multi-epoch run converges most of the time while every failure route is
#: exercised across a seed sweep.
DEFAULT_OUTCOME_WEIGHTS: tuple[float, float, float] = (0.6, 0.2, 0.2)


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
    A clean outcome writes exactly what the task CLAIMS (each ``file_ownership`` path
    for implement, the ``artifact_out`` for a non-write task) plus a free-form
    ``handoff.md``, so the deterministic gate passes by construction; a ``failed``
    outcome writes NOTHING (a zero-diff / missing-artifact gate failure that exhausts
    to an escalate). The critic dispatch always rubber-stamps PASS (a clean outcome
    here is honest work; the rubber-stamp SAFETY net is tested separately against a
    critic that passes BROKEN work).
    """

    seed: int
    weights: tuple[float, float, float] = DEFAULT_OUTCOME_WEIGHTS
    rate_limit_once: bool = False
    _rl_fired: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def _outcome(self, task_id: str) -> str:
        rng = random.Random(f"{self.seed}|{task_id}")
        return rng.choices(STOCHASTIC_OUTCOMES, weights=list(self.weights), k=1)[0]

    def run(self, request: WorkerRequest) -> str:
        if request.critic is not None:
            (request.scratch / CRITIC_VERDICT_FILENAME).write_text(
                json.dumps({"outcome": "PASS", "reason": "stochastic critic PASS"}),
                encoding="utf-8",
            )
            return ""

        with self._lock:
            if self.rate_limit_once and not self._rl_fired:
                self._rl_fired = True
                raise RateLimited("stochastic worker one-shot rate limit")

        outcome = self._outcome(request.task_id)
        is_retry = bool(request.failure_context)
        clean = outcome == "pass" or (outcome == "retry_pass" and is_retry)
        if not clean:
            # No work and no report: the deterministic gate rejects the attempt.
            return ""
        self._write_work(request)
        (request.scratch / HANDOFF_FILENAME).write_text(
            _handoff_md(request), encoding="utf-8"
        )
        return ""

    def _write_work(self, request: WorkerRequest) -> None:
        """Write exactly the claimed deliverables (mirrors the ``LoopWorker`` happy
        path) so the deterministic gate passes."""

        task = request.task
        if request.mode == "implement":
            for rel in task.file_ownership:
                path = request.scratch / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    f"# {request.task_id}\nvalue = {request.task_id!r}\n",
                    encoding="utf-8",
                )
        else:
            assert task.artifact_out is not None
            # The worker writes the BASENAME in its CWD; Python owns the publish key.
            out = request.scratch / Path(task.artifact_out).name
            out.write_text(f"# artifact {request.task_id}\n", encoding="utf-8")


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

    def run(self, request: WorkerRequest) -> str:
        if request.critic is None:
            with self._lock:
                self._n += 1
                if self._n == self.crash_on:
                    raise SimulatedKill("simulated kill (mid-epoch crash)")
        return self.inner.run(request)
