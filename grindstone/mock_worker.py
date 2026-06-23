"""Deterministic mock worker, scripted failures for the S1 loop tests.

The script is a list of behaviors consumed one entry per ``run()`` call, so a
test pins the exact failure sequence with zero randomness (ARCHITECTURE.md / S1):
``["rate_limit", "bad_json", "ok"]`` raises, then writes garbage, then behaves.
The five behaviors are the failure taxonomy the loop must survive. A separate
seeded ``fuzz_script`` generates unscripted sequences for the one fuzz test
that proves the loop always terminates, the only place randomness is allowed.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field

from grindstone.worker import (
    REVIEW_FILENAME,
    RateLimited,
    SessionLimited,
    WorkerRequest,
    WorkerTimeout,
)

#: The exact behavior vocabulary (S1). Order is the scripting order.
BEHAVIORS: tuple[str, ...] = ("ok", "rate_limit", "bad_json", "empty", "timeout")
#: ``session_limit`` is a scriptable behavior too (the long quota-window limit:
#: the attempt driver PARKS hourly on it rather than burning the attempt), but it
#: is deliberately KEPT OUT of ``BEHAVIORS`` so the fuzz generator never emits it,
#: a real hourly park would make the termination-fuzz test sleep 24 wall-clock
#: hours. Tests script it explicitly with an injected fake ``sleep_fn``.


def _valid_handoff(request: WorkerRequest, cited: list[str]) -> dict[str, object]:
    """A schema- and semantic-valid DONE handoff echoing the task's checks."""

    checks = [
        {"check": c.cmd if hasattr(c, "cmd") else "artifact", "exit_code": 0}
        for c in request.task.done_when
    ]
    return {
        "schema_version": "1",
        "task_id": request.task_id,
        "status": "DONE",
        "what_changed": [{"kind": "file", "ref": f} for f in cited] or [],
        "resulting_state": "toy work complete",
        "downstream_needs": [],
        "not_done": [],
        "citations": [{"file": f} for f in cited],
        "checks": checks,
        "occupancy": {"compacted": False, "subagent_splits": 0},
    }


@dataclass
class MockWorker:
    """A worker whose every ``run()`` follows the next scripted behavior.

    ``artifacts`` maps a scratch-relative path to its content; on ``ok`` the
    worker writes them (the files the task's ``done_when`` checks and the
    handoff citations point at) and emits a valid handoff. The script must be
    long enough for the calls the loop will make; running past its end raises.
    """

    script: list[str]
    artifacts: dict[str, str] = field(default_factory=dict)
    _calls: int = 0

    def run(self, request: WorkerRequest) -> None:
        if self._calls >= len(self.script):
            raise AssertionError("mock worker script exhausted")
        behavior = self.script[self._calls]
        self._calls += 1
        handoff = request.scratch / "handoff.json"

        if behavior == "session_limit":
            raise SessionLimited("mock session limit . resets 2:20am")
        if behavior == "rate_limit":
            raise RateLimited("mock 429")
        if behavior == "bad_json":
            handoff.write_text("{ this is not valid json", encoding="utf-8")
            return
        if behavior == "empty":
            return
        if behavior == "timeout":
            # Hung-then-killed: a partial file lands, then the supervisor kills
            # the worker. No real sleep, the kill is modelled as a raise.
            handoff.write_text('{"schema_version": "1"', encoding="utf-8")
            raise WorkerTimeout("mock hang killed")
        if behavior == "ok":
            cited: list[str] = []
            for rel, content in self.artifacts.items():
                path = request.scratch / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")
                cited.append(rel)
            if request.mode == "implement":
                # A compliant implement worker satisfies the review gate the
                # core appends to done_when (`test -s review.md`).
                (request.scratch / REVIEW_FILENAME).write_text(
                    "mock review: no findings\n", encoding="utf-8"
                )
            handoff.write_text(
                json.dumps(_valid_handoff(request, cited)), encoding="utf-8"
            )
            return
        raise ValueError(f"unknown mock behavior: {behavior!r}")


def fuzz_script(seed: int, length: int) -> list[str]:
    """Generate a deterministic-from-seed random behavior script (fuzz only).

    The ONLY sanctioned randomness in the test suite (ARCHITECTURE.md): a seeded
    generator that explores unscripted failure sequences. The invariant the
    fuzz test asserts is termination, not a specific outcome.
    """

    rng = random.Random(seed)
    return [rng.choice(BEHAVIORS) for _ in range(length)]
