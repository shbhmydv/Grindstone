"""Deterministic mock planner — scripted decisions + scripted failures (S3).

Same consumed-per-call discipline as ``MockWorker`` (ruling 10): the script is a
list consumed one entry per ``plan()`` call, so a test pins the exact decision /
failure sequence with zero randomness. An entry is either:

  - a ``dict`` — a decision payload, returned as JSON text (optionally fence- or
    prose-wrapped so the core's extractor is exercised end-to-end); or
  - a failure token from ``FAILURES`` — raising the matching transport exception
    or returning malformed text.

Failure taxonomy (the planner analogue of the worker's): ``rate_limit`` →
backoff, ``transient`` / ``timeout`` → retry, ``hard`` → human, ``bad_json`` /
``empty`` / ``invalid`` → un-gateable output the core re-asks on.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Union

from grindstone.planner import PlannerHardError, RateLimited, TransportError, WorkerTimeout


#: An ``invalid`` token returns valid JSON that fails the decision gate (an
#: implement decision with no tasks — schema-rejected), exercising the re-ask
#: ladder without any transport error.
_INVALID_DECISION = {"schema_version": "1", "tool": "implement", "args": {}}

ScriptEntry = Union[dict[str, object], str]


@dataclass
class MockPlanner:
    """A planner whose every ``plan()`` follows the next scripted entry.

    ``wrap`` chooses how decision dicts are rendered: ``"bare"`` (plain JSON),
    ``"fence"`` (```json fenced), or ``"prose"`` (reasoning before + after) — so
    a single test can prove the extractor survives codex's wrapping habits.
    """

    script: list[ScriptEntry]
    wrap: str = "bare"
    _calls: int = field(default=0)

    def plan(self, prompt: str) -> str:
        if self._calls >= len(self.script):
            raise AssertionError("mock planner script exhausted")
        entry = self.script[self._calls]
        self._calls += 1
        if isinstance(entry, str):
            return self._failure(entry)
        return self._render(entry)

    def _failure(self, token: str) -> str:
        if token == "rate_limit":
            raise RateLimited("mock planner 429")
        if token == "transient":
            raise TransportError("mock planner 5xx")
        if token == "timeout":
            raise WorkerTimeout("mock planner hang killed")
        if token == "hard":
            raise PlannerHardError("mock planner auth failure")
        if token == "bad_json":
            return "here is the decision: { not valid json"
        if token == "empty":
            return ""
        if token == "invalid":
            return json.dumps(_INVALID_DECISION)
        raise ValueError(f"unknown mock planner failure token: {token!r}")

    def _render(self, decision: dict[str, object]) -> str:
        body = json.dumps(decision)
        if self.wrap == "fence":
            return f"Here is my decision:\n```json\n{body}\n```\n"
        if self.wrap == "prose":
            return (
                "Let me think. The skeleton looks fine, so I will proceed.\n\n"
                f"{body}\n\nThat is my single tool call."
            )
        return body
