"""Deterministic mock planner: scripted decisions + scripted failures.

A test double for the loop's planner seam. The script is a list consumed one entry
per ``plan()`` call, so a test pins the exact decision / failure sequence with zero
randomness. An entry is either:

  - a ``dict`` (a decision payload) returned as JSON text, optionally fence- or
    prose-wrapped so the core's extractor is exercised end-to-end; or
  - a failure token from ``FAILURES``, raising the matching transport exception or
    returning malformed text.

Failure taxonomy (the bones two-node model): ``rate_limit`` / ``session_limit`` ->
``RateLimited`` (back off and re-issue); ``transient`` / ``timeout`` / ``hard`` ->
``PlannerError`` (the cannot-continue catch-all); ``bad_json`` / ``empty`` /
``invalid`` -> un-gateable output the core re-asks on.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Union

from grindstone.contracts.models import Decision
from grindstone.planner import PlannerError, RateLimited

if TYPE_CHECKING:
    from grindstone.loop import PlannerContext

#: An ``invalid`` token returns valid JSON that fails the decision gate (an epoch
#: decision with an empty epoch: no title, no tasks), exercising the re-ask ladder
#: without any transport error.
_INVALID_DECISION = {"kind": "epoch", "epoch": {}}

#: The scriptable failure tokens.
FAILURES = (
    "rate_limit",
    "session_limit",
    "transient",
    "timeout",
    "hard",
    "bad_json",
    "empty",
    "invalid",
)

ScriptEntry = Union[dict[str, object], str]


@dataclass
class MockPlanner:
    """A planner whose every ``plan()`` follows the next scripted entry.

    ``wrap`` chooses how decision dicts are rendered: ``"bare"`` (plain JSON),
    ``"fence"`` (```json fenced), or ``"prose"`` (reasoning before + after), so a
    single test can prove the extractor survives a model's wrapping habits.
    """

    script: list[ScriptEntry]
    wrap: str = "bare"
    _calls: int = field(default=0)

    def plan(self, prompt: str, *, workdir: Path | None = None) -> str:
        # A pure scripted transport: no worktree to grind in, so ``workdir`` is
        # accepted for protocol parity and deliberately ignored.
        if self._calls >= len(self.script):
            raise AssertionError("mock planner script exhausted")
        entry = self.script[self._calls]
        self._calls += 1
        if isinstance(entry, str):
            return self._failure(entry)
        return self._render(entry)

    def _failure(self, token: str) -> str:
        if token in ("rate_limit", "session_limit"):
            raise RateLimited(f"mock planner {token}")
        if token in ("transient", "timeout", "hard"):
            raise PlannerError(f"mock planner {token}")
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
                "Let me think. The tip looks ready, so I will proceed.\n\n"
                f"{body}\n\nThat is my single decision."
            )
        return body


#: A decision-level script entry: a typed ``Decision`` returned verbatim, or a
#: failure token (``rate_limit`` -> ``RateLimited`` node #1, ``error`` ->
#: ``PlannerError`` node #2).
DecisionEntry = Union[Decision, str]


@dataclass
class MockDecisionPlanner:
    """The loop's planner seam as a test double (symmetric to the worker mocks):
    ``decide`` returns the next scripted ``Decision`` (or raises a scripted failure)
    instead of dispatching a real rig. It records every ``PlannerContext`` it was
    handed so a test can assert the loop rebuilt the context from disk (e.g. the
    prior epoch's carried failures landed on the next boundary)."""

    script: list[DecisionEntry]
    contexts: list["PlannerContext"] = field(default_factory=list)
    _calls: int = field(default=0)

    def decide(self, context: "PlannerContext") -> Decision:
        self.contexts.append(context)
        if self._calls >= len(self.script):
            raise AssertionError("mock decision planner script exhausted")
        entry = self.script[self._calls]
        self._calls += 1
        if isinstance(entry, str):
            if entry in ("rate_limit", "session_limit"):
                raise RateLimited(f"mock planner {entry}")
            if entry in ("error", "hard", "transient"):
                raise PlannerError(f"mock planner {entry}")
            raise ValueError(f"unknown mock decision token: {entry!r}")
        return entry
