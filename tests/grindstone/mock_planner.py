"""Deterministic mock planner: scripted decisions + scripted failures.

Three test doubles for the planner seams the rewrite exposes:

* ``MockPlanner`` -- the RAW-TEXT transport double (``plan(prompt) -> str``): scripts
  decision dicts (optionally fence-/prose-wrapped so the core's extractor is
  exercised) or failure tokens. Used where only the text channel matters.
* ``MockPlannerTransport`` -- the ``PlannerTransport`` seam (``dispatch(request) ->
  str``, symmetric to ``mock_worker``): each scripted ``MockRig`` writes a decision
  to any of the three result channels (``decision.json`` in the workdir, the
  ``--out`` file, stdout), so ``ScriptPlanner.decide``'s read-priority + re-ask loop
  is driven with zero randomness. A failure token raises instead.
* ``MockDecisionPlanner`` -- the loop's ``Planner`` seam (``decide(context) ->
  Decision``): scripts typed decisions, recording each context so a loop test can
  assert the boundary was rebuilt from disk.

Failure taxonomy (the bones two-node model): ``rate_limit`` / ``session_limit`` ->
``RateLimited`` (back off and re-issue); ``transient`` / ``timeout`` / ``hard`` /
``error`` -> ``PlannerError`` (the cannot-continue catch-all); ``bad_json`` /
``empty`` / ``invalid`` -> un-gateable output the core re-asks on.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Union

from grindstone.check_decision import DECISION_FILE
from grindstone.contracts.models import (
    Decision,
    EndDecision,
    Epoch,
    EpochDecision,
    Task,
)
from grindstone.planner import PlannerDispatch, PlannerError, RateLimited

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


def _decision_text(decision: dict[str, object]) -> str:
    return json.dumps(decision)


@dataclass(frozen=True)
class MockRig:
    """One simulated rig run for ``MockPlannerTransport``: the content it writes to
    each result channel. ``decision_json`` lands at ``workdir/decision.json`` (the
    self-validated disk contract), ``out`` at the ``--out`` file, ``stdout`` is
    returned. ``None`` leaves a channel un-written, so a test pins exactly which
    channel the read-priority must pick (and can put DIFFERENT content on each)."""

    decision_json: str | None = None
    out: str | None = None
    stdout: str = ""

    @classmethod
    def from_decision(
        cls, decision: dict[str, object], *, channel: str = "decision_json"
    ) -> MockRig:
        """A rig that writes ``decision`` to one channel (default the disk contract)."""

        body = _decision_text(decision)
        if channel == "decision_json":
            return cls(decision_json=body)
        if channel == "out":
            return cls(out=body)
        if channel == "stdout":
            return cls(stdout=body)
        raise ValueError(f"unknown channel: {channel!r}")


#: A transport-level script entry: a ``MockRig`` (writes channels) or a failure token.
RigEntry = Union[MockRig, str]


@dataclass
class MockPlannerTransport:
    """The ``PlannerTransport`` seam as a test double: every ``dispatch`` follows the
    next scripted ``MockRig`` (writing the channels a real rig would), or raises a
    scripted failure. Drives ``ScriptPlanner.decide`` end to end without a real rig:
    a ``[invalid_rig, valid_rig]`` script proves the re-ask loop; differing
    per-channel content proves the ``decision.json`` > ``--out`` > stdout priority."""

    script: list[RigEntry]
    _calls: int = field(default=0)

    def dispatch(self, request: PlannerDispatch) -> str:
        if self._calls >= len(self.script):
            raise AssertionError("mock planner transport script exhausted")
        entry = self.script[self._calls]
        self._calls += 1
        if isinstance(entry, str):
            return self._failure(entry)
        if entry.decision_json is not None:
            (request.workdir / DECISION_FILE).write_text(
                entry.decision_json, encoding="utf-8"
            )
        if entry.out is not None:
            request.out_file.write_text(entry.out, encoding="utf-8")
        return entry.stdout

    @staticmethod
    def _failure(token: str) -> str:
        if token in ("rate_limit", "session_limit"):
            raise RateLimited(f"mock planner {token}")
        if token in ("error", "transient", "hard", "timeout"):
            raise PlannerError(f"mock planner {token}")
        raise ValueError(f"unknown mock rig failure token: {token!r}")


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

    script: Sequence[DecisionEntry]
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


@dataclass
class GoalPlanner:
    """A GOAL-DRIVEN loop planner for the stochastic convergence E2E.

    Unlike ``MockDecisionPlanner`` (a fixed script), this one self-steers like the
    real planner: it reads the boundary's DISK STATE (the keyed-log index + the
    integration-tip file list) and proposes the next epoch toward a fixed goal,
    reacting to a stochastic worker's failures. It plans a realistic multi-epoch
    job, RESEARCH (one artifact) -> IMPLEMENT fan-out (one disjoint task per missing
    module) -> REVIEW (one artifact) -> END, and retries a stage that did not land
    (a failed/blocked task leaves no artifact / no file on the tip), each retry on a
    FRESH epoch index so the stochastic worker re-draws (a different full task id ->
    a different outcome), so the run converges.

    Termination is guaranteed two ways: each stage is bounded by ``stage_cap``
    attempts (after which the planner moves on / ends with what it has), and the
    loop's ``max_epochs`` backstop forces a clean partial-end regardless. It records
    every ``PlannerContext`` + the integration tip it saw each boundary so a test can
    assert the run-branch fast-forward invariant.
    """

    impl_files: tuple[str, ...]
    stage_cap: int = 4
    contexts: list["PlannerContext"] = field(default_factory=list)
    tip_history: list[str | None] = field(default_factory=list)
    _research_tries: int = 0
    _impl_tries: int = 0
    _review_tries: int = 0

    def decide(self, context: "PlannerContext") -> Decision:
        self.contexts.append(context)
        self.tip_history.append(context.tip_ref)
        n = context.epoch_index

        if not self._has_artifact(context, "research.md") and (
            self._research_tries < self.stage_cap
        ):
            self._research_tries += 1
            return self._epoch(
                "research",
                [
                    Task(
                        id="T1",
                        mode="research",
                        goal="investigate the job and lay groundwork",
                        artifact_out=f"E{n}/T1/research.md",
                    )
                ],
            )

        missing = [f for f in self.impl_files if f not in context.tip_files]
        if missing and self._impl_tries < self.stage_cap:
            self._impl_tries += 1
            tasks = [
                Task(
                    id=f"T{i + 1}",
                    mode="implement",
                    goal=f"build {f}",
                    file_ownership=[f],
                )
                for i, f in enumerate(missing[:8])
            ]
            return self._epoch("build the missing modules", tasks)

        if not self._has_artifact(context, "review.md") and (
            self._review_tries < self.stage_cap
        ):
            self._review_tries += 1
            return self._epoch(
                "review",
                [
                    Task(
                        id="T1",
                        mode="review",
                        goal="re-derive the job and reconcile against the built modules",
                        artifact_out=f"E{n}/T1/review.md",
                    )
                ],
            )

        built = [f for f in self.impl_files if f in context.tip_files]
        return EndDecision(
            kind="end",
            summary=f"built {len(built)} of {len(self.impl_files)} modules",
        )

    @staticmethod
    def _epoch(title: str, tasks: list[Task]) -> EpochDecision:
        return EpochDecision(kind="epoch", epoch=Epoch(title=title, tasks=tasks))

    @staticmethod
    def _has_artifact(context: "PlannerContext", name: str) -> bool:
        return any(key.endswith("/" + name) for key in context.log_index)
