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

Failure taxonomy: ``rate_limit`` / ``session_limit`` -> ``RateLimited`` (node #1, the
loop parks and re-issues); ``timeout`` -> ``PlannerTimeout`` (the loop retries once then
backs off); ``transient`` / ``hard`` / ``error`` -> ``PlannerError`` (a generic transport
fault the loop retries under its cap); ``bad_json`` / ``empty`` / ``invalid`` ->
un-gateable output the core re-asks on. All planner failures auto-recover on the PLAN
call now; only the consecutive-failure cap ends the run.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Union

from grindstone import worktree as wt
from grindstone.check_decision import DECISION_FILE
from grindstone.planner import BATON_FILE
from grindstone.contracts.models import (
    Decision,
    EndDecision,
    Epoch,
    EpochDecision,
    Task,
)
from grindstone.planner import (
    PlannerDispatch,
    PlannerError,
    PlannerTimeout,
    RateLimited,
)

if TYPE_CHECKING:
    from grindstone.loop import CloseoutContext, PlannerContext


def render_mock_baton(context: "CloseoutContext") -> str:
    """A deterministic four-section baton the mock planners write at close-out, so a
    loop test can drive the full lifecycle AND assert the planner's memory carries the
    epoch's outcomes (the escalation reasons, the integration conflict, the setup error)
    into the next boundary's ``context.baton``."""

    done = [o for o in context.task_outcomes if o.outcome == "passed"]
    pending = [o for o in context.task_outcomes if o.outcome != "passed"]
    lines = [
        "## Project summary",
        f"epoch {context.epoch_id} ({context.title}); job in progress.",
        "## Tasks done",
    ]
    lines += [f"- {o.task_id} ({o.mode}) passed" for o in done] or ["- (none this epoch)"]
    lines.append("## Pending")
    lines += [
        f"- {o.task_id} {o.outcome}: {o.reason}" for o in pending
    ] or ["- (none)"]
    lines.append("## Current status")
    if context.setup_error is not None:
        lines.append(f"setup_error: {context.setup_error}")
    if context.integration_conflict is not None:
        lines.append(f"conflict: {context.integration_conflict}")
    for o in pending:
        lines.append(f"{o.task_id} {o.outcome}: {o.reason}")
    if not (pending or context.setup_error or context.integration_conflict):
        lines.append("everything passed.")
    return "\n".join(lines) + "\n"

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
        if token == "timeout":
            raise PlannerTimeout(f"mock planner {token}")
        if token in ("transient", "hard"):
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
    each result channel. ``decision_json`` lands at ``workdir/decision.json`` and
    ``baton`` at ``workdir/baton.md`` (the self-validated / free-form disk contracts a
    writable rig grinds in the worktree), ``out`` at the ``--out`` file, ``stdout`` is
    returned. ``None`` leaves a channel un-written, so a test pins exactly which channel
    the read-priority must pick (and can put DIFFERENT content on each)."""

    decision_json: str | None = None
    baton: str | None = None
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
        if entry.baton is not None:
            (request.workdir / BATON_FILE).write_text(entry.baton, encoding="utf-8")
        if entry.out is not None:
            request.out_file.write_text(entry.out, encoding="utf-8")
        return entry.stdout

    @staticmethod
    def _failure(token: str) -> str:
        if token in ("rate_limit", "session_limit"):
            raise RateLimited(f"mock planner {token}")
        if token == "timeout":
            raise PlannerTimeout(f"mock planner {token}")
        if token in ("error", "transient", "hard"):
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
    instead of dispatching a real rig, and ``close_out`` writes a deterministic baton.
    It records every ``PlannerContext`` and ``CloseoutContext`` it was handed so a test
    can assert the loop rebuilt the context from disk (e.g. the prior epoch's baton
    landed on the next boundary). ``closeout_rate_limit_once`` makes the FIRST
    ``close_out`` raise ``RateLimited`` exactly once (the node-#1 raze + epoch-restart
    proof), then behave."""

    script: Sequence[DecisionEntry]
    contexts: list["PlannerContext"] = field(default_factory=list)
    closeout_contexts: list["CloseoutContext"] = field(default_factory=list)
    closeout_rate_limit_once: bool = False
    _calls: int = field(default=0)
    _closeout_rl_fired: bool = field(default=False)

    def decide(self, context: "PlannerContext") -> Decision:
        self.contexts.append(context)
        if self._calls >= len(self.script):
            raise AssertionError("mock decision planner script exhausted")
        entry = self.script[self._calls]
        self._calls += 1
        if isinstance(entry, str):
            if entry in ("rate_limit", "session_limit"):
                raise RateLimited(f"mock planner {entry}")
            if entry == "timeout":
                raise PlannerTimeout(f"mock planner {entry}")
            if entry in ("error", "hard", "transient"):
                raise PlannerError(f"mock planner {entry}")
            raise ValueError(f"unknown mock decision token: {entry!r}")
        return entry

    def close_out(self, context: "CloseoutContext") -> str:
        self.closeout_contexts.append(context)
        if self.closeout_rate_limit_once and not self._closeout_rl_fired:
            self._closeout_rl_fired = True
            raise RateLimited("mock planner close-out rate limit")
        return render_mock_baton(context)


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
    closeout_contexts: list["CloseoutContext"] = field(default_factory=list)
    tip_history: list[str | None] = field(default_factory=list)
    _research_tries: int = 0
    _impl_tries: int = 0
    _review_tries: int = 0

    def decide(self, context: "PlannerContext") -> Decision:
        self.contexts.append(context)
        self.tip_history.append(context.tip_ref)
        n = context.epoch_index
        tree = self._tip_tree(context)

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

        missing = [f for f in self.impl_files if f not in tree]
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

        built = [f for f in self.impl_files if f in tree]
        return EndDecision(
            kind="end",
            summary=f"built {len(built)} of {len(self.impl_files)} modules",
        )

    def close_out(self, context: "CloseoutContext") -> str:
        self.closeout_contexts.append(context)
        return render_mock_baton(context)

    @staticmethod
    def _tip_tree(context: "PlannerContext") -> set[str]:
        """The integration-tip tree, read straight from git (the real planner greps its
        workdir; the mock reads the tree it has repo + tip_ref for)."""

        if context.repo is None or context.tip_ref is None:
            return set()
        return set(wt.list_tree(context.repo, context.tip_ref))

    @staticmethod
    def _epoch(title: str, tasks: list[Task]) -> EpochDecision:
        return EpochDecision(kind="epoch", epoch=Epoch(title=title, tasks=tasks))

    @staticmethod
    def _has_artifact(context: "PlannerContext", name: str) -> bool:
        return any(key.endswith("/" + name) for key in context.log_index)
