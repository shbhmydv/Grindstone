"""The real planner: stateless one-shot, self-validated on disk.

BONES: the planner is STATELESS per call. Each boundary the loop rebuilds a bounded
``PlannerContext`` from disk (job + integration tip + keyed log + the prior epoch's
carried failures); the planner renders ONE prompt from it, grounds itself in a
throwaway checkout of the tip (its ``_planner_tip`` worktree), self-validates its
``decision.json`` against the SAME core gate the orchestrator will apply, and
returns ONE typed ``Decision`` (an epoch, or an end). Model proposes, state machine
disposes.

This module owns everything the model does not:

* ``build_planner_input`` (PURE): a byte-stable CORE (the epochs-only operating
  instructions, identical every call so a backend can prefix-cache it) plus the
  volatile tail (the job, the integration-tip file list, the keyed-log index, the
  carried failures, the domain-skill catalogue index, and the read-tools note).
* ``ScriptPlanner.decide``: manage the ``_planner_tip`` worktree, arm the on-disk
  validator, dispatch the rig through the ``PlannerTransport`` seam, read the result
  by the priority ``decision.json`` > ``--out`` > stdout, and parse it. An invalid
  or un-extractable decision is re-asked (the rig already self-corrects on disk;
  this is the defense-in-depth budget); an exhausted budget is ``PlannerError``.
* The two-node failure taxonomy: ``RateLimited`` (node #1, the loop parks ~1/hr and
  re-issues the boundary) and ``PlannerError`` (node #2, the loop ends cleanly).
  ``decide`` NEVER returns an unvalidated decision.

The real subprocess transport is ``script_planner.ScriptPlannerTransport`` (mirrors
``script_worker``); tests drive ``decide`` through a mock transport.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from grindstone import worktree as wt
from grindstone.check_decision import (
    DECISION_FILE,
    extract_decision_json,
    write_validator,
)
from grindstone.contracts.models import Decision, parse_decision
from grindstone.domain_skills import load_domain_skill_index

if TYPE_CHECKING:  # pragma: no cover - typing only (avoids the loop<->planner cycle)
    from grindstone.loop import PlannerContext


# --- the two-node failure taxonomy ---------------------------------------------


class PlannerError(Exception):
    """Any planner failure that cannot be recovered (auth, transport, a decision the
    planner could not make valid within its budget). Routes to the run's clean
    partial-end (BONES failure model #2)."""


class RateLimited(PlannerError):
    """A rate-limit / quota refusal (BONES failure model #1): the loop backs off
    (~1/hr) and re-issues the boundary call; nothing is burned."""


# --- the transport seam (the rig dispatch boundary) ----------------------------


@dataclass(frozen=True)
class PlannerDispatch:
    """One fully-resolved planner dispatch. The rig grinds in ``workdir`` (the
    in-repo ``_planner_tip`` checkout of the integration tip) and writes its decision
    to ``workdir/decision.json`` (self-validate) and/or to ``out_file`` (its final
    message); ``dispatch`` returns the rig's raw stdout. The core reads the three
    channels back in priority order, so the transport only RUNS the rig."""

    prompt: str
    workdir: Path
    out_file: Path


class PlannerTransport(Protocol):
    """The uniform rig boundary the planner dispatches through. ``dispatch`` runs the
    rig and returns its raw stdout; it raises ``RateLimited`` (node #1) or
    ``PlannerError`` (node #2). It NEVER parses or validates (that stays in core)."""

    def dispatch(self, request: PlannerDispatch) -> str: ...


# --- input construction (PURE: byte-stable CORE + volatile tail) ---------------

#: Cap on the integration-tip file listing rendered in the tail: a bounded
#: reference, not a payload (the planner greps its workdir for anything further).
TIP_FILES_CAP = 200

#: The defense-in-depth re-ask budget on top of the rig's own on-disk self-validate
#: loop: a rig that lands an invalid (or no) decision is re-asked this many times
#: before the boundary fails to ``PlannerError`` (node #2). The rig usually
#: self-corrects on disk, so this rarely fires.
MAX_REASKS = 2

#: The byte-stable CORE: the epochs-only operating instructions, identical every
#: call of every run (so a backend prefix-caches it). All that MOVES rides the
#: volatile tail. No phases, no skeleton, no budget, no verify-command authoring.
PLANNER_CORE = """\
You are the planner for Grindstone, a deterministic epoch-based orchestrator. You
are a STATELESS one-shot call: the fixed state machine runs the loop, runs every
check, integrates the work, and disposes of your decision. You decide only the
NEXT step. Model proposes, state machine disposes.

Emit EXACTLY ONE JSON object and nothing else (no prose, no second object, no
markdown fence). It is ONE of two shapes:
  EPOCH: {"kind":"epoch","epoch":{"title":..,"rationale":..,"tasks":[ ... ],"setup":[ ... ]}}
  END:   {"kind":"end","summary":".."}

These rules hold for EVERY decision:
- YOU own all sequencing. There are no phases, no skeleton, no epoch budget: you
  steer the run yourself, one epoch at a time, until the job is met, then emit END.
- Propose the SINGLE next epoch as 1 to 8 INDEPENDENT tasks that fan out in
  parallel. Each task carries: an id ("T1".."T8"); a mode (implement | research |
  review | artifact); a routing tier ("local", the default, for mechanical or
  checkable work; "senior" for judgment, taste, or synthesis); and a prose goal
  that states the task's OWN notion of done.
  * implement tasks declare file_ownership: a list of CONCRETE files (>= 1) the
    task may create or edit. Ownership across the epoch's tasks MUST be DISJOINT,
    no two tasks may touch the same file; the state machine enforces disjointness
    and refuses to integrate an overlap.
  * research / review / artifact tasks declare artifact_out: the ONE log key the
    deliverable lands at. They do not own or edit tree files.
- A FRESH run usually leans research-first (understand the job, lay groundwork),
  but that is NOT forced: a small job may be a single epoch, and once the job is
  met you simply emit END.
- Declare host mutations as SETUP. If an epoch needs an install or any host-level
  command (npm ci, pip install, creating a shared dir), list it in the epoch's
  "setup": the TRUSTED state machine runs those, in order, BEFORE the tasks. The
  untrusted worker NEVER mutates the host, so anything a task needs outside its own
  worktree MUST be declared as setup.
- Do NOT author verify or test commands as a gate. You write no done_when and no
  check commands: an independent agentic CRITIC re-derives each task's goal and
  judges the work against it. Carry acceptance in the task's prose goal, never as a
  shell command.
- Keep tasks BOUNDED and ownership DISJOINT. Enumerate concrete files; never claim
  a whole subtree or a wildcard you cannot bound.
- Select domain skills per task. When the <domain_skills> catalogue below lists a
  skill relevant to a task, name it in that task's "skills" (retrieve, do not
  attach the whole catalogue); the core delivers only the selected skill to that
  task's worker.
- inputs are log keys that ALREADY EXIST in the <keyed_log> index below; never
  invent one. Produced artifacts are referenced by their log key, never inlined.
- END when the job is met (summary = what was accomplished) OR when you cannot make
  progress (summary = a phase handoff that seeds the next appendable run).

To LAND your decision: you run inside a throwaway checkout of the current
integration tip (your workdir). Write your decision to ./decision.json, run
`python3 check_decision.py decision.json`, FIX every violation it prints, and loop
until it exits 0. That gate-clean decision.json is your ONLY output.
"""


def _state_block(context: PlannerContext) -> str:
    log = "\n".join(f"  - {k}" for k in context.log_index) or "  (empty)"
    return (
        "<state>\n"
        f"epoch {context.epoch_index} of at most {context.max_epochs}\n"
        "<keyed_log>\n"
        f"{log}\n"
        "</keyed_log>\n"
        "</state>\n"
    )


def _tip_block(context: PlannerContext) -> str:
    files = context.tip_files
    if not files:
        return "<integration_tip>\n  (none yet, the repo is empty or unborn)\n</integration_tip>\n"
    shown = files[:TIP_FILES_CAP]
    more = "" if len(shown) >= len(files) else f" (showing {len(shown)})"
    body = "\n".join(f"  - {f}" for f in shown)
    return f"<integration_tip files={len(files)}{more}>\n{body}\n</integration_tip>\n"


def _carried_block(context: PlannerContext) -> str:
    if not context.carried:
        return "<carried>\n  (none, this is the first epoch)\n</carried>\n"
    body = "\n".join(f"  - {c}" for c in context.carried)
    return (
        "<carried>\n"
        "The prior epoch left these outcomes UNRESOLVED (a worker blocked, a critic "
        "escalated, or an ownership conflict). Steer around them or end on them:\n"
        f"{body}\n"
        "</carried>\n"
    )


def _domain_skills_block(index: dict[str, str]) -> str:
    if not index:
        return ""
    lines = "\n".join(f"  - {name}: {desc}" for name, desc in sorted(index.items()))
    return (
        "<domain_skills>\n"
        "Domain skills this target repo provides. SELECT the relevant ones for a task "
        "by listing their NAMES in that task's \"skills\" field; the core delivers the "
        "selected skill TEXT to that task's worker. Keep selection MINIMAL (retrieve, "
        "do not attach the whole catalogue):\n"
        f"{lines}\n"
        "</domain_skills>\n"
    )


_TOOLS_BLOCK = (
    "<tools>\n"
    "Your workdir is a checkout of the current integration tip. GREP and READ it to "
    "ground your plan (what already exists, where things live). For a large or "
    "unfamiliar area, the grindstone repo-map (grindstone/repomap.py) ranks the "
    "most-referenced files and symbols. Reading is your own internal step; your turn "
    "still ends with exactly one decision written to ./decision.json.\n"
    "</tools>\n"
)


def build_planner_input(
    context: PlannerContext,
    *,
    domain_skill_index: dict[str, str],
    reask_errors: tuple[str, ...] = (),
) -> str:
    """Render the full planner prompt from ``context`` (PURE, no I/O).

    ``PLANNER_CORE`` (byte-stable) then the volatile tail: the job spec, the running
    state + keyed-log index, the integration-tip file listing, the carried failures,
    the domain-skill catalogue index (when the repo ships one), the read-tools note,
    any re-ask feedback, and the request. References, not payloads: only names and
    log keys, never file bodies.
    """

    errors = ""
    if reask_errors:
        joined = "\n".join(f"  - {e}" for e in reask_errors)
        errors = (
            "<errors>\nYour previous decision was REJECTED. Fix and re-emit:\n"
            f"{joined}\n</errors>\n"
        )
    return (
        f"<system>\n{PLANNER_CORE}</system>\n"
        f"<job>\n{context.job}\n</job>\n"
        f"{_state_block(context)}"
        f"{_tip_block(context)}"
        f"{_carried_block(context)}"
        f"{_domain_skills_block(domain_skill_index)}"
        f"{_TOOLS_BLOCK}"
        f"{errors}"
        "<request>\n"
        "Emit exactly one decision (epoch or end) as a single JSON object. No prose.\n"
        "</request>\n"
    )


# --- the planner (the loop's stateless boundary) -------------------------------

#: The in-repo planner-READ/WRITE worktree (a checkout of the integration tip a
#: sandboxed rig must reach inside the repo, distinct from the external task
#: worktrees). Lives under the run dir; refreshed at the tip, reused across
#: boundaries.
_TIP_DIRNAME = "_planner_tip"
#: The rig's ``--out`` fallback file (the second read-priority channel).
_OUT_FILENAME = "_planner_out.txt"


def _read_result(decision_path: Path, out_file: Path, stdout: str) -> str:
    """Read the rig's result by priority: ``decision.json`` > ``--out`` > stdout.

    A self-validating rig wrote a gate-clean ``decision.json`` (the real proof, the
    file it looped the check on); else its final message is in ``--out``; else only
    stdout has bytes. A present-but-empty file falls through to the next channel."""

    for path in (decision_path, out_file):
        if path.is_file():
            text = path.read_text(encoding="utf-8")
            if text.strip():
                return text
    return stdout


@dataclass
class ScriptPlanner:
    """The loop's ``Planner``: ``decide(context) -> Decision`` behind a transport.

    Stateless per boundary except the ``_planner_tip`` worktree it reuses (refreshed
    only when the integration tip moves). Construct it with a ``PlannerTransport``
    (the real subprocess rig, or a mock); ``grindstone_python`` is baked into the
    on-disk validator the rig runs (defaults to the running interpreter).
    """

    transport: PlannerTransport
    max_reasks: int = MAX_REASKS
    grindstone_python: str = field(default_factory=lambda: sys.executable)
    _tip_ref: str | None = field(default=None, init=False, repr=False)

    def decide(self, context: PlannerContext) -> Decision:
        """Render the prompt, ground + self-validate on disk, return ONE Decision.

        ``RateLimited`` (node #1) and ``PlannerError`` (node #2) propagate; a decision
        is NEVER returned unvalidated. An invalid / un-extractable decision is re-asked
        up to ``max_reasks`` times (the rig already self-corrects on disk), then the
        boundary fails to ``PlannerError``."""

        workdir = self._ensure_tip(context)
        write_validator(workdir, grindstone_python=self.grindstone_python)
        index = (
            load_domain_skill_index(context.repo) if context.repo is not None else {}
        )
        decision_path = workdir / DECISION_FILE
        out_file = context.run_dir.root / _OUT_FILENAME

        reask: tuple[str, ...] = ()
        last_error = "no attempts"
        for _ in range(self.max_reasks + 1):
            prompt = build_planner_input(
                context, domain_skill_index=index, reask_errors=reask
            )
            # Clear stale channels so a rig that silently writes nothing can never
            # feed us a previous boundary's decision.
            decision_path.unlink(missing_ok=True)
            out_file.unlink(missing_ok=True)
            stdout = self.transport.dispatch(
                PlannerDispatch(prompt=prompt, workdir=workdir, out_file=out_file)
            )
            raw = _read_result(decision_path, out_file, stdout)
            json_text = extract_decision_json(raw)
            if json_text is None:
                last_error = "planner output carried no JSON decision object"
                reask = (last_error,)
                continue
            try:
                return parse_decision(json.loads(json_text))
            except ValueError as exc:
                last_error = str(exc)
                reask = (last_error,)
        raise PlannerError(
            f"planner produced no valid decision after {self.max_reasks + 1} "
            f"attempts: {last_error}"
        )

    def _ensure_tip(self, context: PlannerContext) -> Path:
        """The in-repo ``_planner_tip`` worktree, refreshed at the integration tip.

        Reused across boundaries: only re-checked-out when the tip moves (a completed
        epoch). With no repo / an unborn HEAD there is nothing to check out, so a plain
        scratch dir gives the rig a CWD to write ``decision.json`` + the validator."""

        tip = context.run_dir.root / _TIP_DIRNAME
        repo, tip_ref = context.repo, context.tip_ref
        if repo is None or tip_ref is None:
            tip.mkdir(parents=True, exist_ok=True)
            return tip
        if self._tip_ref == tip_ref and tip.is_dir():
            return tip
        wt.add_worktree_detached(repo, tip, ref=tip_ref)
        self._tip_ref = tip_ref
        return tip
