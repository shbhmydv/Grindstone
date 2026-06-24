"""The real planner: stateless one-shot, two roles (PLAN + CLOSE-OUT).

BONES: the planner is STATELESS per call. At an epoch START the loop rebuilds a
bounded ``PlannerContext`` from disk (job + integration tip + keyed log + the prior
epoch's BATON); ``decide`` renders ONE prompt from it, grounds itself in a throwaway
checkout of the tip (its ``_planner_tip`` worktree), self-validates its
``decision.json`` against the SAME core gate the orchestrator will apply, and returns
ONE typed ``Decision`` (an epoch, or an end). At an epoch END the loop hands a
``CloseoutContext`` (the per-task outcomes + the staging tree); ``close_out`` renders
the close-out prompt, reads the staging tree + the pointed-to handoffs/verdicts, and
returns the updated BATON markdown (free-form, NEVER parsed, like the handoff). Model
proposes, state machine disposes.

This module owns everything the model does not:

* ``build_planner_input`` / ``build_closeout_input`` (PURE): a byte-stable preamble
  (the operating instructions, identical every call so a backend can prefix-cache it)
  plus the volatile tail (the job, the keyed-log index, the baton / the epoch report,
  the domain-skill catalogue index, the read-tools note).
* ``ScriptPlanner.decide`` / ``ScriptPlanner.close_out``: manage the ``_planner_tip``
  worktree (the tip for PLAN, the staging tree for CLOSE-OUT), dispatch the rig through
  the ``PlannerTransport`` seam, and read the result by the priority
  ``decision.json`` / ``baton.md`` > ``--out`` > stdout. ``decide`` arms the on-disk
  validator and re-asks an invalid decision (an exhausted budget is ``PlannerError``);
  ``close_out`` does NO self-validate loop and NO JSON parse (free-form prose).
* The two-node failure taxonomy: ``RateLimited`` (node #1, the loop parks ~1/hr) and
  ``PlannerError`` (node #2, the loop ends cleanly). ``decide`` NEVER returns an
  unvalidated decision; ``close_out`` NEVER hard-fails on content.

The real subprocess transport is ``script_planner.ScriptPlannerTransport`` (mirrors
``script_worker``); tests drive both roles through a mock transport.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol

from grindstone import worktree as wt
from grindstone.check_decision import (
    DECISION_FILE,
    extract_decision_json,
    write_validator,
)
from grindstone.contracts.models import Decision, parse_decision
from grindstone.domain_skills import load_domain_skill_index
from grindstone.rundir import RunDir

if TYPE_CHECKING:  # pragma: no cover - typing only (avoids the loop<->planner cycle)
    from grindstone.loop import CloseoutContext, PlannerContext


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
    """One fully-resolved planner dispatch. The rig grinds in ``workdir`` (the in-repo
    ``_planner_tip`` checkout of the integration tip for PLAN, or the epoch's staging
    tree for CLOSE-OUT) and writes its output to ``workdir/decision.json`` /
    ``workdir/baton.md`` and/or to ``out_file`` (its final message); ``dispatch``
    returns the rig's raw stdout. ``purpose`` tells the rig which role it is running
    (it swaps the self-validate sys-append for the minimal close-out one). The core
    reads the channels back in priority order, so the transport only RUNS the rig."""

    prompt: str
    workdir: Path
    out_file: Path
    purpose: Literal["plan", "closeout"] = "plan"


class PlannerTransport(Protocol):
    """The uniform rig boundary the planner dispatches through. ``dispatch`` runs the
    rig and returns its raw stdout; it raises ``RateLimited`` (node #1) or
    ``PlannerError`` (node #2). It NEVER parses or validates (that stays in core)."""

    def dispatch(self, request: PlannerDispatch) -> str: ...


# --- input construction (PURE: byte-stable preamble + volatile tail) -----------

#: The defense-in-depth re-ask budget on top of the rig's own on-disk self-validate
#: loop: a rig that lands an invalid (or no) decision is re-asked this many times
#: before the boundary fails to ``PlannerError`` (node #2). The rig usually
#: self-corrects on disk, so this rarely fires.
MAX_REASKS = 2

#: The PLAN preamble: the byte-stable operating instructions for the forward (PLAN)
#: role, identical every call of every run (so a backend prefix-caches it). All that
#: MOVES rides the volatile tail. No phases, no skeleton, no budget, no verify-command
#: authoring.
PLAN_PREAMBLE = """\
You are the planner for Grindstone, a deterministic epoch-based orchestrator. A fixed
state machine runs the loop, runs every check, integrates the work, and disposes of
your decision. You are a STATELESS one-shot call: you decide only the NEXT step. Model
proposes, state machine disposes.

HOW THE RUN WORKS. The run advances one EPOCH at a time. Each epoch is 1 to 8
INDEPENDENT tasks that local + cloud workers grind in parallel, each in its own isolated
git worktree. Deterministic checks gate the work (a disjoint-ownership merge of the git
diff, and one final acceptance of the job's own done_when); an independent agentic critic
triages each task. When you closed the previous epoch you wrote a BATON - your living
plan - and it is given to you below as your memory. At the end of THIS epoch you will be
asked to update it. There are no phases and no fixed budget: you steer, epoch by epoch,
until the job is met, then you END.

WHAT YOU CAN SEE AND READ. Your CWD is a throwaway checkout of the current integration
tip. GREP and READ it to ground every decision in what actually exists. You can SEE:
read images directly (screenshots, mockups, diagrams, rendered UI) - never plan blind to
a visual you could open. The keyed log (indexed below) holds prior tasks' handoffs,
critic verdicts, and produced artifacts; read any of them. Reading is your own internal
step; your turn still ends with exactly one decision.

YOUR DECISION. Emit EXACTLY ONE JSON object and nothing else (no prose, no second
object, no markdown fence), ONE of two shapes:
  EPOCH: {"kind":"epoch","epoch":{"title":"..","tasks":[ .. ],"setup":[ .. ]}}
  END:   {"kind":"end","summary":".."}

Rules for every decision:
- YOU own all sequencing. Propose the SINGLE next epoch as 1 to 8 tasks that fan out in
  parallel with NO dependency on each other (a dependency means a later epoch, not a
  task in this one). Each task carries: an id ("T1".."T8"); a mode (implement | research
  | review | artifact); a routing tier ("local", the default, for mechanical or
  checkable work; "senior" for judgment, taste, synthesis, or visual quality); and a
  prose goal that states the task's OWN notion of done.
  * implement tasks declare file_ownership: a list of CONCRETE files (>= 1) the task may
    create or edit. Ownership across the epoch MUST be DISJOINT; the state machine
    refuses to integrate an overlap. Enumerate real files; never claim a subtree or a
    wildcard you cannot bound.
  * research / review / artifact tasks declare artifact_out: the ONE log key the
    deliverable lands at (a report, a verdict, a rendered image - artifacts may be
    visual). They do not own or edit tree files.
- Declare HOST-GLOBAL prep as SETUP. If an epoch needs a host-level mutation (a
  system-wide tool, a shared directory outside the repo), list it in the epoch's "setup":
  the trusted state machine runs those, in order, before the tasks. Setup runs in a
  throwaway checkout, NOT the task worktrees, so do NOT put the project's own dependency
  installs in setup - they would not reach the isolated worktrees. An implement task
  installs the project dependencies it needs inside its OWN worktree as part of its work.
- Do NOT author verify or test commands as a gate. You write no done_when and no check
  commands: the independent critic re-derives each task's goal and judges the work, and
  the job's own done_when is the single final acceptance. Carry acceptance in each task's
  prose goal.
- Select domain skills per task. When the <domain_skills> catalogue below lists a skill
  relevant to a task, name it in that task's "skills" (retrieve, do not attach the whole
  catalogue); the core delivers only the selected skill to that task's worker.
- inputs are log keys that ALREADY EXIST in the <keyed_log> index below; never invent
  one. Produced artifacts are referenced by their log key, never inlined.
- END when the job is met (summary = what was accomplished) or when you cannot make
  progress (summary = a handoff that seeds the next appendable run).

To LAND your decision: you run inside the throwaway checkout (your workdir). Write your
decision to ./decision.json, run `python3 check_decision.py decision.json`, FIX every
violation it prints, and loop until it exits 0. That gate-clean decision.json is your
ONLY output.
"""

#: The CLOSE-OUT preamble: the operating instructions for the backward (CLOSE-OUT)
#: role. It carries the four-section baton skeleton, so ``build_closeout_input`` does
#: NOT duplicate it.
CLOSEOUT_PREAMBLE = """\
You are the planner for Grindstone, closing out the epoch you just ran. A fixed state
machine ran the epoch's tasks, gated and integrated the work, and now asks you for ONE
thing: the updated BATON - your living plan, the memory you pass to your next self.

You are the only one who can judge what really happened, so judge it. A task that the
machine marks "escalated" might be partial progress, no progress, or a regression - only
you can tell, by reading what was attempted. Do not let the machine's flat label stand
in for your judgment.

WHAT TO READ. Your CWD is a throwaway checkout of this epoch's STAGING tree - the work
that actually merged. Grep and read it to see what now exists. For each task in the epoch
report below, READ its handoff (the worker's own report) and its critic verdict at the
keyed-log paths given. You can SEE: VIEW any image the work produced or referenced
(screenshots, rendered UI) and judge it with your eyes, not a description. Reconcile all
of it against your prior baton.

WHAT TO WRITE. Write your updated baton to ./baton.md and nothing else (do not print it).
Free-form markdown, but ALWAYS these four sections:

  ## Project summary
  Where the whole job stands, in a few sentences. Carry it forward and refine it; this is
  the big picture your next self needs to not re-derive the world.

  ## Tasks done
  What is genuinely complete and merged (not just attempted). Be concrete - name the
  capability or files, not the task ids.

  ## Tasks pending
  What still needs doing to meet the job, including anything a failure this epoch left
  undone. This is your to-do list to your next self.

  ## Current status
  The honest now: what this epoch changed, and for every failure, YOUR read of its nature
  (partial progress and what remains / no progress and why / a regression to undo) and how
  to steer next. This is where the nuance lives. If everything passed, say so plainly.

Keep it tight and high-signal - it is a baton, not a log. Your next self reads ONLY this
plus the tree and the keyed log, so put here everything it needs and nothing it can
re-derive by looking.
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


def _baton_block(context: PlannerContext) -> str:
    if not context.baton.strip():
        return "<baton>\n  (none yet, this is the first epoch)\n</baton>\n"
    return (
        "<baton>\nThe living plan you wrote when you closed the previous epoch. It "
        "is your memory across this run: the project so far, what is done, what is "
        "pending, and the current status (including any failures and their nature). "
        "Reconcile it against the actual tree (grep your workdir) - the tree is "
        "ground truth, the baton is your intent:\n"
        f"{context.baton}\n</baton>\n"
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
    "ground your plan (what already exists, where things live). Reading is your own "
    "internal step; your turn still ends with exactly one decision written to "
    "./decision.json.\n"
    "</tools>\n"
)


def build_planner_input(
    context: PlannerContext,
    *,
    domain_skill_index: dict[str, str],
    reask_errors: tuple[str, ...] = (),
) -> str:
    """Render the full PLAN prompt from ``context`` (PURE, no I/O).

    ``PLAN_PREAMBLE`` (byte-stable) then the volatile tail: the job spec, the running
    state + keyed-log index, the prior epoch's BATON, the domain-skill catalogue index
    (when the repo ships one), the read-tools note, any re-ask feedback, and the
    request. References, not payloads: only the baton text, names, and log keys, never
    file bodies (the planner greps its workdir for the tree).
    """

    errors = ""
    if reask_errors:
        joined = "\n".join(f"  - {e}" for e in reask_errors)
        errors = (
            "<errors>\nYour previous decision was REJECTED. Fix and re-emit:\n"
            f"{joined}\n</errors>\n"
        )
    return (
        f"<system>\n{PLAN_PREAMBLE}</system>\n"
        f"<job>\n{context.job}\n</job>\n"
        f"{_state_block(context)}"
        f"{_baton_block(context)}"
        f"{_domain_skills_block(domain_skill_index)}"
        f"{_TOOLS_BLOCK}"
        f"{errors}"
        "<request>\n"
        "Emit exactly one decision (epoch or end) as a single JSON object. No prose.\n"
        "</request>\n"
    )


def _prior_baton_block(context: CloseoutContext) -> str:
    if not context.prior_baton.strip():
        return "<prior_baton>\n  (none, first epoch)\n</prior_baton>\n"
    return f"<prior_baton>\n{context.prior_baton}\n</prior_baton>\n"


def _epoch_report_block(context: CloseoutContext) -> str:
    """Render the deterministic per-task outcomes + keyed-log pointers the close-out
    planner OPENS and judges (Python labels nothing; the model reads the pointed-to
    handoff/verdict and writes the nuance)."""

    lines = [
        f"<epoch_report epoch=\"{context.epoch_id}\" title=\"{context.title}\">",
        "For each task: its deterministic outcome, the keyed-log files to READ (the "
        "worker handoff + the critic verdict), and the verbatim reason. OPEN those "
        "files (and VIEW any images) and judge what really happened, then write the "
        "baton.",
    ]
    if context.setup_error is not None:
        lines.append(f"  setup_error: {context.setup_error}")
    if context.integration_conflict is not None:
        lines.append(f"  integration_conflict: {context.integration_conflict}")
    if not context.task_outcomes:
        lines.append("  (no tasks ran this epoch)")
    for o in context.task_outcomes:
        lines.append(f"  - {o.task_id} ({o.mode}): {o.outcome}")
        lines.append(f"      handoff: {o.handoff_key or '(none)'}")
        lines.append(f"      verdict: {o.verdict_key or '(none)'}")
        if o.reason:
            lines.append(f"      reason: {o.reason}")
    lines.append("</epoch_report>\n")
    return "\n".join(lines)


_CLOSEOUT_TOOLS_BLOCK = (
    "<tools>\n"
    "Your workdir is a checkout of this epoch's staging tree. GREP and READ it, and "
    "READ the keyed-log handoffs + verdicts named above (VIEW any images - you can "
    "see). Then write ./baton.md and stop.\n"
    "</tools>\n"
)


def build_closeout_input(
    context: CloseoutContext,
    *,
    domain_skill_index: dict[str, str] | None = None,
) -> str:
    """Render the full CLOSE-OUT prompt from ``context`` (PURE, no I/O).

    ``CLOSEOUT_PREAMBLE`` (byte-stable, and it already carries the four-section baton
    skeleton) then the volatile tail: the job, the prior baton, the epoch report (the
    deterministic outcomes + keyed-log pointers), the domain-skill catalogue index
    (so the baton's pending list can name a skill the next epoch will select), the
    tools/vision note, and the request to write ``./baton.md``.
    """

    skills = _domain_skills_block(domain_skill_index or {})
    return (
        f"<system>\n{CLOSEOUT_PREAMBLE}</system>\n"
        f"<job>\n{context.job}\n</job>\n"
        f"{_prior_baton_block(context)}"
        f"{_epoch_report_block(context)}"
        f"{skills}"
        f"{_CLOSEOUT_TOOLS_BLOCK}"
        "<request>\n"
        "Write your updated baton (the four sections) to ./baton.md and nothing else. "
        "Do not print it.\n"
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
#: The free-form close-out BATON the rig writes in its workdir (the disk contract,
#: NEVER parsed, like the worker handoff).
BATON_FILE = "baton.md"


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
    """The loop's ``Planner``: ``decide`` (PLAN) + ``close_out`` (CLOSE-OUT) behind one
    transport.

    Stateless per boundary except the ``_planner_tip`` worktree it reuses (refreshed
    only when the checked-out ref moves: the integration tip for PLAN, the staging tree
    for CLOSE-OUT). Construct it with a ``PlannerTransport`` (the real subprocess rig,
    or a mock); ``grindstone_python`` is baked into the on-disk validator the rig runs
    (defaults to the running interpreter).
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

        workdir = self._ensure_worktree(context.repo, context.tip_ref, context.run_dir)
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
                last_error = "planner output contained no JSON decision object"
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

    def close_out(self, context: CloseoutContext) -> str:
        """Render the close-out prompt over a checkout of the epoch's staging tree,
        dispatch the rig (``purpose="closeout"``), and return the BATON markdown read
        back by priority ``baton.md`` > ``--out`` > stdout.

        Mirrors ``decide`` minus the self-validate loop and the JSON parse: the baton
        is FREE-FORM (never parsed, like the handoff). ``RateLimited`` (node #1)
        propagates so the loop razes + restarts the epoch; any other transport hard
        error propagates as ``PlannerError`` (the loop's abort path). It NEVER raises
        on content: whatever prose the rig produced is returned."""

        workdir = self._ensure_worktree(
            context.repo, context.staging_ref, context.run_dir
        )
        index = (
            load_domain_skill_index(context.repo) if context.repo is not None else {}
        )
        baton_path = workdir / BATON_FILE
        out_file = context.run_dir.root / _OUT_FILENAME
        baton_path.unlink(missing_ok=True)
        out_file.unlink(missing_ok=True)
        prompt = build_closeout_input(context, domain_skill_index=index)
        stdout = self.transport.dispatch(
            PlannerDispatch(
                prompt=prompt, workdir=workdir, out_file=out_file, purpose="closeout"
            )
        )
        return _read_result(baton_path, out_file, stdout)

    def _ensure_worktree(
        self, repo: Path | None, ref: str | None, run_dir: RunDir
    ) -> Path:
        """The in-repo ``_planner_tip`` worktree, refreshed to ``ref`` (the integration
        tip for PLAN, the staging tree for CLOSE-OUT).

        Reused across calls: only re-checked-out when ``ref`` moves. With no repo / an
        unborn HEAD there is nothing to check out, so a plain scratch dir gives the rig
        a CWD to write its ``decision.json`` / ``baton.md`` + the validator."""

        tip = run_dir.root / _TIP_DIRNAME
        if repo is None or ref is None:
            tip.mkdir(parents=True, exist_ok=True)
            return tip
        if self._tip_ref == ref and tip.is_dir():
            return tip
        wt.add_worktree_detached(repo, tip, ref=ref)
        self._tip_ref = ref
        return tip
