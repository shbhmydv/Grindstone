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
import shutil
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
from grindstone.repo_map import load_repo_map
from grindstone.rundir import RunDir
from grindstone.strategy_skill import load_strategy
from grindstone.strikes import CarriedItem

if TYPE_CHECKING:  # pragma: no cover - typing only (avoids the loop<->planner cycle)
    from grindstone.loop import CloseoutContext, PlannerContext


# --- the two-node failure taxonomy ---------------------------------------------


class PlannerError(Exception):
    """A planner failure (auth, transport, a decision the planner could not make valid
    within its budget). On the PLAN call the loop RETRIES it under a consecutive-failure
    cap (a transport fault is a transient, not the planner's judgment to end); only when
    the cap is exhausted does it fall to the run's clean partial-end (BONES failure model
    #2). On the CLOSE-OUT call it routes to the epoch abort node."""


class RateLimited(PlannerError):
    """A rate-limit / quota refusal (BONES failure model #1): the loop backs off
    (~1/hr) and re-issues the boundary call; nothing is burned."""


class PlannerTimeout(PlannerError):
    """A planner wall-clock timeout: a TRANSIENT (often a flaky 1-in-N), NOT the
    planner's judgment to end. A ``PlannerError`` subclass so the loop's unified
    planner-retry gives it the immediate-retry-once treatment (then a backoff on a
    repeat) without a string match, rather than halting an unattended run."""


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

WHAT YOU CAN SEE, READ, AND RUN. Your CWD is a throwaway checkout of the current
integration tip, and you have a FULL shell in it - nothing you do here touches durable
state, so investigate freely. GREP and READ it to ground every decision in what actually
exists. You can SEE: read images directly (screenshots, mockups, diagrams, rendered UI) -
never plan blind to a visual you could open. The keyed log (indexed below) holds prior
tasks' handoffs, critic verdicts, and produced artifacts; read any of them. Your last
close-out persisted an EVIDENCE bundle at E<n-1>/baton-artifacts/ (pointed to below when
it exists) - the render it built and viewed, the gate output, the diffs: OPEN it and VIEW
the render as PRIMARY ground truth, ahead of the baton's prose. And you MAY RUN
verification commands here (build, test, render, inspect) to see the real state with your
own eyes before you decide. This is you informing YOURSELF: it does NOT change the rule
below that you author no verify or check commands as worker tasks. Reading and running
are your own internal step; your turn still ends with exactly one decision.

YOUR DECISION. Emit EXACTLY ONE JSON object and nothing else (no prose, no second
object, no markdown fence), ONE of two shapes:
  EPOCH: {"kind":"epoch","epoch":{"title":"..","tasks":[ .. ],"setup":[ .. ]},"pending":[ .. ]}
  END:   {"kind":"end","summary":".."}

Rules for every decision:
- YOU own all sequencing, and the run is EXPECTED to span many epochs - you are not
  trying to finish in one. Propose the SINGLE next epoch as 1 to 8 tasks that fan out in
  parallel with NO dependency on each other. Sibling tasks grind in ISOLATED worktrees
  off the same base and CANNOT see each other's output; the only sync point is the merge
  at the epoch boundary. So if a task would need anything another task produces (code
  under test, a module to import, an interface to build on), it does NOT belong in this
  epoch: schedule the producer now and let the NEXT epoch, which sees it merged, do the
  consumer. Tests come after the code they test; integration after its parts.
  SPLIT LIBERALLY within that constraint: prefer MANY small independent tasks over a few
  big ones - decompose to the finest grain that stays dependency-free (one task per
  screen, per module, per component group, per report) and fill the fan-out toward the
  8-task ceiling whenever independent work exists. A task that bundles several independent
  pieces is a wasted fan-out: break it apart. More tasks puts more work on the cheap local
  tier, leaves fewer pieces big enough to need senior, and takes fewer TOTAL epochs - so
  fewer planner boundaries, each of which would otherwise reprocess a growing baton and log.
  Each task
  carries: an id ("T1".."T8"); a mode (implement | research
  | review | artifact); a routing tier ("local", the default, for mechanical or
  checkable work; "senior" for judgment, taste, synthesis, or visual quality); and a
  prose goal that states the task's OWN notion of done. Default every task to "local"
  and JUSTIFY each "senior" - it is the scarce, expensive tier. A taste-critical FEATURE
  is not a wholesale-"senior" feature: split its mechanical substrate (literal
  transcription of spec'd values into tokens/constants, type definitions, config,
  scaffolding - work with one correct answer) into "local" tasks, and reserve "senior"
  for the slices where judgment changes the output (component feel, layout, composition,
  visual quality). Do not route a whole subsystem to "senior" because part of it needs
  taste.
  * implement tasks declare file_ownership: a list of CONCRETE files (>= 1) the task may
    create or edit. Ownership across the epoch MUST be DISJOINT; the state machine
    refuses to integrate an overlap. Enumerate real files; never claim a subtree or a
    wildcard you cannot bound.
  * research / review / artifact tasks declare artifact_out: the ONE log key the
    deliverable lands at (a report, a verdict, a rendered image - artifacts may be
    visual). They do not own or edit tree files.
- Carry a living BACKLOG across epochs. The baton below has a "## Pending" section: the
  deferred work you recorded in earlier epochs but have not done yet. READ it and treat it
  as your standing to-do list. THIS epoch, SCHEDULE a SUBSET of those pending items as
  tasks (fill the fan-out toward the 8-task ceiling whenever the pending items are
  independent of each other), and record any NEW deferred work you are NOT doing yet in
  this decision's "pending" field - one short prose line per item (it MAY list more than
  you scheduled this epoch). There is no drain quota: the backlog is self-balancing (a
  heavy backlog simply fills all 8 slots and drains fast), so do not try to empty it in
  one epoch. The close-out reconciles the backlog deterministically (it removes a pending
  item only when a task that addressed it PASSED the gate), so a pending item you schedule
  but that fails is carried for you automatically; just keep recording genuinely new work.
- SCAFFOLD now, REFINE later to keep judgment work small (a GENERIC decomposition). When a
  task would route to "senior" but most of its body is routine implementation that cannot
  be cleanly carved into its own "local" task, SPLIT IT ACROSS EPOCHS: a SCAFFOLD task
  THIS epoch ("local") that builds the complete, correct structure, and a REFINE task in a
  LATER epoch ("senior") that owns the SAME files and elevates ONLY the judgment layer
  against the merged result. Record the refine as a "pending" addition now; do NOT
  schedule it in the same epoch as its scaffold (a refine owning the scaffold's files is a
  same-epoch ownership overlap the merge gate refuses, and it could not see the scaffold
  anyway - siblings cannot read each other's output).
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
thing: the updated BATON - your living plan, the memory you pass to your next self. You
are the deep-survey right-hand man: do not take the field reports at face value, go and
SEE for yourself what really happened, then report ground truth forward.

You are the only one who can judge what really happened, so judge it. A task that the
machine marks "escalated" might be partial progress, no progress, or a regression - only
you can tell, by reading what was attempted. A worker handoff claims success; the critic
verdict is one lenient read. Neither is proof. VERIFY: do not let a flat label or a
worker's own claim stand in for your judgment.

WHAT TO READ. Your CWD is a throwaway checkout of this epoch's STAGING tree - the work
that actually merged. Grep and read it to see what now exists. For each task in the epoch
report below, READ its handoff (the worker's own report) and its critic verdict at the
keyed-log paths given. Reconcile all of it against your prior baton.

GO VERIFY (you have a full shell in this throwaway checkout, and nothing you run here
touches durable state). Actively confirm what the workers claim rather than trusting it:
- For UI / visual work, BUILD and RENDER the merged result and VIEW the screenshot with
  your own eyes (you can SEE images) - never sign off on a look you only read described.
- For functional work, RUN the relevant checks (build, type-check, tests, a quick probe)
  and watch them pass or fail for real.

WHAT TO PERSIST AS EVIDENCE. Alongside the baton, save a ``baton-artifacts/`` directory
in your CWD holding the EVIDENCE you gathered: the render PNG(s) you produced and viewed,
the key gate / check output, the notable diffs. The state machine relocates this bundle
into the keyed log, so your NEXT self opens GROUND TRUTH - the actual render, the actual
output - not your prose description of it. Keep it small and high-signal; omit it
entirely for a purely functional epoch that produced nothing worth showing.

WHAT TO WRITE. Write your updated baton to ./baton.md and nothing else printed (do not
print the baton). Free-form markdown, but ALWAYS these four sections:

  ## Project summary
  Where the whole job stands, in a few sentences. Carry it forward and refine it; this is
  the big picture your next self needs to not re-derive the world.

  ## Tasks done
  What is genuinely complete and merged (not just attempted). Be concrete - name the
  capability or files, not the task ids.

  ## Pending
  The persisted work BACKLOG your next PLAN schedules from: short, actionable lines of
  deferred work (one bullet each; name the routing tier when it matters, e.g. "(senior)").
  RECONCILE it deterministically, do not rewrite it from scratch: the new backlog is the
  prior baton's ## Pending, PLUS this epoch's planned additions (listed below as
  <pending_additions>), PLUS any new undone work this epoch surfaced, MINUS every prior
  backlog item that was scheduled as a task this epoch AND whose task PASSED the gate.
  Base "done" ONLY on the per-task outcomes in the epoch report (read the handoffs to see
  which backlog item a task addressed); NEVER on your own guess. A backlog item whose
  scheduled task FAILED (escalated, or never merged) STAYS IN THE BACKLOG - carry it.

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


def _baton_artifacts_block(keys: tuple[str, ...]) -> str:
    """The prior epoch's persisted EVIDENCE bundle: the keyed-log paths the deep-survey
    close-out left at ``E<n-1>/baton-artifacts/`` (the render it built + viewed, the gate
    output it ran, the diffs it kept). Point the plan-step at it as PRIMARY ground truth -
    open it, VIEW the render - rather than trusting the baton's prose description of it.

    Empty ``keys`` -> ``""`` (epoch 1, or a run that persisted none: byte-identical to a
    run without the bundle, so the cacheable prefix and the absent path are untouched)."""

    if not keys:
        return ""
    lines = "\n".join(f"  - {k}" for k in keys)
    return (
        "<baton_artifacts>\n"
        "The EVIDENCE your last close-out persisted: the render(s) it built and viewed, "
        "the gate/check output it ran, the notable diffs - ground truth, not a "
        "description of it. OPEN these keyed-log files and VIEW the render(s) before you "
        "decide; trust them over the baton's prose:\n"
        f"{lines}\n"
        "</baton_artifacts>\n"
    )


def _carried_block(items: tuple[CarriedItem, ...]) -> str:
    """The strike-ladder NUDGE (soft planner guidance): the task lineages that failed
    the WHOLE in-epoch tier ladder (both local and senior already tried, in one epoch)
    and were carried unfinished, each flagged with what the DETERMINISTIC state machine
    will do if you re-issue an overlapping task (one reframe chance at strike 1, BLOCK
    at strike 2).

    Empty ``items`` -> ``""`` (so a run that never carried a task is byte-identical to
    today and the cacheable system prefix is untouched). The instruction line is the
    repair-epoch guidance: do NOT re-issue the same framing, REFRAME or RE-DECOMPOSE;
    a re-decomposed child inherits its parent's strikes, so splitting cannot dodge the
    block."""

    if not items:
        return ""
    lines = []
    for it in items:
        if it.parked:
            tag = "BLOCKED by the state machine - do NOT re-issue this; it was dropped"
        else:
            tag = (
                "carried unfinished once (BOTH the local and senior tiers already "
                "failed it in-epoch) - REFRAME or RE-DECOMPOSE into smaller pieces; a "
                "re-decomposed child INHERITS the strike and a second failure BLOCKS "
                "the lineage"
            )
        reason = f"; last failure: {it.reason}" if it.reason else ""
        lines.append(
            f"  - {it.descriptor} [{it.mode}], carried unfinished "
            f"{it.strikes}x: {tag}{reason}"
        )
    body = "\n".join(lines)
    return (
        "<carried>\n"
        "These task LINEAGES failed the WHOLE in-epoch tier ladder (each attempt was "
        "already retried locally AND escalated to the senior tier within its epoch) and "
        "were carried unfinished (one strike per failed epoch). Re-issuing the SAME "
        "framing has not worked: when you schedule one again, REFRAME or RE-DECOMPOSE it "
        "into smaller, differently-shaped tasks. A re-decomposed child INHERITS its "
        "parent's strikes (splitting does not reset the ladder), and a SECOND full-ladder "
        "failure BLOCKS the lineage: the state machine drops it from the active set.\n"
        f"{body}\n</carried>\n"
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


def _strategy_block(text: str) -> str:
    """The target repo's always-on PLANNER strategy overlay, framed as an ADVISORY
    extension of the operating preamble - NEVER an override of the mechanics.

    Empty text -> ``""`` (so the no-strategy prompt is byte-identical to today and the
    byte-stable ``<system>...</system>`` prefix stays cacheable). Non-empty -> a tagged
    block whose first line subordinates the strategy to the operating rules / decision
    contract / gates above (preferences, not permissions)."""

    if not text:
        return ""
    return (
        "<strategy_skill>\n"
        "Repo-specific PLANNING guidance (cadence, focus, decomposition emphasis). It "
        "REFINES how you plan; it does NOT override the operating rules, the decision "
        "contract, or the gates above - those WIN on any conflict. Preferences, not "
        "permissions.\n"
        f"{text}\n"
        "</strategy_skill>\n"
    )


def _repo_map_block(text: str) -> str:
    """The target repo's OPTIONAL navigation map, framed as a starting orientation for
    grepping the workdir - a reference, NEVER ground truth (the tree wins on conflict).

    Empty text -> ``""`` (so the no-map prompt is byte-identical to today). Non-empty -> a
    tagged block placed in the volatile tail alongside the other repo context, so the
    cacheable system prefix and the always-on strategy seam are unchanged."""

    if not text:
        return ""
    return (
        "<repo_map>\n"
        "A map of this repo (where things live) to orient your grep. It is a STARTING "
        "POINT, not ground truth: the actual tree you read WINS on any conflict.\n"
        f"{text}\n"
        "</repo_map>\n"
    )


_TOOLS_BLOCK = (
    "<tools>\n"
    "Your workdir is a checkout of the current integration tip with a FULL shell - "
    "nothing you run here touches durable state. GREP and READ it to ground your plan "
    "(what already exists, where things live), OPEN the prior epoch's baton-artifacts "
    "bundle (VIEW its render), and you MAY RUN verification commands (build, test, "
    "render, inspect) to see real state before deciding - this informs YOU, it does not "
    "make verification a worker task. Reading and running are your own internal step; "
    "your turn still ends with exactly one decision written to ./decision.json.\n"
    "</tools>\n"
)


def build_planner_input(
    context: PlannerContext,
    *,
    domain_skill_index: dict[str, str],
    strategy: str = "",
    repo_map: str = "",
    reask_errors: tuple[str, ...] = (),
) -> str:
    """Render the full PLAN prompt from ``context`` (PURE, no I/O).

    ``PLAN_PREAMBLE`` (byte-stable) then the repo's always-on STRATEGY overlay (advisory,
    injected right after ``</system>`` so the cacheable system prefix is unchanged; empty
    when the repo ships none) then the volatile tail: the job spec, the running state +
    keyed-log index, the prior epoch's BATON, the pointer to its persisted EVIDENCE bundle
    (when it produced one), the optional repo-navigation map (when the repo ships one), the
    domain-skill catalogue index (when the repo ships one), the read-tools note, any re-ask
    feedback, and the request. References, not payloads: only
    the baton text, names, and log keys, never file bodies (the planner greps its workdir
    for the tree). An absent ``repo_map`` adds zero bytes (byte-identical to today).
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
        f"{_strategy_block(strategy)}"
        f"<job>\n{context.job}\n</job>\n"
        f"{_state_block(context)}"
        f"{_baton_block(context)}"
        f"{_baton_artifacts_block(context.baton_artifacts)}"
        f"{_carried_block(context.carried)}"
        f"{_repo_map_block(repo_map)}"
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


def _pending_additions_block(context: CloseoutContext) -> str:
    """The plan's ``decision.pending`` additions (this epoch's new deferred work). The
    close-out folds them INTO the baton's ## Pending backlog (union with the prior
    ## Pending, minus prior items a task this epoch scheduled AND passed). Always
    rendered (the empty case is noted) so the model is never left guessing."""

    if not context.pending_additions:
        return (
            "<pending_additions>\n"
            "  (none: the plan recorded no new deferred work this epoch)\n"
            "</pending_additions>\n"
        )
    items = "\n".join(f"  - {p}" for p in context.pending_additions)
    return (
        "<pending_additions>\n"
        "NEW deferred work the plan recorded this epoch (the decision's pending field). "
        "Fold these INTO the baton's ## Pending backlog (UNION with the prior ## Pending, "
        "then MINUS any prior backlog item a task this epoch scheduled AND passed):\n"
        f"{items}\n</pending_additions>\n"
    )


def _parked_block(context: CloseoutContext) -> str:
    """The strike-ladder BLOCK note for close-out: lineages the state machine dropped
    this epoch (strike 2). The baton's ## Pending / ## Current status should record
    them as "could not close" so the next self does not keep re-proposing them. Empty
    ``parked`` -> ``""`` (byte-identical to a run that parked nothing)."""

    if not context.parked:
        return ""
    items = "\n".join(f"  - {d}" for d in context.parked)
    return (
        "<parked>\n"
        "The state machine BLOCKED these task lineages this epoch (they failed the WHOLE "
        "in-epoch tier ladder - local AND senior - across two epochs and were dropped, so "
        "the run can still reach a clean end). Note them in the baton as unclosed - do "
        "NOT keep re-proposing them:\n"
        f"{items}\n</parked>\n"
    )


_CLOSEOUT_TOOLS_BLOCK = (
    "<tools>\n"
    "Your workdir is a checkout of this epoch's staging tree with a FULL shell - nothing "
    "you run here touches durable state. GREP and READ it, READ the keyed-log handoffs + "
    "verdicts named above (VIEW any images - you can see), and GO VERIFY: build / render "
    "/ run the relevant checks to confirm what the workers claim. Save the evidence you "
    "gather (renders, gate output, diffs) under ./baton-artifacts/, then write ./baton.md "
    "and stop.\n"
    "</tools>\n"
)


def build_closeout_input(
    context: CloseoutContext,
    *,
    domain_skill_index: dict[str, str] | None = None,
    strategy: str = "",
    repo_map: str = "",
) -> str:
    """Render the full CLOSE-OUT prompt from ``context`` (PURE, no I/O).

    ``CLOSEOUT_PREAMBLE`` (byte-stable, and it already carries the four-section baton
    skeleton) then the repo's always-on STRATEGY overlay (advisory, injected right after
    ``</system>`` so the cacheable system prefix is unchanged; empty when absent - the
    baton-writer steers cadence too: what to prioritise next) then the volatile tail: the
    job, the prior baton, the epoch report (the deterministic outcomes + keyed-log
    pointers), this epoch's ``decision.pending`` additions (so the ## Pending backlog can
    be reconciled here, the sole baton write), the optional repo-navigation map (when the
    repo ships one), the domain-skill catalogue index (so the baton's pending list can
    name a skill the next epoch will select), the tools/vision note, and the request to
    write ``./baton.md``. An absent ``repo_map`` adds zero bytes (byte-identical to today).
    """

    skills = _domain_skills_block(domain_skill_index or {})
    return (
        f"<system>\n{CLOSEOUT_PREAMBLE}</system>\n"
        f"{_strategy_block(strategy)}"
        f"<job>\n{context.job}\n</job>\n"
        f"{_prior_baton_block(context)}"
        f"{_epoch_report_block(context)}"
        f"{_pending_additions_block(context)}"
        f"{_parked_block(context)}"
        f"{_repo_map_block(repo_map)}"
        f"{skills}"
        f"{_CLOSEOUT_TOOLS_BLOCK}"
        "<request>\n"
        "Verify what really happened, then write your updated baton (the four sections) "
        "to ./baton.md, and save the evidence you gathered under ./baton-artifacts/. Do "
        "not print the baton.\n"
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
#: The close-out's EVIDENCE bundle dir (render PNGs, gate/check output, notable diffs)
#: written alongside the baton in its throwaway workdir, relocated to the keyed log
#: (``E<n>/baton-artifacts/``) so the next plan-step opens ground truth, not a
#: description of it. Free-form, NEVER parsed, like the baton + handoff.
BATON_ARTIFACTS_DIRNAME = "baton-artifacts"


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


def _relocate_baton_artifacts(
    workdir: Path, *, run_dir: RunDir, epoch_index: int
) -> Path | None:
    """Relocate the close-out's EVIDENCE bundle (``baton-artifacts/``) from the throwaway
    planner worktree to the keyed log at ``E<n>/baton-artifacts/``, so the next plan-step
    OPENS ground truth (the render PNG, the gate output, the diffs), not a description of
    it. Mirrors ``worker._relocate_handoff``: free-form evidence, MOVED verbatim, never
    parsed.

    A no-op (returns ``None``) when the dir is absent or holds no files (a functional run,
    or a close-out that produced none): nothing is relocated and no keyed-log key appears.
    Returns the dest dir when files moved."""

    src = workdir / BATON_ARTIFACTS_DIRNAME
    if not src.is_dir():
        return None
    if not any(p.is_file() for p in src.rglob("*")):
        return None
    dest = run_dir.baton_artifacts_dir(epoch_index)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dest))
    return dest


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
        strategy = load_strategy(context.repo)
        repo_map = load_repo_map(context.repo)
        decision_path = workdir / DECISION_FILE
        out_file = context.run_dir.root / _OUT_FILENAME

        reask: tuple[str, ...] = ()
        last_error = "no attempts"
        for _ in range(self.max_reasks + 1):
            prompt = build_planner_input(
                context,
                domain_skill_index=index,
                strategy=strategy,
                repo_map=repo_map,
                reask_errors=reask,
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
        strategy = load_strategy(context.repo)
        repo_map = load_repo_map(context.repo)
        baton_path = workdir / BATON_FILE
        out_file = context.run_dir.root / _OUT_FILENAME
        baton_path.unlink(missing_ok=True)
        out_file.unlink(missing_ok=True)
        prompt = build_closeout_input(
            context, domain_skill_index=index, strategy=strategy, repo_map=repo_map
        )
        stdout = self.transport.dispatch(
            PlannerDispatch(
                prompt=prompt, workdir=workdir, out_file=out_file, purpose="closeout"
            )
        )
        baton = _read_result(baton_path, out_file, stdout)
        # Relocate the deep-survey EVIDENCE bundle out of the throwaway worktree (where it
        # would evaporate) into the durable keyed log, so the next plan-step sees it. A
        # no-op when the close-out produced none (functional runs). The baton itself is
        # persisted by the loop's _finalize_epoch, as before.
        _relocate_baton_artifacts(
            workdir, run_dir=context.run_dir, epoch_index=context.epoch_index
        )
        return baton

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
