"""Worker transport interface, the uniform task-in / disk-out boundary.

ARCHITECTURE.md: the orchestrator's only levers are pre-dispatch (task sizing,
input resolution) and post-return (validate, escalate). It never inspects a
worker's internals. A transport receives a fully-resolved ``WorkerRequest``,
does its work in the request's scratch CWD, and writes ``handoff.json`` THERE.

The disk file is the only output channel (§7 disk contract): ``run`` returns
nothing. Raising signals a transport-level failure (rate limit, process error,
kill); the task loop maps any exception to a failed attempt exactly as it maps
a missing or invalid handoff. Transport-owned supervision (timeouts, process
kills) lives in the transport, never in the loop (§10).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Union

from grindstone.contracts.models import (
    ArtifactExistsCheck,
    ArtifactTask,
    CmdCheck,
    ImplementTask,
)
from grindstone.contracts.semantics import HANDOFF_MAX_BYTES, HandoffMode

#: A dispatched task is one of the two typed contract task shapes.
Task = Union[ImplementTask, ArtifactTask]

#: The implement-mode review gate: the checked artifact a fresh-context review
#: must produce in the attempt CWD, and the done_when command that gates it.
#: (Verified mechanism: a review demanded as a gated artifact fires; prose
#: instructions don't.) Shared by the loop (appends the check, exempts the file
#: from commit) and the prompt builder (explains the step to the worker).
REVIEW_FILENAME = "review.md"
REVIEW_CHECK_COMMAND = f"test -s {REVIEW_FILENAME}"

#: Per-cwd pi settings the transport drops in the attempt CWD to pin spawned
#: subagents to the same model as the parent worker. pi-subagents reads
#: ``<project-root>/.pi/settings.json`` and a dir is "project root" iff it holds
#: a ``.pi/``, so writing here makes the attempt CWD that root (verified in
#: pi-subagents@0.23.1 src/agents/agents.ts: findNearestProjectRoot +
#: getProjectAgentSettingsPath). Shared so the loop knows what to strip before
#: commit (orchestration metadata never enters the diff) without importing any
#: model/provider knowledge (doer-empty-output RCA); the transport, which owns
#: provider/model, writes the content.
PI_SETTINGS_RELPATH = ".pi/settings.json"


class TransportError(Exception):
    """Base for transport-level failures the loop treats as a failed attempt."""


class RateLimited(TransportError):
    """The model endpoint refused the call for rate-limit reasons."""


class WorkerTimeout(TransportError):
    """The worker hung and was killed by its own transport-level supervisor."""


@dataclass(frozen=True)
class WorkerRequest:
    """Everything a transport needs for one attempt; the disk is its output.

    ``task`` is the typed contract model (never stringly JSON). ``inputs`` maps
    each declared input log key to its resolved on-disk path (the core resolves
    keys at dispatch, §8 working window). ``scratch`` is the CWD the worker
    runs in and writes ``handoff.json`` into. ``task_id`` is the fully-qualified
    ``P<phase>/E<epoch>/T<task>`` key the handoff must echo. ``attempt`` is the
    1-based global attempt number; ``failure_context`` carries the rejection
    reasons from every prior attempt so the worker can correct course. ``mode``
    is the epoch's decision tool: research/review/artifact all dispatch the same
    ``ArtifactTask`` shape, so the task type alone cannot select the worker
    plan, the prompt builder dispatches on ``mode``.
    """

    task: Task
    task_id: str
    inputs: dict[str, Path]
    scratch: Path
    attempt: int
    failure_context: list[str]
    mode: HandoffMode
    #: A PageRank-ranked SUBTREE of the target repo, personalized on this task's
    #: files (a navigation aid for large repos). ``None`` below threshold / on any
    #: failure / for tasks with no seed files; rendered only when present.
    repo_map: str | None = None


class WorkerTransport(Protocol):
    """The uniform worker interface: run one attempt, output to disk only."""

    def run(self, request: WorkerRequest) -> None:
        """Execute the task in ``request.scratch``; write ``handoff.json`` there.

        Returns nothing. Raise to signal a transport-level failure.
        """
        ...


# --- prompt construction (orchestration: *what* to ask) ------------------------
#
# ``build_worker_prompt`` is a pure function, *what* the orchestrator asks a
# worker to do, independent of *which* transport runs it (ARCHITECTURE.md). It lives
# in core so every transport (pi, script) builds the identical prompt; only the
# disk-out boundary differs.

#: The builtin pi-subagent the implement plan spawns (the fresh-context review
#: step). Pinning it to the parent's model keeps reviews on the local rig:
#: pi-subagents does NOT inherit the parent --provider/--model, so an unpinned
#: child silently falls back to the cloud default. agentOverrides is keyed by
#: exact agent name (no wildcard form), so we name it explicitly. The transport
#: that owns provider/model writes this into ``.pi/settings.json`` (PI_SETTINGS_RELPATH).
_PINNED_SUBAGENT = "reviewer"


def _render_checks(request: WorkerRequest) -> str:
    lines: list[str] = []
    for check in request.task.done_when:
        if isinstance(check, CmdCheck):
            lines.append(f"  - command `{check.cmd}` must exit {check.expect_exit}")
        elif isinstance(check, ArtifactExistsCheck):
            lines.append(f"  - artifact `{check.artifact_exists}` must exist")
    return "\n".join(lines)


def _render_inputs(request: WorkerRequest) -> str:
    if not request.inputs:
        return "  (none)"
    return "\n".join(
        f"  - `{key}` -> {path}" for key, path in sorted(request.inputs.items())
    )


def _implement_plan(task: ImplementTask) -> str:
    """The implement worker plan: the verified solo skill discipline.

    Contract-first ordering, verbatim-spec re-reads, a mandatory bake whose
    review step is a fresh-context subagent gated by a non-empty review.md,
    handoff written last, and the ownership lane (the S4 attempt-2 root cause).
    In-task implementation splitting is deliberately absent: it measured
    correctness-neutral at 2-9x token cost; a too-big task escalates to the
    planner via a truthful FAILED handoff instead.
    """

    globs = "\n".join(f"  - {g}" for g in task.file_ownership)
    return f"""
<implement_plan>
Work this plan in order. You implement everything yourself; subagents are for
the review step only. A saturated context silently drops shared-contract
details, the discipline below is what keeps a large solo build coherent.
  1. CONTRACT FIRST. Identify the shared pieces every other file depends on,
     constants, interface signatures, exception types, schemas. Implement them
     first, completely, and verify they import/load cleanly before moving on.
  2. WORK IN DEPENDENCY ORDER, ANCHORED TO THE CONTRACT. Before each unit ask:
     "must this agree with the internals of another unit?" Wherever two units
     must agree on something the contract files do not fully fix, pin that
     convention explicitly and apply it identically in both places yourself.
     Run the relevant done_when checks as each unit lands.
  3. VERBATIM SPEC. Before each unit, re-read its authoritative spec in the
     task goal and inputs end to end. Never work from a paraphrase or from
     memory, paraphrase silently drops requirements. If your context was
     compacted, recover by re-reading the spec and the files on disk.
  4. BAKE BEFORE HANDOFF, mandatory, all of (a)-(c) BEFORE handoff.json:
     (a) run EVERY done_when check yourself and fix every failure you see
         (exception: `python3 check_handoff.py` validates handoff.json itself,
         it cannot pass yet; you satisfy it in step 5);
     (b) re-read the full task goal once more and audit the seams, implement
         anything no earlier step clearly covered;
     (c) get ONE fresh-context review: spawn the registered `reviewer`
         subagent with the goal, the done_when checks and a summary of what
         you changed; its findings must be written to `{REVIEW_FILENAME}` in
         this directory (non-empty, `{REVIEW_CHECK_COMMAND}` is one of your
         checks). ACT on what it finds.
  5. Write handoff.json LAST, as its own final step, only after the bake, then
     run `python3 check_handoff.py` and fix violations until it exits 0.
If after honest effort the checks cannot pass, write a truthful FAILED or
PARTIAL handoff with `not_done` and `downstream_needs` filled in, the planner
re-plans from that. Never claim DONE on failing checks: every check is re-run
by the orchestrator and a false DONE is always caught.
</implement_plan>

<file_ownership>
You may create or edit files ONLY within these globs:
{globs}
Changing ANY other file fails the attempt. (`handoff.json`, `check_handoff.py`
and `{REVIEW_FILENAME}` are orchestration files, not repo work, write them in
the CWD as instructed; the orchestrator excludes them from this rule.)
</file_ownership>
"""


#: Shared containment line for the non-implement plans: their scratch is a plain
#: dir nested INSIDE the target repo's working tree, so a wandering worker can
#: reach the operator's checkout (E2E gate2 P0: an artifact worker checked out
#: the integration branch in the live repo). The env fence (GIT_CEILING_DIRECTORIES)
#: is the mechanism; this line keeps the model from wandering by path at all.
_CWD_CONTAINMENT = """Your CWD is your entire workspace: read your resolved inputs, write your
artifact and handoff here. Never cd above it, never read or modify the
surrounding repository, and do not run git, there is no repository here."""


def _research_plan() -> str:
    return f"""
<research_plan>
This is a research task: investigate and report; do not modify code.
{_CWD_CONTAINMENT}
  1. Read the resolved inputs; they contain everything the goal requires.
  2. Write your findings into the artifact named above, that artifact is the
     deliverable the planner reads.
  3. Ground every claim: the handoff's `citations` MUST contain at least one
     real file (with line numbers where useful). A research handoff with no
     citations is rejected.
</research_plan>
"""


def _review_plan(task: ArtifactTask) -> str:
    targets = "\n".join(f"  - {t}" for t in task.targets or [])
    return f"""
<review_plan>
This is a review task: judge the targets; do not modify them.
{_CWD_CONTAINMENT}
Targets under review:
{targets}
  1. Examine each target against the question in the goal.
  2. Write your findings AND an explicit verdict into the artifact named above.
  3. Ground every finding: the handoff's `citations` MUST contain at least one
     real file/line. A review handoff with no citations is rejected.
</review_plan>
"""


def _artifact_plan() -> str:
    return f"""
<artifact_plan>
Produce the artifact named above so that every done_when check passes. The
artifact is the deliverable; keep the handoff to references, not payloads.
{_CWD_CONTAINMENT}
</artifact_plan>
"""


def build_worker_prompt(request: WorkerRequest) -> str:
    """Construct the worker prompt (pure function, no pi, no I/O).

    Common skeleton: goal, resolved inputs, the done_when checks, prior-failure
    context, and a handoff block pinning the disk contract (write handoff.json
    in the CWD, the exact task_id, the byte cap, the citation requirement).
    Between done_when and the handoff block sits the PER-MODE worker plan
    (owner ruling 2026-06-11): do->verify->review discipline belongs to the
    implement plan ONLY; research/review/artifact get their own lean plans, no
    verify theater. We own the model-facing format (§7: XML-tagged sections).
    """

    task = request.task
    artifact_line = ""
    if isinstance(task, ArtifactTask):
        artifact_line = (
            f"\nProduce the artifact at log key `{task.artifact_out}`.\n"
        )
    if isinstance(task, ImplementTask):
        plan_block = _implement_plan(task)
        occupancy_line = (
            "  - occupancy: {\"compacted\": <bool>, \"subagent_splits\": <int>}, "
            "report honestly;\n    your reviewer spawn counts as a split."
        )
    else:
        if request.mode == "research":
            plan_block = _research_plan()
        elif request.mode == "review":
            plan_block = _review_plan(task)
        else:
            plan_block = _artifact_plan()
        occupancy_line = (
            "  - occupancy: {\"compacted\": <bool>, \"subagent_splits\": <int>}."
        )
    context_block = ""
    if request.failure_context:
        joined = "\n".join(f"  - {c}" for c in request.failure_context)
        context_block = (
            "\n<prior_failures>\nEarlier attempts failed for these reasons; "
            f"fix them:\n{joined}\n</prior_failures>\n"
        )
    repo_map_block = ""
    if request.repo_map:
        repo_map_block = (
            "\n<repo_map>\nStructural map of the target repo near this task's "
            "files (most-referenced symbols first). A navigation aid, not "
            f"exhaustive; verify against the actual files.\n{request.repo_map}\n"
            "</repo_map>\n"
        )
    return f"""<task id="{request.task_id}">
{task.goal}
</task>
{artifact_line}
<inputs>
{_render_inputs(request)}
</inputs>
{repo_map_block}
<done_when>
These deterministic checks will be re-run by the orchestrator. They MUST pass:
{_render_checks(request)}
</done_when>
{plan_block}{context_block}
<stop_rule>
If the done_when checks cannot be satisfied in THIS verification environment (a
required tool/dependency is missing, or the task as specified is not achievable
here), do NOT loop or keep editing. STOP, write handoff.json with status FAILED
(or PARTIAL) and a concise diagnosis of WHY it cannot be satisfied, then exit.
Always write handoff.json as your final act before stopping, even when out of
ideas or running low on budget: a missing handoff tells the planner nothing.
</stop_rule>
<scope>
You edit ONLY files within your file_ownership (or, for non-implement tasks, only
your CWD). Grindstone (the orchestrator core) handles all git staging and
committing and may keep its own bookkeeping files in the tree: do NOT git-commit,
do NOT touch orchestration files, and do not worry about a "working tree clean"
state, that is grindstone's concern, not yours.
</scope>
<handoff>
When finished, write a file named exactly `handoff.json` in your current working
directory. It is the ONLY thing the orchestrator reads, stdout is ignored.
Requirements:
  - JSON object, schema_version "1".
  - task_id MUST be exactly "{request.task_id}".
  - status: "DONE" if every done_when check passes, else "FAILED" or "PARTIAL".
  - resulting_state: one short sentence for the planner (references, not file bodies).
  - what_changed: list of {{"kind": "file"|"interface"|"artifact", "ref": <path or name>}}
    objects, one per thing you changed, NOT free prose strings.
  - not_done / downstream_needs: lists of short strings (downstream_needs holds
    log keys); fill both on FAILED or PARTIAL.
  - checks: echo each done_when as {{"check": <text>, "exit_code": <int>}}.
  - citations: list of {{"file": <path>, "line": <int optional>}} grounding your
    claims in real files. Paths resolve against your CWD, or against the
    target repo root for research/review/artifact tasks, and must stay inside
    those roots; a citation outside them (or hallucinated) fails the handoff.
{occupancy_line}
  - The whole file must serialize under {HANDOFF_MAX_BYTES} bytes, references, not payloads.
After writing handoff.json, run `python3 check_handoff.py` (it is in your CWD
and in done_when) and fix every violation it prints until it exits 0. It can
only pass once handoff.json exists, earlier check runs cannot cover it.
</handoff>
"""
