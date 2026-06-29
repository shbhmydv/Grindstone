"""The per-task EXECUTION UNIT (``run_task``) + the lenient CRITIC.

BONES: one task is run to a triage verdict, reusable and safe to call
concurrently. ``run_task`` isolates the work in a throwaway worktree off an
EXTERNAL base (so a worker that strips its CWD to the repo root cannot reach the
operator checkout), dispatches the tier's rig behind the uniform ``WorkerTransport``
seam, then disposes on DETERMINISTIC FACTS plus the critic's lenient verdict, and
NOTHING ELSE. The worker's ``handoff.md`` is a FREE-FORM prose report for the
critic; the state machine never parses or schema-gates it.

The two deterministic invariants the Python owns (everything else is agentic):

* implement: after the grind, is there a non-empty commit, and are the changed
  paths within the declared ``file_ownership`` (the GIT DIFF is scope-checked via
  ``worktree.scope_violations``, never the handoff)? A zero-diff or out-of-scope
  attempt is a failed attempt; in-scope work then disjoint-merges (Part 3).
* non-write (research / review / artifact): does the ``artifact_out`` file exist
  at its keyed-log path? Missing -> a failed attempt.

A gate-clean attempt ALWAYS runs the tier-matched CRITIC, which now OWNS every
judgment that used to be a Python gate: done-vs-blocked-vs-incomplete, grounding /
citation quality (research / review must cite real files under the read-tip;
ungrounded -> RETRY / ESCALATE), and retry-worthiness. An unrecoverable
environmental blocker the worker reports in ``handoff.md`` becomes a critic
ESCALATE (this replaces the old worker-self-declared BLOCKED path).

The control flow, per BONES failure model (two nodes, everything routes to #2):

* worker RATE-LIMITED -> the exception escapes (NOT a burned attempt): the epoch
  loop parks and re-runs (#1).
* attempt fails the deterministic gate (missing / zero-diff / out-of-scope /
  missing artifact) -> a failed attempt: bounded same-tier retry, the retry
  INHERITS the prior attempt's stripped wip; exhausted -> escalate (#2).
* critic PASS -> return the merge-ready worktree result (Part 3 integrates).
* critic RETRY -> bounded same-tier retry (a defect the SAME worker can fix).
* critic ESCALATE -> surface to the planner (#2).

The epoch-level FAN-OUT and the disjoint-merge INTEGRATION are PART 3; the
``Backends`` per-rig-endpoint semaphore seam (local = 1 slot) is defined here so a
task acquires its backend's slot around each dispatch, but the run-wide map is built
and shared by Part 3.
"""

from __future__ import annotations

import json
import shutil
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Literal, Protocol

from grindstone import worktree as wt
from grindstone.check_decision import extract_verdict_json
from grindstone.contracts.models import (
    HandoffMode,
    Task,
    Verdict,
    VerdictOutcome,
    parse_verdict,
)
from grindstone.domain_skills import load_domain_skill
from grindstone.repo_map import load_repo_map
from grindstone.rundir import RunDir

#: The worker's FREE-FORM report, written in its CWD: a short prose note (what I
#: did, what is done, what is blocked / unfinished, which files I touched, grounding
#: as prose). NOT a wire contract: never parsed or schema-gated. Relocated to the
#: keyed log verbatim (the planner's optional context + the journal) and handed to
#: the critic as text. Stripped before an implement commit like every orchestration
#: file.
HANDOFF_FILENAME = "handoff.md"

#: The critic's only result channel: a ``verdict.json`` written in the critic's CWD
#: (a re-read disk contract). Relocated into the run dir and re-validated with
#: Pydantic; stdout is never parsed.
CRITIC_VERDICT_FILENAME = "verdict.json"

#: Per-tier attempt budgets for the IN-EPOCH tier ladder. A local task gets a few
#: same-tier self-heal attempts, then (when the rig has a DISTINCT senior endpoint)
#: escalates to senior for a final attempt on the carried wip. A senior-only task
#: (planner-routed senior, or every-tier-local rig) runs just its own stage.
LOCAL_MAX_ATTEMPTS = 3  # 1 initial + 2 retries
SENIOR_MAX_ATTEMPTS = 2  # 1 initial + 1 retry

#: The CRITIC FAILURE NODE's same-tier budget: a critic that chatted its verdict (wrote
#: no parseable verdict.json AND no recoverable JSON in stdout) gets one MORE dispatch
#: at the task tier before the node bumps to the senior critic. Distinct from the worker
#: ladder above: this re-dispatches ONLY the critic on the SAME already-passed work.
CRITIC_MAX_ATTEMPTS = 2  # 1 initial + 1 retry at the task tier

#: Cap on the model's chat output echoed into a ``CriticFailed`` reason / event (so an
#: operator can see WHY the critic could not land a verdict, without dumping a transcript).
_CRITIC_SNIPPET_MAX = 200

#: DoS sanity backstop on the disk reads (reject, never truncate, a pathological /
#: corrupt multi-megabyte file before reading it into memory). A real handoff /
#: verdict serializes far under this; it only ever fires on a corrupt file.
_DISK_READ_MAX_BYTES = 1_048_576


# --- transport seam (the uniform task-in / disk-out boundary) ------------------


class TransportError(Exception):
    """Any worker transport failure. The loop treats it as a failed attempt that,
    if unrecoverable, routes to the planner (BONES failure model #2)."""


class RateLimited(TransportError):
    """A rate-limit / quota refusal (BONES failure model #1): the work is blocked on
    a quota window, NOT a failure of the work, so it escapes ``run_task`` un-burned
    and the epoch loop parks (~1/hr) and re-runs."""


@dataclass(frozen=True)
class CriticBrief:
    """Marks a dispatch as the CRITIC pass (not a worker grind).

    Carries the task's OWN claimed ``goal`` (the anchor the critic judges against,
    not its own taste), the worker's free-form ``handoff_text`` report, and a pointer
    to the produced work: ``diff_base`` is the base ref an implement critic diffs HEAD
    against (it runs in the post-commit worktree), and ``artifact_path`` is the
    CWD-relative deliverable a non-write critic reads. ``read_root`` is the repository
    a non-write critic reads to VERIFY the artifact's citations are grounded in real
    files (None for implement, which judges the diff directly).
    """

    goal: str
    mode: HandoffMode
    handoff_text: str = ""
    diff_base: str | None = None
    artifact_path: str | None = None
    read_root: str | None = None


@dataclass(frozen=True)
class WorkerRequest:
    """One fully-resolved dispatch: the typed ``task``, its full keyed-log
    ``task_id`` (``E*/T*``), the worker ``mode``, the scratch CWD it writes its
    output into, the resolved ``inputs`` (prior keyed-log artifacts), the SELECTED
    domain skills (retrieve-not-concatenate), the repo's OPTIONAL navigation map
    (``repo_map``, empty when the repo ships none -> byte-identical prompt),
    prior-attempt ``failure_context``, and whether a prior attempt's work is already
    present (incremental retry).

    When ``critic`` is set the dispatch is the CRITIC pass: the prompt builder
    branches to ``build_critic_prompt`` and the result channel is ``verdict.json``,
    not a handoff.

    ``read_root`` is the integration-tip checkout a NON-WRITE task reads + grounds
    its citations against (set by the epoch driver to a read-only worktree of the
    run-branch tip, so a research / review task sees what prior epochs built, not the
    stale base). ``None`` for implement tasks (they read + write their own worktree).
    """

    task: Task
    task_id: str
    mode: HandoffMode
    scratch: Path
    inputs: dict[str, Path] = field(default_factory=dict)
    domain_skills: dict[str, str] = field(default_factory=dict)
    repo_map: str = ""
    failure_context: tuple[str, ...] = ()
    prior_work_present: bool = False
    critic: CriticBrief | None = None
    read_root: Path | None = None


class WorkerTransport(Protocol):
    """The uniform dispatch boundary the per-task unit runs through (worker AND
    critic). ``run`` does its work in ``request.scratch`` and writes its disk artifact
    there; it RETURNS the dispatch's stdout (the gate reads the disk artifact, but the
    CRITIC falls back to this stdout when the model chatted its verdict instead of
    writing ``verdict.json``) and raises to signal a transport failure."""

    def run(self, request: WorkerRequest) -> str: ...


class AttemptEvents(Protocol):
    """Optional per-attempt event sink the epoch driver threads in so the gate +
    triage land in the journal LIVE during a real run (default ``None`` keeps
    ``run_task`` standalone for the unit tests). The driver's adapter maps these to
    ``work_gate_passed`` / ``work_gate_rejected`` / ``verdict`` journal events.
    ``work_gate_passed`` now fires when the DETERMINISTIC gate passes (a non-empty
    in-scope commit, or a present artifact), before the critic runs. The ``critic_*``
    hooks surface the CRITIC FAILURE NODE: a verdict recovered from chat, a bump to the
    senior critic, or the node exhausting without a verdict."""

    def work_gate_passed(self, task_id: str) -> None: ...
    def work_gate_rejected(self, task_id: str, reason: str) -> None: ...
    def verdict(self, task_id: str, outcome: VerdictOutcome, reason: str) -> None: ...
    def tier_escalated(self, task_id: str, to_tier: str, attempt: int) -> None: ...
    def critic_recovered(self, task_id: str, tier: str) -> None: ...
    def critic_escalated(self, task_id: str, to_tier: str) -> None: ...
    def critic_failed(self, task_id: str, tier: str, snippet: str) -> None: ...


# --- per-backend concurrency seam (BONES concurrency ruling) -------------------


@dataclass
class _Endpoint:
    transport: WorkerTransport
    sem: threading.Semaphore


class Backends:
    """The per-backend concurrency map: ONE semaphore per rig ENDPOINT (keyed by the
    resolved request-script identity), local (:8080, ``--parallel 1``) sized to 1
    slot, claude to N. A task acquires its tier's backend slot around EACH dispatch
    (worker + the tier-matched critic), so same-backend tasks serialize on the GPU
    while cross-backend tasks run concurrently, and two tiers that resolve to the
    SAME endpoint share ONE semaphore (no double-booking). The epoch fan-out (Part 3)
    builds one ``Backends`` from config and shares it across every task; here it
    bounds a single task's dispatches.
    """

    def __init__(
        self, endpoints: dict[str, _Endpoint], tier_endpoint: dict[str, str]
    ) -> None:
        self._endpoints = endpoints
        self._tier_endpoint = tier_endpoint

    @classmethod
    def single(cls, transport: WorkerTransport, *, slots: int = 1) -> Backends:
        """A one-endpoint map both tiers resolve to (uniform rig / tests)."""

        endpoints = {"only": _Endpoint(transport, threading.Semaphore(slots))}
        return cls(endpoints, {"local": "only", "senior": "only"})

    @contextmanager
    def slot(self, tier: str) -> Iterator[WorkerTransport]:
        """Acquire the backend slot for ``tier`` and yield its transport.

        A ``senior`` tier with no senior endpoint falls back to ``local`` (BONES:
        a rig with no senior tier grinds every tier locally). Released on exit even
        when the dispatch raises (so a RateLimited never strands a slot)."""

        key = self._tier_endpoint.get(tier) or self._tier_endpoint["local"]
        endpoint = self._endpoints[key]
        with endpoint.sem:
            yield endpoint.transport

    def has_distinct_tier(self, tier: str) -> bool:
        """Does ``tier`` resolve to a DISTINCT endpoint from local (a real, separate
        rig), so an in-epoch escalation to it actually changes which model grinds?

        False when the tier is unmapped or shares local's endpoint (every tier grinds
        on the one backend) - in that case ``slot(tier)`` silently falls back to local,
        so escalating to it would just re-run the same model and the ladder must not."""

        key = self._tier_endpoint.get(tier)
        return key is not None and key != self._tier_endpoint["local"]


# --- result type ---------------------------------------------------------------


@dataclass(frozen=True)
class TaskResult:
    """The terminal of one task. ``outcome`` routes the epoch loop:

    * ``passed``: merge-ready. For an implement task ``branch`` (+ ``head``) is the
      wip the loop disjoint-merges into the run branch; for a non-write task
      ``artifact_key`` is the deliverable already relocated into the run dir.
    * ``escalated``: a critic ESCALATE (an environmental blocker the worker reported,
      an ambiguous spec, a decision needed) or an exhausted retry ladder -> the
      planner. ``reason`` is the surfaced context.

    ``handoff_path`` is the relocated free-form report (the planner's optional context
    + the journal), ``None`` when the worker wrote none.
    """

    task_id: str
    outcome: Literal["passed", "escalated"]
    attempts: int
    handoff_path: Path | None = None
    verdict: Verdict | None = None
    branch: str | None = None
    head: str | None = None
    artifact_key: str | None = None
    reason: str = ""


# --- internal attempt-failure signal -------------------------------------------


class _Rejected(Exception):
    """One attempt failed the deterministic gate (missing / zero-diff / out-of-scope
    write / missing artifact). ``chainable`` is False for a poisoned worktree (an
    out-of-scope write): the next retry restarts clean from the epoch base rather
    than inheriting the poison."""

    def __init__(self, reason: str, *, chainable: bool = True) -> None:
        super().__init__(reason)
        self.reason = reason
        self.chainable = chainable


class CriticError(Exception):
    """The critic produced no valid verdict (transport raise / missing / invalid).
    A fail-safe: never a silent pass; ``run_task`` surfaces it to the planner."""


# --- prompts (pure functions; *what* to ask, independent of the transport) -----

#: BONES safety + isolation contract, stated to every worker. Lifted from the v7
#: build_worker_prompt: write only relative to the CWD, never absolute, never
#: outside the worktree; the orchestrator inspects ONLY this worktree.
_WORKTREE_CONTRACT = """<worktree>
You run inside an ISOLATED, throwaway git worktree that is your current working directory
and IS the repository for this task. Create and edit every file with paths RELATIVE to
your CWD; never write to an absolute path and never write outside your CWD. There is no
other repository you may touch - do not go looking for one, this worktree is it. The
orchestrator inspects ONLY this worktree to gate and integrate your work, so anything you
write elsewhere is invisible, discarded, and corrupts the run. If something you depend on
(a module to import, code under test) is NOT present in your worktree, it is owned by
another task that has not merged yet - do NOT create it to unblock yourself: write against
it as if it exists and record the missing dependency in your handoff. Reaching beyond your
own files only fails the gate. You can SEE: read any
image in your worktree or inputs directly (screenshots, mockups, designs) and produce
images where visual proof helps your reviewer - view, do not guess.
</worktree>"""

#: Concise per-mode guidance (BONES drops the operating-skill scenario split; the
#: dynamic per-task lanes are appended in code). Each is the worker's plan for ONE
#: intent. The executor is ONE role: a senior-tier task reuses the same guidance
#: (the tier only selects which rig SCRIPT runs it).
_MODE_GUIDANCE: dict[HandoffMode, str] = {
    "implement": (
        "Make the change inside your file_ownership, then COMMIT it (the orchestrator "
        "gates the git diff in this worktree, not your words). If your checks need the "
        "project's dependencies, you MAY install them INSIDE THIS WORKTREE as part of "
        "your work (setup does not reach here). Run whatever checks you write to "
        "convince yourself it works; if the work is visual, render and LOOK at the "
        "result."
    ),
    "research": (
        "Investigate, then write your findings to the artifact log key below. Ground "
        "every claim in a real file (cite file + line) or a real image you viewed; do "
        "not speculate."
    ),
    "review": (
        "Re-derive the question yourself and reconcile it against the real files and "
        "rendered output; do not just check that sections are present. View any "
        "screenshots or UI the work produced. Write your verdict to the artifact log "
        "key below and cite what you judged."
    ),
    "artifact": (
        "Produce the deliverable at the artifact log key below (it may be a document "
        "or a rendered image); ground it in real files or visuals where it makes "
        "claims about the repo."
    ),
}


def _render_inputs(request: WorkerRequest) -> str:
    if not request.inputs:
        return "  (none)"
    return "\n".join(
        f"  - `{key}` -> {path}" for key, path in sorted(request.inputs.items())
    )


def _domain_skills_block(request: WorkerRequest) -> str:
    """Compose only the SELECTED domain skills (retrieve-not-concatenate)."""

    if not request.domain_skills:
        return ""
    blocks = "\n".join(
        f'<skill name="{name}">\n{text}\n</skill>'
        for name, text in sorted(request.domain_skills.items())
    )
    return (
        "\n<domain_skills>\nRepo-specific skills selected for THIS task; treat them "
        "as authoritative guidance for this repo's conventions:\n"
        f"{blocks}\n</domain_skills>\n"
    )


def _repo_map_block(request: WorkerRequest) -> str:
    """The target repo's OPTIONAL navigation map: a starting orientation for finding
    things in the tree, NEVER ground truth (the files you read win on conflict). Empty
    when the repo ships none, so the no-map prompt is byte-identical to today."""

    if not request.repo_map:
        return ""
    return (
        "\n<repo_map>\nA map of this repo (where things live) to orient you. It is a "
        "STARTING POINT, not ground truth: the actual files you read WIN on any "
        f"conflict.\n{request.repo_map}\n</repo_map>\n"
    )


def _critic_skills_block(request: WorkerRequest) -> str:
    """The CRITIC-framed view of the task's SELECTED domain skills: the rubric the
    work CLAIMS to meet. Distinct from ``_domain_skills_block`` (worker-facing
    "authoritative guidance"); here the framing is enforcement, and the critic must
    NOT be lenient on conformance. Empty when no skills were selected, so the
    lenient-router prompt stays byte-identical."""

    if not request.domain_skills:
        return ""
    blocks = "\n".join(
        f'<skill name="{name}">\n{text}\n</skill>'
        for name, text in sorted(request.domain_skills.items())
    )
    return (
        "\n<rubric>\nThese repo skills were SELECTED for this task: they are THE "
        "RUBRIC THE WORK CLAIMS TO MEET. VERIFY the work conforms to them and do NOT "
        "be lenient on that conformance - a task that ignored its selected rubric did "
        "NOT meet its claimed goal: RETRY if the SAME worker can bring it into "
        "conformance, ESCALATE if not. (This is the one place strictness is required; "
        "everywhere else the lenient-router bias below still holds.)\n"
        f"{blocks}\n</rubric>\n"
    )


def _file_ownership_block(task: Task) -> str:
    globs = "\n".join(f"  - {g}" for g in task.file_ownership)
    return f"""
<file_ownership>
You may create or edit files ONLY within these globs:
{globs}
Changing ANY other file fails the attempt. (`{HANDOFF_FILENAME}` is an
orchestration file, write it in the CWD as instructed; the orchestrator excludes
it from this rule and from your commit.)
</file_ownership>
"""


#: The free-form handoff-report instruction, identical on a fresh OR a resume prompt
#: (the report contract never changes). A module constant so both prompt shapes reuse
#: the one wording.
_HANDOFF_BLOCK = f"""<handoff>
When finished, write a SHORT free-form report named exactly `{HANDOFF_FILENAME}` in
your current working directory, for the independent reviewer who reads your work
next. Plain prose (no required schema): what you did, what is DONE, what is still
blocked or unfinished and why, which files you touched, and any grounding /
citations as prose. If a hard ENVIRONMENTAL blocker stopped you (a dependency you
cannot install, a host change you may not make, a decision only a human can take),
SAY SO plainly here so the reviewer can route it onward. This report is for the
reviewer; the orchestrator gates your actual work (the committed diff or the
produced artifact), never this file.
</handoff>
"""


def build_worker_prompt(request: WorkerRequest) -> str:
    """Construct the worker prompt (pure, no transport, no I/O).

    Two shapes, branched on ``prior_work_present``:

    * FRESH (no prior work): leads with ``<task>{goal}</task>`` - the full original
      brief as the active instruction - then the worktree contract, inputs, per-mode
      guidance + ownership, skills, the optional repo map, and the handoff. A fresh
      prompt is BYTE-IDENTICAL to the historical prompt (a poisoned-tree retry that
      restarts clean is fresh: it shows ``<prior_failures>`` but no resume frame).
    * RESUME (a prior attempt's wip is on the tree, from a same-tier retry or a tier
      ESCALATION): leads with a ``<resume>`` frame + the full untruncated
      ``<prior_failures>``, so a small local model reads "fix THESE, on the SAME tree"
      as the dominant instruction; the original goal follows as marked REFERENCE
      context only, NOT a re-build command.
    """

    if request.prior_work_present:
        return _resume_worker_prompt(request)
    task = request.task
    context_block = ""
    if request.failure_context:
        joined = "\n".join(f"  - {c}" for c in request.failure_context)
        context_block = (
            "\n<prior_failures>\nEarlier attempts failed for these reasons; fix "
            f"them:\n{joined}\n</prior_failures>\n"
        )
    return f"""<task id="{request.task_id}">
{task.goal}
</task>
{_artifact_line(task)}
{_WORKTREE_CONTRACT}
{_read_root_block(request)}<inputs>
{_render_inputs(request)}
</inputs>
{_plan_block(task)}{_domain_skills_block(request)}{_repo_map_block(request)}{context_block}
{_HANDOFF_BLOCK}"""


def _resume_worker_prompt(request: WorkerRequest) -> str:
    """The RESUME shape (a prior attempt's wip is present): lead with the fix frame so
    the worker repairs the same tree rather than rebuilding. Pure (no I/O)."""

    task = request.task
    if request.failure_context:
        joined = "\n".join(f"  - {c}" for c in request.failure_context)
    else:
        joined = "  - (the prior attempt did not land; see the work already on the tree)"
    return f"""<resume id="{request.task_id}">
A PRIOR attempt at this task FAILED and you are RESUMING it on the SAME working tree -
all prior work is already present here. Your ONLY goal now is to FIX the specific
failures listed below; do NOT rebuild from scratch. You MAY reset and redo a file if
the prior approach was genuinely wrong, but default to a minimal targeted fix.
</resume>

<prior_failures>
These are the failures you must fix; address each one:
{joined}
</prior_failures>

<original_task>
For reference, the original brief was the following. This is CONTEXT, not the active
instruction - the active instruction is to fix the failures above on the existing tree:
{task.goal}
</original_task>
{_artifact_line(task)}
{_WORKTREE_CONTRACT}
{_read_root_block(request)}<inputs>
{_render_inputs(request)}
</inputs>
{_plan_block(task)}{_domain_skills_block(request)}{_repo_map_block(request)}
{_HANDOFF_BLOCK}"""


def _artifact_line(task: Task) -> str:
    if task.mode != "implement" and task.artifact_out is not None:
        return f"\nProduce the artifact at log key `{task.artifact_out}`.\n"
    return ""


def _read_root_block(request: WorkerRequest) -> str:
    task = request.task
    if task.mode != "implement" and request.read_root is not None:
        return (
            "\n<read_root>\nRead the repository AT ITS CURRENT IN-RUN STATE under "
            f"`{request.read_root}` (the integration tip, including everything prior "
            "epochs built). Ground your citations in files under that path; do NOT "
            "read the operator's base checkout.\n</read_root>\n"
        )
    return ""


def _plan_block(task: Task) -> str:
    plan = "\n" + _MODE_GUIDANCE[task.mode]
    if task.mode == "implement":
        plan += _file_ownership_block(task)
    return plan


def build_critic_prompt(request: WorkerRequest, brief: CriticBrief) -> str:
    """The lenient CRITIC prompt (pure, no transport, no I/O).

    Encodes the BONES triage: (1) anchor on the task's OWN claimed goal, strict on
    "did it accomplish what it claimed", lenient on polish/style; (2) the bar is
    "good enough to build on", not perfect, so PASS-with-notes when torn; (3) the
    single retry-vs-escalate question, "can the SAME worker plausibly fix this
    itself?" yes -> RETRY, no -> ESCALATE. The critic now OWNS the judgments the
    Python used to gate: done-vs-blocked, grounding / citation quality (research /
    review must cite real files), and an environmental blocker -> ESCALATE. Bias to
    PASS when unsure. The critic ROUTES, it does not grade.

    ONE additive strictness: when the task SELECTED domain skills, they render as a
    rubric (``_critic_skills_block``) the work must conform to (skill-less tasks keep
    this byte-identical lenient prompt).
    """

    if brief.mode == "implement":
        work_line = (
            f"The work is a diff. Run `git diff {brief.diff_base or 'HEAD~1'}` in "
            "this worktree to see exactly what the worker changed, and read the "
            "changed files. If the change is visual and the worktree has a rendered "
            "screenshot, view it."
        )
        grounding = ""
    else:
        work_line = (
            f"The work is an artifact at `{brief.artifact_path}` (relative to this "
            "directory). Read it in full and judge its actual content, not a summary."
        )
        root = brief.read_root or "the repository"
        grounding = (
            "\n<grounding>\nThis is a research / review artifact: its claims MUST be "
            f"grounded in real files under `{root}`. VERIFY the citations resolve to "
            "files that exist and actually support each claim; a citation may be to an "
            "image, which you must actually OPEN and verify shows what is claimed. A "
            "claim with no citation, or one citing a file or image that does not exist "
            "or does not say what is claimed, is UNGROUNDED: if the worker can "
            "plausibly fix it (add or correct citations) -> RETRY; if the underlying "
            "claim is wrong or unverifiable -> ESCALATE.\n</grounding>\n"
        )
    report = brief.handoff_text.strip() or "(the worker wrote no handoff report)"
    return f"""<critic id="{request.task_id}">
You are an INDEPENDENT critic. You did NOT write this work. Do NOT edit anything.
Your job is to TRIAGE it into one of three routes, not to grade it. You can SEE - if
the work is or produces a visual (a screenshot, a rendered screen, a diagram), VIEW it
and judge it with your eyes, not the worker's description.
</critic>

<claimed_goal>
Judge the work against the task's OWN claimed goal below, NOT against your personal
taste. Be strict on "did it accomplish what it claimed"; be lenient on polish and
style. Intermediate red (a reference to something a later task will build) is fine.
{brief.goal}
</claimed_goal>

<worker_report>
The worker's own free-form report (its claims, not the ground truth; verify against
the actual work). If it reports a hard environmental blocker it could not resolve,
that is an ESCALATE, not a RETRY:
{report}
</worker_report>

<the_work>
{work_line}
</the_work>
{grounding}{_critic_skills_block(request)}
<triage>
The bar is "GOOD ENOUGH TO BUILD ON", not "perfect". Decide ONE outcome:
  - PASS: it accomplished the claimed goal well enough to build on. Minor
    imperfections are NOTES that carry forward, not blockers. When you are torn
    between failing and noting, PASS-with-notes (a false fail wastes a retry; a
    noted imperfection is cheap and a final review catches what matters).
  - RETRY vs ESCALATE: only if it did NOT meet the claimed goal. Ask the SINGLE
    question "can the SAME worker plausibly fix this itself?":
      * YES (a typo, a wrong value, a missing piece the worker can add) -> RETRY.
      * NO (a missing dependency, an ambiguous or wrong spec, a decision is needed,
        anything environmental the worker cannot fix) -> ESCALATE to the planner.
Bias to PASS when unsure.
</triage>

<verdict>
Write a file named exactly `{CRITIC_VERDICT_FILENAME}` in your CWD as your ONLY
output (stdout is ignored). A JSON object:
  - "outcome": "PASS" | "RETRY" | "ESCALATE".
  - "reason": one short sentence (the note to carry forward, or what is unfixable).
Write `{CRITIC_VERDICT_FILENAME}` and nothing else, then stop.
</verdict>
"""


def build_prompt(request: WorkerRequest) -> str:
    """Dispatch to the worker or critic prompt by whether a brief is attached."""

    if request.critic is not None:
        return build_critic_prompt(request, request.critic)
    return build_worker_prompt(request)


# --- handoff relocation (free-form; NEVER parsed or schema-gated) --------------


def _relocate_handoff(
    scratch: Path, *, run_dir: RunDir, task_id: str
) -> tuple[Path | None, str]:
    """Relocate the worker's FREE-FORM ``handoff.md`` to the keyed log and return its
    ``(path, text)``.

    The report is NEVER parsed or schema-gated: it is the worker's prose for the
    critic + the planner's optional context. A missing or pathologically-large report
    is fine (the deterministic gate is the diff / artifact, not this file), so return
    ``(None, "")`` and let the critic judge the actual work."""

    src = scratch / HANDOFF_FILENAME
    if not src.is_file():
        return None, ""
    if src.stat().st_size > _DISK_READ_MAX_BYTES:
        src.unlink()  # oversized report: dropped (never parsed), never committed
        return None, ""
    try:
        text = src.read_text(encoding="utf-8", errors="replace")
    except OSError:
        src.unlink(missing_ok=True)
        return None, ""
    dest = run_dir.resolve(f"{task_id}/{HANDOFF_FILENAME}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dest))
    return dest, text


# --- the per-task unit ---------------------------------------------------------


def _load_domain_skills(repo: Path | None, task: Task) -> dict[str, str]:
    """The SELECTED domain skills' text (retrieve-not-concatenate). A named-but-
    missing skill raises ``_Rejected`` (the planner selected a skill the repo does
    not ship), never a silent empty block."""

    if repo is None or not task.skills:
        return {}
    out: dict[str, str] = {}
    for name in task.skills:
        try:
            out[name] = load_domain_skill(repo, name)
        except (FileNotFoundError, ValueError) as exc:
            raise _Rejected(f"domain skill {name!r} could not be loaded: {exc}") from exc
    return out


def _critic_verdict(
    task: Task,
    task_id: str,
    *,
    tier: str,
    scratch: Path,
    base: str | None,
    artifact_rel: str | None,
    handoff_text: str,
    critic_read_root: Path | None,
    run_dir: RunDir,
    backends: Backends,
    domain_skills: dict[str, str],
    events: AttemptEvents | None = None,
) -> Verdict:
    """Dispatch the critic at the attempt's STAGE ``tier`` in the task's own scratch,
    read the lenient ``Verdict`` by CHANNEL PRIORITY (mirroring the planner's resilient
    read), persist it to the keyed log, and return it. The critic runs at the SAME tier
    the attempt did, so an escalated (senior) attempt is judged by the senior critic.

    Channel priority: (a) ``verdict.json`` in scratch (the disk contract, the 90% path,
    byte-identical to before); ELSE (b) a JSON verdict object sniffed out of the
    dispatch's STDOUT (the local model answered in CHAT instead of writing the file).
    The recovered verdict is PERSISTED to the keyed-log dest either way (a moved file,
    or the recovered JSON written there). Only when NEITHER channel yields a parseable
    lenient verdict is it a ``CriticError`` (fail-safe, never a silent pass); its message
    carries a short snippet of the model's chat output for observability. A stdout
    recovery emits ``critic_recovered`` when an event sink is threaded in.

    The task's SELECTED ``domain_skills`` ride along so the critic can VERIFY the work
    against them as a rubric (the "analyse" step of the composition loop); when empty
    the critic prompt is byte-identical to the lenient-router prompt."""

    implement = task.mode == "implement"
    brief = CriticBrief(
        goal=task.goal,
        mode=task.mode,
        handoff_text=handoff_text,
        diff_base=base if implement else None,
        artifact_path=artifact_rel,
        read_root=None if implement or critic_read_root is None else str(critic_read_root),
    )
    request = WorkerRequest(
        task=task, task_id=task_id, mode=task.mode, scratch=scratch, critic=brief,
        domain_skills=domain_skills,
    )
    try:
        with backends.slot(tier) as transport:
            stdout = transport.run(request)
    except RateLimited:
        raise
    except Exception as exc:
        raise CriticError(f"critic transport failed: {type(exc).__name__}: {exc}") from exc

    dest = run_dir.resolve(f"{task_id}/{CRITIC_VERDICT_FILENAME}")
    dest.parent.mkdir(parents=True, exist_ok=True)

    # (a) The disk contract: the critic wrote verdict.json in its CWD (the normal path).
    verdict_file = scratch / CRITIC_VERDICT_FILENAME
    if verdict_file.is_file():
        try:
            payload = json.loads(verdict_file.read_text(encoding="utf-8"))
            verdict = parse_verdict(payload)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise CriticError(f"{CRITIC_VERDICT_FILENAME} invalid: {exc}") from exc
        shutil.move(str(verdict_file), str(dest))  # remove from scratch: never committed
        return verdict

    # (b) Fallback: the critic CHATTED its verdict (no verdict.json). Sniff stdout.
    recovered = extract_verdict_json(stdout)
    if recovered is not None:
        try:
            recovered_verdict = parse_verdict(json.loads(recovered))
        except (json.JSONDecodeError, ValueError):
            recovered_verdict = None
        if recovered_verdict is not None:
            dest.write_text(recovered, encoding="utf-8")
            if events is not None:
                events.critic_recovered(task_id, tier)
            return recovered_verdict

    snippet = stdout.strip()[:_CRITIC_SNIPPET_MAX] or "(no output)"
    raise CriticError(
        f"critic wrote no {CRITIC_VERDICT_FILENAME} and no parseable verdict in its "
        f"chat output: {snippet}"
    )


def _resilient_critic_verdict(
    task: Task,
    task_id: str,
    *,
    tier: str,
    scratch: Path,
    base: str | None,
    artifact_rel: str | None,
    handoff_text: str,
    critic_read_root: Path | None,
    run_dir: RunDir,
    backends: Backends,
    domain_skills: dict[str, str],
    events: AttemptEvents | None,
) -> Verdict:
    """The CRITIC FAILURE NODE: judge an already-gate-passed attempt to a lenient
    ``Verdict``, resiliently, NEVER faulting the run.

    A single critic dispatch can fail to land a verdict (a local model that answers in
    chat without even a sniffable JSON object). This node re-dispatches ONLY the critic
    (never the worker / ``_run_attempt``, the work already passed the deterministic
    gate) up an escalating ladder on the SAME scratch:

    1. up to ``CRITIC_MAX_ATTEMPTS`` dispatches at the task ``tier``;
    2. then, if the rig has a DISTINCT senior endpoint and this is not already senior,
       ONE senior-critic dispatch (the senior model judges the same passed diff /
       artifact, no re-grind, no new worktree).

    On success it returns the verdict (``critic_recovered`` already emitted by
    ``_critic_verdict`` if it came from chat). If every dispatch fails it emits
    ``critic_failed`` (with a short chat snippet) and raises ``CriticError`` carrying
    that snippet, which ``run_task`` turns into the existing ``escalated`` planner route.
    ``RateLimited`` is NOT a critic failure: it escapes un-caught so the loop parks."""

    def _dispatch(at_tier: str) -> Verdict:
        return _critic_verdict(
            task, task_id, tier=at_tier, scratch=scratch, base=base,
            artifact_rel=artifact_rel, handoff_text=handoff_text,
            critic_read_root=critic_read_root, run_dir=run_dir,
            backends=backends, domain_skills=domain_skills, events=events,
        )

    last_exc: CriticError | None = None
    for _ in range(CRITIC_MAX_ATTEMPTS):
        try:
            return _dispatch(tier)
        except CriticError as exc:
            last_exc = exc

    if tier != "senior" and backends.has_distinct_tier("senior"):
        if events is not None:
            events.critic_escalated(task_id, "senior")
        try:
            return _dispatch("senior")
        except CriticError as exc:
            last_exc = exc

    snippet = (str(last_exc) if last_exc else "critic produced no verdict")[
        :_CRITIC_SNIPPET_MAX
    ]
    if events is not None:
        events.critic_failed(task_id, tier, snippet)
    raise CriticError(snippet)


@dataclass
class _AttemptOutput:
    """A gate-clean attempt: the free-form report (path + text for the critic) and
    the produced work (implement HEAD, or the published artifact log key)."""

    handoff_path: Path | None
    handoff_text: str
    head: str | None
    artifact_rel: str | None


def _run_attempt(
    task: Task,
    task_id: str,
    *,
    tier: str,
    scratch: Path,
    repo: Path | None,
    read_root: Path | None,
    epoch_base: str | None,
    backends: Backends,
    domain_skills: dict[str, str],
    repo_map: str,
    failure_context: tuple[str, ...],
    prior_work_present: bool,
    run_dir: RunDir,
    events: AttemptEvents | None,
) -> _AttemptOutput:
    """Dispatch one worker attempt and gate it on DETERMINISTIC FACTS. For an
    implement task: relocate the free-form handoff (always removed from scratch),
    commit, scope-check against the EPOCH base, and reject a zero-diff commit. For a
    non-write
    task: ensure the ``artifact_out`` file exists and publish it to its log key.
    Raises ``_Rejected`` on any gate failure; lets ``RateLimited`` escape. The
    handoff report is NEVER parsed.

    A NON-WRITE task grounds its citations against ``read_root`` (the integration tip)
    when set, never the stale base. The backend slot is held ONLY around the model
    dispatch (not the git/disk work), so a concurrent same-backend task cannot land a
    second call on the single local slot mid-grind, while cross-backend tasks run
    free."""

    implement = task.mode == "implement"
    #: A non-write task reads + cites the integration tip (``read_root``) when the
    #: driver provided one, else the base repo (the standalone-test fallback).
    cite_root = repo if implement else (read_root if read_root is not None else repo)
    inputs = {key: run_dir.resolve(key) for key in task.inputs}
    request = WorkerRequest(
        task=task,
        task_id=task_id,
        mode=task.mode,
        scratch=scratch,
        inputs=inputs,
        domain_skills=domain_skills,
        repo_map=repo_map,
        failure_context=failure_context,
        prior_work_present=prior_work_present,
        read_root=None if implement else cite_root,
    )
    try:
        with backends.slot(tier) as transport:
            transport.run(request)
    except RateLimited:
        raise
    except Exception as exc:
        raise _Rejected(f"transport error: {type(exc).__name__}: {exc}") from exc

    handoff_path, handoff_text = _relocate_handoff(
        scratch, run_dir=run_dir, task_id=task_id
    )

    if implement:
        assert repo is not None and epoch_base is not None
        wt.commit_all(scratch, f"grind({task_id}): {task.goal.splitlines()[0][:72]}")
        changed = wt.changed_paths(scratch, epoch_base)
        if not changed:
            if handoff_path is not None:
                handoff_path.unlink(missing_ok=True)
            raise _Rejected(
                "no committed work: implement task left a zero-diff branch"
            )
        out_of_scope = wt.scope_violations(changed, list(task.file_ownership))
        if out_of_scope:
            if handoff_path is not None:
                handoff_path.unlink(missing_ok=True)
            raise _Rejected(
                f"out-of-scope writes: {', '.join(out_of_scope)}", chainable=False
            )
        if events is not None:
            events.work_gate_passed(task_id)
        return _AttemptOutput(
            handoff_path=handoff_path, handoff_text=handoff_text,
            head=wt.head_commit(scratch), artifact_rel=None,
        )

    # Non-write: the produced deliverable must exist; publish it to its log key.
    assert task.artifact_out is not None
    produced = scratch / task.artifact_out
    if not produced.is_file():
        if handoff_path is not None:
            handoff_path.unlink(missing_ok=True)
        raise _Rejected(f"artifact_out not produced in CWD: {task.artifact_out}")
    published = run_dir.resolve(task.artifact_out)
    published.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(produced, published)
    if events is not None:
        events.work_gate_passed(task_id)
    return _AttemptOutput(
        handoff_path=handoff_path, handoff_text=handoff_text,
        head=None, artifact_rel=task.artifact_out,
    )


def run_task(
    task: Task,
    task_id: str,
    *,
    run_dir: RunDir,
    repo: Path | None,
    base: str | None,
    backends: Backends,
    read_root: Path | None = None,
    events: AttemptEvents | None = None,
) -> TaskResult:
    """Run ONE task to a triage verdict (reusable, safe to call concurrently).

    The task climbs an IN-EPOCH TIER LADDER of STAGES. A planner-``local`` task on a
    rig with a DISTINCT senior endpoint runs ``[local x LOCAL_MAX_ATTEMPTS, senior x
    SENIOR_MAX_ATTEMPTS]``; a planner-``senior`` task runs ``[senior x
    SENIOR_MAX_ATTEMPTS]``; a ``local`` task on an every-tier-local rig runs
    ``[local x LOCAL_MAX_ATTEMPTS]`` (escalating would just re-run the same model).
    Each attempt grinds in a fresh isolated worktree (implement) or scratch dir
    (non-write) off the EXTERNAL base, dispatches the STAGE's rig, and gates on
    DETERMINISTIC FACTS (a non-empty in-scope commit, or a present artifact). A
    gate-clean attempt ALWAYS runs the stage-tier critic, which routes: PASS ->
    merge-ready, RETRY -> a bounded retry within the stage, ESCALATE -> the next stage
    (or the planner from the last stage). The incremental wip carries ACROSS attempts
    AND across the stage boundary, so senior RESUMES the local attempt's tree rather
    than rebuilding. A failed gate is a failed attempt. ``RateLimited`` escapes
    un-burned.

    A non-write task reads + grounds against ``read_root`` (the integration tip) when
    the driver supplies one. ``events`` (optional) lands the gate + triage + tier
    escalation in the journal live; ``None`` keeps this callable standalone for the
    unit tests.
    """

    implement = task.mode == "implement"
    slug = task_id.replace("/", "-")
    critic_read_root = read_root if read_root is not None else repo
    try:
        domain_skills = _load_domain_skills(repo, task)
    except _Rejected as rej:
        return TaskResult(task_id, "escalated", attempts=0, reason=rej.reason)
    repo_map = load_repo_map(repo)

    if task.tier == "local" and backends.has_distinct_tier("senior"):
        stages = [("local", LOCAL_MAX_ATTEMPTS), ("senior", SENIOR_MAX_ATTEMPTS)]
    elif task.tier == "senior":
        stages = [("senior", SENIOR_MAX_ATTEMPTS)]
    else:
        stages = [("local", LOCAL_MAX_ATTEMPTS)]

    failure_context: list[str] = []
    prior_branch: str | None = None  # implement incremental-retry/escalation chain base
    attempts = 0  # TOTAL dispatch count across every stage (for TaskResult.attempts)
    for stage_idx, (tier, budget) in enumerate(stages):
        last_stage = stage_idx == len(stages) - 1
        if stage_idx > 0 and events is not None:
            # The ladder just escalated to a stronger tier on the carried wip.
            events.tier_escalated(task_id, tier, attempts + 1)
        stage_attempt = 0
        while stage_attempt < budget:
            stage_attempt += 1
            attempts += 1
            attempt_base = prior_branch if prior_branch is not None else base
            if implement:
                assert repo is not None and attempt_base is not None
                scratch = run_dir.worktrees_root / slug / f"attempt-{attempts}"
                branch = f"grind-wip/{slug}/attempt-{attempts}"
                wt.add_worktree(repo, scratch, branch=branch, base=attempt_base)
            else:
                scratch = run_dir.artifacts_dir(f"{task_id}/attempt-{attempts}")
                branch = None

            try:
                attempt = _run_attempt(
                    task, task_id, tier=tier,
                    scratch=scratch, repo=repo, read_root=read_root, epoch_base=base,
                    backends=backends,
                    domain_skills=domain_skills,
                    repo_map=repo_map,
                    failure_context=tuple(failure_context),
                    prior_work_present=prior_branch is not None,
                    run_dir=run_dir,
                    events=events,
                )
            except _Rejected as rej:
                if events is not None:
                    events.work_gate_rejected(task_id, rej.reason)
                failure_context.append(rej.reason)
                prior_branch = _carry_or_discard(
                    rej, repo=repo, scratch=scratch, branch=branch,
                    implement=implement, task_id=task_id,
                )
                continue

            # Gate-clean -> the stage-tier critic triages it (in the same scratch: the
            # post-commit worktree for implement, the artifact dir for a non-write task).
            # The resilient node recovers a chatted verdict + escalates a stuck critic
            # to senior, and only raises CriticError when it cannot land any verdict.
            try:
                verdict = _resilient_critic_verdict(
                    task, task_id, tier=tier, scratch=scratch, base=base,
                    artifact_rel=attempt.artifact_rel,
                    handoff_text=attempt.handoff_text,
                    critic_read_root=critic_read_root, run_dir=run_dir,
                    backends=backends, domain_skills=domain_skills, events=events,
                )
            except CriticError as exc:
                _discard(repo, scratch, branch, implement)
                return TaskResult(
                    task_id, "escalated", attempts=attempts,
                    handoff_path=attempt.handoff_path, reason=str(exc),
                )
            if events is not None:
                events.verdict(task_id, verdict.outcome, verdict.reason)

            if verdict.outcome == "PASS":
                return TaskResult(
                    task_id, "passed", attempts=attempts,
                    handoff_path=attempt.handoff_path, verdict=verdict,
                    branch=branch, head=attempt.head, artifact_key=attempt.artifact_rel,
                )
            if verdict.outcome == "ESCALATE":
                if last_stage:
                    _discard(repo, scratch, branch, implement)
                    return TaskResult(
                        task_id, "escalated", attempts=attempts,
                        handoff_path=attempt.handoff_path, verdict=verdict,
                        reason=verdict.reason or "critic escalated to planner",
                    )
                # Not the last stage: carry the wip up to the next (senior) tier.
                failure_context.append(f"critic ESCALATE: {verdict.reason}")
                prior_branch = _carry_partial(repo, scratch, branch, implement, task_id)
                break
            # RETRY: a defect the same worker can fix; chain the committed wip.
            failure_context.append(f"critic RETRY: {verdict.reason}")
            prior_branch = _carry_partial(repo, scratch, branch, implement, task_id)
        # The stage ended without a PASS (budget exhausted, or ESCALATE broke out of a
        # non-last stage): the loop advances to the next stage, which resumes the
        # carried ``prior_branch``. The last stage falls through to the terminal below.

    reason = failure_context[-1] if failure_context else "no attempts"
    return TaskResult(
        task_id, "escalated", attempts=attempts,
        reason=f"ladder exhausted: {reason}",
    )


def _carry_or_discard(
    rej: _Rejected,
    *,
    repo: Path | None,
    scratch: Path,
    branch: str | None,
    implement: bool,
    task_id: str,
) -> str | None:
    """Route a rejected attempt's worktree: a chainable failure keeps the partial
    wip as the next retry's base; a poisoned (non-chainable) one is discarded so the
    retry restarts clean. Returns the new chain base (a branch name) or None."""

    if not implement:
        return None
    if rej.chainable:
        return _carry_partial(repo, scratch, branch, implement, task_id)
    _discard(repo, scratch, branch, implement)
    return None


def _carry_partial(
    repo: Path | None,
    scratch: Path,
    branch: str | None,
    implement: bool,
    task_id: str,
) -> str | None:
    """Commit an implement attempt's partial work to its branch (the core only
    commits on success, so otherwise it dies with the worktree), tear down the
    worktree checkout, and KEEP the branch as the next retry's chain base."""

    if not implement or repo is None or branch is None:
        return None
    wt.commit_all(scratch, f"grind-wip({task_id}): partial kept for retry")
    wt.remove_worktree(repo, scratch)
    return branch


def _discard(
    repo: Path | None, scratch: Path, branch: str | None, implement: bool
) -> None:
    """Tear down an attempt that leaves nothing to merge (ESCALATE / a poisoned
    worktree): worktree removed + branch deleted (zero dead artifacts)."""

    if implement and repo is not None and branch is not None:
        wt.discard_attempt(repo, scratch, branch)
