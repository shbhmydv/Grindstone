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

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, Union

from grindstone.config import load_operating_skill
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
    """The model endpoint refused the call for rate-limit reasons.

    The SHORT, transient kind: a throttle (429) that clears within seconds to
    minutes, handled by a short exponential backoff. Distinct from a
    ``SessionLimited`` quota-window exhaustion (see below), which resets in hours.
    """


class SessionLimited(RateLimited):
    """A long quota-window SESSION limit, not a transient throttle.

    The Claude / codex / opencode-go "session limit" (e.g. claude's "You've hit
    your session limit . resets 2:20am") is a usage WINDOW that resets in HOURS,
    fundamentally different from a 429 throttle that clears in seconds. It must
    PARK the run and retry hourly WITHOUT counting against the transient-retry or
    rate-limit-wait budgets. Subclassing ``RateLimited`` keeps every existing
    ``except RateLimited`` catching it; the policy layer distinguishes the two via
    ``isinstance`` and ``classify_failure``.
    """


class WorkerTimeout(TransportError):
    """The worker hung and was killed by its own transport-level supervisor."""


#: Signatures of a long quota-window SESSION limit in a CLI's stdout/stderr. A
#: module-level constant so both transports (planner + worker) detect identically,
#: and extensible: add codex / opencode-go wording here as it is observed. NOTE the
#: phrasing differs from a transient 429: a plain "rate limit (429)" must NOT match
#: any of these (it stays on the short-backoff ``RateLimited`` path).
_SESSION_LIMIT_SIGNATURES: tuple[str, ...] = ("session limit", "usage limit")


#: The HOURLY session-limit park policy, shared by BOTH retry paths (the planner
#: loop in run_loop and the worker/senior attempt driver in task_loop). A long
#: quota-window limit resets in hours, so we retry once an hour (never against the
#: transient/rate-limit/attempt budgets), bounded by a high finite ceiling so a
#: stuck window escalates rather than hanging forever. Defined here (next to
#: ``SessionLimited``) so task_loop can import them without the planner-import cycle
#: (planner -> epoch_loop -> task_loop); ``planner`` re-exports them for its callers.
SESSION_LIMIT_RETRY_S = 3600.0
MAX_SESSION_LIMIT_WAITS = 24


def is_session_limit(text: str) -> bool:
    """True iff ``text`` carries a long quota-window SESSION-limit signature.

    Matches any of ``_SESSION_LIMIT_SIGNATURES`` ("session limit", "usage limit"),
    plus the claude phrasing where a "limit" "resets" at a wall-clock time
    (e.g. "resets 2:20am"). Case-insensitive. A plain transient "rate limit (429)"
    deliberately does NOT match: that stays on the short-backoff ``RateLimited``
    path; only the session/usage window goes to the hourly ``SessionLimited`` path.
    """

    blob = (text or "").lower()
    if any(sig in blob for sig in _SESSION_LIMIT_SIGNATURES):
        return True
    return "limit" in blob and "resets" in blob


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
    #: The SELECTED domain skills for this task (name -> the ``.md`` text), loaded
    #: by ``task_loop`` from the target repo's ``.grindstone/skills/`` catalogue for
    #: each name in ``task.skills``. Retrieve-not-concatenate: only the skills the
    #: planner selected for THIS task ride here, never the whole catalogue. Empty
    #: for a task that selected none / a repo with no catalogue (the common case);
    #: rendered as a ``<domain_skills>`` block only when present.
    domain_skills: dict[str, str] = field(default_factory=dict)
    #: An INFRA-REPAIR brief (gate-rebalance G3): when set, the dispatch is not a
    #: feature task but a focused senior repair of a structurally-broken gate
    #: ENVIRONMENT. Carrying it on the request keeps the transport unchanged, the
    #: prompt builder branches to ``build_infra_repair_prompt``. ``None`` for every
    #: ordinary task.
    infra_repair: "InfraRepairBrief | None" = None
    #: A VERIFICATION brief (gate-rebalance G4): when set, the dispatch is not a
    #: feature task but the end-of-epoch adversarial verification pass that judges
    #: the epoch's natural-language ``criteria`` against the produced artifacts and
    #: writes ``verdict.json`` (NOT a handoff). Carrying it on the request keeps the
    #: transport unchanged, the prompt builder branches to
    #: ``build_verification_prompt``. ``None`` for every ordinary task.
    verification: "VerificationBrief | None" = None
    #: Incremental retry (implement tasks): True when this attempt's worktree was
    #: based on a PRIOR attempt's branch, so the prior attempt's work is already
    #: present in ``scratch``. The prompt tells the worker it may fix that work in
    #: place OR reset and redo if it judges the prior attempt bad. False for a first
    #: attempt and for a clean tier-escalation start (a fresh worktree from base).
    prior_work_present: bool = False


@dataclass(frozen=True)
class InfraRepairBrief:
    """The focused brief a senior infra-repair worker is dispatched with (G3).

    ``failing_commands`` are the gate commands that failed environmentally;
    ``output_tail`` is their captured stderr/stdout (so the repair knows WHY);
    ``reason`` is the classifier's matched signature. ``allow_host_commands`` is
    the host-command guard's allowlist (default empty): repo-local fixes are
    automatic, host-level actions are reported, not run, unless allowlisted.
    """

    failing_commands: list[str]
    output_tail: str
    reason: str
    allow_host_commands: list[str]


#: The verdict file the verification pass writes (re-read disk contract, NOT a
#: handoff). The core relocates + Pydantic-re-validates it; stdout is never parsed.
VERDICT_FILENAME = "verdict.json"


@dataclass(frozen=True)
class VerificationBrief:
    """The focused brief the end-of-epoch verification pass is dispatched with (G4).

    ``epoch_goal`` is the epoch title/rationale; ``criteria`` is each task's
    natural-language acceptance statement (aggregated across the epoch); ``artifacts``
    is a per-task pointer to what was produced (the handoff state PLUS, for an
    artifact-mode task, the absolute PATH to the relocated deliverable the verifier must
    READ), so the verifier judges the REAL artifacts on disk, never a byte-capped
    paraphrase embedded in the prompt. The verifier is adversarial, hunts for gaps,
    defaults to FAIL on uncertainty, and writes ``verdict.json`` only.
    """

    epoch_goal: str
    criteria: list[str]
    artifacts: list[str]


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


#: The worker scenario names: one operating skill each under
#: ``skills/operating/worker/<scenario>.md``, selected by ``select_worker_scenario``.
#: The scenario IS the epoch mode. The executor is ONE role: ``build_worker_prompt``
#: builds the identical prompt for a worker- or senior-tier task (the tier only
#: selects which request SCRIPT runs it), so the senior reuses these same skills,
#: there is no separate senior namespace.
WORKER_SCENARIOS: frozenset[str] = frozenset(
    {"implement", "research", "review", "artifact"}
)


def select_worker_scenario(request: WorkerRequest) -> str:
    """Pick the worker's scenario skill from the task shape + mode (pure, no I/O).

    Mirrors the prompt builder's original branch: an ``ImplementTask`` is always
    the implement scenario (the only write task); every non-write ``ArtifactTask``
    routes by the epoch ``mode`` (research / review / artifact). An unexpected
    mode on a non-write task falls to ``artifact`` (the original ``else`` branch).
    """

    if isinstance(request.task, ImplementTask):
        return "implement"
    if request.mode == "research":
        return "research"
    if request.mode == "review":
        return "review"
    return "artifact"


def _file_ownership_block(task: ImplementTask) -> str:
    """The DYNAMIC implement ownership lane, built from ``task.file_ownership``.

    Kept in code (not the static skill) because the globs are per-task, mirrored
    on the planner's volatile-tail pattern. Names ``REVIEW_FILENAME`` from the live
    constant so the orchestration-file exemption never drifts.
    """

    globs = "\n".join(f"  - {g}" for g in task.file_ownership)
    return f"""
<file_ownership>
You may create or edit files ONLY within these globs:
{globs}
Changing ANY other file fails the attempt. (`handoff.json`, `check_handoff.py`
and `{REVIEW_FILENAME}` are orchestration files, not repo work, write them in
the CWD as instructed; the orchestrator excludes them from this rule.)
</file_ownership>
"""


def _review_targets_block(task: ArtifactTask) -> str:
    """The DYNAMIC review targets list, built from ``task.targets``.

    Kept in code (not the static review skill) because the targets are per-task;
    the skill prose points at this ``<review_targets>`` block.
    """

    targets = "\n".join(f"  - {t}" for t in task.targets or [])
    return f"""
<review_targets>
{targets}
</review_targets>
"""


def _domain_skills_block(request: WorkerRequest) -> str:
    """Compose the SELECTED domain skills into a ``<domain_skills>`` block (pure).

    Retrieve-not-concatenate: ``request.domain_skills`` already holds ONLY the
    skills the planner selected for this task (the loader fetched them in
    ``task_loop``); the worker never sees the whole catalogue. Each skill is its
    repo-authored ``.md`` text under a named child tag. Empty -> empty string.
    """

    if not request.domain_skills:
        return ""
    blocks = "\n".join(
        f'<skill name="{name}">\n{text}\n</skill>'
        for name, text in sorted(request.domain_skills.items())
    )
    return (
        "\n<domain_skills>\nRepo-specific skills selected for this task. Treat them "
        "as authoritative guidance for THIS repo's conventions and patterns:\n"
        f"{blocks}\n</domain_skills>\n"
    )


def build_worker_prompt(request: WorkerRequest) -> str:
    """Construct the worker prompt (pure function, no pi, no I/O).

    Common skeleton: goal, resolved inputs, the done_when checks, prior-failure
    context, and a handoff block pinning the disk contract (write handoff.json
    in the CWD, the exact task_id, the byte cap, the citation requirement).
    Between done_when and the handoff block sits the PER-MODE worker plan
    (owner ruling 2026-06-11): do->verify->review discipline belongs to the
    implement plan ONLY; research/review/artifact get their own lean plans, no
    verify theater. The STATIC discipline of each mode is an operating skill
    loaded verbatim from ``skills/operating/worker/<scenario>.md`` (selected by
    ``select_worker_scenario``) and concatenated as-is; the DYNAMIC per-task data
    (the implement ownership globs, the review targets) is built here in code and
    appended around it, mirroring the planner's CORE + scenario + volatile-tail
    split. We own the model-facing format (§7: XML-tagged sections).
    """

    if request.infra_repair is not None:
        return build_infra_repair_prompt(request, request.infra_repair)
    if request.verification is not None:
        return build_verification_prompt(request, request.verification)
    task = request.task
    artifact_line = ""
    if isinstance(task, ArtifactTask):
        artifact_line = (
            f"\nProduce the artifact at log key `{task.artifact_out}`.\n"
        )
    scenario = select_worker_scenario(request)
    skill = load_operating_skill("worker", scenario)
    if isinstance(task, ImplementTask):
        # The static implement discipline (skill) + the dynamic ownership lane.
        plan_block = "\n" + skill + _file_ownership_block(task)
        occupancy_line = (
            "  - occupancy: {\"compacted\": <bool>, \"subagent_splits\": <int>}, "
            "report honestly;\n    your reviewer spawn counts as a split."
        )
    else:
        # research / artifact: skill is self-contained; review appends its targets.
        plan_block = "\n" + skill
        if scenario == "review":
            plan_block += _review_targets_block(task)
        occupancy_line = (
            "  - occupancy: {\"compacted\": <bool>, \"subagent_splits\": <int>}."
        )
    domain_skills_block = _domain_skills_block(request)
    context_block = ""
    if request.failure_context:
        joined = "\n".join(f"  - {c}" for c in request.failure_context)
        context_block = (
            "\n<prior_failures>\nEarlier attempts failed for these reasons; fix "
            "them. Each line is a SHORT summary followed by absolute PATH(s) to the "
            "full detail on disk (the complete failure text and, where present, that "
            "attempt's rejected handoff.json). You MAY read those paths for the full "
            f"detail if the summary is not enough:\n{joined}\n</prior_failures>\n"
        )
    repo_map_block = ""
    if request.repo_map:
        repo_map_block = (
            "\n<repo_map>\nStructural map of the target repo near this task's "
            "files (most-referenced symbols first). A navigation aid, not "
            f"exhaustive; verify against the actual files.\n{request.repo_map}\n"
            "</repo_map>\n"
        )
    prior_work_block = ""
    if request.prior_work_present:
        prior_work_block = (
            "\n<prior_work>\nA PRIOR attempt at this task left its work IN PLACE in "
            "this working directory (the files it created or edited are already "
            "here). Build on it: fix what the <prior_failures> below report and "
            "complete what is unfinished, rather than starting from scratch. If you "
            "judge the prior attempt's approach wrong, you MAY reset and redo it, but "
            "do not lose correct work for no reason.\n</prior_work>\n"
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
{plan_block}{domain_skills_block}{prior_work_block}{context_block}
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


def _render_host_guard(brief: "InfraRepairBrief") -> str:
    """The host-command guard clause: repo-local automatic, host-level reported.

    Repo-local / in-worktree fixes (installing a dep into package.json/lockfile,
    editing config inside the repo) are fully automatic. Host-level / privileged
    actions (``sudo``, ``apt``, system-wide installs, writes outside the repo) are
    DENY by default: the worker must REPORT "needs host command X" in the handoff
    rather than run it, UNLESS X is on the allowlist below. The allowlist is the
    operator's explicit opt-in for a trusted box; empty means nothing host-level.
    """

    if brief.allow_host_commands:
        allowed = ", ".join(f"`{c}`" for c in brief.allow_host_commands)
        host_line = (
            f"The operator has ALLOWLISTED these host-level commands, you may run "
            f"them: {allowed}. Any OTHER host-level / privileged action (sudo, apt, "
            f"system-wide install, a write outside this repo) is still forbidden."
        )
    else:
        host_line = (
            "NO host-level commands are allowlisted. Every fix must be repo-local "
            "(stay inside this worktree). Do NOT run sudo, apt, a system-wide "
            "install, or write outside the repo."
        )
    return (
        "<host_guard>\n"
        "Make ONLY repo-local fixes that land in committed repo files (e.g. install a\n"
        "missing dependency so it is recorded in package.json / requirements / the\n"
        "lockfile, add a config file, fix a path inside the repo). " + host_line + "\n"
        "If the gate genuinely cannot be made satisfiable without a forbidden\n"
        "host-level action, do NOT run it: write a FAILED handoff whose not_done\n"
        "names the exact host command needed (\"needs host command: <cmd>\") so the\n"
        "operator can allowlist it. Never silently run a forbidden command.\n"
        "</host_guard>"
    )


def build_infra_repair_prompt(
    request: WorkerRequest, brief: "InfraRepairBrief"
) -> str:
    """The senior infra-repair prompt (pure function, no transport, no I/O).

    A FOCUSED brief, distinct from a feature task: the gate's deterministic checks
    failed for an ENVIRONMENTAL reason (a missing tool / dependency / a broken
    install), not because the application logic is wrong. The senior's job is to
    make the gate environment satisfiable WITHOUT rewriting application logic, then
    leave the worktree so the SAME commands pass. The failing commands + their
    captured output tell it exactly what broke; the host guard bounds what it may
    do; the disk contract (a handoff in the CWD) is the only result channel, but
    the authoritative judge is the core RE-RUNNING the gate, never this handoff.
    """

    commands = "\n".join(f"  - `{c}`" for c in brief.failing_commands)
    output = brief.output_tail.strip() or "(no output captured)"
    return f"""<infra_repair id="{request.task_id}">
A deterministic GATE check failed for an ENVIRONMENTAL reason ({brief.reason}), not
because the code is wrong. Your job: make the gate environment satisfiable so the
failing command(s) below pass, WITHOUT rewriting application logic.
</infra_repair>

<failing_gate_commands>
These commands are run by the gate in this worktree and currently fail
environmentally. After your repair they MUST pass (exit 0):
{commands}
</failing_gate_commands>

<captured_output>
{output}
</captured_output>

<skill>
This is an INFRA REPAIR, not a feature. Diagnose WHY the command cannot run, then
fix only the environment: install the missing dependency (so it is recorded in the
repo's manifest/lockfile), restore a missing config file, or correct a path INSIDE
the repo. Do not change application behavior, do not edit feature code to make a
test pass, do not delete or weaken the failing check. When done, run the failing
command(s) yourself and confirm they exit 0.
</skill>

{_render_host_guard(brief)}

<handoff>
Write `handoff.json` in your CWD as your final act (the only result channel; stdout
is ignored). schema_version "1", task_id exactly "{request.task_id}", status "DONE"
when the failing command(s) now pass else "FAILED"/"PARTIAL" with not_done filled
in (name any needed host command there). The orchestrator RE-RUNS the gate to
judge you, a false DONE is always caught.
</handoff>
"""


def build_verification_prompt(
    request: WorkerRequest, brief: "VerificationBrief"
) -> str:
    """The end-of-epoch verification prompt (pure function, no transport, no I/O).

    An ADVERSARIAL acceptance pass, distinct from a feature task and from the worker
    that produced the artifacts: a SEPARATE invocation given only the epoch goal,
    the natural-language ``criteria``, and the produced artifacts, told to judge
    whether EVERY criterion is met by the ACTUAL artifacts (read them, do not trust
    a summary), hunt for gaps, and DEFAULT TO FAIL on any uncertainty. It makes NO
    edits. Its only result channel is ``verdict.json`` (a re-read disk contract); the
    core relocates + re-validates it. A pass here can never override a failing
    deterministic floor, the floor ran first and already cleared.

    Because this pass already READS every diff/handoff/artifact to judge the criteria,
    it is ALSO asked to emit a descriptive ``digest`` (G10) in the same verdict: a
    factual steering summary for the planner choosing the NEXT epoch. The digest is
    purely descriptive and never affects the adversarial pass/fail.
    """

    criteria = "\n".join(f"  - {c}" for c in brief.criteria) or "  (none)"
    artifacts = "\n".join(f"  - {a}" for a in brief.artifacts) or "  (none reported)"
    return f"""<verification id="{request.task_id}">
This is an ADVERSARIAL VERIFICATION pass, not a feature task. You did NOT write this
work. Judge whether the epoch's acceptance criteria are met by the ACTUAL artifacts
in this worktree. Do NOT edit anything. Read the real files; never trust a summary.
</verification>

<epoch_goal>
{brief.epoch_goal}
</epoch_goal>

<criteria>
Each statement below is an acceptance criterion. For EACH one, decide whether the
artifacts in this worktree actually satisfy it:
{criteria}
</criteria>

<produced_artifacts>
What the epoch's tasks reported producing. Each entry is a POINTER, not the content:
where a line gives an absolute PATH to a relocated deliverable, READ that file in full
and judge its actual content. Never trust the one-line state; verify against the files:
{artifacts}
</produced_artifacts>

<stance>
Be adversarial. Hunt for gaps: a criterion only half-covered, a screen/case the work
never maps, a claim with no artifact behind it. DEFAULT TO FAIL on any uncertainty,
an unmet or unverifiable criterion means `pass` is false. Ground every judgement in a
real file you read. A false pass is worse than a false fail (the deterministic floor
already cleared; this pass exists only to CATCH semantic gaps that floor cannot).
</stance>

<verdict>
Write a file named exactly `{VERDICT_FILENAME}` in your CWD as your only output
(stdout is ignored). A JSON object:
  - "pass": true ONLY if EVERY criterion is met by the actual artifacts, else false.
  - "per_criterion": list of {{"criterion": <verbatim>, "met": <bool>, "evidence":
    <the file/line or quote that proves your judgement>}}, one per criterion.
  - "gaps": list of short strings, the concrete unmet-criterion gaps (empty when pass).
  - "digest": a short FACTUAL summary (a few sentences, NOT a grade) for the planner
    deciding the NEXT epoch: what this epoch actually produced (key files/structure/
    decisions you saw) and what is notably incomplete or risky. Describe, do not judge,
    the pass/fail above is the judgement; this digest is descriptive steering only.
Write `{VERDICT_FILENAME}` and nothing else, then stop.
</verdict>
"""
