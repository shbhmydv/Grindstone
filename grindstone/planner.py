"""Planner core: transport interface, failure classification, pure input
construction, and the decision validation pipeline (ARCHITECTURE.md, PLANNER_CONTRACT).

The planner is a stateless one-shot call. The deterministic core owns everything
the model does not: it CONSTRUCTS the input (stable head + volatile tail),
CLASSIFIES transport failures, and GATES the raw output through schema → typed →
semantic validation before dispatch. The transport only returns raw text.

Journal mapping for an INVALID decision (ruling 8): the vocabulary has no
"invalid decision" event. A decision that fails the gate is journaled by the
*caller* as ``planner_call_succeeded`` NOT emitted + ``planner_call_failed`` with
classification ``"transient"``, it is a retryable bad output, re-asked up to
twice before the run escalates. (Transport-level failures are journaled with
their ``classify_failure`` result; this module owns that classification.)

Input construction is a PURE function over durable state (ruling 3):

  - **Stable head**, byte-identical across every call of a run: a fixed system
    preamble, an (empty at S3) ``<skills>`` digest, an (empty at S3, S5 seam)
    ``<repo_memory>`` digest, and the stored ``<skeleton>`` as compact JSON. The
    head changes ONLY when ``propose_skeleton`` / ``revise_phases`` change the
    skeleton, then, and only then, the prefix cache legitimately resets.
  - **Volatile tail**, ``<state>`` (phase id, epoch counter, keyed-log index),
    ``<last_epoch>`` (the prior outcome flattened: per-task status/attempts/tier,
    each DONE task's handoff resulting_state + downstream_needs, each FAILED
    task's last failure reason), optional ``<errors>`` (re-ask feedback), and
    ``<request>``. References, not payloads: only log keys, never file bodies.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol

from grindstone.config import load_operating_skill
from grindstone.contracts.gate import decision_schema_errors
from grindstone.contracts.models import (
    ArtifactEpochArgs,
    EpochDecision,
    ImplementDecision,
    ImplementEpochArgs,
    Phase,
    parse_decision,
    parse_handoff,
)
from grindstone.contracts.semantics import (
    content_grep_check_violations,
    epoch_decision_violations,
    implement_task_size_violations,
)
from grindstone.epoch_loop import EpochOutcome
from grindstone.rundir import RunDir

# The transport exception family is the workers' (ruling 1), re-exported here so
# planner code imports one surface. ``PlannerHardError`` is the planner-only
# addition for auth/config: it is deliberately NOT a ``TransportError`` so
# ``classify_failure`` reads it as ``hard`` (the default branch).
from grindstone.worker import (
    MAX_SESSION_LIMIT_WAITS,
    SESSION_LIMIT_RETRY_S,
    RateLimited,
    SessionLimited,
    TransportError,
    WorkerTimeout,
)

__all__ = [
    "BACKOFF_BASE_S",
    "BACKOFF_CAP_S",
    "BACKOFF_FACTOR",
    "DEFAULT_LOCAL_MAX_TASK_FILES",
    "DEFAULT_SENIOR_MAX_TASK_FILES",
    "FailedEpochInfo",
    "FailureClass",
    "GateResult",
    "MAX_RATE_LIMIT_WAITS",
    "MAX_REASKS",
    "MAX_SESSION_LIMIT_WAITS",
    "MAX_TRANSIENT_RETRIES",
    "PLANNER_CORE",
    "PLANNER_SCENARIOS",
    "PhaseTailInfo",
    "PlannerHardError",
    "PlannerTransport",
    "RateLimited",
    "SESSION_LIMIT_RETRY_S",
    "SessionLimited",
    "TOOL_NAMES",
    "TransportError",
    "WorkerTimeout",
    "WorkspaceInfo",
    "backoff_delay",
    "build_planner_input",
    "classify_failure",
    "extract_decision_json",
    "flatten_last_epoch",
    "is_decision_like",
    "select_planner_scenario",
    "stable_head",
    "validate_decision",
    "volatile_tail",
]

#: The decision tool names (mirrors schemas/epoch_decision.json $.tool enum).
TOOL_NAMES: frozenset[str] = frozenset(
    {
        "propose_skeleton",
        "research",
        "implement",
        "review",
        "artifact",
        "revise_phases",
        "handle_failed_epoch",
        "escalate_run",
        "complete_run",
    }
)


class PlannerHardError(Exception):
    """A planner failure that needs a human: auth, config, unknown model error.

    Outside the transport exception family on purpose, ``classify_failure``
    returns ``hard`` for it (and for any other unrecognized exception).
    """


# --- transport interface -------------------------------------------------------


class PlannerTransport(Protocol):
    """One-shot planner: constructed input in, raw final-message text out.

    NEVER parses or validates (ruling 1). Raises the transport exception family
    on failure: ``RateLimited`` (→ backoff), ``TransportError`` / ``WorkerTimeout``
    (→ transient retry), ``PlannerHardError`` or anything else (→ human).

    ``workdir`` is the boundary's writable planner worktree when one exists: a rig
    that self-validates runs IN it (writing ``decision.json`` + looping on
    ``check_decision.py``); a read-only rig ignores it. ``None`` leaves the rig on
    its read-only fallback (no worktree: artifact-only run or unborn HEAD).
    """

    def plan(self, prompt: str, *, workdir: Path | None = None) -> str: ...


# --- failure classification + backoff (ruling 2) -------------------------------

FailureClass = Literal["rate_limit", "transient", "hard", "session_limit"]

#: Exponential backoff for rate-limit waits: 30s, 60s, 120s, 240s, 480s, 600s…
BACKOFF_BASE_S = 30.0
BACKOFF_FACTOR = 2.0
BACKOFF_CAP_S = 600.0
MAX_RATE_LIMIT_WAITS = 6
MAX_TRANSIENT_RETRIES = 3
MAX_REASKS = 2

# The HOURLY session-limit park policy (``SESSION_LIMIT_RETRY_S`` /
# ``MAX_SESSION_LIMIT_WAITS``) is imported from ``grindstone.worker`` above, where
# it lives next to ``SessionLimited`` so the task_loop worker path can share the
# same values WITHOUT the planner-import cycle (planner -> epoch_loop -> task_loop).
# It is re-exported here (in ``__all__``) so planner callers keep one import surface:
# a long quota-window limit resets in HOURS, so ``SessionLimited`` PARKS the run and
# retries once an hour, never against the transient/rate-limit budgets, bounded by a
# high finite ceiling (24 hourly waits ~= 24h, past any single reset window, then escalate).

#: Default tier-aware ceilings on a fresh implement task's ``file_ownership``
#: glob count (the planner-decomposition size gate). Mirror the config defaults
#: (``GrindstoneConfig.local_max_task_files`` / ``senior_max_task_files``); the
#: run loop passes the configured values, these are the seam default when a
#: caller validates a decision standalone.
DEFAULT_LOCAL_MAX_TASK_FILES = 5
DEFAULT_SENIOR_MAX_TASK_FILES = 12


def classify_failure(exc: BaseException) -> FailureClass:
    """Four-way classification of a planner-call failure (ARCHITECTURE.md).

    ``SessionLimited`` → session_limit (a long quota-window limit, park the run
    and retry hourly, NEVER against the transient/rate-limit budgets). It is
    checked FIRST because it subclasses ``RateLimited`` (a session limit is the
    long kind). ``RateLimited`` → rate_limit (a transient 429, short backoff, never
    auto-spill to a different planner). ``TransportError`` / ``WorkerTimeout`` →
    transient (retry the same call). Anything else (auth/config/unknown) → hard.
    """

    if isinstance(exc, SessionLimited):
        return "session_limit"
    if isinstance(exc, RateLimited):
        return "rate_limit"
    if isinstance(exc, (TransportError, WorkerTimeout)):
        return "transient"
    return "hard"


def backoff_delay(wait_index: int) -> float:
    """Backoff seconds for the ``wait_index``-th (0-based) rate-limit wait."""

    return min(BACKOFF_BASE_S * (BACKOFF_FACTOR**wait_index), BACKOFF_CAP_S)


# --- input construction: stable head (byte-identical across a run) -------------

#: The THIN, always-on planner core: the role identity + output discipline + the
#: cross-cutting facts true for EVERY planner call regardless of scenario (the
#: structural-check contract, the content-grep prohibition, the verification
#: floor, references-not-payloads, read-capable planning, the verdict-by-reference
#: steering, and the authoritative output envelope + args-by-tool table). It is
#: ALWAYS included, in the byte-identical stable head; the per-call SCENARIO skill
#: (selected by ``select_planner_scenario`` and loaded from ``skills/operating/
#: planner/``) carries the situation-specific guidance and is composed after it.
#:
#: WHY a constant here and not a loaded markdown file (unlike the scenarios): the
#: core lives in the stable head, whose byte-identity contract (ruling 3) must hold
#: independent of on-disk state, and it is tightly coupled to this module's exports
#: (the schema/tool names). The VARYING part is what benefits from being external
#: and selectable, so only the scenarios are files; ``load_operating_skill`` stays
#: role-generic for the worker/senior cores in the next round.
PLANNER_CORE = """\
You are the planner for Grindstone, a deterministic epoch-based orchestrator.
You are a stateless one-shot call. The fixed state machine runs the loop, runs
all checks, and integrates work; you only decide the NEXT step.

A SCENARIO skill follows (after the job + skeleton): it carries the guidance for
THIS call's situation. These CORE rules hold for EVERY decision:
- Emit EXACTLY ONE tool call per turn, as a single JSON object matching the
  epoch_decision schema. No prose, no second object, no commentary.
- The first decision of a run MUST be propose_skeleton. After that, every call
  is an epoch boundary: choose one of implement / research / review / artifact
  (1-8 independent fan-out tasks), revise_phases, escalate_run, or complete_run.
- You define exit criteria and done_when as DETERMINISTIC STRUCTURAL checks (a
  command with an expected exit code, or a required artifact log key). They are
  for structural facts ONLY: the project's own build / test / type-check
  commands, and file existence (`test -f`). The state machine evaluates them; you
  never claim a phase or task is done.
- Do NOT author CONTENT-GREP checks. A check that greps a file's CONTENT for a
  token (`rg`/`grep`/`egrep`/`fgrep`/`ag`/`ack` for a string or pattern) is a
  brittle proxy that fails for environmental reasons and is REJECTED by the gate.
  Express content or semantic acceptance ("the plan maps every ramp to an RN
  equivalent", "the report cites the source for each claim") as natural-language
  `criteria` on the task instead, an agentic pass judges those; a shell command
  never does. `criteria` is an optional list of prose acceptance statements; use
  it whenever the real bar is semantic rather than structural.
- Do NOT restate the verification FLOOR. The clean worktree, a valid handoff,
  committed work, and the repo's own build/test are owned by the repo config and
  the core and run on every gate automatically; your `checks`/`done_when` add
  task-specific STRUCTURAL facts on top, never a copy of the floor.
- References, not payloads. A task's `inputs` are log keys that already EXIST in
  the keyed-log index below; never invent a key. Resulting artifacts are named
  by their log key, never inlined. An accepted artifact task's `artifact_out`
  file is published to the keyed log, and an `artifact_exists` check may name
  it either by its exact log key or by its bare filename, use the bare
  filename in phase exit criteria (the P*/E*/T*/ placement is decided epochs
  later); it passes only while exactly one logged artifact carries that name.
- The last epoch's verification VERDICT (when one ran) is in the `<workspace>`
  manifest as a `.../verdict.json` path. It holds the verifier's per-criterion
  judgement, the unmet `gaps`, and a DESCRIPTIVE steering `digest` (what the epoch
  produced, what is incomplete or risky). READ that file to inform your NEXT decision.
  It is steering, NOT a gate: the pass/fail came from the deterministic floor and the
  criteria, never from this text. It is delivered BY REFERENCE (the full content lives
  on disk; you read what you need) rather than embedded, so nothing is truncated.
- READ-CAPABLE PLANNING. You run as a read-capable agent in the target repo: you
  MAY (and when the last-epoch rows or the phase checks are not enough to decide
  well, you SHOULD) grep and read on disk before deciding. The
  `<workspace>` block below gives you ABSOLUTE paths you may read: the
  integration-tip tree (a checkout of the CURRENT integration tip, the exact code
  the gate evaluates), the keyed-log root, and a manifest mapping each live log key
  (handoffs, the verdicts, relocated artifacts, the captured check output) to its
  absolute path. The `<workspace>` `repo_map` path, when present, is a ranked
  structural map of the current integration tip on disk (most-referenced
  files/symbols first); READ it when doing STRUCTURAL planning (deciding what to
  build, where things live, what already exists). It is large, so it is referenced
  by path, not inlined; on a focused failed-epoch disposition you can skip it. Use
  these handles to inspect the ACTUAL code, diffs, handoffs and artifacts rather
  than steering blind on the digest alone. This is for STEERING ONLY: the
  deterministic floor and the task criteria still DISPOSE of every gate, reading the
  tree never changes what passes, and reading is your OWN internal step. Whatever
  you read, your turn STILL ends with EXACTLY ONE epoch_decision tool call and
  nothing else.

Output format, emit EXACTLY ONE JSON object with this envelope, nothing else:
  {"schema_version":"1","tool":"<one tool name>","args":{ ... }}
The keys schema_version, tool, args are ALL mandatory. Do NOT use the shorthand
{"<tool>":{...}} and do NOT omit schema_version.

`args` by tool (the schema is authoritative):
- propose_skeleton: {"phases":[{"id":"P1..","title":..,"exit_criterion":[check..],"epoch_budget":int}]}
- implement:        {"epoch_title":..,"rationale":..,"tasks":[{"id":"T1..","goal":..,"done_when":[check..],"file_ownership":[concrete file path..],"senior"?:bool,"inputs"?,"skills"?,"criteria"?}]}
- research/review/artifact: {"epoch_title":..,"rationale":..,"tasks":[{"id","goal","done_when","artifact_out","targets"?,"senior"?:bool,"inputs"?,"skills"?,"criteria"?}]}
  (`senior`:true routes THAT task to the senior tier for judgment/taste/synthesis; default false runs it locally. `file_ownership` must enumerate CONCRETE files, no wildcard globs.)
- revise_phases:    {"reason":..,"phases":[..]}  (the PHASE STRUCTURE is wrong; replaces the current phase onward, never completed phases)
- handle_failed_epoch: {"action":"retry","hint":..,"escalate_tier"?:bool} | {"action":"escalate_senior","diagnosis":..} | {"action":"halt","reason":..}  (legal ONLY when an epoch has failed and is awaiting disposition)
- escalate_run:     {"reason":..,"needed_from_human"?}
- complete_run:     {"summary":..,"evidence":[check..]}
A check is {"cmd":..,"expect_exit"?:int} or {"artifact_exists":"<log key>"} or
{"vision_review":{"screenshot":"<eval-worktree-relative path>","criteria":..}}
(taste gate, phase exit criteria only, after a cmd check renders the shot).
"""


#: The planner scenario names: one operating skill each under
#: ``skills/operating/planner/<scenario>.md``, selected by ``select_planner_scenario``.
PLANNER_SCENARIOS: frozenset[str] = frozenset(
    {"plan_skeleton", "plan_epoch", "repair_epoch"}
)


def select_planner_scenario(
    *, skeleton_exists: bool, failed_epoch_active: bool
) -> str:
    """Pick the planner's scenario skill from durable run state (pure, no I/O).

    A state-machine lookup over the SAME signals the position-legality gate keys
    on, so the skill the planner reads always matches the decision the gate will
    accept:

    * no skeleton yet -> ``plan_skeleton`` (propose_skeleton is the only legal
      tool, decompose the job into the phase skeleton).
    * a failed epoch awaiting disposition -> ``repair_epoch`` (handle_failed_epoch
      is the only legal tool).
    * otherwise -> ``plan_epoch`` (the steady-state work decision).

    PRECEDENCE mirrors ``_position_legality``: the skeleton question is settled
    FIRST (a failed epoch cannot exist before a skeleton, since epochs require
    one), so ``skeleton_exists == False`` dominates; only once a skeleton exists
    does ``failed_epoch_active`` route to repair, else the default plan_epoch.
    """

    if not skeleton_exists:
        return "plan_skeleton"
    if failed_epoch_active:
        return "repair_epoch"
    return "plan_epoch"


def _compact(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def stable_head(job: str, skeleton: list[Phase] | None, repo_memory: str | None = None) -> str:
    """The byte-identical prefix of every planner call in a run (ruling 3).

    Depends ONLY on the (frozen-at-run-start) job spec, the skeleton, and the
    repo-memory digest (S5 seam: the run-loop freezes ``load_digest`` at run
    start so byte-identity holds run-long; ``skills`` stays empty). Identical
    bytes while those are unchanged; changes exactly when ``propose_skeleton`` /
    ``revise_phases`` move the skeleton. The ``<job>`` slot is the spec the
    planner plans against, frozen at run start, hence part of the head (an
    addition to ruling 3's enumerated slots: a planner cannot plan without the
    ask). The ``<repo_memory>`` slot is empty-byte-identical when no digest
    exists (the common case), so the seam adds zero cost to a memory-less repo.
    """

    skeleton_json = (
        _compact([p.model_dump(mode="json") for p in skeleton])
        if skeleton is not None
        else "null"
    )
    memory_slot = (
        f"<repo_memory>\n{repo_memory}\n</repo_memory>\n"
        if repo_memory
        else "<repo_memory>\n</repo_memory>\n"
    )
    return (
        f"<system>\n{PLANNER_CORE}</system>\n"
        f"<job>\n{job}\n</job>\n"
        "<skills>\n</skills>\n"
        f"{memory_slot}"
        f"<skeleton>\n{skeleton_json}\n</skeleton>\n"
    )


def _scenario_block(scenario: str) -> str:
    """Wrap the selected operating skill in its delimited ``<scenario>`` block.

    The scenario varies per call (it keys on the VOLATILE failed-epoch signal), so
    it lives here in ``build_planner_input`` between the byte-identical head and the
    volatile tail, NOT in ``stable_head`` (whose byte-identity contract must not key
    on volatile state). The skill text is loaded verbatim from
    ``skills/operating/planner/<scenario>.md`` through the role-generic loader."""

    skill = load_operating_skill("planner", scenario)
    return f'<scenario name="{scenario}">\n{skill}</scenario>\n'


# --- input construction: volatile tail -----------------------------------------


def flatten_last_epoch(run_dir: RunDir, outcome: EpochOutcome) -> list[dict[str, object]]:
    """Flatten an EpochOutcome into compact per-task rows for ``<last_epoch>``.

    DONE tasks contribute their handoff ``resulting_state`` + ``downstream_needs`` PLUS
    ``what_changed`` (each as ``kind:ref``) and ``not_done`` (G10), read from the keyed
    log, references, never bodies; FAILED tasks contribute their last failure reason.
    Pure over durable state (the relocated handoffs). The handoff's own schema already
    bounds these worker-written fields legitimately (``what_changed.ref`` / ``not_done``
    are <=256), so the rows carry them IN FULL: the extra embedding-truncation was a
    band-aid (silent information loss), and the whole handoff is referenceable by path via
    the ``<workspace>`` manifest. No diffs/file content here (the planner reads those from
    the workspace handles when it needs them).
    """

    rows: list[dict[str, object]] = []
    for task in outcome.tasks:
        row: dict[str, object] = {
            "task": task.task_id,
            "status": task.status,
            "attempts": task.attempts,
            "tier": task.tier,
        }
        if task.status == "done" and task.handoff_key:
            try:
                payload = json.loads(run_dir.resolve(task.handoff_key).read_text(encoding="utf-8"))
                handoff = parse_handoff(payload)
            except (ValueError, OSError):
                pass
            else:
                row["resulting_state"] = handoff.resulting_state
                row["downstream_needs"] = list(handoff.downstream_needs)
                row["what_changed"] = [
                    f"{wc.kind}:{wc.ref}" for wc in handoff.what_changed
                ]
                row["not_done"] = list(handoff.not_done)
        elif task.status == "failed":
            row["failure_reason"] = task.failure_reason
        rows.append(row)
    return rows


@dataclass(frozen=True)
class PhaseTailInfo:
    """Cumulative-state surfacing for the volatile tail (S4 ruling 3).

    References, never payloads: per-check pass/fail of the current phase's exit
    criterion (freshly evaluated by the core before this call), the integration
    tip's file LISTING (names + total, capped), the ids already passed, and
    whether the current phase is under an escalation demand (budget exhausted,
    only revise_phases / escalate_run are legal until it clears).
    """

    title: str
    check_results: list[tuple[str, bool]]
    budget_used: int
    budget: int
    passed_ids: list[str]
    escalation_active: bool
    tip_files: list[str]
    tip_total: int


@dataclass(frozen=True)
class FailedEpochInfo:
    """The focused context for a ``handle_failed_epoch`` decision (Part B).

    Carried in the volatile tail ONLY when an epoch has failed and is awaiting
    disposition. ``failed_tasks`` is each failed task's id + last reason;
    ``failed_checks`` is the phase exit checks that fail (label + captured command
    output, Part A); ``passing_handoffs`` is each DONE task's id +
    resulting_state, the workers' HONEST pass claim the planner must weigh against
    the still-failing gate (the gate-skepticism evidence)."""

    epoch_id: str
    failed_tasks: list[tuple[str, str]]
    failed_checks: list[str]
    passing_handoffs: list[tuple[str, str]]
    disposed_count: int
    cap: int
    #: Concrete unmet-criterion gaps from the end-of-epoch agentic verification pass
    #: (G4): the epoch cleared its deterministic floor but a natural-language
    #: acceptance criterion was judged UNMET. Empty for a task-failure / gate-failure
    #: epoch; non-empty when the semantic verification pass is what failed the epoch.
    verification_gaps: list[str] = field(default_factory=list)


#: Cap on the manifest rows surfaced in the `<workspace>` block: a bounded
#: reference (key -> absolute path), not a payload. The full keyed-log COUNT is
#: always reported alongside; the planner resolves any further key itself via the
#: keyed-log root + the `<state>` index. Mirrors the integration-tip listing cap.
WORKSPACE_MANIFEST_CAP = 200


@dataclass(frozen=True)
class WorkspaceInfo:
    """The read-capable workspace surfaced to the planner (its pull-access handles).

    Built deterministically from the run-dir layout + the integration tip. The
    planner runs as a read-capable agent (codex read-only ``-C repo`` / claude
    Read+Grep with cwd=repo), so these ABSOLUTE paths, all inside ``$repo`` and thus
    inside each rig's read sandbox, let it grep/read the actual code, handoffs,
    diffs and artifacts for STEERING. Reading is purely the planner's own internal
    step: the deterministic floor + criteria still dispose of every gate.

    ``integration_tip`` is a checked-out tree of the CURRENT integration tip (the
    exact code the gate evaluates), or ``None`` when no tip exists yet (the run has
    not produced an integration branch). ``keyed_log_root`` is the run dir's keyed-log
    root. ``manifest`` maps each live log key to its absolute path (handoffs, the
    verdicts, relocated artifacts, captured check output), already resolved + bounded
    by the caller; an empty manifest renders cleanly.

    ``repo_map_path`` is the on-disk PageRank-ranked structural map of the current
    integration tip, written once per boundary by the caller to a stable file under
    the run dir; the planner reads it for structural planning. The map is delivered BY
    REFERENCE (this path) rather than inlined into the prompt, so the prompt does not
    pay the per-boundary token tax for the whole map. ``None`` below the size threshold
    (first epoch / tiny repo): no file is written and the entry is omitted cleanly.
    """

    integration_tip: Path | None
    keyed_log_root: Path
    manifest: list[tuple[str, Path]]
    repo_map_path: Path | None = None


def _workspace_block(ws: WorkspaceInfo) -> str:
    tip_line = (
        f"integration_tip (checkout of the current integration tip, the exact code "
        f"the gate evaluates): {ws.integration_tip.resolve()}\n"
        if ws.integration_tip is not None
        else "integration_tip: (none yet, no integration branch has been built)\n"
    )
    shown = ws.manifest[:WORKSPACE_MANIFEST_CAP]
    more = "" if len(shown) >= len(ws.manifest) else f" (showing {len(shown)})"
    manifest_lines = (
        "\n".join(f"  - {key} -> {path}" for key, path in shown) or "  (empty)"
    )
    repo_map_line = (
        f"repo_map (PageRank-ranked structural map of the current integration tip, "
        f"most-referenced files/symbols first; READ it for structural planning, it is "
        f"large so it is referenced not inlined): {ws.repo_map_path.resolve()}\n"
        if ws.repo_map_path is not None
        else ""
    )
    return (
        "<workspace>\n"
        "Read-capable handles (ABSOLUTE paths you MAY grep/read for STEERING; the "
        "deterministic floor + criteria still dispose, reading never changes a "
        "verdict).\n"
        f"{tip_line}"
        f"keyed_log_root (the run dir's keyed log; every log key resolves under it): "
        f"{ws.keyed_log_root.resolve()}\n"
        f"{repo_map_line}"
        f"log_manifest (live log keys -> absolute paths{more}):\n"
        f"{manifest_lines}\n"
        "</workspace>\n"
    )


def _domain_skills_block(index: dict[str, str]) -> str:
    """The available domain-skill catalogue, rendered for the planner's SELECTION.

    A bounded reference list (name -> one-line description), the target repo's
    own ``.grindstone/skills/index.md``. The planner attaches the relevant skills
    to a task by NAME via the task's ``skills`` field; the core delivers only the
    SELECTED skill text into that task's worker prompt (retrieve, not concatenate).
    Empty index -> empty string (the common case: most repos ship no catalogue), so
    a memory-less repo pays zero bytes and the block simply never appears.
    """

    if not index:
        return ""
    lines = "\n".join(f"  - {name}: {desc}" for name, desc in sorted(index.items()))
    return (
        "<domain_skills>\n"
        "Domain skills this target repo provides. SELECT the relevant ones for a "
        "task by listing their NAMES in that task's `skills` field; the core "
        "delivers the selected skill TEXT to that task's worker. Keep selection "
        "MINIMAL, name only the skills the task actually needs (retrieve, do not "
        "attach the whole catalogue):\n"
        f"{lines}\n"
        "</domain_skills>\n"
    )


def _failed_epoch_block(info: FailedEpochInfo) -> str:
    tasks = (
        "\n".join(f"  - {tid}: {reason}" for tid, reason in info.failed_tasks)
        or "  (none, the phase gate failed though every task passed)"
    )
    checks = "\n".join(f"  - {label}" for label in info.failed_checks) or "  (none)"
    gaps_block = ""
    if info.verification_gaps:
        gaps = "\n".join(f"  - {g}" for g in info.verification_gaps)
        gaps_block = (
            "semantic_gaps (the agentic verification pass judged these acceptance "
            "criteria UNMET by the actual artifacts, the deterministic floor passed "
            "but the work is incomplete, retry with these as corrective feedback):\n"
            f"{gaps}\n"
        )
    handoffs = (
        "\n".join(f"  - {tid}: {state}" for tid, state in info.passing_handoffs)
        or "  (none)"
    )
    return (
        f"<failed_epoch epoch={info.epoch_id} "
        f"disposed={info.disposed_count}/{info.cap}>\n"
        "This epoch FAILED. The ONLY legal decision now is handle_failed_epoch "
        "(retry / escalate_senior / halt). revise_phases is NOT for this, the "
        "phase STRUCTURE is unchanged.\n"
        "failed_tasks (retry ladder exhausted):\n"
        f"{tasks}\n"
        "failing_phase_checks (deterministic, WITH captured output, env-vs-code "
        "evidence):\n"
        f"{checks}\n"
        f"{gaps_block}"
        "worker handoffs that claimed an HONEST pass (weigh these against the "
        "still-failing gate, a passing worker + a failing gate that repeats the "
        "SAME way points at the gate/environment, prefer halt over another "
        "identical repair):\n"
        f"{handoffs}\n"
        f"This phase has disposed of {info.disposed_count} of {info.cap} permitted "
        "failed epochs; at the cap the state machine halts to a human regardless.\n"
        "</failed_epoch>\n"
    )


def _phase_status_block(phase_id: str | None, phase: PhaseTailInfo) -> str:
    checks = (
        "\n".join(
            f"  - [{'PASS' if ok else 'FAIL'}] {label}"
            for label, ok in phase.check_results
        )
        or "  (none)"
    )
    passed = ", ".join(phase.passed_ids) or "(none)"
    shown = "\n".join(f"  - {p}" for p in phase.tip_files) or "  (empty)"
    more = "" if len(phase.tip_files) >= phase.tip_total else f" (showing {len(phase.tip_files)})"
    status = (
        "<phase_status>\n"
        f"current_phase: {phase_id or 'none'}, {phase.title}\n"
        f"epoch_budget: {phase.budget_used}/{phase.budget} epochs used\n"
        "exit_criterion (deterministic, evaluated by the state machine):\n"
        f"{checks}\n"
        f"passed_phases: {passed}\n"
        "</phase_status>\n"
        f"<integration_tip files={phase.tip_total}{more}>\n"
        f"{shown}\n"
        "</integration_tip>\n"
    )
    if phase.escalation_active:
        status += (
            "<escalation>\n"
            "This phase's epoch_budget is EXHAUSTED and its exit criterion still "
            "fails. The ONLY legal decisions now are revise_phases (replace the "
            "current phase with a better-scoped plan) or escalate_run.\n"
            "</escalation>\n"
        )
    return status


def volatile_tail(
    *,
    phase_id: str | None,
    epoch_counter: int,
    log_index: list[str],
    last_epoch_rows: list[dict[str, object]] | None,
    reask_errors: list[str],
    phase: PhaseTailInfo | None = None,
    failed_epoch: FailedEpochInfo | None = None,
    workspace: "WorkspaceInfo | None" = None,
    domain_skills: dict[str, str] | None = None,
) -> str:
    """The per-call suffix: running state, phase status, last-epoch report,
    re-ask feedback, request. Never byte-stable, it carries everything that
    moves (S4 adds the phase-status + integration-tip surfacing, ruling 3).

    The PageRank-ranked structural map of the target repo's CURRENT tip is no longer
    inlined here; it is delivered BY REFERENCE via the ``<workspace>`` ``repo_map``
    path (the planner reads the on-disk file for structural planning), so the prompt
    does not pay the per-boundary token tax for the whole map.

    The verifier's descriptive steering digest (G10) is likewise no longer embedded
    here: the full ``verdict.json`` (digest + per-criterion evidence + gaps) is
    persisted on disk and surfaced in the ``<workspace>`` manifest, so the planner reads
    it by reference (nothing truncated) rather than from an inlined ``<epoch_digest>``
    block."""

    log_lines = "\n".join(f"  - {k}" for k in log_index) or "  (empty)"
    state = (
        "<state>\n"
        f"phase: {phase_id or 'none'}\n"
        f"epoch_counter: {epoch_counter}\n"
        "keyed_log:\n"
        f"{log_lines}\n"
        "</state>\n"
    )
    phase_status = _phase_status_block(phase_id, phase) if phase is not None else ""
    failed_epoch_block = (
        _failed_epoch_block(failed_epoch) if failed_epoch is not None else ""
    )
    workspace_block = _workspace_block(workspace) if workspace is not None else ""
    domain_block = _domain_skills_block(domain_skills) if domain_skills else ""
    if last_epoch_rows is None:
        last = "<last_epoch>\n  (none, this is the first decision)\n</last_epoch>\n"
    else:
        last = f"<last_epoch>\n{_compact(last_epoch_rows)}\n</last_epoch>\n"
    errors = ""
    if reask_errors:
        joined = "\n".join(f"  - {e}" for e in reask_errors)
        errors = (
            "<errors>\nYour previous decision was REJECTED. Fix and re-emit:\n"
            f"{joined}\n</errors>\n"
        )
    request = (
        "<request>\n"
        "Emit exactly one tool call as a single JSON object matching the "
        "epoch_decision schema. No prose.\n"
        "</request>\n"
    )
    return (
        state + phase_status + failed_epoch_block + workspace_block
        + domain_block + last + errors + request
    )


def build_planner_input(
    *,
    job: str,
    skeleton: list[Phase] | None,
    phase_id: str | None,
    epoch_counter: int,
    log_index: list[str],
    last_epoch_rows: list[dict[str, object]] | None,
    reask_errors: list[str],
    phase: PhaseTailInfo | None = None,
    repo_memory: str | None = None,
    failed_epoch: FailedEpochInfo | None = None,
    workspace: "WorkspaceInfo | None" = None,
    domain_skill_index: dict[str, str] | None = None,
) -> str:
    """Full constructed input: ``stable_head`` + ``<scenario>`` + ``volatile_tail``.

    The stable head carries the always-on ``PLANNER_CORE``; between it and the tail,
    the ONE deterministically-selected operating skill is composed in. The scenario
    is derived from the SAME durable signals the gate keys on (a skeleton exists iff
    ``skeleton is not None``; an epoch awaits failure disposition iff ``failed_epoch
    is not None``), so the guidance the planner reads always matches the decision the
    gate will accept, no new state is invented.

    ``workspace`` (when present) is injected into the volatile tail only; the stable
    head stays byte-identical across a run regardless of it. The structural repo-map and
    the verifier's verdict (digest + evidence + gaps) are delivered BY REFERENCE through
    the ``workspace`` manifest (``repo_map`` + the ``verdict.json`` path), not inlined.

    ``domain_skill_index`` (name -> one-line description) is the target repo's domain
    skill catalogue (``.grindstone/skills/index.md``); when non-empty it renders an
    ``<domain_skills>`` selection block in the volatile tail so the planner can attach
    relevant skills to a task by name. Empty / ``None`` (most repos) renders nothing."""

    scenario = select_planner_scenario(
        skeleton_exists=skeleton is not None,
        failed_epoch_active=failed_epoch is not None,
    )
    return (
        stable_head(job, skeleton, repo_memory)
        + _scenario_block(scenario)
        + volatile_tail(
            phase_id=phase_id,
            epoch_counter=epoch_counter,
            log_index=log_index,
            last_epoch_rows=last_epoch_rows,
            reask_errors=reask_errors,
            phase=phase,
            failed_epoch=failed_epoch,
            workspace=workspace,
            domain_skills=domain_skill_index,
        )
    )


# --- decision validation pipeline (gate → typed → semantic) --------------------


@dataclass(frozen=True)
class GateResult:
    """Outcome of validating one raw decision: a typed decision XOR errors."""

    decision: EpochDecision | None
    errors: list[str]


def _shorthand_tool(payload: dict[str, object]) -> str | None:
    """The tool name of a ``{tool: args}`` shorthand object, else ``None``."""

    tool_keys = [k for k in payload if k in TOOL_NAMES]
    if len(tool_keys) == 1 and isinstance(payload[tool_keys[0]], dict):
        return tool_keys[0]
    return None


def is_decision_like(obj: object) -> bool:
    """Heuristic for picking the decision object out of a noisy final message.

    True for the canonical ``{tool, args}`` envelope AND for the model's common
    function-call shorthand ``{<tool>: {...}}`` (one tool-named key, object
    value). Used by the extractor to prefer the real decision over example/
    reasoning objects.
    """

    return isinstance(obj, dict) and ("tool" in obj or _shorthand_tool(obj) is not None)


# --- tolerant JSON extraction (core parsing, ruling 1) ------------------------


def _balanced_object_spans(text: str) -> list[tuple[int, int]]:
    """Yield (start, end) of every TOP-LEVEL ``{...}`` region, string-aware.

    Brace counting that ignores braces inside JSON string literals (honouring
    backslash escapes), so a model that emits reasoning then a fenced object,
    or prose with braces in quotes, still yields the real object spans. Nested
    objects are part of their enclosing top-level span, not separate entries.
    """

    spans: list[tuple[int, int]] = []
    i, n = 0, len(text)
    while i < n:
        if text[i] != "{":
            i += 1
            continue
        depth, in_str, esc, j = 0, False, False, i
        while j < n:
            c = text[j]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            elif c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    spans.append((i, j + 1))
                    break
            j += 1
        i = j + 1
    return spans


def extract_decision_json(text: str) -> str | None:
    """Return the source text of the decision JSON object, or ``None``.

    Strategy: scan every balanced top-level object, keep those that parse as a
    JSON object, and prefer the LAST one carrying a ``"tool"`` key (codex puts
    the final answer last, after any reasoning). Falls back to the last parsing
    object when none is decision-shaped, and to ``None`` when nothing parses.
    Returns the substring (the core does the authoritative ``json.loads``).
    """

    decision_like: list[str] = []
    any_object: list[str] = []
    for start, end in _balanced_object_spans(text):
        sub = text[start:end]
        try:
            obj = json.loads(sub)
        except ValueError:
            continue
        if isinstance(obj, dict):
            any_object.append(sub)
            if is_decision_like(obj):
                decision_like.append(sub)
    if decision_like:
        return decision_like[-1]
    if any_object:
        return any_object[-1]
    return None


def normalize_tool_call(payload: object) -> object:
    """Canonicalize tolerated tool-call encodings to ``{schema_version,tool,args}``.

    Codex (and most function-calling models) natively emit either the canonical
    envelope, sometimes without ``schema_version``, or the shorthand
    ``{<tool>: <args>}``. Both are unambiguous dispatch-on-tool-name encodings;
    canonicalizing them here (structurally, not via fragile prompt wording) lets
    the authoritative schema stay the single source of truth. ``schema_version``
    defaults to ``"1"`` (the only version in v1) when absent; nothing else is
    inferred, a genuinely malformed object passes through and is rejected.
    """

    if not isinstance(payload, dict):
        return payload
    if "tool" in payload and "args" in payload:
        if "schema_version" not in payload:
            return {"schema_version": "1", **payload}
        return payload
    tool = _shorthand_tool(payload)
    if tool is not None:
        return {"schema_version": "1", "tool": tool, "args": payload[tool]}
    return payload


#: Tools that resolve a phase-escalation demand (S4 ruling 2): re-scope the
#: current phase, or hand the whole run to a human.
_ESCALATION_TOOLS: frozenset[str] = frozenset({"revise_phases", "escalate_run"})


def _position_legality(
    decision: EpochDecision,
    skeleton_exists: bool,
    *,
    phase_escalated: bool,
    failed_epoch_active: bool,
) -> list[str]:
    """Call-position rules the schema cannot express (PLANNER_CONTRACT §3).

    ``propose_skeleton`` is legal ONLY on the first call (no skeleton yet); every
    other tool is legal ONLY once a skeleton exists. When the current phase is
    under an escalation demand (budget exhausted, S4 ruling 2), the ONLY legal
    tools are ``revise_phases`` / ``escalate_run``, any work epoch is rejected
    back to the planner via the same re-ask ladder.

    ``handle_failed_epoch`` is the FOCUSED disposition of an epoch that FAILED:
    it is legal ONLY when a failed epoch is awaiting disposition, and conversely
    when one IS awaiting disposition it is the ONLY legal tool (the planner must
    decide retry / escalate_senior / halt for THIS epoch, not fan out a fresh
    one or blindly revise the phase structure, the dogfood spin-loop fix).

    The failed-epoch disposition TAKES PRECEDENCE over budget phase-escalation:
    when a phase's epoch_budget is exhausted BY a failed epoch, both demands fire
    at the same boundary, but ``handle_failed_epoch`` already covers "what next"
    for that epoch (retry / escalate_senior / halt) more specifically than
    revise_phases / escalate_run. So while a failed epoch is pending, ONLY the
    handle_failed_epoch rule applies and the phase-escalation rule is skipped;
    otherwise the two would deadlock (handle_failed_epoch rejected for not being
    an escalation tool, every other tool rejected for not being handle_failed_epoch).
    The budget-escalation path stays live for its genuine case: a phase whose gate
    never passes while its epochs all complete structurally (no task FAILS), which
    exhausts the budget with NO failed epoch pending.
    """

    is_propose = decision.tool == "propose_skeleton"
    if is_propose and skeleton_exists:
        return ["propose_skeleton is only legal as the first decision of a run"]
    if not is_propose and not skeleton_exists:
        return ["the first decision of a run must be propose_skeleton"]
    if failed_epoch_active:
        if decision.tool != "handle_failed_epoch":
            return [
                f"a failed epoch is awaiting disposition: only handle_failed_epoch is "
                f"legal (retry / escalate_senior / halt), not {decision.tool}"
            ]
        return []
    if decision.tool == "handle_failed_epoch":
        return [
            "handle_failed_epoch is only legal when an epoch has failed and is "
            "awaiting disposition"
        ]
    if phase_escalated and decision.tool not in _ESCALATION_TOOLS:
        return [
            f"phase escalation in force: only revise_phases or escalate_run are "
            f"legal, not {decision.tool}"
        ]
    return []


def _size_gate_violations(
    decision: EpochDecision,
    *,
    failed_epoch_active: bool,
    has_senior: bool,
    local_max_task_files: int,
    senior_max_task_files: int,
) -> list[str]:
    """The deterministic per-task SIZE gate for a FRESH implement decomposition.

    Scoped to fresh decomposition: a ``handle_failed_epoch`` repair re-dispatches
    its originating decision directly (never through this gate) and may need broad
    scope, and while a failed epoch is awaiting disposition the ONLY legal tool is
    ``handle_failed_epoch`` anyway, so ``failed_epoch_active`` short-circuits the
    gate. TIER-AWARE PER TASK: a ``task.senior`` task runs on the senior tier (when
    the rig has one), so it gets the larger senior bound; every other task (and any
    task when the rig has NO senior tier, since it then falls back to local) gets the
    local bound.
    """

    if failed_epoch_active or not isinstance(decision, ImplementDecision):
        return []
    # No senior tier in the ladder -> a senior task falls back to local, so it is
    # bounded by the local cap; pass that as the senior bound in that case.
    senior_bound = senior_max_task_files if has_senior else local_max_task_files
    return implement_task_size_violations(
        list(decision.args.tasks),
        max_files=local_max_task_files,
        senior_max_files=senior_bound,
    )


def _content_grep_violations(decision: EpochDecision) -> list[str]:
    """Reject a task ``done_when`` check that is a content-grep (gate rebalance).

    Mirrors ``_size_gate_violations`` as a deterministic per-task gate, but with a
    DIFFERENT scope: a content-grep is forbidden on every task-carrying decision
    (implement / research / review / artifact) and is NOT exempted on the
    failed-epoch-repair path, a content-grep check is brittle regardless of who
    authored it or when. The rejection steers the planner to express that
    acceptance as natural-language ``criteria`` instead of a shell command.
    """

    args = decision.args
    if isinstance(args, (ImplementEpochArgs, ArtifactEpochArgs)):
        return content_grep_check_violations(list(args.tasks))
    return []


def validate_decision(
    json_text: str | None,
    *,
    existing_log_keys: frozenset[str],
    completed_phase_ids: frozenset[str],
    skeleton_exists: bool,
    phase_escalated: bool = False,
    failed_epoch_active: bool = False,
    has_senior: bool = False,
    local_max_task_files: int = DEFAULT_LOCAL_MAX_TASK_FILES,
    senior_max_task_files: int = DEFAULT_SENIOR_MAX_TASK_FILES,
    known_skill_names: frozenset[str] = frozenset(),
) -> GateResult:
    """Gate raw decision text: extract-input → JSON → schema → typed → semantic.

    ``json_text`` is the candidate JSON the extractor pulled from the transport's
    raw output (``None`` when nothing extractable was found). Returns a typed
    decision on success, else the ordered list of human-readable rejection
    reasons that get appended to the re-ask.

    The size gate (``_size_gate_violations``) additionally REJECTS a fresh
    implement decision whose tasks are not decomposed: a task over its tier's
    file-count bound, or one claiming whole-repo ownership. The content-grep gate
    (``_content_grep_violations``) REJECTS a task ``done_when`` check built on a
    content-grep (``rg`` / ``grep`` for a token), steering the planner to express
    that acceptance as natural-language ``criteria``. A rejection is
    indistinguishable from any other gate failure to the caller, so it rides the
    SAME invalid-decision re-ask ladder (the re-ask names the offending task).
    """

    if json_text is None:
        return GateResult(None, ["planner output contained no extractable decision JSON"])
    try:
        payload = normalize_tool_call(json.loads(json_text))
    except ValueError as exc:
        return GateResult(None, [f"decision is not valid JSON: {exc}"])
    schema_errors = decision_schema_errors(payload)
    if schema_errors:
        return GateResult(None, [f"schema: {m}" for m in schema_errors[:6]])
    try:
        decision = parse_decision(payload)
    except ValueError as exc:
        return GateResult(None, [f"typed parse: {exc}"])
    errors = list(
        epoch_decision_violations(
            decision,
            existing_log_keys=existing_log_keys,
            completed_phase_ids=completed_phase_ids,
            known_skill_names=known_skill_names,
        )
    )
    errors += _position_legality(
        decision,
        skeleton_exists,
        phase_escalated=phase_escalated,
        failed_epoch_active=failed_epoch_active,
    )
    errors += _size_gate_violations(
        decision,
        failed_epoch_active=failed_epoch_active,
        has_senior=has_senior,
        local_max_task_files=local_max_task_files,
        senior_max_task_files=senior_max_task_files,
    )
    errors += _content_grep_violations(decision)
    if errors:
        return GateResult(None, errors)
    return GateResult(decision, [])
