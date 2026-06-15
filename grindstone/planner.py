"""Planner core: transport interface, failure classification, pure input
construction, and the decision validation pipeline (ARCHITECTURE.md, PLANNER_CONTRACT).

The planner is a stateless one-shot call. The deterministic core owns everything
the model does not: it CONSTRUCTS the input (stable head + volatile tail),
CLASSIFIES transport failures, and GATES the raw output through schema → typed →
semantic validation before dispatch. The transport only returns raw text.

Journal mapping for an INVALID decision (ruling 8): the vocabulary has no
"invalid decision" event. A decision that fails the gate is journaled by the
*caller* as ``planner_call_succeeded`` NOT emitted + ``planner_call_failed`` with
classification ``"transient"`` — it is a retryable bad output, re-asked up to
twice before the run escalates. (Transport-level failures are journaled with
their ``classify_failure`` result; this module owns that classification.)

Input construction is a PURE function over durable state (ruling 3):

  - **Stable head** — byte-identical across every call of a run: a fixed system
    preamble, an (empty at S3) ``<skills>`` digest, an (empty at S3, S5 seam)
    ``<repo_memory>`` digest, and the stored ``<skeleton>`` as compact JSON. The
    head changes ONLY when ``propose_skeleton`` / ``revise_phases`` change the
    skeleton — then, and only then, the prefix cache legitimately resets.
  - **Volatile tail** — ``<state>`` (phase id, epoch counter, keyed-log index),
    ``<last_epoch>`` (the prior outcome flattened: per-task status/attempts/tier,
    each DONE task's handoff resulting_state + downstream_needs, each FAILED
    task's last failure reason), optional ``<errors>`` (re-ask feedback), and
    ``<request>``. References, not payloads: only log keys, never file bodies.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal, Protocol

from grindstone.contracts.gate import decision_schema_errors
from grindstone.contracts.models import EpochDecision, Phase, parse_decision, parse_handoff
from grindstone.contracts.semantics import epoch_decision_violations
from grindstone.epoch_loop import EpochOutcome
from grindstone.rundir import RunDir

# The transport exception family is the workers' (ruling 1) — re-exported here so
# planner code imports one surface. ``PlannerHardError`` is the planner-only
# addition for auth/config: it is deliberately NOT a ``TransportError`` so
# ``classify_failure`` reads it as ``hard`` (the default branch).
from grindstone.worker import RateLimited, TransportError, WorkerTimeout

__all__ = [
    "BACKOFF_BASE_S",
    "BACKOFF_CAP_S",
    "BACKOFF_FACTOR",
    "FailureClass",
    "GateResult",
    "MAX_RATE_LIMIT_WAITS",
    "MAX_REASKS",
    "MAX_TRANSIENT_RETRIES",
    "PhaseTailInfo",
    "PlannerHardError",
    "PlannerTransport",
    "RateLimited",
    "TOOL_NAMES",
    "TransportError",
    "WorkerTimeout",
    "backoff_delay",
    "build_planner_input",
    "classify_failure",
    "extract_decision_json",
    "flatten_last_epoch",
    "is_decision_like",
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
        "escalate_run",
        "complete_run",
    }
)


class PlannerHardError(Exception):
    """A planner failure that needs a human: auth, config, unknown model error.

    Outside the transport exception family on purpose — ``classify_failure``
    returns ``hard`` for it (and for any other unrecognized exception).
    """


# --- transport interface -------------------------------------------------------


class PlannerTransport(Protocol):
    """One-shot planner: constructed input in, raw final-message text out.

    NEVER parses or validates (ruling 1). Raises the transport exception family
    on failure: ``RateLimited`` (→ backoff), ``TransportError`` / ``WorkerTimeout``
    (→ transient retry), ``PlannerHardError`` or anything else (→ human).
    """

    def plan(self, prompt: str) -> str: ...


# --- failure classification + backoff (ruling 2) -------------------------------

FailureClass = Literal["rate_limit", "transient", "hard"]

#: Exponential backoff for rate-limit waits: 30s, 60s, 120s, 240s, 480s, 600s…
BACKOFF_BASE_S = 30.0
BACKOFF_FACTOR = 2.0
BACKOFF_CAP_S = 600.0
MAX_RATE_LIMIT_WAITS = 6
MAX_TRANSIENT_RETRIES = 3
MAX_REASKS = 2


def classify_failure(exc: BaseException) -> FailureClass:
    """Three-way classification of a planner-call failure (ARCHITECTURE.md).

    ``RateLimited`` → rate_limit (wait for the window — never auto-spill to a
    different planner). ``TransportError`` / ``WorkerTimeout`` → transient (retry
    the same call). Anything else (auth/config/unknown) → hard (escalate).
    """

    if isinstance(exc, RateLimited):
        return "rate_limit"
    if isinstance(exc, (TransportError, WorkerTimeout)):
        return "transient"
    return "hard"


def backoff_delay(wait_index: int) -> float:
    """Backoff seconds for the ``wait_index``-th (0-based) rate-limit wait."""

    return min(BACKOFF_BASE_S * (BACKOFF_FACTOR**wait_index), BACKOFF_CAP_S)


# --- input construction: stable head (byte-identical across a run) -------------

SYSTEM_PREAMBLE = """\
You are the planner for Grindstone, a deterministic epoch-based orchestrator.
You are a stateless one-shot call. The fixed state machine runs the loop, runs
all checks, and integrates work; you only decide the NEXT step.

Rules:
- Emit EXACTLY ONE tool call per turn, as a single JSON object matching the
  epoch_decision schema. No prose, no second object, no commentary.
- The first decision of a run MUST be propose_skeleton. After that, every call
  is an epoch boundary: choose one of implement / research / review / artifact
  (1-8 independent fan-out tasks), revise_phases, escalate_run, or complete_run.
- You define exit criteria and done_when as DETERMINISTIC checks (a command with
  an expected exit code, or a required artifact log key). The state machine
  evaluates them; you never claim a phase or task is done.
- References, not payloads. A task's `inputs` are log keys that already EXIST in
  the keyed-log index below; never invent a key. Resulting artifacts are named
  by their log key, never inlined. An accepted artifact task's `artifact_out`
  file is published to the keyed log, and an `artifact_exists` check may name
  it either by its exact log key or by its bare filename — use the bare
  filename in phase exit criteria (the P*/E*/T*/ placement is decided epochs
  later); it passes only while exactly one logged artifact carries that name.
- implement tasks carry `file_ownership` globs that must be pairwise DISJOINT
  across the epoch (the merge-correctness mechanism). research/review/artifact
  tasks carry `artifact_out`; review tasks also carry `targets`.
- Choose the mode by the deliverable's DESTINATION, never by its flavor.
  Output the job requires as a COMMITTED file in the repo tree — code, config,
  docs, even prose — is implement work: only implement tasks run in a worktree
  and get committed. Output consumed via the keyed log (an analysis, report,
  or investigation the job does NOT require as a committed file) is research
  or artifact work shipped through `artifact_out`; review judges existing work
  and ships a verdict the same way. Never give a task a worktree its
  deliverable does not need.
- Sequence by tier of thinking. research/review (and visual epochs) run on the
  stronger SENIOR tier; implement/artifact run on the local rig — so a skeleton
  is also a routing choice: put judgment on senior, production on local. For
  heavy or judgment-laden work, SPLIT into phases rather than cramming it into a
  single local epoch, and feed each step forward through the keyed log (a
  non-implement epoch's `artifact_out` becomes a later epoch's `inputs`). Good
  shapes (nudges, not a fixed menu): heavy build = research -> implement ->
  review; report / triage / migration = research -> artifact (do NOT collapse the
  analysis into one local artifact epoch — that downgrades it off senior); UI =
  research -> implement with `visual:true` -> a phase exit criterion that builds,
  screenshots, then `vision_review`s it. A small job can be a single epoch.
- Taste routing: set `"visual": true` on an implement or review epoch whose
  deliverable is FRONT-END / UI / visual / polish output (layout, styling, a
  rendered page, a diagram, anything judged by how it LOOKS). That epoch is
  built by the stronger taste-building senior tier instead of the local default
  (the senior is a text model; the actual image judgment is the vision_review
  gate below). Omit it (defaults false) for non-visual work — backend, logic,
  plain text/config.
- Vision-review (taste gate): a third check `{"vision_review":{"screenshot":
  "<path relative to the eval worktree>","criteria":"<what polished looks
  like>"}}` makes a strong vision model JUDGE a rendered screenshot against
  criteria and emit a pass/fail verdict. Use it ONLY in a PHASE EXIT CRITERION
  for a visual phase: put a cmd check FIRST that builds + screenshots the UI
  into the tip worktree (e.g. `{"cmd":"npm run build && node shot.js
  ui/screen.png"}`), then a `vision_review` of that `ui/screen.png` against the
  design bar. The state machine renders the verdict deterministically (a failed
  taste verdict fails the phase, just like a failed command) — it is not a task
  `done_when` (a worker scratch has no renderer/screenshot).
- done_when is scoped by mode. research/review/artifact tasks run in a scratch
  dir that is NOT a repo checkout: their done_when must verify the
  artifact itself (e.g. `test -s notes.md` in the task CWD, or an
  artifact_exists key) — never repo build/test commands; those can only pass in
  implement tasks or phase exit criteria (run in a checkout of the tip).
- escalate_run only when you genuinely cannot proceed. complete_run only when
  the whole job is done; its `evidence` checks are re-run deterministically and
  rejected if they fail.
- A skeleton has BETWEEN 2 AND 10 phases. Even a small job needs at least two
  (e.g. a build phase then a verify phase); phase ids are "P1","P2",… in order.
  Each epoch has 1-8 tasks with ids "T1"…"T8".

Task sizing, independence, and decomposition:
- Tasks within an epoch run in PARALLEL and MUST NOT consume each other's
  outputs. Anything where one task needs another's result is SEQUENTIAL work —
  put it in a later epoch (or phase), never in a sibling task of the same epoch.
- Each task must fit ONE worker with a ~90k-token working context. Treat 90k as
  the sizing CONTRACT: the worker has headroom above it, but that headroom is
  overrun insurance, never plannable budget. If a task cannot plausibly fit, it
  is two tasks or two epochs.
- `epoch_budget` is how many epochs a phase may consume before the state machine
  fires a phase escalation (forcing you to revise_phases or escalate_run). It is
  a ceiling sized to the phase's real arc — a small phase is 1-2, a broad build
  phase a few more — not a target; unused budget is free.
- Decompose CONSERVATIVELY, at the top level only. You are a powerful planner:
  prefer ONE task whenever the work is even remotely interconnected and shared
  context helps. Split into multiple tasks ONLY when the parts are genuinely
  independent, or genuinely too big for one worker's context. Naive fan-out of
  intertangled work hands the hardest part — cross-file consistency — to the
  least-coordinated agents.
- Carry the relevant job-spec requirements into each task's `goal` VERBATIM, or
  point at the exact input artifacts (by log key) that contain them. Never
  paraphrase or summarize a requirement away — lossy paraphrase silently drops
  requirements. `goal` is capped at 1024 chars: quote exactly what fits and move
  the rest into referenced input artifacts; never compress a requirement into a
  summary.

Output format — emit EXACTLY ONE JSON object with this envelope, nothing else:
  {"schema_version":"1","tool":"<one tool name>","args":{ ... }}
The keys schema_version, tool, args are ALL mandatory. Do NOT use the shorthand
{"<tool>":{...}} and do NOT omit schema_version.

`args` by tool (the schema is authoritative):
- propose_skeleton: {"phases":[{"id":"P1..","title":..,"exit_criterion":[check..],"epoch_budget":int}]}
- implement:        {"epoch_title":..,"rationale":..,"visual"?:bool,"tasks":[{"id":"T1..","goal":..,"done_when":[check..],"file_ownership":[glob..],"inputs"?,"skills"?}]}
- research/review/artifact: {"epoch_title":..,"rationale":..,"visual"?:bool,"tasks":[{"id","goal","done_when","artifact_out","targets"?,"inputs"?,"skills"?}]}
- revise_phases:    {"reason":..,"phases":[..]}  (replaces the current phase onward; never completed phases)
- escalate_run:     {"reason":..,"needed_from_human"?}
- complete_run:     {"summary":..,"evidence":[check..]}
A check is {"cmd":..,"expect_exit"?:int} or {"artifact_exists":"<log key>"} or
{"vision_review":{"screenshot":"<eval-worktree-relative path>","criteria":..}}
(taste gate — phase exit criteria only, after a cmd check renders the shot).

Example first decision (note: TWO phases minimum):
  {"schema_version":"1","tool":"propose_skeleton","args":{"phases":[
    {"id":"P1","title":"Build","exit_criterion":[{"cmd":"test -f out.txt","expect_exit":0}],"epoch_budget":2},
    {"id":"P2","title":"Verify","exit_criterion":[{"cmd":"grep -q DONE out.txt","expect_exit":0}],"epoch_budget":1}]}}

Example implement decision (two GENUINELY INDEPENDENT files, so two tasks with
pairwise-DISJOINT file_ownership; each done_when is machine-checkable; each goal
quotes the spec VERBATIM):
  {"schema_version":"1","tool":"implement","args":{"epoch_title":"Write greeting and version files","rationale":"two independent files, no shared state","tasks":[
    {"id":"T1","goal":"Create greeting.txt. Spec verbatim: 'greeting.txt MUST contain exactly the line HELLO'.","done_when":[{"cmd":"grep -qx HELLO greeting.txt"}],"file_ownership":["greeting.txt"]},
    {"id":"T2","goal":"Create version.txt. Spec verbatim: 'version.txt MUST contain exactly the line 1.0.0'.","done_when":[{"cmd":"grep -qx 1.0.0 version.txt"}],"file_ownership":["version.txt"]}]}}
"""


def _compact(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def stable_head(job: str, skeleton: list[Phase] | None, repo_memory: str | None = None) -> str:
    """The byte-identical prefix of every planner call in a run (ruling 3).

    Depends ONLY on the (frozen-at-run-start) job spec, the skeleton, and the
    repo-memory digest (S5 seam: the run-loop freezes ``load_digest`` at run
    start so byte-identity holds run-long; ``skills`` stays empty). Identical
    bytes while those are unchanged; changes exactly when ``propose_skeleton`` /
    ``revise_phases`` move the skeleton. The ``<job>`` slot is the spec the
    planner plans against — frozen at run start, hence part of the head (an
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
        f"<system>\n{SYSTEM_PREAMBLE}</system>\n"
        f"<job>\n{job}\n</job>\n"
        "<skills>\n</skills>\n"
        f"{memory_slot}"
        f"<skeleton>\n{skeleton_json}\n</skeleton>\n"
    )


# --- input construction: volatile tail -----------------------------------------


def flatten_last_epoch(run_dir: RunDir, outcome: EpochOutcome) -> list[dict[str, object]]:
    """Flatten an EpochOutcome into compact per-task rows for ``<last_epoch>``.

    DONE tasks contribute their handoff ``resulting_state`` + ``downstream_needs``
    (read from the keyed log — references, never bodies); FAILED tasks contribute
    their last failure reason. Pure over durable state (the relocated handoffs).
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
    whether the current phase is under an escalation demand (budget exhausted —
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
        f"current_phase: {phase_id or 'none'} — {phase.title}\n"
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
) -> str:
    """The per-call suffix: running state, phase status, last-epoch report,
    re-ask feedback, request. Never byte-stable — it carries everything that
    moves (S4 adds the phase-status + integration-tip surfacing, ruling 3)."""

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
    if last_epoch_rows is None:
        last = "<last_epoch>\n  (none — this is the first decision)\n</last_epoch>\n"
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
    return state + phase_status + last + errors + request


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
) -> str:
    """Full constructed input: ``stable_head`` + ``volatile_tail`` (ruling 3)."""

    return stable_head(job, skeleton, repo_memory) + volatile_tail(
        phase_id=phase_id,
        epoch_counter=epoch_counter,
        log_index=log_index,
        last_epoch_rows=last_epoch_rows,
        reask_errors=reask_errors,
        phase=phase,
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


# --- tolerant JSON extraction (core parsing — ruling 1) ------------------------


def _balanced_object_spans(text: str) -> list[tuple[int, int]]:
    """Yield (start, end) of every TOP-LEVEL ``{...}`` region, string-aware.

    Brace counting that ignores braces inside JSON string literals (honouring
    backslash escapes), so a model that emits reasoning then a fenced object —
    or prose with braces in quotes — still yields the real object spans. Nested
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
    envelope — sometimes without ``schema_version`` — or the shorthand
    ``{<tool>: <args>}``. Both are unambiguous dispatch-on-tool-name encodings;
    canonicalizing them here (structurally, not via fragile prompt wording) lets
    the authoritative schema stay the single source of truth. ``schema_version``
    defaults to ``"1"`` (the only version in v1) when absent; nothing else is
    inferred — a genuinely malformed object passes through and is rejected.
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
    decision: EpochDecision, skeleton_exists: bool, *, phase_escalated: bool
) -> list[str]:
    """Call-position rules the schema cannot express (PLANNER_CONTRACT §3).

    ``propose_skeleton`` is legal ONLY on the first call (no skeleton yet); every
    other tool is legal ONLY once a skeleton exists. When the current phase is
    under an escalation demand (budget exhausted, S4 ruling 2), the ONLY legal
    tools are ``revise_phases`` / ``escalate_run`` — any work epoch is rejected
    back to the planner via the same re-ask ladder.
    """

    is_propose = decision.tool == "propose_skeleton"
    if is_propose and skeleton_exists:
        return ["propose_skeleton is only legal as the first decision of a run"]
    if not is_propose and not skeleton_exists:
        return ["the first decision of a run must be propose_skeleton"]
    if phase_escalated and decision.tool not in _ESCALATION_TOOLS:
        return [
            f"phase escalation in force: only revise_phases or escalate_run are "
            f"legal, not {decision.tool}"
        ]
    return []


def validate_decision(
    json_text: str | None,
    *,
    existing_log_keys: frozenset[str],
    completed_phase_ids: frozenset[str],
    skeleton_exists: bool,
    phase_escalated: bool = False,
) -> GateResult:
    """Gate raw decision text: extract-input → JSON → schema → typed → semantic.

    ``json_text`` is the candidate JSON the extractor pulled from the transport's
    raw output (``None`` when nothing extractable was found). Returns a typed
    decision on success, else the ordered list of human-readable rejection
    reasons that get appended to the re-ask.
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
        )
    )
    errors += _position_legality(decision, skeleton_exists, phase_escalated=phase_escalated)
    if errors:
        return GateResult(None, errors)
    return GateResult(decision, [])
