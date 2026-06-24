"""The per-task EXECUTION UNIT (``run_task``) + the lenient CRITIC.

BONES: one task is run to a triage verdict, reusable and safe to call
concurrently. ``run_task`` isolates the work in a throwaway worktree off an
EXTERNAL base (so a worker that strips its CWD to the repo root cannot reach the
operator checkout), dispatches the tier's rig behind the uniform ``WorkerTransport``
seam, collects the handoff from disk (the disk file is the gate, stdout is never
parsed), then either routes a self-reported BLOCKED straight to the planner or runs
a tier-matched CRITIC that TRIAGES the work (PASS | RETRY | ESCALATE). A bounded
same-tier retry (no tier escalation) absorbs the cheap self-heals; anything else
becomes context the planner sees next boundary.

The control flow, per BONES failure model (two nodes, everything routes to #2):

* worker RATE-LIMITED -> the exception escapes (NOT a burned attempt): the epoch
  loop parks and re-runs (#1).
* handoff INVALID / FAILED / PARTIAL -> a failed attempt: bounded same-tier retry,
  the retry INHERITS the prior attempt's stripped wip; exhausted -> escalate (#2).
* handoff BLOCKED -> skip the critic, surface to the planner (#2, honest blocker).
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
from grindstone.check_handoff import (
    CHECK_COMMAND,
    CHECK_SCRIPT_NAME,
    generate_check_script,
)
from grindstone.contracts.models import (
    HANDOFF_MAX_BYTES,
    Handoff,
    HandoffMode,
    Task,
    Verdict,
    parse_handoff,
    parse_verdict,
)
from grindstone.domain_skills import load_domain_skill
from grindstone.rundir import RunDir

#: The implement-mode self-review artifact a worker writes in its CWD before the
#: handoff (a quality step; the independent CRITIC is the real gate). Stripped
#: before commit like every other orchestration file. (Verified mechanism: a
#: review demanded as a gated artifact fires; prose instructions do not.)
REVIEW_FILENAME = "review.md"

#: The critic's only result channel: a ``verdict.json`` written in the critic's CWD
#: (a re-read disk contract, like the handoff). Relocated into the run dir and
#: re-validated with Pydantic; stdout is never parsed.
CRITIC_VERDICT_FILENAME = "verdict.json"

#: Bounded SAME-tier retry budget (BONES: keep a tiny 1-2x local retry that absorbs
#: a free self-heal; NO tier escalation). A second RETRY exhausts to escalation.
MAX_ATTEMPTS = 2

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
    not its own taste) and a pointer to the produced work: ``diff_base`` is the base
    ref an implement critic diffs HEAD against (it runs in the post-commit worktree),
    and ``artifact_path`` is the CWD-relative deliverable a non-write critic reads.
    """

    goal: str
    mode: HandoffMode
    diff_base: str | None = None
    artifact_path: str | None = None


@dataclass(frozen=True)
class WorkerRequest:
    """One fully-resolved dispatch: the typed ``task``, its full keyed-log
    ``task_id`` (``P*/E*/T*``), the worker ``mode``, the scratch CWD it writes its
    output into, the resolved ``inputs`` (prior keyed-log artifacts), the SELECTED
    domain skills (retrieve-not-concatenate), prior-attempt ``failure_context``, and
    whether a prior attempt's work is already present (incremental retry).

    When ``critic`` is set the dispatch is the CRITIC pass: the prompt builder
    branches to ``build_critic_prompt`` and the result channel is ``verdict.json``,
    not a handoff.
    """

    task: Task
    task_id: str
    mode: HandoffMode
    scratch: Path
    inputs: dict[str, Path] = field(default_factory=dict)
    domain_skills: dict[str, str] = field(default_factory=dict)
    failure_context: tuple[str, ...] = ()
    prior_work_present: bool = False
    critic: CriticBrief | None = None


class WorkerTransport(Protocol):
    """The uniform dispatch boundary the per-task unit runs through (worker AND
    critic). ``run`` does its work in ``request.scratch`` and writes its disk
    artifact there; it returns nothing and raises to signal a transport failure."""

    def run(self, request: WorkerRequest) -> None: ...


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


# --- result type ---------------------------------------------------------------


@dataclass(frozen=True)
class TaskResult:
    """The terminal of one task. ``outcome`` routes the epoch loop:

    * ``passed``: merge-ready. For an implement task ``branch`` (+ ``head``) is the
      wip the loop disjoint-merges into the run branch; for a non-write task
      ``artifact_key`` is the deliverable already relocated into the run dir.
    * ``blocked``: a worker-reported environment blocker -> the planner (critic
      skipped). ``handoff`` carries the BLOCKED self-report; ``reason`` summarizes.
    * ``escalated``: a critic ESCALATE or an exhausted retry ladder -> the planner.
      ``reason`` is the surfaced context.
    """

    task_id: str
    outcome: Literal["passed", "blocked", "escalated"]
    attempts: int
    handoff: Handoff | None = None
    verdict: Verdict | None = None
    branch: str | None = None
    head: str | None = None
    artifact_key: str | None = None
    reason: str = ""


# --- internal attempt-failure signal -------------------------------------------


class _Rejected(Exception):
    """One attempt was rejected (invalid/missing handoff, a non-DONE status, or an
    out-of-scope write). ``chainable`` is False for a poisoned worktree (an
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
You run inside an ISOLATED, throwaway git worktree that is your current working
directory and IS the repository for this task. Create and edit every file with
paths RELATIVE to your CWD; never write to an absolute path and never write
outside your CWD. There is no other repository you may touch, do not go looking
for one, this worktree is it. The orchestrator inspects ONLY this worktree to
gate and integrate your work, so anything you write elsewhere is invisible,
discarded, and corrupts the run.
</worktree>"""

#: Concise per-mode guidance (BONES drops the operating-skill scenario split; the
#: dynamic per-task lanes are appended in code). Each is the worker's plan for ONE
#: intent. The executor is ONE role: a senior-tier task reuses the same guidance
#: (the tier only selects which rig SCRIPT runs it).
_MODE_GUIDANCE: dict[HandoffMode, str] = {
    "implement": (
        "Make the change inside your file_ownership. Run whatever checks you write "
        "to convince yourself it works, then self-review your diff and record a "
        f"short note in `{REVIEW_FILENAME}` in your CWD before handing off."
    ),
    "research": (
        "Investigate, then write your findings to the artifact log key below. "
        "Ground every claim in a real file (cite file + line); do not speculate."
    ),
    "review": (
        "Re-derive the question yourself and reconcile it against the real files; "
        "do not just check that sections are present. Write your verdict to the "
        "artifact log key below and cite the files you judged."
    ),
    "artifact": (
        "Produce the deliverable at the artifact log key below; ground it in real "
        "files where it makes claims about the repo."
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


def _file_ownership_block(task: Task) -> str:
    globs = "\n".join(f"  - {g}" for g in task.file_ownership)
    return f"""
<file_ownership>
You may create or edit files ONLY within these globs:
{globs}
Changing ANY other file fails the attempt. (`handoff.json`, `{CHECK_SCRIPT_NAME}`
and `{REVIEW_FILENAME}` are orchestration files, write them in the CWD as
instructed; the orchestrator excludes them from this rule.)
</file_ownership>
"""


def build_worker_prompt(request: WorkerRequest) -> str:
    """Construct the worker prompt (pure, no transport, no I/O).

    Skeleton: goal, the worktree-isolation contract, resolved inputs, the per-mode
    guidance, the dynamic per-task lane (implement ownership), selected domain
    skills, prior-failure context, prior-work note, and the handoff disk contract.
    """

    task = request.task
    artifact_line = ""
    if task.mode != "implement" and task.artifact_out is not None:
        artifact_line = f"\nProduce the artifact at log key `{task.artifact_out}`.\n"
    plan = "\n" + _MODE_GUIDANCE[task.mode]
    if task.mode == "implement":
        plan += _file_ownership_block(task)
    prior_work_block = ""
    if request.prior_work_present:
        prior_work_block = (
            "\n<prior_work>\nA PRIOR attempt at this task left its work IN PLACE in "
            "this working directory. Build on it: fix what the prior failures report "
            "and complete what is unfinished rather than starting from scratch; you "
            "MAY reset and redo if you judge the prior approach wrong.\n</prior_work>\n"
        )
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
{artifact_line}
{_WORKTREE_CONTRACT}
<inputs>
{_render_inputs(request)}
</inputs>
{plan}{_domain_skills_block(request)}{prior_work_block}{context_block}
<handoff>
When finished, write a file named exactly `handoff.json` in your current working
directory. It is the ONLY thing the orchestrator reads; stdout is ignored.
  - JSON object, schema_version "1"; task_id MUST be exactly "{request.task_id}".
  - status: "DONE" if you accomplished the goal; "BLOCKED" if a hard environment
    blocker (a missing dependency, a host change you may not make) stops you;
    else "FAILED" / "PARTIAL".
  - resulting_state: one short sentence for the planner (references, not bodies).
  - what_changed: list of {{"kind": "file"|"interface"|"artifact", "ref": <path>}}.
  - downstream_needs / not_done: short strings; fill on FAILED / PARTIAL / BLOCKED.
  - citations: list of {{"file": <path>, "line": <int optional>}} grounding your
    claims in real files (required for research / review). Paths resolve against
    your CWD (or the target repo root for non-implement tasks) and must stay inside.
  - checks: echo any checks you ran as {{"check": <text>, "exit_code": <int>}}.
  - occupancy: {{"compacted": <bool>, "subagent_splits": <int>}}, report honestly.
  - The whole file must serialize under {HANDOFF_MAX_BYTES} bytes (references, not payloads).
After writing handoff.json run `{CHECK_COMMAND}` (it is in your CWD) and fix every
violation it prints until it exits 0.
</handoff>
"""


def build_critic_prompt(request: WorkerRequest, brief: CriticBrief) -> str:
    """The lenient CRITIC prompt (pure, no transport, no I/O).

    Encodes the BONES triage: (1) anchor on the task's OWN claimed goal, strict on
    "did it accomplish what it claimed", lenient on polish/style; (2) the bar is
    "good enough to build on", not perfect, so PASS-with-notes when torn; (3) the
    single retry-vs-escalate question, "can the SAME worker plausibly fix this
    itself?" yes -> RETRY, no -> ESCALATE. Bias to PASS when unsure. The critic
    ROUTES, it does not grade.
    """

    if brief.mode == "implement":
        work_line = (
            f"The work is a diff. Run `git diff {brief.diff_base or 'HEAD~1'}` in "
            "this worktree to see exactly what the worker changed, and read the "
            "changed files."
        )
    else:
        work_line = (
            f"The work is an artifact at `{brief.artifact_path}` (relative to this "
            "directory). Read it in full and judge its actual content, not a summary."
        )
    return f"""<critic id="{request.task_id}">
You are an INDEPENDENT critic. You did NOT write this work. Do NOT edit anything.
Your job is to TRIAGE it into one of three routes, not to grade it.
</critic>

<claimed_goal>
Judge the work against the task's OWN claimed goal below, NOT against your personal
taste. Be strict on "did it accomplish what it claimed"; be lenient on polish and
style. Intermediate red (a reference to something a later task will build) is fine.
{brief.goal}
</claimed_goal>

<the_work>
{work_line}
</the_work>

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


# --- handoff collection + the core gate (the disk file is the gate) ------------


def _read_disk_artifact(path: Path) -> object:
    """Read + JSON-parse a re-read disk artifact, with the DoS guard. Raises on a
    missing / oversized / non-JSON file (the caller maps that to a rejection)."""

    if not path.is_file():
        raise _Rejected(f"no {path.name} written")
    if path.stat().st_size > _DISK_READ_MAX_BYTES:
        raise _Rejected(
            f"{path.name} over the {_DISK_READ_MAX_BYTES}-byte DoS guard; rejecting"
        )
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise _Rejected(f"{path.name} is not valid JSON: {exc}") from exc


#: Modes whose DONE handoff must carry >= 1 citation (mirrors check_handoff).
_CITATION_MODES: tuple[HandoffMode, ...] = ("research", "review")


def _gate_handoff(
    handoff: Handoff,
    *,
    mode: HandoffMode,
    expected_task_id: str,
    scratch: Path,
    repo: Path | None,
) -> list[str]:
    """The core handoff gate beyond the Pydantic schema (BONES minimal checks):
    exact task_id echo, the research/review citation requirement (DONE only), every
    cited file exists inside an allowed root, and the canonical size cap. The
    grounding roots mirror the worker-facing validator: the scratch CWD always, plus
    the target repo for a non-implement task (its scratch is a plain dir)."""

    out: list[str] = []
    if handoff.task_id != expected_task_id:
        out.append(f"task_id {handoff.task_id!r} != dispatched {expected_task_id!r}")
    if handoff.status == "DONE" and mode in _CITATION_MODES and not handoff.citations:
        out.append("research/review handoff requires >= 1 citation")
    roots = [scratch.resolve()]
    if repo is not None and mode != "implement":
        roots.append(repo.resolve())
    for cite in handoff.citations:
        ok = False
        for root in roots:
            c = (root / cite.file).resolve()
            if c.is_file() and c.is_relative_to(root):
                ok = True
                break
        if not ok:
            out.append(f"citation missing or outside allowed roots: {cite.file}")
    canonical = json.dumps(
        handoff.model_dump(exclude_none=True), sort_keys=True, separators=(",", ":")
    ).encode()
    if len(canonical) > HANDOFF_MAX_BYTES:
        out.append(f"handoff exceeds {HANDOFF_MAX_BYTES} bytes: {len(canonical)}")
    return out


def _collect_handoff(
    scratch: Path,
    *,
    run_dir: RunDir,
    expected_task_id: str,
    mode: HandoffMode,
    repo: Path | None,
) -> Handoff:
    """Relocate + fully validate one attempt's handoff; raise ``_Rejected``.

    Read the scratch handoff, relocate parseable bytes to the keyed log, parse with
    Pydantic, then run the core gate. On any failure the relocated record is removed
    (zero dead artifacts). The DISK FILE is the gate; stdout is never parsed."""

    src = scratch / "handoff.json"
    payload = _read_disk_artifact(src)
    dest = run_dir.resolve(f"{expected_task_id}/handoff.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dest)
    try:
        handoff = parse_handoff(payload)
    except ValueError as exc:
        dest.unlink(missing_ok=True)
        raise _Rejected(f"handoff parse failed: {exc}") from exc
    violations = _gate_handoff(
        handoff, mode=mode, expected_task_id=expected_task_id, scratch=scratch, repo=repo
    )
    if violations:
        dest.unlink(missing_ok=True)
        raise _Rejected("; ".join(violations))
    return handoff


# --- orchestration-file hygiene ------------------------------------------------


def _strip_orchestration_files(scratch: Path) -> None:
    """Drop the metadata an implement worker leaves in its scratch (handoff.json,
    the validator, the self-review, any critic verdict, and the worker's per-cwd
    ``.pi/settings.json``) so none of it enters a commit or trips the scope check."""

    for name in ("handoff.json", CHECK_SCRIPT_NAME, REVIEW_FILENAME, CRITIC_VERDICT_FILENAME):
        (scratch / name).unlink(missing_ok=True)
    settings = scratch / ".pi" / "settings.json"
    settings.unlink(missing_ok=True)
    pi_dir = settings.parent
    if pi_dir.is_dir() and not any(pi_dir.iterdir()):
        pi_dir.rmdir()


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


def _install_validator(
    scratch: Path, *, task_id: str, mode: HandoffMode, repo: Path | None
) -> None:
    """Drop the worker-facing ``check_handoff.py`` validator in the attempt CWD so
    the worker can self-validate before handing off. The core re-validates the disk
    file independently. Non-implement tasks bake the repo as a second citation root."""

    (scratch / CHECK_SCRIPT_NAME).write_text(
        generate_check_script(
            task_id=task_id, mode=mode, repo_root=repo if mode != "implement" else None
        ),
        encoding="utf-8",
    )


def _critic_verdict(
    task: Task,
    task_id: str,
    *,
    scratch: Path,
    base: str | None,
    artifact_rel: str | None,
    run_dir: RunDir,
    backends: Backends,
) -> Verdict:
    """Dispatch the tier-matched critic in the task's own scratch, read + relocate
    its ``verdict.json``, return the parsed lenient ``Verdict``. A missing / invalid
    verdict or a transport raise is a ``CriticError`` (fail-safe, never a pass)."""

    brief = CriticBrief(
        goal=task.goal,
        mode=task.mode,
        diff_base=base if task.mode == "implement" else None,
        artifact_path=artifact_rel,
    )
    request = WorkerRequest(
        task=task, task_id=task_id, mode=task.mode, scratch=scratch, critic=brief
    )
    try:
        with backends.slot(task.tier) as transport:
            transport.run(request)
    except RateLimited:
        raise
    except Exception as exc:
        raise CriticError(f"critic transport failed: {type(exc).__name__}: {exc}") from exc
    verdict_file = scratch / CRITIC_VERDICT_FILENAME
    if not verdict_file.is_file():
        raise CriticError(f"critic wrote no {CRITIC_VERDICT_FILENAME}")
    try:
        payload = json.loads(verdict_file.read_text(encoding="utf-8"))
        verdict = parse_verdict(payload)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise CriticError(f"{CRITIC_VERDICT_FILENAME} invalid: {exc}") from exc
    dest = run_dir.resolve(f"{task_id}/{CRITIC_VERDICT_FILENAME}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(verdict_file, dest)
    return verdict


@dataclass
class _AttemptOutput:
    handoff: Handoff
    head: str | None
    artifact_rel: str | None


def _run_attempt(
    task: Task,
    task_id: str,
    *,
    scratch: Path,
    repo: Path | None,
    epoch_base: str | None,
    backends: Backends,
    domain_skills: dict[str, str],
    failure_context: tuple[str, ...],
    prior_work_present: bool,
    run_dir: RunDir,
) -> _AttemptOutput:
    """Dispatch one worker attempt and collect its handoff. For a DONE implement
    task: strip orchestration, commit, scope-check against the EPOCH base, and reject
    a zero-diff DONE. For a DONE non-write task: publish the deliverable to its log
    key. Raises ``_Rejected`` on any failure; lets ``RateLimited`` escape.

    The backend slot is held ONLY around the model dispatch (not the git/disk work),
    so a concurrent same-backend task cannot land a second call on the single local
    slot mid-grind, while cross-backend tasks run free."""

    implement = task.mode == "implement"
    _install_validator(scratch, task_id=task_id, mode=task.mode, repo=repo)
    inputs = {key: run_dir.resolve(key) for key in task.inputs}
    request = WorkerRequest(
        task=task,
        task_id=task_id,
        mode=task.mode,
        scratch=scratch,
        inputs=inputs,
        domain_skills=domain_skills,
        failure_context=failure_context,
        prior_work_present=prior_work_present,
    )
    try:
        with backends.slot(task.tier) as transport:
            transport.run(request)
    except RateLimited:
        raise
    except Exception as exc:
        raise _Rejected(f"transport error: {type(exc).__name__}: {exc}") from exc

    handoff = _collect_handoff(
        scratch, run_dir=run_dir, expected_task_id=task_id, mode=task.mode, repo=repo
    )
    if handoff.status != "DONE":
        # BLOCKED / FAILED / PARTIAL are honest non-DONE statuses returned to the
        # caller (which routes BLOCKED to the planner and retries the others).
        return _AttemptOutput(handoff=handoff, head=None, artifact_rel=None)

    if implement:
        assert repo is not None and epoch_base is not None
        _strip_orchestration_files(scratch)
        wt.commit_all(scratch, f"grind({task_id}): {task.goal.splitlines()[0][:72]}")
        changed = wt.changed_paths(scratch, epoch_base)
        if not changed:
            raise _Rejected(
                "no committed work: implement task handed off DONE but left a "
                "zero-diff branch"
            )
        out_of_scope = wt.scope_violations(changed, list(task.file_ownership))
        if out_of_scope:
            raise _Rejected(
                f"out-of-scope writes: {', '.join(out_of_scope)}", chainable=False
            )
        return _AttemptOutput(handoff=handoff, head=wt.head_commit(scratch), artifact_rel=None)

    # Non-write DONE: publish the produced deliverable to its log key.
    assert task.artifact_out is not None
    produced = scratch / task.artifact_out
    if not produced.is_file():
        run_dir.resolve(f"{task_id}/handoff.json").unlink(missing_ok=True)
        raise _Rejected(f"artifact_out not produced in CWD: {task.artifact_out}")
    published = run_dir.resolve(task.artifact_out)
    published.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(produced, published)
    return _AttemptOutput(handoff=handoff, head=None, artifact_rel=task.artifact_out)


def run_task(
    task: Task,
    task_id: str,
    *,
    run_dir: RunDir,
    repo: Path | None,
    base: str | None,
    backends: Backends,
) -> TaskResult:
    """Run ONE task to a triage verdict (reusable, safe to call concurrently).

    Each attempt grinds in a fresh isolated worktree (implement) or scratch dir
    (non-write) off the EXTERNAL base, dispatches the tier's rig, collects the
    handoff from disk, and routes: BLOCKED -> planner (skip critic); DONE -> the
    tier-matched critic (PASS -> merge-ready, RETRY -> bounded same-tier retry that
    inherits the prior wip, ESCALATE -> planner); INVALID / FAILED / PARTIAL -> a
    failed attempt (bounded retry, then escalate). ``RateLimited`` escapes un-burned.
    """

    implement = task.mode == "implement"
    slug = task_id.replace("/", "-")
    try:
        domain_skills = _load_domain_skills(repo, task)
    except _Rejected as rej:
        return TaskResult(task_id, "escalated", attempts=0, reason=rej.reason)

    failure_context: list[str] = []
    prior_branch: str | None = None  # implement incremental-retry chain base
    attempts = 0
    while attempts < MAX_ATTEMPTS:
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
                task, task_id,
                scratch=scratch, repo=repo, epoch_base=base, backends=backends,
                domain_skills=domain_skills,
                failure_context=tuple(failure_context),
                prior_work_present=implement and prior_branch is not None,
                run_dir=run_dir,
            )
        except _Rejected as rej:
            failure_context.append(rej.reason)
            prior_branch = _carry_or_discard(
                rej, repo=repo, scratch=scratch, branch=branch, implement=implement,
                task_id=task_id,
            )
            continue

        handoff = attempt.handoff
        if handoff.status == "BLOCKED":
            _discard(repo, scratch, branch, implement)
            return TaskResult(
                task_id, "blocked", attempts=attempts, handoff=handoff,
                reason=_blocked_reason(handoff),
            )
        if handoff.status != "DONE":  # FAILED / PARTIAL: a failed attempt
            reason = "; ".join(handoff.not_done) or handoff.resulting_state
            failure_context.append(f"handoff status {handoff.status}: {reason}")
            prior_branch = _carry_partial(repo, scratch, branch, implement, task_id)
            continue

        # DONE -> the tier-matched critic triages it (in the same scratch: the
        # post-commit worktree for implement, the artifact dir for a non-write task).
        try:
            verdict = _critic_verdict(
                task, task_id, scratch=scratch, base=base,
                artifact_rel=attempt.artifact_rel, run_dir=run_dir, backends=backends,
            )
        except CriticError as exc:
            _discard(repo, scratch, branch, implement)
            return TaskResult(
                task_id, "escalated", attempts=attempts, handoff=handoff,
                reason=str(exc),
            )

        if verdict.outcome == "PASS":
            return TaskResult(
                task_id, "passed", attempts=attempts, handoff=handoff, verdict=verdict,
                branch=branch, head=attempt.head, artifact_key=attempt.artifact_rel,
            )
        if verdict.outcome == "ESCALATE":
            _discard(repo, scratch, branch, implement)
            return TaskResult(
                task_id, "escalated", attempts=attempts, handoff=handoff,
                verdict=verdict, reason=verdict.reason or "critic escalated to planner",
            )
        # RETRY: a defect the same worker can fix; chain the committed wip.
        failure_context.append(f"critic RETRY: {verdict.reason}")
        prior_branch = _carry_partial(repo, scratch, branch, implement, task_id)

    reason = failure_context[-1] if failure_context else "no attempts"
    return TaskResult(
        task_id, "escalated", attempts=attempts,
        reason=f"retries exhausted: {reason}",
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
    _strip_orchestration_files(scratch)
    wt.commit_all(scratch, f"grind-wip({task_id}): partial kept for retry")
    wt.remove_worktree(repo, scratch)
    return branch


def _discard(
    repo: Path | None, scratch: Path, branch: str | None, implement: bool
) -> None:
    """Tear down an attempt that leaves nothing to merge (BLOCKED / ESCALATE / a
    poisoned worktree): worktree removed + branch deleted (zero dead artifacts)."""

    if implement and repo is not None and branch is not None:
        wt.discard_attempt(repo, scratch, branch)


def _blocked_reason(handoff: Handoff) -> str:
    return "; ".join(handoff.not_done) or handoff.resulting_state or "worker reported BLOCKED"
