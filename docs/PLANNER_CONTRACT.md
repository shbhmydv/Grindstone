# Grindstone Planner Contract v1

The planner is the only model that steers a run. This document is the reference for
**how Grindstone calls it, what it is allowed to emit, and how the state machine
validates and disposes of every decision**.

The wire format is owned by `schemas/epoch_decision.json`; that file mirrors the
Pydantic types in `grindstone/contracts/models.py`, which are the source of truth.
Where prose here and the model disagree, the model wins.

---

## 1. Call model

The planner is a **stateless one-shot call** (`grindstone/planner.py`,
`ScriptPlanner.decide`). There is no warm planner instance and no planner-held
state: Grindstone reconstructs the full input fresh from durable disk state on
every call, so model drift has nowhere to accumulate. The shipped default adapter
(`models/claude/planner_request.sh`) drives Claude (Opus) via `claude -p`
read-only; the role is swappable behind the script by a bundled `rig:` name (e.g.
the all-Qwen `models/local/`) or the operator's gitignored `models/personal/`.

Grindstone invokes the planner once at **every epoch boundary**, starting from the
first. There are no phases and no skeleton step: the very first call already
proposes a real epoch (or, for a trivial job, ends). The planner emits **exactly
one decision per call**.

Each call rebuilds a bounded `PlannerContext` from disk and the planner returns one
typed `Decision`. The two-node failure taxonomy governs what can go wrong:

- **`RateLimited`** (failure node 1): a rate-limit / quota refusal. The loop parks,
  backs off (about once an hour), and re-issues the same boundary call. Nothing is
  burned.
- **`PlannerError`** (failure node 2): any other unrecoverable planner failure
  (auth, transport, or a decision the planner could not make valid within its
  budget). The loop ends the run cleanly (a resumable partial-end).

A decision that fails validation is treated as a retryable bad output and
**re-asked up to twice** (`MAX_REASKS`; the failing reasons are appended to the
next input). The rig usually self-corrects on disk before this fires (see the
self-validate loop, below); an exhausted re-ask budget raises `PlannerError`.

## 2. Input construction

Grindstone owns the input; it does **not** mirror the output format. Every call is
built as a byte-stable head plus a volatile tail so a backend prefix cache can reuse
the head across a run (`build_planner_input` in `grindstone/planner.py`):

```
[BYTE-STABLE CORE: byte-identical every call of every run]
  PLANNER_CORE             role identity + output discipline + the cross-cutting
                           rules true for every decision (fixed per Grindstone version)
[VOLATILE TAIL]
  job spec                 the full job text
  state                    the epoch index + its max-epochs backstop, and the
                           keyed-log index (the log keys a task may name as inputs)
  integration tip          a bounded listing of the tracked files at the current tip
                           (capped at TIP_FILES_CAP; the planner greps for the rest)
  carried failures         the prior epoch's UNRESOLVED outcomes (a critic escalate,
                           an ownership conflict, a setup failure) to steer around or end on
  domain skills            the target repo's <repo>/.grindstone/skills catalogue index
                           (name -> one-line description; empty when the repo ships none)
  tools note               the workdir is a checkout of the tip; grep + read it to ground
  re-ask errors            only when a prior decision was rejected
  decision request
```

**References, not payloads.** The tail carries *names* and *log keys*, never inlined
file bodies. The planner names an artifact by its log key in a task's `inputs`;
Grindstone resolves keys to file bodies only when building the *worker's* input. The
listed `inputs` must already exist in the keyed-log index; an invented key is
rejected.

## 3. The decision wire contract

A decision is a single JSON object discriminated on `kind`, one of two shapes:

```
EPOCH: {"kind":"epoch","epoch":{"title":..,"rationale":..,"tasks":[ ... ],"setup":[ ... ]}}
END:   {"kind":"end","summary":".."}
```

The planner **owns all sequencing**. There are no phases, no skeleton, no epoch
budget, and no exit criteria: it steers the run itself, one epoch at a time, until
the job is met, then emits **END**. A fresh run often leans research-first to
understand the job, but that is not forced; a small job may be a single epoch.

- **EPOCH** proposes the single next epoch as **1 to 8 independent tasks** that fan
  out in parallel (`title`, an optional `rationale`, the `tasks`, and an optional
  `setup` list).
- **END** stops the run. `summary` is the pending-summary / resume seed: either what
  was accomplished (a satisfied done) or, when the planner cannot make progress, a
  handoff that lets the work continue as the next appendable run. END is also how
  the planner disposes of an unresolvable carried failure.

The planner **does not author verify or test commands as a gate**: it writes no
`done_when` and no check commands. An independent agentic critic re-derives each
task's goal and judges the work against it. Acceptance is carried in the task's
prose `goal`, never as a shell command. *Model proposes, state machine disposes.*

### Setup: the trusted host-mutation seam

If an epoch needs a **host-global** mutation (a system package, a globally installed
tool, a shared directory outside the repo), the planner lists the command in the
epoch's `setup`: the **trusted** state machine runs those, in order, before the
tasks (the untrusted worker never mutates the host). Setup runs in a throwaway
checkout, **not** the task worktrees, so project-**local** dependency installs
(the project's own package manager) do **not** belong in `setup`: they would not reach
the isolated task worktrees. An implement task installs the project deps it needs
inside its own worktree as part of its work. This one field replaces an entire
infra-repair subsystem.

## 4. Tasks

Each task in an epoch is independent and fans out concurrently; tasks **must not
consume each other's outputs** (anything sequential belongs in a later epoch).
1 to 8 tasks per epoch, ids `T1`..`T8`; the fully-qualified log-key prefix is
`<phase>/<epoch>/<task>` (the keyed log keeps a single fixed phase prefix, since the
system is epochs-only).

Every task carries:

- `id`: `T1`..`T8`.
- `mode`: `implement` | `research` | `review` | `artifact`. Picks the worker prompt
  and the critic's grounding bar.
- `goal`: prose (<= 2048 chars), stating the task's **own** notion of done. Carry
  the relevant job-spec requirements verbatim or point at the exact input artifacts
  (by log key); lossy paraphrase silently drops requirements.
- `tier`: `local` (default, the Qwen rig, mechanical or checkable work) or `senior`
  (the Claude rig, judgment / taste / synthesis). Routing is **per task**: split a
  mechanical slice from a judgment slice and flag only the judgment slice `senior`,
  so the senior quota is spent only where it is needed. A rig with no senior tier
  falls back to local. The critic runs on the same tier as its task.
- `skills`: optional domain-skill names (max 6) selected from the target repo's
  `<repo>/.grindstone/skills/index.md` catalogue. The validator rejects any name the
  catalogue does not advertise, and absent a catalogue `skills` must be empty. The
  worker composes only the selected skills into its prompt (retrieve, do not
  concatenate the whole catalogue).
- `inputs`: optional log keys (max 12) this task reads. Each must already exist in
  the keyed-log index; an invented key is rejected.

Mode-specific shape (the one cross-field rule, enforced by the Pydantic model):

- **implement** requires `file_ownership`: 1 to 32 concrete files or globs the task
  may create or edit, and **must not** set `artifact_out`. Ownership across the
  epoch's tasks must be **pairwise disjoint**; this is the merge-correctness
  mechanism (Section 6), not metadata. Enumerate concrete files; never claim a whole
  subtree you cannot bound.
- **research / review / artifact** require `artifact_out` (the one log key the
  deliverable lands at) and **own no tree files**. Two tasks may not declare the
  **same** `artifact_out` (the artifact analogue of disjoint ownership; a collision
  is rejected so the planner re-emits with distinct keys). These tasks get no
  worktree: a non-write task is never handed the live repo as its CWD. A `review`
  should independently re-derive what it judges and reconcile it against the real
  files, not merely confirm that sections are present.

## 5. Choosing the mode and the tier

Pick the **mode** by the deliverable's destination:

- Output the job needs as a **committed file in the repo tree** (code, config, docs,
  even prose) -> `implement`. Only implement tasks run in a worktree and get
  committed.
- Output consumed **through the keyed log** (an analysis the job does not need
  committed) -> `research` / `artifact`, shipped via `artifact_out`.
- A judgment over existing work -> `review`, also via `artifact_out`.

Pick the **tier** by the kind of thinking: mechanical or factual work (scaffolding,
tokens, boilerplate, exports, web-search fact gathering, a structural check) is
`local`; judgment or taste (layout, polish, an approach synthesis, a design-quality
verdict) is `senior`. A non-implement epoch's `artifact_out` persists to the keyed
log, so it becomes a downstream epoch's `inputs`: that keyed-log handoff is how
findings flow from a senior investigation into a worker build or write-up.

Split sequential steps across epochs liberally (each boundary is a free planner
checkpoint), but split parallel tasks within an epoch conservatively (they must be
truly independent and disjoint). Do not invent artificial steps; every epoch costs a
planner call, bounded by `max_epochs`.

## 6. Integration policy (implement epochs)

1. Every implement task gets a worktree branched from the **epoch base** (the run
   branch tip at epoch dispatch).
2. `file_ownership` sets are **pairwise disjoint within the epoch**. An overlap in
   the realized git diff is detected at integration and surfaced to the planner as a
   carried failure (the planner mis-scoped); the run branch is left untouched.
3. On task return, a deterministic scope check: every committed path must fall
   inside the task's `file_ownership`. An out-of-scope write or a zero-diff commit is
   a failed attempt; workers never negotiate scope.
4. Integration is fast-forward merges in task-id order onto a staging branch, then a
   fast-forward of the durable run branch `grind/<run-id>`. Given (2) and (3) the
   merges commute; a conflict is treated as a structural error that aborts
   integration (carried to the planner), never papered over.
5. Grindstone (never the model) commits each successful task.

## 7. The critic and the worker handoff

There is no Python-parsed handoff contract. The worker writes a **free-form**
`handoff.md` report in its CWD (plain prose: what it did, what is done, what is
blocked, which files it touched, grounding as prose). Grindstone relocates it
verbatim to the keyed log for the critic and the planner's optional context; it is
**never parsed or schema-validated**, and stdout is never parsed. The deterministic
gate is the committed diff or the produced artifact, not this file.

A gate-clean attempt always runs the tier-matched **critic** (an independent agentic
pass). It emits a lenient `Verdict` (`schemas/verdict.json`): an `outcome` of
**PASS** (merge; notes carry forward), **RETRY** (a defect the same worker can fix;
the bounded same-tier retry), or **ESCALATE** (anything the worker cannot fix, which
routes to the planner as a carried failure). An environmental blocker the worker
reports in `handoff.md` becomes a critic ESCALATE; there is no separate Python
BLOCKED gate. A research / review critic also verifies the artifact's citations
resolve to real files. A missing or invalid verdict is a fail-safe escalate, never a
silent pass.

## 8. Failure flow

```
failed deterministic gate (zero-diff / out-of-scope / missing artifact)
                              -> a failed attempt: bounded same-tier retry (MAX_ATTEMPTS), then escalate
critic RETRY                  -> bounded same-tier retry, inheriting the prior attempt's wip
critic ESCALATE               -> the task escalates; its reason is CARRIED to the planner next boundary
retries exhausted             -> the task escalates (carried)
ownership overlap / merge conflict -> integration aborts; carried to the planner (run branch untouched)
setup command failed          -> carried to the planner; the epoch is skipped
planner RATE-LIMITED          -> failure node 1: park, back off ~1/hr, re-issue the boundary
worker RATE-LIMITED           -> failure node 1: raze the in-flight epoch, restart it whole after the backoff
max_epochs reached            -> failure node 2: an involuntary clean partial-end (resumable)
```

There is **no tier escalation**, no infra-repair node, and no separate
failed-epoch state machine. Every non-rate-limit failure is **carried** as context
the planner reads at the next boundary and either steers around (re-plan) or ends on
(a clean partial-end). The `max_epochs` backstop guarantees a planner that spins
without progress is bounded.

## 9. The self-validate-on-disk loop and read priority

The planner grounds itself the same way a worker does. It runs inside a throwaway
checkout of the current integration tip (its `_planner_tip` workdir, refreshed only
when the tip moves). It may grep and read that workdir to ground its plan, then it
writes its decision to `./decision.json`, runs `python3 check_decision.py
decision.json` (which re-execs the **real** core validator,
`grindstone/check_decision.py`, so the on-disk gate cannot drift from the
orchestrator's), fixes every violation it prints, and loops until the check exits 0.
That gate-clean `decision.json` is its output.

Grindstone reads the rig's result back in a fixed **priority order**
(`_read_result`): `decision.json` (the file the rig looped the check on, the real
proof) > the rig's `--out` final-message file > raw stdout. A present-but-empty
channel falls through to the next. Whatever is read is extracted to JSON and parsed
through the same `parse_decision` the validator uses; an unparseable or invalid
result is re-asked (Section 1), and an exhausted budget is a `PlannerError`. The
planner never returns an unvalidated decision.
