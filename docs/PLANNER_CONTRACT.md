# Grindstone Planner Contract v1

The planner is the only model that steers a run. This document is the reference for
**how Grindstone calls it, what it is allowed to emit, and how the state machine
validates and disposes of every decision**.

The wire format is owned by `schemas/epoch_decision.json`; that file mirrors the
Pydantic types in `grindstone/contracts/models.py`, which are the source of truth.
Where prose here and the model disagree, the model wins.

---

## 1. Call model

The planner is a **stateless one-shot call** with two roles (`grindstone/planner.py`):
**PLAN** (`ScriptPlanner.decide`, forward) proposes the next epoch, and **CLOSE-OUT**
(`ScriptPlanner.close_out`, back) writes the epoch's baton. There is no warm planner
instance and no planner-held state: Grindstone reconstructs the full input fresh from
durable disk state on every call, so model drift has nowhere to accumulate. Its only
memory across the run is the **baton** it writes at close-out and re-reads at the next
PLAN. The shipped default adapter (`models/claude/planner_request.sh`) drives Claude
(Opus) via `claude -p` read-only and serves both roles (switched by a `--purpose`
flag); the role is swappable behind the script by a bundled `rig:` name (e.g. the
all-Qwen `models/local/`) or the operator's gitignored `models/personal/`.

Grindstone invokes PLAN once at **every epoch boundary**, starting from the first.
There are no phases and no skeleton step: the very first call already proposes a real
epoch (or, for a trivial job, ends). PLAN emits **exactly one decision per call**.
After the epoch's tasks are gated and integrated onto a staging branch, CLOSE-OUT is
invoked once to write the baton (Section 10).

Each call rebuilds a bounded context from disk (`PlannerContext` for PLAN,
`CloseoutContext` for close-out); PLAN returns one typed `Decision`, close-out returns
the free-form baton markdown. The two-node failure taxonomy governs what can go wrong:

- **`RateLimited`** (failure node 1): a rate-limit / quota refusal. The loop parks,
  backs off (about once an hour), and re-enters. A PLAN rate-limit re-issues the same
  boundary call; a close-out rate-limit razes the in-flight epoch (staging and wip)
  and restarts the epoch whole. Nothing is burned.
- **`PlannerError`** (failure node 2): any other unrecoverable PLAN failure (auth,
  transport, or a decision the planner could not make valid within its budget). The
  loop ends the run cleanly (a resumable partial-end). Close-out never hard-fails on
  content (it reads whatever prose the rig produced, like the handoff); a transport
  hard error there takes the epoch abort path (raze and restart).

A decision that fails validation is treated as a retryable bad output and
**re-asked up to twice** (`MAX_REASKS`; the failing reasons are appended to the
next input). The rig usually self-corrects on disk before this fires (see the
self-validate loop, below); an exhausted re-ask budget raises `PlannerError`.

## 2. Input construction

Grindstone owns the input; it does **not** mirror the output format. The PLAN call is
built as a byte-stable head plus a volatile tail so a backend prefix cache can reuse
the head across a run (`build_planner_input` in `grindstone/planner.py`); the
close-out input (Section 10) mirrors the same head-plus-tail shape:

```
[BYTE-STABLE HEAD: byte-identical every call of every run]
  PLAN_PREAMBLE            role identity + output discipline + the cross-cutting
                           rules true for every decision (fixed per Grindstone version)
[VOLATILE TAIL]
  job spec                 the full job text
  state                    the epoch index + its max-epochs backstop, and the
                           keyed-log index (the log keys a task may name as inputs)
  baton                    the prior completed epoch's baton.md text (the planner's
                           living plan: its memory; empty on the first epoch)
  domain skills            the target repo's <repo>/.grindstone/skills catalogue index
                           (name -> one-line description; empty when the repo ships none)
  tools note               the workdir is a checkout of the tip; grep + read it to ground
  re-ask errors            only when a prior decision was rejected
  decision request
```

There is **no file-name dump**: the planner greps its own workdir checkout of the
tip for the tree. Its only carried state is the baton, which it reconciles against
that actual tree (the tree is ground truth, the baton is intent).

**References, not payloads.** The tail carries *names* and *log keys* (and the baton
text the planner itself wrote), never inlined file bodies. The planner names an
artifact by its log key in a task's `inputs`; Grindstone resolves keys to file bodies
only when building the *worker's* input. The listed `inputs` must already exist in the
keyed-log index; an invented key is rejected.

## 3. The decision wire contract

A decision is a single JSON object discriminated on `kind`, one of two shapes:

```
EPOCH: {"kind":"epoch","epoch":{"title":..,"tasks":[ ... ],"setup":[ ... ]}}
END:   {"kind":"end","summary":".."}
```

The planner **owns all sequencing**. There are no phases, no skeleton, no epoch
budget, and no exit criteria: it steers the run itself, one epoch at a time, until
the job is met, then emits **END**. A fresh run often leans research-first to
understand the job, but that is not forced; a small job may be a single epoch.

- **EPOCH** proposes the single next epoch as **1 to 8 independent tasks** that fan
  out in parallel (`title`, the `tasks`, and an optional `setup` list). There is no
  `rationale` field: the planner's reasoning lives in the baton it carries across
  epochs, not in the decision.
- **END** stops the run. `summary` is the pending-summary / resume seed: either what
  was accomplished (a satisfied done) or, when the planner cannot make progress, a
  handoff that lets the work continue as the next appendable run. END is also how
  the planner stops on a failure it judges unresolvable (the baton records why).

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
`<epoch>/<task>` (e.g. `E3/T2`), with no phase prefix, since the system is
epochs-only.

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
   the realized git diff is detected at integration and recorded by close-out in the
   baton (the planner mis-scoped); the run branch is left untouched.
3. On task return, a deterministic scope check: every committed path must fall
   inside the task's `file_ownership`. An out-of-scope write or a zero-diff commit is
   a failed attempt; workers never negotiate scope.
4. Integration is fast-forward merges in task-id order onto a **staging branch**
   (`_integrate_to_staging`); the durable run branch `grind/<run-id>` is
   fast-forwarded to that staging tip later, in `_finalize_epoch`, after close-out has
   read the staging tree. Given (2) and (3) the merges commute; a conflict is treated
   as a structural error that aborts integration (close-out records it in the baton),
   never papered over.
5. Grindstone (never the model) commits each successful task.

## 7. The critic and the worker handoff

There is no Python-parsed handoff contract. The worker writes a **free-form**
`handoff.md` report in its CWD (plain prose: what it did, what is done, what is
blocked, which files it touched, grounding as prose). Grindstone relocates it
verbatim to the keyed log for the critic and the close-out planner to read; it is
**never parsed or schema-validated**, and stdout is never parsed. The deterministic
gate is the committed diff or the produced artifact, not this file.

A gate-clean attempt always runs the tier-matched **critic** (an independent agentic
pass). It emits a lenient `Verdict` (`schemas/verdict.json`): an `outcome` of
**PASS** (merge; notes carry forward), **RETRY** (a defect the same worker can fix;
the bounded same-tier retry), or **ESCALATE** (anything the worker cannot fix; the
task's work is discarded and the outcome surfaces to the close-out planner, which
reads the verdict and records what really happened in the baton). An environmental
blocker the worker reports in `handoff.md` becomes a critic ESCALATE; there is no
separate Python BLOCKED gate. A research / review critic also verifies the artifact's
citations resolve to real files. A missing or invalid verdict is a fail-safe escalate,
never a silent pass.

The gate events in the journal are `work_gate_passed` / `work_gate_rejected` (the
deterministic gate), followed by the `verdict` (the critic's triage).

## 8. Failure flow

```
failed deterministic gate (zero-diff / out-of-scope / missing artifact)
                              -> a failed attempt: bounded same-tier retry (MAX_ATTEMPTS), then escalate
critic RETRY                  -> bounded same-tier retry, inheriting the prior attempt's wip
critic ESCALATE               -> the task's work is discarded; close-out reads the outcome and records it in the baton
retries exhausted             -> the task escalates (same path as ESCALATE)
ownership overlap / merge conflict -> integration aborts (run branch untouched); close-out records it in the baton
setup command failed          -> grind skipped; the epoch still completes with a baton noting the error, then advances
PLAN RATE-LIMITED             -> failure node 1: park, back off ~1/hr, re-issue the boundary
worker / close-out RATE-LIMITED -> failure node 1: raze the in-flight epoch (staging + wip), restart it whole after the backoff
unexpected epoch-body error   -> raze + restart the SAME epoch (an aborted epoch has no baton); K consecutive aborts clean-end the run
max_epochs reached            -> failure node 2: an involuntary clean partial-end (resumable)
```

There is **no tier escalation**, no infra-repair node, and no separate
failed-epoch state machine. Every non-rate-limit failure is read by the close-out
planner and written into the baton, which the next PLAN call reads and either steers
around (re-plan) or ends on (a clean partial-end). The `max_epochs` backstop
guarantees a planner that spins without progress is bounded.

## 9. The self-validate-on-disk loop and read priority (PLAN)

The PLAN call grounds itself the same way a worker does. It runs inside a throwaway
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

## 10. Close-out and the baton

At the **end** of every epoch, after the passing tasks are merged onto a staging
branch but **before** the durable run-branch fast-forward, the loop invokes the
planner's **CLOSE-OUT** role (`ScriptPlanner.close_out`, `build_closeout_input`). It
hands a `CloseoutContext` and asks for one thing: the updated **baton**.

The close-out input mirrors the PLAN head-plus-tail shape: a byte-stable
`CLOSEOUT_PREAMBLE` (which carries the four-section baton skeleton) then the volatile
tail of the job, the **prior baton**, the **epoch report**, the domain-skill index,
and the request to write `./baton.md`. The epoch report lists, per task: its id, mode,
the deterministic outcome (`passed` / `escalated`), and the keyed-log paths of the
worker handoff and the critic verdict to **read**, plus the verbatim reason; then any
`setup_error` and `integration_conflict`. These are pure **references**: Python
characterizes nothing. The close-out planner's CWD is a throwaway checkout of the
epoch's **staging tree** (the work that merged); it greps that tree, opens the named
handoffs and verdicts, and judges what really happened, writing the
partial-progress / no-progress / regression nuance itself.

The **baton** is a **free-form** markdown living plan with four sections the prompt
enforces (Project summary / Tasks done / Tasks pending / Current status). Python
persists it **verbatim and never parses it** (the same status as the handoff), at
`E<n>/baton.md` in the keyed log. There is **no self-validate loop and no JSON
parse**: close-out reads its result back by priority `baton.md` > `--out` > stdout
and returns whatever prose the rig produced (it never `PlannerError`s on content; a
`RateLimited` razes and restarts the epoch, and a transport hard error takes the
epoch abort path). Writing the baton marks the epoch DONE; the next epoch's PLAN call
re-reads it as the planner's memory.

Because close-out reads the staging tree and runs before the fast-forward, the single
durable commit point (the fast-forward + the baton write + `epoch_completed`) already
includes the baton: there is no "integrated-but-not-summarized" limbo, and a
completed epoch always has a baton.

### Vision is first-class

The PLAN call, the close-out call, the worker, and the critic all **see**. Images
(screenshots, mockups, rendered UI, diagrams) ride the **same file contract** as text:
an agent views an image file with its Read tool, with no separate machinery and no
schema. An image artifact a task produces is a first-class log key, and an image input
is named by its log key like any other input. The prompts tell every agent it can see
and should view rather than describe. Generating the visual proof (rendering a screen,
capturing a screenshot) is the **target repo's** job, declared in a task's goal;
Grindstone's job is only to make the pipeline able to see it.
