# Grindstone Planner Contract v1

The planner is the only model that steers a run. This document is the reference
for **how Grindstone calls it, what it is allowed to emit, and how the state
machine validates and disposes of every decision**.

The wire format is owned by `schemas/epoch_decision.json` and
`schemas/handoff.json`: those files are the single source of truth, and the
Pydantic types + validators in `grindstone/contracts/` are kept in lockstep with
them. Where prose here and the schema disagree, the schema wins.

---

## 1. Call model

The planner is a **stateless one-shot call**. There is no warm planner instance
and no planner-held state; Grindstone reconstructs the full input from durable
state on every call, so model drift has nowhere to accumulate. The shipped default
adapter (`models/default/planner_request.sh`) drives Claude (Opus) via `claude -p`
read-only; a bundled `codex exec` alternative lives at `models/codex/planner_request.sh`
(opt in with `grindstone init --rig codex`), and the role is swappable behind the
script via `models/override/`.

Grindstone invokes the planner at exactly two kinds of moment:

1. **Run start**: the input carries the job spec; the only legal decision is
   `propose_skeleton`.
2. **Epoch boundary**: the previous epoch reached its done-predicate (task
   queue empty AND nothing in flight), or a **phase escalation** fired (epoch
   budget exhausted with the exit criterion still failing). The planner emits
   exactly **one decision per call** from the tool set in §3.

A `propose_skeleton` or `revise_phases` decision is applied, journaled, and the
planner is **immediately re-invoked** against the updated skeleton. One decision
per call keeps the audit trail linear, with no compound decisions.

Planner-call failures are classified three ways before any retry
(`classify_failure`, in `grindstone/planner.py`):

- **rate-limit** → exponential backoff, wait for the window (never auto-spill to
  a different planner);
- **transient** (network / 5xx) → retry the same call;
- **hard** (auth / config / unknown) → escalate the run to a human.

A decision that fails validation is treated as a retryable bad output and
**re-asked up to twice** (the failing reasons are appended to the next input);
exhausting the re-ask budget escalates the run.

## 2. Input construction

Grindstone owns the input; it does **not** mirror the output format. Every call
is built as a stable head + a volatile tail so a server-side prefix cache can
reuse the head across a run (`build_planner_input` in `grindstone/planner.py`):

```
[STABLE HEAD: byte-identical across the run]
  system preamble          (fixed per Grindstone version)
  job spec                 (frozen at run start)
  skills digest            (reserved; empty in v1)
  repo memory digest       (frozen at run start; empty when the repo has none)
  phase skeleton           (changes ONLY on propose_skeleton / revise_phases)
[VOLATILE TAIL]
  running state            (phase id, epoch counter, keyed-log index)
  phase status             (per-check pass/fail of the exit criterion, budget,
                            integration-tip file listing, escalation demand)
  last epoch report        (per-task status/attempts/tier, each DONE task's
                            resulting_state + downstream_needs, each FAILED
                            task's last reason)
  re-ask errors            (only when a prior decision was rejected)
  decision request
```

**References, not payloads.** The head and tail carry *log keys* into the
durable keyed log, never inlined artifact bodies. The planner asks for an
artifact by listing its key in a task's `inputs`; Grindstone resolves keys to
file bodies only when constructing the *worker's* input.

## 3. Decision tool set

A decision is a single JSON object `{schema_version, tool, args}`. **Dispatch is
on the tool name**: modes are function names, not a field a model can fudge.

| tool | when legal | effect |
|---|---|---|
| `propose_skeleton` | first call of a run only | creates the phase skeleton |
| `implement` | epoch boundary | plans an epoch whose deliverable is committed repo files (worktree + commit) |
| `research` / `artifact` | epoch boundary | plans an epoch that ships an analysis/report through the keyed log (no worktree) |
| `review` | epoch boundary | judges existing work and ships a verdict through the keyed log |
| `revise_phases` | epoch boundary | the **phase STRUCTURE** is wrong; replaces the current phase onward (never a completed phase); separately journaled |
| `handle_failed_epoch` | only when an epoch has **failed** and is awaiting disposition | a focused disposition of that epoch: `retry` (with a `hint`, optionally `escalate_tier`), `escalate_senior` (with a `diagnosis`), or `halt` (with a `reason`) |
| `escalate_run` | epoch boundary | the planner cannot proceed → hand to a human |
| `complete_run` | epoch boundary | the whole job is done; carries deterministic `evidence` checks |

There is **no `advance_phase` tool**: phase exit is decided by Grindstone running
the phase's deterministic exit criterion against the integration tip, never by a
planner claim. The planner *defines* the criteria (in `propose_skeleton` /
`revise_phases`); the state machine *evaluates* them. *Model proposes, state
machine disposes.* The planner never executes anything inline.

### Choosing the mode: by destination, not flavor

The planner picks the mode from the deliverable's **destination**:

- Output the job requires as a **committed file in the repo tree** (code,
  config, docs, even prose) → `implement`. Only implement tasks run in a
  worktree and get committed.
- Output consumed **through the keyed log** (an analysis or investigation the
  job does not need committed) → `research` / `artifact`, shipped via
  `artifact_out`.
- A judgment over existing work → `review` (also shipped via `artifact_out`,
  plus `targets`).

Never give a task a worktree its deliverable does not need.

### Decomposition is three skills, one per level

The planner decomposes at three distinct levels, in order, with different units
and biases. Keeping them separate is what makes a failure localizable: a single
giant epoch with a single giant task cannot be diagnosed when it fails.

1. **PHASING** (`propose_skeleton` / `revise_phases`): split the *job* into
   phases. **One phase = one MODE** (research / implement / test / review); a
   phase that mixes modes is two phases. 2–10 phases (§4).
2. **EPOCH** (one work decision per call): split a *phase* into epochs. **One
   epoch = one coherent FEATURE or milestone.** For an **implement** phase the
   **FIRST epoch is an explicit baseline-dependencies epoch**: it stands up the
   project skeleton and produces the **committed dependency manifest/lockfile**
   (e.g. `package.json` + its lockfile); later epochs build features on it. (A
   separate `prepare` mechanism, §config, *installs* from that lockfile when
   gates run; the baseline epoch only *creates* the manifest, it does not
   install.) Split sequential steps across epochs *liberally* (below).
3. **TASK** (the fan-out within an epoch): split an *epoch* into tasks. **One
   task = one bounded slice, kept SMALL** (a few files), with disjoint
   `file_ownership`. Split parallel tasks *conservatively* (§5), but a single
   task may not swallow the whole epoch: the **size gate** (§5) rejects an
   oversized or whole-repo task.

### Sequencing: decompose heavy work by tier of thinking

The mode also picks the **tier**: `research` / `review` (and any `visual` epoch)
start on the stronger **senior** tier; `implement` / `artifact` start on the
local rig (see §5). So a phase skeleton is also a *routing* decision: put the
judgment on senior and the production on local. These are gentle defaults, not
mandates: a small job is fine as a single epoch. But for heavy or judgment-laden
work, splitting pays off: each tier does what it is best (and cheapest) at, and
a non-implement epoch's `artifact_out` persists to the keyed log, so it becomes a
downstream epoch's `inputs`. That keyed-log handoff is how findings flow from a
senior investigation into a local build or write-up.

Splitting is driven by **two independent axes**: a **tier change** (judgment vs.
production, above) and a **data dependency or checkpoint**. The second matters
even when the tier does not change: if step B consumes step A's `artifact_out`,
or a meaningful gate sits between them, B belongs in its **own epoch** even when
both are local `implement` work. Be *conservative* splitting parallel tasks
within an epoch (§5), but *liberal* splitting sequential steps across epochs: each
boundary is a free planner checkpoint and a deterministic gate. Do not, however,
invent artificial steps; every epoch costs a planner call, bounded by
`epoch_budget`.

Good shapes (compose freely; these are nudges, not a fixed menu):

- **Heavy implementation** → `research` (map the area + constraints, on senior,
  cited) → `implement` (build it, on local, consuming the research artifact as an
  input) → `review` (on senior, re-derive a sample of the result's claims and
  reconcile them against the inputs, not merely confirm the expected sections
  exist).
- **Report / triage / migration plan** → `research` (investigate + classify, on
  senior, cited) → `artifact` (write the final report from the findings, on
  local). Do not collapse the judgment into a single local `artifact` epoch when
  the analysis is the hard part, which silently downgrades it off senior.
- **UI / polish** → `research` (gather the design intent + tokens) → `implement`
  with `visual: true` (build on senior) → a phase `exit_criterion` that builds +
  screenshots the UI and then `vision_review`s it (the taste gate, §5).

Counter-example: framing a judgment job as "produce `report.md`" makes it look
like local production and routes it off senior. If the analysis is the point, say
so: use a `research` epoch first.

## 4. Phases

A phase = `{id, title, exit_criterion, epoch_budget}` (`id` is `P1`…`P99`; a
skeleton has 2–10 phases).

- `exit_criterion` is a list of deterministic **checks** (the same shape as a
  task's `done_when`, §5). A planner that cannot express phase-done as commands
  + expected exits and/or required artifacts is mis-scoping the phase; the
  validator rejects prose-only criteria structurally.
- `epoch_budget` caps how many epochs the phase may consume. Exhaustion does not
  loop silently: it fires a **phase escalation**; the next call may legally emit
  only `revise_phases` or `escalate_run` until the demand clears.
- `revise_phases` may not touch a completed phase (a semantic check, not just
  the schema).

## 5. Tasks

All tasks in an epoch are independent and fan out concurrently; they **must not
consume each other's outputs** (anything sequential belongs in a later epoch).
1–8 tasks per epoch, ids `T1`…`T8`; the fully-qualified log-key prefix is
`<phase>/<epoch>/<task>`.

Common shape:

- `goal`: byte-capped prose (≤1024 chars). Carry the relevant job-spec
  requirements in **verbatim**, or point at the exact input artifacts (by log
  key) that contain them; lossy paraphrase silently drops requirements.
- `inputs`: **log keys only.** The validator rejects a key that does not exist
  in the keyed log at validation time, the structural guard against the planner
  hallucinating an upstream artifact.
- `done_when`: 1–6 deterministic checks (see *Checks* below). The worker runs
  them, then Grindstone re-runs them on return; a task whose checks cannot run is
  FAILED, never vibes-DONE. `done_when` is scoped by mode: a research / review /
  artifact task runs in a scratch dir that is **not** a repo checkout, so its
  `done_when` must verify the artifact itself (e.g. `test -s notes.md`), never a
  repo build/test command (those can only pass in an implement task or a phase
  exit criterion).
- `skills`: optional catalog names (reserved seam).

Mode-specific:

- **implement** additionally requires `file_ownership`: 1–32 path globs that must
  be **pairwise disjoint across the epoch**; this is the merge-correctness
  mechanism (§6), not metadata. A **deterministic size gate** further bounds a
  *fresh* implement task: its `file_ownership` glob count may not exceed a
  **tier-aware** ceiling (`local_max_task_files`, default **5**; the larger
  `senior_max_task_files`, default **12**, for a `visual` epoch that starts on
  senior), and a **whole-repo glob** (`**`, `**/*`, or a bare `*`) is rejected
  outright as "not decomposed". A task over its bound (or claiming the whole
  repo) bounces back through the same invalid-decision re-ask path (§1), naming
  the offending task. The gate is **scoped to fresh decomposition**: a
  `handle_failed_epoch` repair re-dispatches its originating decision directly
  and may carry broad scope, so it is **exempt** (a repair cannot predict its
  files). The bounds are config fields (both ≥ 1).
- **research / review / artifact** additionally require `artifact_out` (the log
  key the task will create). Review tasks also take `targets` (the paths under
  review). These tasks get **no worktree**: a non-write task is never handed the
  live repo as its CWD. A `review` must INDEPENDENTLY RE-DERIVE a sample of the
  claims or verdicts it judges and RECONCILE them against the upstream
  artifact(s) it consumes via `inputs`; confirming only that required sections or
  fields are present is mis-scoped (it spends a planner call yet catches no wrong
  answer). When a review consumes an upstream artifact, surfacing any
  contradiction between the reviewed work and that artifact is a primary job of
  the review.

### Taste routing: the `visual` epoch flag

An `implement` / `review` / `artifact` epoch may set `visual: true` (default
`false`) on its args when its deliverable is **front-end / UI / visual / polish**
output: anything judged by how it *looks*. A visual epoch starts its workers on
the stronger taste-building **senior** tier instead of the local default. (The
senior is a text model; the genuine image judgment is the vision-review gate
below. A rig with no senior tier falls back to local so a visual epoch grinds
rather than crash.) Omit the flag for backend / logic / plain-text work.

### Checks

A check is one of three shapes:

- `{"cmd": "...", "expect_exit": 0}`: a shell command; passes when its exit code
  equals `expect_exit` (default 0).
- `{"artifact_exists": "<log key>"}`: passes when the keyed log holds that
  artifact (an exact key, or a bare filename that matches exactly one logged
  artifact, useful in a phase exit criterion written before the producing task's
  `P*/E*/T*` placement is known).
- `{"vision_review": {"screenshot": "<eval-worktree-relative path>", "criteria":
  "<what polished looks like>"}}`, the **taste gate**: a strong vision model
  judges a rendered screenshot against the criteria and emits a pass/fail
  verdict.

The `vision_review` check is legal **only in a phase exit criterion** (never a
task `done_when`, a worker scratch has no renderer). Put a `cmd` check *first*
in the same criterion that builds + screenshots the UI into the tip worktree
(e.g. `{"cmd": "npm run build && node shot.js ui/screen.png"}`), then a
`vision_review` of that `ui/screen.png` against the design bar. Grindstone runs
the gate through a request script (the rig's `vision_review.sh`), re-reads the
returned `vision_verdict.json` (a disk contract, never stdout), and treats a
failed taste verdict exactly like a failed command.

## 6. Integration policy (implement epochs)

1. Every task gets a worktree branched from the **epoch base** (the run branch
   tip at epoch dispatch).
2. `file_ownership` sets are **pairwise disjoint within the epoch**, enforced at
   plan time; overlap is rejected back to the planner with the overlap named.
3. On task return, a deterministic scope check: every committed path ⊆
   `file_ownership`. An out-of-scope write is a failed attempt (counts toward the
   retry budget); workers never negotiate scope.
4. Integration is fast-forward merges in task-id order. Given (2) + (3) the
   merges **commute and cannot conflict**; any conflict is a structural bug, not
   a runtime code path, and aborts the epoch rather than being papered over.
5. Grindstone (never the model) commits each successful task.

## 7. Worker handoff (`schemas/handoff.json`)

The handoff is the result channel; Grindstone never parses a worker's stdout.

- **Disk contract.** The worker writes `handoff.json` in its **own CWD** and
  self-validates with the generated validator before finishing. Grindstone
  relocates the file to the task's log key and re-validates it. The relocated,
  re-validated file is the gate.
- **References, not payloads**, enforced by a total serialized **byte cap (8
  KiB)** plus per-field caps in the validator.
- **Grounding spot-check.** Grindstone resolves the handoff's `citations`
  (`{file, line}`) against the allowed roots and rejects a citation that does not
  exist, the backstop against a plausible-but-wrong handoff poisoning the next
  planner call. Research / review handoffs require ≥1 citation.

Fields (see the schema): `status` (DONE / FAILED / PARTIAL), `what_changed`,
`resulting_state`, `downstream_needs` (log keys), `not_done`, `citations`,
`checks` (echo of the `done_when` results), `occupancy` (post-hoc context
telemetry). Only `status == DONE` whose checks all pass is accepted.

## 8. Failure flow

```
check / validation failure  → re-queue with failure context (retry ≤ 3 on the starting tier)
starting tier exhausted     → escalate the task to the next ladder tier (one attempt each)
ladder exhausted            → mark the task FAILED; the epoch continues, the failure lands in the report
epoch with a FAILED task    → FOCUSED handle_failed_epoch decision (the ONLY legal tool until disposed):
                              retry (hint, optional tier bump) | escalate_senior (diagnosis) | halt (reason)
per-phase failed-epoch cap  → after `max_failed_epochs_per_phase` (default 3) failed epochs in one phase the
                              state machine FORCES a halt-to-human regardless of the planner's choice
epoch budget exhausted      → phase escalation → planner (revise_phases | escalate_run only)
```

When an epoch fails the planner is asked a **focused** `handle_failed_epoch`
decision, not left to react with a blind `revise_phases`. The decision input
carries the failing phase checks **with their captured command output** (so the
planner can tell an environment/gate problem from a code bug) and the worker
handoffs that claimed an honest pass. If the workers keep reporting a pass while
the gate fails the **same** way, the contract directs the planner to suspect the
gate/environment and prefer `halt` over ordering yet another identical repair.
`revise_phases` is reserved for a genuine phase-structure error.

`planner_calls_per_run` is a first-class journal metric; every CLI-driven run
carries a `max_planner_calls` backstop AND a deterministic
`max_failed_epochs_per_phase` cap, so a stuck repair loop can never drain the
planner subscription nor spin unattended.

## 9. Versioning

Every payload carries `schema_version` (currently `"1"`). The validator
hard-rejects an unknown version; no silent best-effort parsing. Byte caps are
enforced as UTF-8 bytes in the generated validator; the JSON-Schema `maxLength`
values are the character-level approximation of the same limits.
