# Grindstone Architecture

Grindstone is an **epoch-based deep-work orchestrator**. You hand it a job spec;
a stateless planner proposes one small epoch at a time; workers fan out and grind
through the epoch's tasks in isolated git worktrees; an independent critic triages
each task; and the loop repeats until the planner declares the job done. It is
Python with Pydantic at every boundary and `mypy --strict` clean.

It is a subagent that does not lose context, does not have to be babysat, and uses
cheap local compute when the work is checkable. Nothing more.

## The thesis: model proposes, state machine disposes

The differentiator is the loop itself. It is **not** a stateful "leader" model that
owns the run and drifts over a long horizon. The only model that steers the run,
the planner, is a **stateless one-shot call**: each boundary Grindstone
reconstructs its full input fresh from durable disk state, so context is
re-derived, never accumulated, and no window rots or goes quadratic. The model
*proposes* one decision as constrained JSON; the state machine *validates and
disposes* of it.

Python here is a pure function of disk state plus a handful of enum/boolean
signals, mapping to the next node. It makes **no quality judgment**. It owns
exactly:

- the synchronous state machine (the epoch loop, the two failure nodes, resume);
- **two deterministic invariants** and nothing more (the disjoint-ownership merge
  of the git diff, and one final acceptance running the job's own `done_when`
  once);
- the durable append-only event log, rebuild-from-disk, and crash-only resume;
- assembling the planner's bounded fresh-from-disk context (it **lists**
  references, never summarizes them);
- the trust boundary (the untrusted worker confined to its worktree, host-global
  mutations only from the trusted planner's declared list, and an RCE guard on
  every configured repo script);
- infra (spawning rig subprocesses, the worktree lifecycle, per-backend
  semaphores, relocating artifacts and handoffs, rendering the journal).

Every other judgment (is this code good, is this research grounded, is this task
done or blocked or retryable) is delegated to a model.

## The five properties that define it

1. **Externalized context.** No model holds the whole history. The memory is the
   git tip plus the durable keyed log plus the event log. Each epoch reconstructs
   a bounded, fresh window from disk: the job, the integration-tip file list, the
   keyed-log index, and the prior epoch's carried failures. The planner is
   stateless per call.
2. **Local when it can.** Per-task tier: `local` (Qwen) for mechanical or
   checkable work, `senior` (Claude) for judgment and taste. The planner picks the
   tier; Python just maps it to a rig.
3. **Right skills.** A repo-owned domain-skill catalogue, selected per task.
   Retrieve, do not concatenate.
4. **Verifiable checkpoints.** Every step is gated by exactly two deterministic
   invariants plus an agentic review. The worker writes a free-form report the
   critic reads; the state machine never parses or schema-gates it. Deterministic
   facts (git, file existence) and one lenient verdict are the only things Python
   disposes on, so a run can be trusted and left alone.
5. **Parallel fan-out.** Multiple workers per epoch on disjoint file ownership.
   The throughput win and the local-GPU leverage.

## The three roles, each behind a script

Grindstone is a pure orchestrator that knows only **three role names** and reaches
each through a request **script** behind a file contract. It never learns the
transport, model identity, or GPU assignment hiding behind a script; those live
entirely in `models/`.

| role | what it does | shipped default adapter (`models/claude/`) |
|---|---|---|
| **planner** | proposes one epoch at a time as constrained JSON | `planner_request.sh` -> Claude (Opus) via `claude -p`, read-only |
| **worker** | the on-rig grinders that fan out across an epoch's tasks | `worker_request.sh` -> Claude (Opus) via `claude -p` in the worktree |
| **senior** | optional judgment / taste / synthesis tier | `senior_request.sh` -> Claude (Opus) via `claude -p`, web search on |

There is **one executor role**: a `senior`-tier task reuses the same worker
prompts (the tier only selects which rig script runs it). The critic is dispatched
on the task's own tier.

`models/` is layered as a rig stack: `claude/` is the tracked Claude rig (the
shipped floor) a fresh clone runs with zero setup; `local/` is a bundled all-Qwen
rig; and `personal/` (gitignored) is the operator's personal per-file rig.
Resolution splits on whether a rig is named: an explicit rig (e.g. `--rig local`)
searches `[rig, claude, _common]`, never `personal/`, so a selected rig is exact
and reproducible; the implicit default searches `[personal, claude, _common]`,
letting the operator's scripts win where present. `_common/` (e.g. `stop.sh`) is
the shared-helper floor under both modes. The per-repo `.grindstone/config.yaml`
names each role by a bundled `rig:` name OR an explicit `script:` path, plus its
`slots` (per-role concurrency) and `timeout_s`; `planner` + `worker` are required,
`senior` is optional (its absence falls every tier back to the worker rig).

`grindstone/config.py` loads that YAML into a frozen, unknown-key-rejecting
Pydantic object, and refuses any configured `script:` path that does not resolve
under the bundled `models/` dir (a cloned repo's config is attacker-controlled,
and every configured script is executed) unless `GRINDSTONE_ALLOW_REPO_SCRIPTS=1`
opts a trusted repo in. The core ships no rig-specific defaults: an absent config
is a hard error toward `grindstone init`.

## The run lifecycle

The spine is `grindstone/loop.py` (`start_run` / `resume_run`), a synchronous,
deterministic loop. There are **no phases, no skeleton, no exit criteria, and no
epoch budget**: the stateless planner self-steers, one epoch at a time, until the
job is met.

```
loop (until the planner ends or the max-epochs backstop fires):
  context  = job + integration tip (file list) + keyed-log index + carried failures
  decision = planner.decide(context)          # ONE typed Decision, self-validated on disk
  if decision is END:
    run the one final acceptance (job done_when) ONCE
    pass -> completed ; otherwise -> a clean partial-end (resumable)
  else (an EPOCH of 1..8 disjoint tasks):
    run the planner-declared setup commands (the trusted host-mutation seam)
    fan the tasks out (tier-routed, each in its own worktree):
      worker grinds -> writes a free-form handoff.md report in its CWD
      deterministic gate (in-scope commit, or the artifact exists)
      tier-matched critic reads the report + the diff/artifact -> a lenient verdict
        PASS -> merge-ready ; RETRY -> bounded same-tier retry ; ESCALATE -> the planner
    integrate the PASSing implement tasks by the disjoint-ownership merge invariant
    fast-forward the durable run branch (only on epoch completion)
    carry any non-merged outcome forward as context for the next boundary
```

### Epochs and tasks

A **decision** is one of two shapes (`grindstone/contracts/models.py`): an
**epoch** (a titled bundle of 1 to 8 independent tasks) or an **end** (a summary
that seeds the next appendable run). Each task carries an `id` (`T1`..`T8`), a
`mode` (`implement` / `research` / `review` / `artifact`), a routing `tier`
(`local` default, `senior` for judgment), a prose `goal` that states its own
notion of done, and the disk-shape fields the orchestrator needs to isolate and
merge it (`file_ownership` for implement, `artifact_out` for the rest, plus
optional `skills` and `inputs`). There is no rigid acceptance schema: semantic
acceptance is judged agentically by the critic against the task's own claimed goal.

Tasks fan out concurrently (`grindstone/loop.py`), bounded by the per-backend
semaphores (below), not a global pool:

- **implement** tasks each grind in a fresh per-attempt git worktree branched from
  the epoch base. After the grind the **core** (never the model) commits the
  worktree and deterministically checks that every committed path falls inside the
  task's `file_ownership` globs; a zero-diff or out-of-scope commit is a failed
  attempt. A passed epoch fast-forwards one durable run branch, `grind/<run-id>`
  (the only ref that survives a boundary); per-attempt and per-epoch staging
  branches all live under a transient `grind-wip/*` namespace and are pruned once
  their work is absorbed.
- **research / review / artifact** tasks run in a plain run-dir scratch directory
  with no worktree and no git: a non-write task is never handed the live repo as
  its CWD. They publish their `artifact_out` to the keyed log. A non-write task is
  given a read-only checkout of the integration tip (`_read_tip`) to read and cite,
  so it sees what prior epochs built, not the stale base.

A task gets a small bounded **same-tier retry** (`MAX_ATTEMPTS`, currently 2);
there is **no tier escalation**. A retry that the critic asked for inherits the
prior attempt's committed work-in-progress (chained off the prior wip branch); an
out-of-scope write poisons its worktree and the retry instead restarts clean. When
the retries are exhausted, the task escalates and becomes context the planner
handles next boundary.

### The two deterministic invariants

Python disposes on exactly two deterministic checks. Everything else is grounded
and judged agentically.

1. **Disjoint-ownership merge** (`_integrate`, `_ownership_overlap`). Parallel
   implement tasks declare the files they own; the core enforces that each wrote
   only within its declared globs and that the realized ownership is pairwise
   disjoint, then fast-forward-merges the passing wip branches (in task order) onto
   a staging branch and fast-forwards the durable run branch to it. An ownership
   overlap or a merge conflict aborts integration as a hard error (carried to the
   planner, the run branch left untouched). This is the one check that prevents
   silent corruption. The artifact analogue is enforced at parse time: two tasks
   may not declare the same `artifact_out`.
2. **One final acceptance** (`make_acceptance`). When the planner emits END,
   Grindstone checks out the integration tip in a throwaway worktree and runs the
   job's own `done_when` **once**: exit 0 means the run is `completed`; any other
   exit makes the END a clean partial-end (`ended`) whose summary seeds the next
   run. This is deliberately the **only** deterministic build gate; it exists so
   "done" still means something when every per-epoch check is agentic. When no
   `done_when` is configured, the planner's END is trusted.

There is **no per-epoch build gate**. Intermediate red is by design: epoch 1 may
write a module epoch 2 will build against. Only judgment can tell "incrementally
incomplete" from "broken", so between-epoch build-health is carried as context the
critic notes and the next epoch resolves, never a deterministic gate.

### The critic: triage, not grade

A gate-clean attempt always runs the **tier-matched critic** (an independent
agentic pass that did not write the work). It **routes**, it does not grade,
emitting a lenient `Verdict` (an `outcome` enum plus a free-text reason):

- **PASS** (including good-enough-with-notes): merge; notes carry forward. Minor
  imperfections are carried information, not a gate. The critic biases here when
  unsure.
- **RETRY**: a defect the **same** worker can plausibly fix (a typo, a wrong value,
  a missing piece). The bounded same-tier retry.
- **ESCALATE**: anything the worker cannot fix on its own (a missing dependency, an
  ambiguous or wrong spec, a decision needed, an environmental blocker). Routes to
  the planner.

The retry-vs-escalate split is one question: "can the same worker plausibly fix
this itself?". A research / review critic additionally verifies that the artifact's
citations resolve to real files under the read tip; an ungrounded claim routes to
RETRY or ESCALATE.

A worker that hits a hard environmental blocker says so in its free-form
`handoff.md`. The critic reads that report, and an unrecoverable blocker becomes a
critic ESCALATE. There is **no separate Python BLOCKED gate**: collapsing blocked
into the critic's ESCALATE keeps one judge of "is this done / blocked / retryable"
and avoids trusting a self-declared status. A missing or invalid verdict is a
fail-safe escalate, never a silent pass.

### The handoff disk convention (not a schema gate)

Every worker writes a **free-form** `handoff.md` report in its own CWD: plain prose
(what I did, what is done, what is blocked or unfinished, which files I touched,
grounding as prose). Grindstone relocates it verbatim to the task's log key for the
critic and the planner's optional context. It is **never parsed or schema-validated
by Python**; stdout is never parsed either. The deterministic gate is the committed
diff or the produced artifact, not this file. A missing or pathologically large
report is fine: the critic judges the actual work.

The worker owns a narrow lane, stated in the worktree-isolation contract: it edits
only files within its `file_ownership`, writing every path relative to its CWD,
never to an absolute path and never outside the worktree (a worker that strips its
CWD back to the repo root must not be able to reach the real checkout). The core
owns all git staging and committing; the worker must not commit and must not touch
orchestration files. The critic's verdict is a `verdict.json` written in the
critic's CWD, relocated and re-validated with Pydantic (a re-read disk contract,
stdout ignored).

## The safety boundary

Review is post-hoc and sees the **diff**: it flags unsafe *code*, not unsafe
*actions* already taken during execution. For an untrusted local worker, "be
lenient, install what you need" equals arbitrary code execution on the host, and
review cannot un-run it. Therefore:

- The **planner** (the trusted tier) declares any **host-global** setup/install
  commands in the epoch's `setup` list; the orchestrator runs them, in order,
  before the tasks, in a throwaway checkout of the epoch base (torn down after, so
  setup can never dirty the operator checkout). The untrusted worker never
  improvises host mutations.
- Project-**local** dependency installs (`npm ci`, `pip install`) do **not** go in
  `setup`: that throwaway checkout is not the task worktrees, so an install there
  would not reach them. An implement task installs the project deps it needs inside
  its **own** worktree as part of its work.
- Worktree isolation contains every worker file write.

The principle: fully agentic on **judgment** (is this code good), a hard boundary
on **actions** (what may touch the host). This replaces an entire infra-repair
state machine with one field in the decision.

## The failure model: two nodes

Every interruption routes to exactly one of two handlers.

1. **Rate limit / quota** (on the planner, the worker, or the critic): **park**,
   back off (about once an hour, injectable), then re-enter. A planner rate-limit
   re-issues the boundary call; a mid-epoch worker rate-limit razes the in-flight
   epoch's throwaway worktrees and **restarts the epoch whole** (partial state is
   never trusted).
2. **Cannot continue** (any other epoch failure): the failure becomes carried
   context the planner sees next boundary and steers around, or the planner ends
   cleanly by writing a summary (the resume seed). The `max_epochs` backstop is the
   **involuntary** trigger of the same clean end, so a planner that spins without
   progress is always bounded. An unexpected error escaping the worktree or
   integration machinery is razed and carried; a run of consecutive such aborts
   clean-ends the run rather than looping forever.

There is no infra-repair node, no session-limited node, no worker-timeout node, and
no tier-escalation state machine. A hung worker is just a task failure that routes
to node 2.

## The run-dir layout

All durable run state lives under `.grindstone/runs/<run-id>/` in the **target**
repo (`grindstone/rundir.py`). Log keys *are* relative paths under this dir,
guarded so nothing resolves outside it:

```
.grindstone/runs/<run-id>/
  events.ndjson        append-only journal, the durable source of truth
  journal.md           human-facing markdown post-mortem (derived; latest run only)
  P1/E1/T1/handoff.md  the keyed log: relocated free-form handoffs, verdicts, artifacts
  artifacts/           scratch CWDs for non-write tasks
  logs/                per-worker / per-critic raw stdout (ephemeral, reaped each epoch)
  _planner_tip/        the orchestrator-managed planner read/write tip checkout
```

The model-written executor worktrees (task attempts, staging) and the
orchestrator's scratch trees do **not** nest under the run dir: they live on an
external base, `/tmp/cache/grindstone/<repo-id>/<run-id>/worktrees`
(`GRINDSTONE_WORKTREE_BASE` to relocate; `rundir.worktrees_root`). A worktree
nested inside the target repo would let a worker that strips its CWD back to the
repo root write into the main checkout instead of its isolated worktree, so hosting
them externally removes the nesting the strip relies on. Only the
orchestrator-managed `_planner_tip` checkout stays under the run dir, since the
sandboxed planner rig must reach it inside the repo and it is never model-written.

Because the durable run branch fast-forwards **only** on epoch completion, its tip
is always at a clean boundary. The raw stdout per task is hundreds of megabytes of
pure debugging scratch, so each epoch's start **reaps** the prior epoch's raw logs
(keeping only the latest epoch's), while the small keyed log and the event journal
are kept forever.

## The journal and resume

The **journal** (`grindstone/events.py`, `grindstone/journal.py`) is the backbone:
a frozen vocabulary of Pydantic events (`run_started`, `epoch_started`,
`task_dispatched`, `handoff_accepted`, `handoff_rejected`, `verdict`,
`epoch_carried`, `epoch_completed`, `run_completed`, `run_ended`, `rate_limited`,
`run_resumed`) written one per line, fsynced, with strictly monotonic `seq`. The
event stream alone is sufficient to render the whole run -> epoch -> task tree;
`journal.md` is a derived view that carries no trust and is never read back into the
loop.

**Resume is the universal crash-only recovery primitive.** Because the run branch
only fast-forwards on completion, the git tip needs no rewind. On resume from a
non-ended epoch, programmatically (with no planner in the cleanup): remove the
run's worktrees and transient `grind-wip/` branches, reap the incomplete epoch's
partial keyed log and raw logs, **preserve** the completed-epoch keyed log and the
append-only journal (appending a "razed incomplete epoch" marker, never
truncating), reconstruct the carried context from the journal, and re-enter the
loop at the planner prompt from the last clean boundary. Every interruption (kill,
rate-limit, crash, any unhandled case) recovers the same way: resume = cleanup +
re-plan, never a rewind.
