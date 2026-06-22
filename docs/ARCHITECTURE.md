# Grindstone Architecture

Grindstone is an **epoch-based deep-work orchestrator**. You hand it a job spec;
a strong cloud planner proposes one small, verifiable epoch at a time; local
workers fan out and grind through the tasks; deterministic checks and disk
contracts gate everything; and the loop repeats until the job's exit criteria
pass. It is Python with Pydantic at every boundary and `mypy --strict` clean.

## The thesis: model proposes, state machine disposes

The differentiator is the loop itself. It is **not** a stateful "leader" model
that owns the run and drifts over a long horizon. It is a fixed, deterministic
state machine with provable invariants (termination, no orphaned tasks, budget
caps). The only model that steers the run, the planner, is a **stateless
one-shot call**: Grindstone reconstructs its full input from durable state every
time, so model drift has nowhere to accumulate. The model *proposes* one decision
as constrained JSON; the state machine *validates and disposes* of it,
re-evaluates every check itself, and never takes a model's word that a task or
phase is done.

## The three roles, each behind a script

Grindstone is a pure orchestrator that knows only **three role names** and reaches
each through a request **script** behind a file contract. It never learns the
transport, model identity, or GPU assignment hiding behind a script; those live
entirely in `models/`.

| role | what it does | shipped default adapter (`models/claude/`) |
|---|---|---|
| **planner** | plans one epoch at a time as a constrained tool-call | `planner_request.sh` → Claude (Opus) via `claude -p`, read-only |
| **worker** | the on-rig grinders that fan out across an epoch's tasks | `worker_request.sh` → Claude (Opus) via `claude -p` in the worktree |
| **senior** | optional escalation / web-research / taste tier | `senior_request.sh` → Claude (Opus) via `claude -p`, web search on |

`models/` is layered as a rig stack: `claude/` is the tracked Claude rig (the
shipped floor) a fresh clone runs with zero setup; `codex/` is a bundled
alternative (a `codex exec` planner, opt in via `grindstone init --rig codex`);
and `personal/` (gitignored) is the operator's personal per-file rig. Resolution
splits on whether a rig is named: an explicit `--rig codex` searches `codex/`
then the `claude/` floor (never `personal/`, so a selected rig is exact and
reproducible), while the implicit default searches `personal/` then `claude/`,
letting the operator's scripts win where present. Either way `init` bakes the
absolute resolved path into `.grindstone/config.yaml`. `stop.sh` +
`_timeout_prefix.sh` are backend-agnostic kill/timeout helpers under `_common/`,
the shared-helper floor under both modes. Two optional scripts back the taste
features when a rig supplies them: `vision_review.sh` (the screenshot-judge gate)
and `codex_polish.sh` (the final-polish pass). All scripts are **reference
adapters**: point them at your own backends, or drop a replacement into
`personal/`. The per-repo `.grindstone/config.yaml` names each role's script plus
its `slots` (per-role concurrency) and `timeout_s`; `planner` + `worker` are
required, `senior` is optional (its absence means a worker-only escalation ladder).

`grindstone/config.py` loads that YAML into a frozen, unknown-key-rejecting
Pydantic object, and refuses any configured `script:` path that does not resolve
under the bundled `models/` dir (a cloned repo's config is attacker-controlled,
and every configured script is executed) unless `GRINDSTONE_ALLOW_REPO_SCRIPTS=1`
opts a trusted repo in.

## The run lifecycle

The spine is `grindstone/run_loop.py` (`run_grind` / `resume_grind`), driving a
stateless planner against the deterministic core:

```
loop:
  input    = stable_head(job, skeleton) + volatile_tail(state, last_epoch, request)
  raw      = planner.plan(input)                         # role script → cloud model
  decision = extract → schema → typed → semantic → position-legality  (re-ask ≤ 2)
  dispatch on tool name:
    propose_skeleton → store the phase skeleton (legal only as the first decision)
    implement | research | review | artifact → run one epoch; its report feeds the next call
    revise_phases    → replace the current phase onward
    escalate_run     → terminal: hand to a human
    complete_run     → re-run the evidence checks deterministically;
                       pass → success, fail → re-ask with the failing evidence
```

### Skeleton and phases

The first decision is always `propose_skeleton`, which lays down a 2–10 phase
skeleton. Each phase carries a deterministic `exit_criterion` and an
`epoch_budget`. On every loop pass, once a skeleton exists, the core freshly
evaluates the current phase's exit criterion **against the integration tip** in a
throwaway worktree; all checks passing advances the phase (the last phase passing
does *not* auto-complete; the planner still owns `complete_run`). A phase whose
budget is exhausted with its criterion still failing fires a one-shot **phase
escalation**, after which only `revise_phases` / `escalate_run` are legal until
it clears.

### Epochs and tasks

Each work decision opens one **epoch** (`grindstone/epoch_loop.py`): 1–8
independent tasks fan out on a bounded thread pool, each running the single-task
state machine in `grindstone/task_loop.py`:

- **implement** tasks each grind in a fresh per-attempt git worktree branched
  from the epoch base. On a successful handoff the **core** (never the model)
  commits the worktree and checks that every committed path falls inside the
  task's `file_ownership` globs. Because ownership is pairwise-disjoint and
  scope-checked, integration is fast-forward merges in task order that *cannot
  conflict*; any conflict is treated as a structural bug and aborts the epoch.
- **research / review / artifact** tasks run in a plain run-dir scratch directory
  with no worktree and no git: a non-write task is never handed the live repo as
  its CWD. They publish their `artifact_out` to the keyed log.

A task gets up to three attempts on its starting tier, then one attempt per
higher ladder rung; exhausting the ladder marks it FAILED and the epoch
continues. The planner picks the *mode*; the core maps mode → starting tier
(`research`/`review`, and any `visual` epoch, start on the senior tier when one
exists; everything else starts on the worker tier).

### The handoff disk contract

Every worker writes `handoff.json` **in its own CWD** and self-validates it.
Grindstone relocates that file to the task's log key and re-validates it from
scratch: schema → typed parse → semantic rules → a **grounding spot-check** that
every cited `{file, line}` actually exists → a re-run of the task's `done_when`.
Only a `DONE` handoff whose checks all pass is accepted; any failure deletes the
relocated record (zero dead artifacts) and re-queues the attempt. **Stdout is
never parsed**; the disk file is the only result channel.

The worker owns a narrow lane: it edits ONLY files within its `file_ownership`
(or, for a non-write task, only its CWD). Grindstone (the core) owns all git
staging and committing and may keep its own bookkeeping files in the tree, so the
worker must NOT git-commit, must NOT touch orchestration files, and "working tree
clean" is grindstone's concern, never the worker's. Because a *missing*
`handoff.json` would otherwise make grindstone retry blind, the role scripts
guarantee one: if the agent exits (out of turn budget, a crash) without writing a
handoff, the script synthesizes a schema-valid `FAILED` handoff carrying a
diagnosis and a tail of the agent logs, and exits 0 so the core consumes it as a
reasoned failed attempt. A genuine infra failure (rate limit / 429) is exempt: it
keeps propagating a non-zero exit so the transport raises `RateLimited`.

### Deterministic gates

`run_loop.evaluate_checks` is the single evaluator behind both phase exit
criteria and `complete_run` evidence: command checks run in a tip worktree,
`artifact_exists` checks resolve against the keyed log, and `vision_review`
checks render a verdict (below). `complete_run` is never trusted on the planner's
word; its `evidence` is re-run deterministically and the completion is rejected
(and re-asked) if anything fails.

## Gate rebalance: three verification sources

Three consecutive dogfood failures were the same shape: a planner-authored
`done_when` failed for an *environmental* reason while the work itself was
correct (a build gate re-run with no `node_modules`, a `test -f package-lock.json`
on a partial attempt, a `rg`-based content-grep returning exit 127 because
ripgrep is not a host binary). The planner is a poor author of verification
commands, so it stops authoring them. Verification now splits into three sources
by *who owns each*, and a failure is classified before it is charged to anyone.

1. **The deterministic floor** (repo config + core invariants). The floor is
   owned by the repo and the core, never the planner. The **core invariants** run
   on every gate: the worktree is clean (nothing written outside a task's
   `file_ownership`), `handoff.json` is present and schema-valid, and the work is
   actually committed on the task branch. The repo's own canonical commands live
   in the `floor:` config block (below) and are re-run in the eval worktree
   *after* `prepare` materializes dependencies, with the same pass/fail semantics
   as a `done_when` (exit 0 == pass, captured output surfaced on failure). A
   floor-check failure fails the gate exactly like a failed exit criterion. The
   planner never restates the floor.
2. **Structural `checks`** (per task, authored by the planner). Deterministic
   *structural* facts only: a project build / test / type-check command's exit
   code, or `test -f` file existence. A **content-grep** (`rg` / `grep` / `egrep`
   / `fgrep` / `ag` / `ack` for a token, in any pipeline segment) is **rejected**
   by the planner validator, the brittle proxy class is deleted at the source.
3. **Natural-language `criteria`** (per task). Prose acceptance statements ("the
   plan maps every Honey/Sky/Pink/Ink ramp to a React Native equivalent"). No
   commands; these feed the agentic pass below.

### The end-of-epoch agentic verification pass

After every task in an epoch clears its deterministic floor and the epoch would
otherwise complete, *if* the epoch carries any `criteria` and a worker tier is
wired, the core runs one **adversarial** verification pass on the **worker** tier
(`grindstone/verify.py`). It runs in a worktree of the epoch's integration tip
with dependencies materialized, is a *separate* invocation from the worker (given
only the epoch goal + criteria + the produced artifacts, told to find gaps and
**default to FAIL** on uncertainty), and its only output is `verdict.json`, a
re-read disk contract the core re-validates with Pydantic (`EpochVerdict`); stdout
is never parsed, the same pattern as `vision_review`. The pass **can only fail an
epoch the floor already cleared, never rubber-stamp past it**: a verdict that
cannot be produced (the transport raised, no file, an invalid one) is itself a
fail-safe FAIL. A `pass=false` verdict's `gaps` become a `FailedEpochInfo` and
route through `handle_failed_epoch` (the same machinery as a task-failure epoch),
so the planner sees the unmet criteria and disposes: `retry` with the gaps as
feedback, `escalate_senior`, or `halt`. The pass is gated by `verify_epochs`
(default on) and the senior `review` epoch stays the deeper phase/run-level pass;
this worker pass is the cheap per-epoch semantic filter.

### The automatic infra-repair loop

Before each boundary the core re-evaluates the current phase gate *infra-aware*.
A failed `cmd` check is run through the shared classifier (`grindstone/infra.py`,
the single source of truth used by both the gate evaluator and the task-loop
`done_when` re-run, so the two can never drift), which is deliberately
**conservative**: a check is INFRA only on exit 127 or a narrow environmental
signature (`command not found`, `Cannot find module`, `ModuleNotFoundError`, an
`npm`/`pip`/`cargo` install failure), and a plain `exit 1` carrying ordinary test
output stays *semantic* so a real assertion failure is never mistaken for infra.

When the gate is infra-failing and an `infra_repair:` policy + a senior tier
exist, the core does **not** charge the worker or open a semantic failed epoch.
It auto-dispatches a **senior** infra-repair against a worktree of the gate tip
(a focused brief: the failing commands, their captured output, and the host
guard), told to make the environment satisfiable *without* rewriting application
logic. The core (never the model) commits the edits and re-runs only the failing
commands against the repair commit, the authoritative judge, never the senior's
handoff. A repair that sticks is adopted as the new integration tip so the
ordinary gate that follows now passes; the run proceeds with no worker charged.
The loop is bounded by `infra_repair.attempts`; on exhaustion the run escalates to
a human, naming the unsatisfiable command. A **host-command guard**
(`allow_host_commands`) keeps repo-local fixes (an `npm install` landing in
`package.json`, editing config inside the repo) fully automatic while host-level /
privileged actions (`sudo`, `apt`, writes outside the repo) are **deny by
default**; the allowlist is carried into the repair dispatch and surfaced in the
prompt.

### The new config blocks

All three are optional blocks in `.grindstone/config.yaml`; absent, every
existing run is byte-unchanged.

- **`floor:`** the repo's canonical verification commands.
  ```yaml
  floor:
    checks:
      - "npx tsc --noEmit"
      - "npm test --silent"
  ```
  `checks` may be **empty** (a fresh project starts with a minimal floor and grows
  it); an empty *command string* in the list is a config typo and is rejected.
  Absent (`None`) means only the core invariants apply.
- **`infra_repair:`** the automatic senior infra-repair policy.
  ```yaml
  infra_repair:
    attempts: 2            # repair cycles per gate (>= 0; 0 disables auto-repair)
    allow_host_commands:   # the host-command guard, deny-by-default (empty)
      - "apt-get"
  ```
  `attempts` defaults to **2** (0 disables auto-repair, so an infra fail escalates
  immediately). `allow_host_commands` defaults to **empty** (nothing host-level
  allowed). Absent (`None`) means no auto-repair, an infra fail routes through the
  ordinary failed-epoch path.
- **`verify_epochs:`** a bool (default **`true`**) toggling the end-of-epoch
  agentic pass. The pass never runs (and never errors) when an epoch has no
  `criteria` or there is no worker tier; set `false` to disable it entirely (the
  deterministic floor + planner `review` epochs still gate).

## The run-dir layout

All run state lives under `.grindstone/runs/<run-id>/` in the **target** repo
(`grindstone/rundir.py`). Log keys *are* relative paths under this dir, guarded
so nothing resolves outside it:

```
.grindstone/runs/<run-id>/
  events.ndjson        append-only journal, the durable source of truth
  run_state.json       run-level cursor (RunState): skeleton, phase, counters
  state.json           in-flight epoch cursor (EpochState): per-task status
  journal.md           human-facing markdown post-mortem (derived; latest run only)
  P1/E1/T1/handoff.json …   the keyed log: relocated handoffs, outcomes, artifacts
  worktrees/           throwaway per-attempt + integration worktrees
  artifacts/           scratch CWDs for non-write tasks
  worker_logs/         per-worker stdout/stderr
  vision/, polish/     scratch for the taste gates
```

The **journal** (`grindstone/events.py`, `grindstone/journal.py`) is the
backbone: a frozen vocabulary of Pydantic events (`run_started`, `phase_passed`,
`epoch_started`, `task_done`, `handoff_rejected`, …) written one-per-line,
flushed and fsynced, with strictly monotonic `seq`. `replay()` folds the event
stream into a run → phase → epoch → task tree; the same fold powers the live
`watch` TUI and the post-mortem `journal.md`. The event stream alone is
sufficient to render the whole run; `journal.md` is a derived view that carries
no trust and is never read back into the loop.

## Repo-map: navigating large repos

A job spec and a usually-empty repo-memory digest are not enough to plan against
a thousand-file codebase. `grindstone/repomap.py` builds a structural map of the
**target** repo on demand: it extracts definition and reference tags per file
with tree-sitter, ranks files and symbols with a PageRank over the def/ref graph,
and renders the spine (the most-referenced files and their key symbols) to a
token budget.

- **The planner** gets a whole-repo map, built once per planner boundary against
  the current integration tip. It rides the **volatile tail** of the planner
  input, never the byte-stable head, so prefix caching is unaffected and the map
  always reflects the latest committed code.
- **Each worker** gets a personalized **subtree**: the same PageRank with a
  restart distribution seeded on the task's own files (an implement task's
  `file_ownership`, a review or research task's `targets`), which collapses the
  map to the neighborhood the worker will touch.

The map is an enhancement, not a gate, and is built to disappear quietly. A repo
below a small file-count threshold is skipped entirely, so demos and small repos
see no map and pay nothing. Any failure (a missing grammar, an unparseable file,
a read-only repo) returns no map rather than raising, so a run proceeds
identically whether or not the map built. The tag cache lives under
`.grindstone/`, keyed on each file's mtime and size so warm rebuilds are
sub-second, with an in-memory fallback when the repo is read-only.

PageRank is a small pure-Python power iteration, so the feature pulls in no
numpy/scipy/networkx stack; it depends on tree-sitter (with grammars for Python,
JavaScript/TypeScript, Go, Rust, Java, C, C++, C#, Ruby, Dart, and Swift),
tiktoken for token counting, and diskcache for the tag cache. The tree-sitter tag
queries under `grindstone/_repomap_queries/` are pure data vendored from aider
(Apache-2.0; see the attribution there).

## Taste and vision features

Grindstone can build and judge work that is evaluated by how it *looks*:

- **The `visual` flag** routes a UI/visual epoch's build to the **senior** tier
  (the stronger taste-builder) instead of the worker default.
- **The `vision_review` gate** is a deterministic phase check: after a `cmd`
  check builds and screenshots the UI into the tip worktree, a `vision_review`
  check shows that screenshot to a vision model (via the rig's `vision_review.sh`)
  with criteria for "what polished looks like". The script writes a
  `vision_verdict.json` that the core re-reads and validates (a disk contract,
  never stdout, `grindstone/script_vision.py`); a failed taste verdict fails the
  phase exactly like a failed command. The gate is "always fail, never crash":
  any error degrades to a deterministic FAIL.
- **The optional final-polish pass** (`grindstone/script_polish.py`, off unless
  the config opts in): after a run's `complete_run` evidence passes, `codex` runs
  in `workspace-write` mode against a throwaway worktree of the final branch and
  edits it in place. The edits are **kept only if the same evidence still
  passes**, otherwise discarded, leaving the original completion standing. They
  are committed to a branch but **never auto-pushed**, and the pass can never turn
  a completed run into a failure.

## Durability and resume

The run is fully resumable after a kill at any point (`resume_grind`), and the
journal leads. `RunState` and the epoch's `EpochState` are atomically rewritten
to *distinct* files, so the multi-epoch loop and the in-flight epoch never clobber
each other.

- A kill **while a planner call is in flight** leaves nothing on disk (planner
  calls are side-effect-free), so resume simply re-issues the call, no work
  burned.
- A kill **mid-epoch** burns only the single in-flight worker attempt (a killed
  worker cannot be trusted): its worktree and branch are torn down and the attempt
  is abandoned, while DONE tasks keep their branches and handoffs and are never
  re-run. Integration then finishes idempotently: already-merged branches are
  ancestors of the integration branch and are skipped.

The keyed log, the atomic state files, and the append-only journal together mean
every transition is recoverable from disk, and a run that ends abruptly can always
be re-entered exactly where it left off.
