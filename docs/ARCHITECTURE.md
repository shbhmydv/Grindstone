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
caps). The only model that steers the run — the planner — is a **stateless
one-shot call**: Grindstone reconstructs its full input from durable state every
time, so model drift has nowhere to accumulate. The model *proposes* one decision
as constrained JSON; the state machine *validates and disposes* of it,
re-evaluates every check itself, and never takes a model's word that a task or
phase is done.

## The three roles, each behind a script

Grindstone is a pure orchestrator that knows only **three role names** and reaches
each through a request **script** behind a file contract. It never learns the
transport, model identity, or GPU assignment hiding behind a script — those live
entirely in `models/`.

| role | what it does | reference adapter (`models/`) |
|---|---|---|
| **planner** | plans one epoch at a time as a constrained tool-call | `planner_request.sh` → a strong cloud model (`codex exec`) |
| **local** | the on-rig grinders that fan out across an epoch's tasks | `local_request.sh` → a local LLM server (Qwen via llama-server, driven by the `pi` CLI) |
| **senior** | optional cloud escalation / web-research / taste tier | `senior_request.sh` → `opencode` (kimi) with web search |

Two more optional scripts back the taste features: `vision_review.sh` (the
screenshot-judge gate) and `codex_polish.sh` (the final-polish pass);
`stop.sh` + `_timeout_prefix.sh` are shared kill/timeout helpers. All scripts are
**reference adapters** — point them at your own backends, or replace them. The
per-repo `.grindstone/config.yaml` names each role's script plus its `slots`
(per-role concurrency) and `timeout_s`; `planner` + `local` are required,
`senior` is optional (its absence means a local-only escalation ladder).

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
does *not* auto-complete — the planner still owns `complete_run`). A phase whose
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
  conflict* — any conflict is treated as a structural bug and aborts the epoch.
- **research / review / artifact** tasks run in a plain run-dir scratch directory
  with no worktree and no git: a non-write task is never handed the live repo as
  its CWD. They publish their `artifact_out` to the keyed log.

A task gets up to three attempts on its starting tier, then one attempt per
higher ladder rung; exhausting the ladder marks it FAILED and the epoch
continues. The planner picks the *mode*; the core maps mode → starting tier
(`research`/`review`, and any `visual` epoch, start on the senior tier when one
exists; everything else starts local).

### The handoff disk contract

Every worker writes `handoff.json` **in its own CWD** and self-validates it.
Grindstone relocates that file to the task's log key and re-validates it from
scratch — schema → typed parse → semantic rules → a **grounding spot-check** that
every cited `{file, line}` actually exists → a re-run of the task's `done_when`.
Only a `DONE` handoff whose checks all pass is accepted; any failure deletes the
relocated record (zero dead artifacts) and re-queues the attempt. **Stdout is
never parsed** — the disk file is the only result channel.

### Deterministic gates

`run_loop.evaluate_checks` is the single evaluator behind both phase exit
criteria and `complete_run` evidence: command checks run in a tip worktree,
`artifact_exists` checks resolve against the keyed log, and `vision_review`
checks render a verdict (below). `complete_run` is never trusted on the planner's
word — its `evidence` is re-run deterministically and the completion is rejected
(and re-asked) if anything fails.

## The run-dir layout

All run state lives under `.grindstone/runs/<run-id>/` in the **target** repo
(`grindstone/rundir.py`). Log keys *are* relative paths under this dir, guarded
so nothing resolves outside it:

```
.grindstone/runs/<run-id>/
  events.ndjson        append-only journal — the durable source of truth
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
sufficient to render the whole run — `journal.md` is a derived view that carries
no trust and is never read back into the loop.

## Taste and vision features

Grindstone can build and judge work that is evaluated by how it *looks*:

- **The `visual` flag** routes a UI/visual epoch's build to the **senior** tier
  (the stronger taste-builder) instead of the local default.
- **The `vision_review` gate** is a deterministic phase check: after a `cmd`
  check builds and screenshots the UI into the tip worktree, a `vision_review`
  check shows that screenshot to a vision model (via `models/vision_review.sh`)
  with criteria for "what polished looks like". The script writes a
  `vision_verdict.json` that the core re-reads and validates (a disk contract,
  never stdout — `grindstone/script_vision.py`); a failed taste verdict fails the
  phase exactly like a failed command. The gate is "always fail, never crash":
  any error degrades to a deterministic FAIL.
- **The optional final-polish pass** (`grindstone/script_polish.py`, off unless
  the config opts in): after a run's `complete_run` evidence passes, `codex` runs
  in `workspace-write` mode against a throwaway worktree of the final branch and
  edits it in place. The edits are **kept only if the same evidence still
  passes** — otherwise discarded, leaving the original completion standing. They
  are committed to a branch but **never auto-pushed**, and the pass can never turn
  a completed run into a failure.

## Durability and resume

The run is fully resumable after a kill at any point (`resume_grind`), and the
journal leads. `RunState` and the epoch's `EpochState` are atomically rewritten
to *distinct* files, so the multi-epoch loop and the in-flight epoch never clobber
each other.

- A kill **while a planner call is in flight** leaves nothing on disk (planner
  calls are side-effect-free), so resume simply re-issues the call — no work
  burned.
- A kill **mid-epoch** burns only the single in-flight worker attempt (a killed
  worker cannot be trusted): its worktree and branch are torn down and the attempt
  is abandoned, while DONE tasks keep their branches and handoffs and are never
  re-run. Integration then finishes idempotently — already-merged branches are
  ancestors of the integration branch and are skipped.

The keyed log, the atomic state files, and the append-only journal together mean
every transition is recoverable from disk, and a run that ends abruptly can always
be re-entered exactly where it left off.
