# BONES: the minimal Grindstone

A spec for the from-scratch rewrite. Copy the bones, rebuild the loop thin,
delete everything else. Target: under 10k lines all-in (core + tests).

## North star

A long-running agent whose context lives ON DISK and is rebuilt fresh each
epoch, that fans work to local workers when it can, pulls the right skills, and
emits verifiable checkpoints so you can trust it unwatched.

It is a subagent that does not lose context, does not have to be babysat, and
uses cheap local compute when the work is checkable. Nothing more.

### The five properties that define it (and nothing else)

1. EXTERNALIZED CONTEXT. No model holds the whole history. The memory is the git
   tip + handoffs + the event log. Each epoch reconstructs a bounded, fresh
   window from disk (job + current repo state + a digest of what is done). The
   planner is stateless per call. This is the real meaning of "does not lose
   context": context is re-derived, never accumulated, so no window rots or goes
   quadratic.
2. LOCAL WHEN IT CAN. Per-task tier: local (Qwen) for mechanical or checkable
   work, senior (Claude) for judgment and taste. The planner picks the tier.
3. RIGHT SKILLS. Repo-owned domain-skill catalogue, selected per task. Retrieve,
   do not concatenate.
4. VERIFIABLE CHECKPOINTS. Every step produces a checked artifact (the handoff
   disk contract) gated by two deterministic invariants + an agentic review, so
   the run can be trusted and left alone.
5. PARALLEL FAN-OUT. Multiple workers per epoch on disjoint file ownership. The
   throughput win and the local-GPU leverage.

## The state machine (epochs only, no phases)

    loop:
        planner sees {job, integrated tip, digest of done work} -> proposes an
            epoch (1..N tasks, each with disjoint file ownership) OR emits done
        if done: run final acceptance once; exit
        for each task (fan out, tier-routed, in its own worktree):
            worker grinds -> writes handoff.json in its CWD
            review (agentic critic, re-derives) -> pass | fail-with-gaps
            on fail: retry same tier -> escalate tier -> abort after K
        integrate passing tasks: disjoint-merge check, then merge to run branch
        next epoch

No phases. No floors. No epoch budget. The planner self-steers to done.

## Bones to COPY (already minimal and correct; port near-verbatim)

- Handoff disk contract: worker writes `handoff.json` in its CWD, orchestrator
  relocates + re-validates the copy in the run dir, stdout is NEVER parsed.
  (today: `grindstone/check_handoff.py`, the handoff half of contracts/models.py)
- `grindstone/worktree.py`: git worktree create + run branch + merge + the
  disjoint-ownership check. Keep the external-base location (rundir.worktrees_root
  lesson: worktrees outside the repo so a worker cannot strip CWD to the repo).
- `models/`: the per-backend rig scripts (claude / local) behind the file
  contract, selected by per-role `rig:` config.
- Skills: the domain-skill catalogue (`<repo>/.grindstone/skills/` + index.md)
  and the per-task selection function. NOT the operating-skill scenario split.
- Pydantic models for the wire contracts (the decision the planner emits, the
  handoff the worker writes). Lenient verdict (see DROP, below).
- The event log: append-only `events.ndjson` as the single source of truth;
  resume can be deferred but the log cannot.
- The planner / worker request SHAPE (input on disk, self-validate, decision.json
  out), minus the phase machinery.

## Components to BUILD thin (face a blank page; do not port)

- The epoch driver (replaces run_loop.py 2463 + epoch_loop.py 750): ~400 lines.
- Task exec (replaces task_loop.py 1573): worktree + dispatch + collect handoff.
- The review step: dispatch an independent critic, read a LENIENT verdict.
- Planner input construction: integrated tip + result digest + file-tree. The
  planner PULLS the repo-map (and grep/read) on demand; not pushed into every call.

## DROP (everything else)

- Phases, `exit_criterion` floors, `epoch_budget`, phase-complete grounding.
- Per-task convergent verification with the rigid `EpochVerdict` JSON schema.
  REPLACE with a lenient agentic verdict (prose + a clear pass/fail). The rigid
  schema is a live failure source: run 051645Z T1, the local verifier emitted a
  schema-invalid verdict.json (`extra_forbidden`) and the task was rejected for a
  MACHINERY fault, not a work fault. Senior/Claude passed the same schema; the
  brittleness is forcing a weak model into a strict structure.
- Infra-repair subsystem (auto-dispatch repair epoch, `prepare:` materialization,
  host allowlist). REPLACE with: the PLANNER declares setup/install commands.
  Rationale below.
- Polish epochs, decomposition/size gate, operating-skill scenario split, TUI.
- Repo-map: NOT orchestrator-injected. Kept as a standalone CLI bone (repomap.py)
  the planner/worker RUN on demand when they need to navigate (the prompt mentions
  it). Pull-not-push: computed only when needed, scoped by the model (whole repo or
  subtree), ~0.4s warm on 1k files, 2-8k tokens only when invoked. Removes the
  always-on cost AND un-defers the capability. Discipline: the prompt frames it as
  "reach for it when navigating an unfamiliar area", not "every boundary".

## The two deterministic invariants we KEEP (minimal checks, not zero)

1. Disjoint-ownership merge: parallel tasks declare the files they own; the
   orchestrator enforces disjointness + that the worker wrote only what it claimed
   + a clean merge. This is the one check that prevents silent corruption.
2. One final acceptance: when the planner says done, run the job's own done_when
   / build ONCE. Keeps "done" meaning something even when every per-epoch check is
   agentic.

Everything else is grounded and checked agentically.

## Safety boundary (ruling)

Review is POST-HOC and sees the DIFF: it flags unsafe CODE, not unsafe ACTIONS
already taken during execution. For an untrusted local worker, "be lenient,
install what you need" equals arbitrary code execution on the host, and review
cannot un-run it. Therefore:

- The PLANNER (trusted Claude tier) declares setup/install commands; the
  orchestrator runs those. The untrusted local worker NEVER improvises host
  mutations.
- Worktree isolation contains worker file writes.
- Principle: fully agentic on JUDGMENT (is this code good), hard boundary on
  ACTIONS (what may touch the host).

This replaces the entire infra-repair state machine with one line in the decision
schema.

## Line budget (the contract: under 10k all-in)

    Core (excl. models/ shell ~1k):
      epoch driver                  400
      task exec (worktree/dispatch) 600
      handoff contract + relocate   400
      worktree.py                   350
      review step                   250
      planner (input + dispatch)    600
      pydantic models               350
      events + resume               300
      skills (catalogue + select)   200
      config + cli                  450
      core total                  ~3900

    Tests (stochastic-first):     ~4500
    skills/schemas/docs:          ~1500
    all-in:                       ~9900

Today for comparison: core 12,596 / tests 17,638 / total tracked 34,377.

## Testing (from scratch, stochastic-first)

The bugs that bit us were emergent and stochastic (worktree escape, single-slot
contention, the verdict-schema fumble), and 17.6k of unit tests caught none of
them. So:

- CONVERGENCE E2E: run the same small job N times against the live stack, assert
  it reaches done and the output passes acceptance. This is the only thing that
  catches a rubber-stamping critic or a flaky worker.
- INVARIANT unit tests: disjoint-merge, handoff validation, worktree isolation,
  skill selection. A handful, not hundreds.

## Sequencing

1. Current run (051645Z) passes -> push current repo to remote as the fallback
   tag (recoverable restore point).
2. Raze-in-branch is the chosen mechanic (keeps git history + remote + the
   published lineage): tag the machinery version `v0-machinery`, branch `bones`,
   `git rm -rq .`, restore only the bone paths from the tag, commit "raze to
   bones". A `git worktree add ../Grindstone-ref v0-machinery` gives a disposable
   read reference. The discipline is in the razing, not in a separate .git.
3. Copy the bones. Build the loop thin. Stochastic tests alongside, not after.

## Operator decisions (2026-06-24 session)

### Concurrency / fan-out: rewrite simply, do not port
The old model carried the double-duty bug (the epoch pool was sized by worker
slots alone, serializing everything). The clean replacement is a per-BACKEND
semaphore map: `{backend_endpoint: Semaphore(slots)}`, each task acquires its
backend's semaphore. Local (:8080, --parallel 1) gets a 1-slot semaphore, claude
gets N. No global pool, no per-tier ScriptWorker semaphore layer on top. ~20
lines. Keep the capability (local serial, claude parallel, both concurrent across
backends); drop the cli._resolve_concurrency + ThreadPoolExecutor + per-tier
double layer.

### Skills: keep (confirmed)
Domain-skill catalogue + per-task selection from the index. Already in the COPY
list; this is a definite keep, not a maybe.

### Log reaping: durable log small, raw stdout ephemeral
The pi/claude raw stdout is 200-500 MB PER TASK and is pure debugging scratch
(the meaningful output is already captured in the handoff + events + keyed log).
So: keep the top-level keyed log + events forever (small, and cross-run append
needs them); reap the PRIOR epoch's raw worker/planner stdout when the NEXT epoch
starts, keeping only the LATEST epoch's full raw logs. Reaping is tied to the
epoch boundary, not the task. (Optional: keep a small tail, last few KB, of a
reaped log for post-mortem.) This keeps an overnight run from filling the disk.

### Failure model: exactly TWO nodes (everything else routes to #2)
1. RATE LIMIT / quota (on planner, senior, or worker): back off and retry about
   once per hour. Two sub-cases, same handler: at a boundary, just retry the
   planner call; mid-epoch, the in-flight epoch is RESTARTED whole after the
   backoff (partial epoch state is not trusted).
2. CANNOT CONTINUE (ANY other failure in an epoch): the failure becomes context
   the planner sees next boundary; the planner either steers (re-plan around it)
   or ends cleanly by writing a PHASE HANDOFF (the pending-summary = the resume
   seed, so runs stay appendable). The max-epochs backstop is the INVOLUNTARY
   trigger of #2: if the planner itself spins without progress, the cap forces
   the same clean partial-end.

   No infra-repair node, no session-limited node, no worker-timeout node, no
   tier-escalation state machine. A hung worker (timeout) is just a task failure
   that routes to #2. Keep it that simple.

   DECIDED: keep a tiny SAME-tier local retry (1-2x) before a task failure routes
   to #2; NO tier escalation. A task that exhausts its local retries just becomes
   context the planner handles in the NEXT epoch. (Run 051645Z self-healed on retry
   twice for free -- the verdict-schema fumble, then the token-hex fixes -- with no
   planner round-trip; that is the only reason retry stays.)

### Resume is the universal recovery primitive
Every interruption -- kill, rate-limit, crash, any unhandled case -- recovers the
SAME way, because the only trusted checkpoint is the last COMPLETED epoch boundary.
The run branch only fast-forwards on epoch completion, so its tip is ALWAYS at a
clean boundary; mid-epoch worker writes live in throwaway worktrees / transient
wip branches, never on the durable branch.

On resume from a NON-ENDED epoch, programmatically (no planner in the cleanup):
1. rm the run's worktrees (the partial task attempts).
2. delete the transient wip branches of the incomplete epoch.
3. reap the incomplete epoch's raw logs + partial task scratch.
4. PRESERVE the durable keyed log of all COMPLETED epochs (that IS the done-list
   the re-planned epoch reads) and the append-only events journal (append a
   "resumed: razed incomplete epoch E_n" marker, do NOT truncate it).
5. re-enter at the planner prompt from the last clean boundary -> clean restart.

The git tip needs NO rewind: it is already at the last boundary by the
merge-only-on-completion invariant. So resume = cleanup + re-plan, not a rewind.

This collapses the failure handlers into ONE recovery path: #1 rate-limit = resume
WITH a backoff timer (~1/hr); kill or crash = resume immediately; #2 planner-ends
= a CLEAN end (phase handoff), not an interruption, resumable as the next
appendable run.

### Build-health is terminal-only; intermediate red is by-design
A deterministic build gate BETWEEN epochs is not just unnecessary, it is wrong: it
cannot tell "incrementally incomplete" from "broken". Epoch 1 may write screens
that import a nav module epoch 2 will build -- red on purpose. A tsc gate sees red
and fails; only JUDGMENT can distinguish "dependency not built yet, it is in
pending" from "typo, fix it". So between-epoch build-health is agentic-and-carried,
never deterministic-and-gated.
- Intermediate red / dirty state is INFORMATION the critic notes and the next epoch
  resolves, NOT a failure-node trigger (same principle as retry-not-escalate:
  failures are carried, not gated).
- The per-task critic gates on "sound incremental progress / did it do what it
  claimed", NOT on "is it green". Red because a dependency is unbuilt can PASS.
- A worker may still run tsc itself while working -- that is just an agent with a
  shell, not an orchestrator gate.
- Build-health is a TERMINAL concern: the review epoch checks it (prompt-driven),
  backed by the single deterministic final acceptance. OPEN: keep that deterministic
  backstop (parent leans yes, as the verifiable-walk-away trust anchor) or fold it
  into the review prompt.
- The disjoint-merge invariant stays, but it is write-SAFETY (did parallel tasks
  collide), orthogonal to build-health.

### Critic verdict: triage, not grade
The critic ROUTES, it does not grade pass/fail. A binary verdict conflates three
different next-steps, and that conflation is what makes it brittle (failing on minor
mistakes; retrying things a retry cannot fix). Three outcomes, each routing to a
destination the failure model already has:
- PASS (including good-enough-with-notes): merge; notes carry forward. Minor
  imperfections are carried information, NOT a gate. Bias here when unsure.
- RETRY: the bounded local retry. ONLY for a defect the SAME worker can plausibly
  fix (typo, wrong value, missing piece).
- ESCALATE -> planner (#2): anything the worker CANNOT fix on its own (missing dep,
  ambiguous or wrong spec, needs a decision, environmental). A missing dep routes
  to the planner who owns setup, never to a futile worker retry.
The retry-vs-escalate split is ONE question: "can the same worker plausibly fix
this itself?" yes -> retry, no -> planner.

Prompt principles (keep it from becoming a crutch):
1. Anchor on the task's own claimed intent / done_when, not the critic's taste.
   Strict on "did it accomplish what it claimed", lenient on polish and style.
2. Bar = "good enough to build on", not "perfect". When torn between fail and note,
   PASS-with-notes (a false-fail wastes a retry; a noted imperfection is cheap and
   the final review catches what matters).
3. Make "is this worker-fixable?" an explicit step so an unfixable blocker cannot
   masquerade as a retry.

Worker self-report: a worker that hits a hard blocker writes BLOCKED (e.g. missing
dep) in its handoff -> routes straight to the planner, SKIPPING the critic (no point
critiquing env-blocked work). The critic is the backstop for a dishonest or mistaken
DONE. Two honest entry points to the planner: worker-reported-blocked and
critic-escalate.

Verdict FORMAT stays lenient: an outcome enum + a short reason. NOT the rigid
multi-field schema the local model fumbled in run 051645Z.
