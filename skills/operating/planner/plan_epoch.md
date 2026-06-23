SCENARIO: plan_epoch. A skeleton exists and no epoch is awaiting failure
disposition. Plan the NEXT epoch within the current phase: choose one of
implement / research / review / artifact (1-8 fan-out tasks), or revise_phases /
escalate_run / complete_run.

- implement tasks carry `file_ownership` globs that must be pairwise DISJOINT
  across the epoch (the merge-correctness mechanism). research/review/artifact
  tasks carry `artifact_out`; review tasks also carry `targets`.
- Choose the mode by the deliverable's DESTINATION, never by its flavor.
  Output the job requires as a COMMITTED file in the repo tree, code, config,
  docs, even prose, is implement work: only implement tasks run in a worktree
  and get committed. Output consumed via the keyed log (an analysis, report,
  or investigation the job does NOT require as a committed file) is research
  or artifact work shipped through `artifact_out`; review judges existing work
  and ships a verdict the same way. Never give a task a worktree its
  deliverable does not need.
- A review epoch must INDEPENDENTLY RE-DERIVE a sample of the claims or verdicts
  it judges and RECONCILE them against the upstream artifact(s) it consumes via
  `inputs`, not merely confirm that the expected sections or fields are present
  (a presence-only review spends a planner call yet catches no wrong answer).
  When a review consumes an upstream artifact, surfacing any contradiction
  between the reviewed work and that artifact is a primary job of the review.
- Tier routing is PER TASK, by whether the task needs JUDGMENT/TASTE. Set
  `"senior": true` on a task whose work is taste composition / layout / polish, a
  synthesis of an approach, or a design-quality judgment, so it is built by the
  stronger senior tier. Leave it false (the default) for MECHANICAL or FACTUAL
  work: scaffolding, tokens, boilerplate, exports, web-search fact gathering, a
  structural review. This applies uniformly to implement, research and review.
  Crucially, SPLIT a mechanical slice from a judgment slice and flag `senior` only
  on the judgment slice, so the senior quota is spent only where it is needed (the
  incident: a whole UI epoch run entirely on the senior burned the quota). Note
  research FACT-gathering (incl. web_search / fetch) is local-capable: only a
  synthesis or a taste/decision call is senior.
- Vision-review (taste gate): a third check `{"vision_review":{"screenshot":
  "<path relative to the eval worktree>","criteria":"<what polished looks
  like>"}}` makes a strong vision model JUDGE a rendered screenshot against
  criteria and emit a pass/fail verdict. Use it ONLY in a PHASE EXIT CRITERION
  for a visual phase: put a cmd check FIRST that builds + screenshots the UI
  into the tip worktree (e.g. `{"cmd":"npm run build && node shot.js
  ui/screen.png"}`), then a `vision_review` of that `ui/screen.png` against the
  design bar. The state machine renders the verdict deterministically (a failed
  taste verdict fails the phase, just like a failed command), it is not a task
  `done_when` (a worker scratch has no renderer/screenshot).
- Domain skills: when a `<domain_skills>` catalogue is shown above, attach the
  relevant skills to a task by listing their NAMES in that task's `skills` field;
  the core delivers the selected skill text into that task's worker. Keep selection
  MINIMAL, name only the skills a task actually needs (none is fine), and never a
  name the catalogue does not advertise (the gate rejects an unknown skill).
- done_when is scoped by mode. research/review/artifact tasks run in a scratch
  dir that is NOT a repo checkout: their done_when must verify the
  artifact itself (e.g. `test -s notes.md` in the task CWD, or an
  artifact_exists key), never repo build/test commands; those can only pass in
  implement tasks or phase exit criteria (run in a checkout of the tip).
- revise_phases means the PHASE STRUCTURE/plan is wrong (wrong milestones, wrong
  exit criteria, a missing or mis-scoped phase), NOT that one epoch's work
  failed. Do NOT use revise_phases to react to a failed epoch, the state machine
  asks you a separate, focused handle_failed_epoch decision for that.
- escalate_run only when you genuinely cannot proceed. complete_run only when
  the whole job is done; its `evidence` checks are re-run deterministically and
  rejected if they fail.

Decomposition is THREE distinct skills, one per level; this scenario is LEVELS 2
and 3. Apply them in order, and keep them separate, the bias and unit of work
differ at each level:

[LEVEL 2: EPOCH] Split a PHASE into epochs (one work decision per call).
- One epoch = one coherent FEATURE or milestone, not a whole phase at once and
  not a single file. Each epoch boundary is a free planner checkpoint plus a
  deterministic gate.
- For an IMPLEMENT phase, the FIRST epoch is an explicit BASELINE DEPENDENCIES
  epoch: stand up the project skeleton and produce the COMMITTED dependency
  manifest/lockfile (e.g. package.json + its lockfile, pyproject + lockfile).
  Later feature epochs build ON that baseline. Do NOT fold dependency setup into
  a feature epoch, and do NOT try to install/build inside it, just create the
  manifest as committed files; a separate prepare mechanism installs from the
  lockfile when gates run.
- Split SEQUENTIAL work across epochs LIBERALLY: give a step its own epoch
  whenever it needs an earlier step's `artifact_out`, OR a real checkpoint/gate
  sits between steps, even at the SAME tier. Do not fuse a genuine A-then-B
  dependency into one opaque epoch; do not manufacture artificial steps either
  (every epoch costs a planner call, bounded by `epoch_budget`).

[LEVEL 3: TASK] Split an EPOCH into tasks (the parallel fan-out within it).
- One task = one bounded SLICE, kept SMALL: a few files, with DISJOINT
  file_ownership. A task that owns the whole repo (or a dozen unrelated files) is
  NOT decomposed, the size gate will REJECT it and make you split further.
- ENUMERATE the concrete files each implement task owns. `file_ownership` must be
  a list of CONCRETE file paths (e.g. `src/theme.ts`, `src/tokens.ts`), NEVER a
  wildcard glob (`src/design-system/**`, `dir/*`, `*.ts`): the gate rejects any
  wildcard so the per-tier file cap actually bounds the task and so you can tell
  mechanical files from taste files when you tier them. If you cannot yet name the
  files, the slice is not understood well enough to dispatch, split or research first.
- ROUTE each task by judgment vs mechanism (see "Tier routing" above): keep the
  mechanical scaffolding/tokens/exports as a LOCAL task (senior false), carve the
  taste/layout/polish/synthesis into a SEPARATE task with `"senior": true`. Do this
  for implement AND research AND review. One epoch may mix local and senior tasks.
- Prefer PRECISE, machine-checkable `done_when` checks so a mechanical gap is caught
  cheaply IN the worker (it bakes against them before handoff), not by an expensive
  rebuild. Express structural facts as the project's own commands / file-existence;
  content/semantic acceptance rides `criteria`.
- Tasks within an epoch run in PARALLEL and MUST NOT consume each other's
  outputs. Anything where one task needs another's result is SEQUENTIAL work,
  put it in a later epoch (or phase), never in a sibling task of the same epoch.
- Two axes, OPPOSITE biases. Splitting SEQUENTIAL work across epochs (LEVEL 2):
  be LIBERAL. Splitting PARALLEL tasks inside one epoch (this level): be
  CONSERVATIVE. Decompose CONSERVATIVELY here, at the top level only: you are a
  powerful planner, prefer ONE task whenever the work is even remotely
  interconnected and shared context helps. Split into multiple tasks ONLY when
  the parts are genuinely independent, or genuinely too big for one worker's
  context. Naive fan-out of intertangled work hands the hardest part, cross-file
  consistency, to the least-coordinated agents. But a single task may NOT swallow
  the whole epoch's files: SMALL and bounded beats one giant task (the size gate
  enforces this, see below).
- Each task must fit ONE worker with a ~90k-token working context. Treat 90k as
  the sizing CONTRACT: the worker has headroom above it, but that headroom is
  overrun insurance, never plannable budget. If a task cannot plausibly fit, it
  is two tasks or two epochs.
- SIZE GATE (deterministic, enforced): a fresh implement task's `file_ownership`
  must ENUMERATE concrete files (any wildcard glob is rejected as "a broad glob"),
  and the count is capped PER TASK by its tier (a small count on the local tier, a
  larger one on a `senior:true` task). A broad-glob or oversized task bounces back
  as an invalid decision naming the offending task, enumerate or split it. (A
  handle_failed_epoch repair may carry broad scope, that path is exempt.)
- `epoch_budget` is how many epochs a phase may consume before the state machine
  fires a phase escalation (forcing you to revise_phases or escalate_run). It is
  a ceiling sized to the phase's real arc, a small phase is 1-2, a broad build
  phase a few more, not a target; unused budget is free. If the budget runs out
  because an epoch FAILED, you get the focused handle_failed_epoch decision for
  that epoch instead (it takes precedence); the revise_phases / escalate_run
  escalation is for a phase whose gate never passes while its epochs all complete.
- Carry the relevant job-spec requirements into each task's `goal` VERBATIM, or
  point at the exact input artifacts (by log key) that contain them. Never
  paraphrase or summarize a requirement away, lossy paraphrase silently drops
  requirements. `goal` is capped at 1024 chars: quote exactly what fits and move
  the rest into referenced input artifacts; never compress a requirement into a
  summary.

Example implement decision (a MECHANICAL slice and a TASTE slice, so two tasks
with pairwise-DISJOINT ENUMERATED file_ownership; the mechanical task stays local,
the taste task is flagged `"senior": true`; each done_when is a STRUCTURAL check,
content acceptance rides `criteria`; each goal quotes the spec VERBATIM):
  {"schema_version":"1","tool":"implement","args":{"epoch_title":"Design tokens and the home screen","rationale":"mechanical tokens vs the screen's taste, split by tier","tasks":[
    {"id":"T1","goal":"Create the design tokens. Spec verbatim: 'tokens.ts MUST export the M3 color + spacing scale'.","done_when":[{"cmd":"test -f src/tokens.ts"}],"criteria":["tokens.ts exports the full M3 color and spacing scale"],"file_ownership":["src/tokens.ts","src/index.ts"]},
    {"id":"T2","goal":"Build the home screen. Spec verbatim: 'the home screen MUST feel calm, with one primary action and >=44px targets'.","done_when":[{"cmd":"test -f src/HomeScreen.tsx"}],"criteria":["the home screen is calm, one primary action, >=44px targets"],"file_ownership":["src/HomeScreen.tsx"],"senior":true}]}}
