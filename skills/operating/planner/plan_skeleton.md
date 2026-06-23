SCENARIO: plan_skeleton. No skeleton exists yet, this is the FIRST decision of
the run. Decompose the JOB SPEC above into the phase skeleton with
propose_skeleton, nothing else is legal yet.

- A skeleton has BETWEEN 2 AND 10 phases. Even a small job needs at least two
  (e.g. a build phase then a verify phase); phase ids are "P1","P2",... in order.
  Each epoch has 1-8 tasks with ids "T1"..."T8".
- Sequence by tier of thinking. Tier is chosen PER TASK by a `senior` flag (set
  later, in the work decisions): judgment/taste/synthesis tasks run on the stronger
  SENIOR tier, mechanical/factual tasks (incl. web-search research) run on the
  WORKER tier (the local rig). So a skeleton is also a routing choice: shape the
  phases so judgment and production can be separated into their own tasks. For
  heavy or judgment-laden work, SPLIT into phases rather than cramming it into one
  epoch, and feed each step forward through the keyed log (a non-implement epoch's
  `artifact_out` becomes a later epoch's `inputs`). Good shapes (nudges, not a
  fixed menu): heavy build = research -> implement -> review; report / triage /
  migration = research -> artifact (flag the analysis/synthesis task senior so it
  is not downgraded); UI = research -> implement (mechanical tokens local, the
  taste/layout task flagged senior) -> a phase exit criterion that builds,
  screenshots, then `vision_review`s it. A small job can be a single epoch.

Decomposition is THREE distinct skills, one per level; this scenario is LEVEL 1.

[LEVEL 1: PHASING] Split the JOB into phases (propose_skeleton / revise_phases).
- One phase = one MODE: research / implement / test / review. Do not mix modes
  in a phase; a phase that "builds and reviews" is two phases.
- A skeleton has BETWEEN 2 AND 10 phases; phase ids "P1","P2",... in order. Even a
  small job needs at least two (e.g. a build phase then a verify phase).
- Sequence by tier of thinking: tier is per TASK (a `senior` flag set in the work
  decisions), judgment/taste/synthesis on the SENIOR tier, mechanical/factual work
  on the WORKER tier (the local rig), so phasing is also a routing choice (shape
  phases so judgment and production land in separate, separately-tiered tasks).
- A phase's GOAL lives in its `title` (and the job spec it derives from); its
  `exit_criterion` is the build-health FLOOR, the deterministic check(s) that prove
  the build is still HEALTHY (it compiles, tests pass, the bundle exports). The
  floor is NECESSARY but NOT SUFFICIENT: a green floor does NOT end a phase. You
  end a phase yourself with phase_complete when you judge its GOAL met (the
  deliverables exist), so do NOT try to encode "the deliverable is complete" as an
  exit_criterion, a generic build-health check (`tsc`, `expo export`) can pass
  before the phase's real work is built, which would be a hollow signal. Keep the
  floor a small, genuine build-health check; carry the deliverable bar in the phase
  title and the tasks' `criteria`.

Example first decision (note: TWO phases minimum; the exit_criterion is the
build-health FLOOR, structural only, never the deliverable-completeness gate):
  {"schema_version":"1","tool":"propose_skeleton","args":{"phases":[
    {"id":"P1","title":"Build the parser","exit_criterion":[{"cmd":"npm test --silent","expect_exit":0}],"epoch_budget":2},
    {"id":"P2","title":"Verify end-to-end","exit_criterion":[{"cmd":"npm run e2e --silent","expect_exit":0}],"epoch_budget":1}]}}
