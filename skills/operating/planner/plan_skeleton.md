SCENARIO: plan_skeleton. No skeleton exists yet, this is the FIRST decision of
the run. Decompose the JOB SPEC above into the phase skeleton with
propose_skeleton, nothing else is legal yet.

- A skeleton has BETWEEN 2 AND 10 phases. Even a small job needs at least two
  (e.g. a build phase then a verify phase); phase ids are "P1","P2",... in order.
  Each epoch has 1-8 tasks with ids "T1"..."T8".
- Sequence by tier of thinking. research/review (and visual epochs) run on the
  stronger SENIOR tier; implement/artifact run on the WORKER tier (the local rig),
  so a skeleton is also a routing choice: put judgment on senior, production on
  the worker tier. For heavy or judgment-laden work, SPLIT into phases rather than
  cramming it into a single worker epoch, and feed each step forward through the
  keyed log (a non-implement epoch's `artifact_out` becomes a later epoch's
  `inputs`). Good shapes (nudges, not a fixed menu): heavy build = research ->
  implement -> review; report / triage / migration = research -> artifact (do NOT
  collapse the analysis into one worker artifact epoch, that downgrades it off
  senior); UI =
  research -> implement with `visual:true` -> a phase exit criterion that builds,
  screenshots, then `vision_review`s it. A small job can be a single epoch.

Decomposition is THREE distinct skills, one per level; this scenario is LEVEL 1.

[LEVEL 1: PHASING] Split the JOB into phases (propose_skeleton / revise_phases).
- One phase = one MODE: research / implement / test / review. Do not mix modes
  in a phase; a phase that "builds and reviews" is two phases.
- A skeleton has BETWEEN 2 AND 10 phases; phase ids "P1","P2",... in order. Even a
  small job needs at least two (e.g. a build phase then a verify phase).
- Sequence by tier of thinking: research/review (and visual phases) run on the
  stronger SENIOR tier, implement/test on the WORKER tier (the local rig), so
  phasing is also a routing choice (judgment on senior, production on the worker
  tier).

Example first decision (note: TWO phases minimum; checks are STRUCTURAL only):
  {"schema_version":"1","tool":"propose_skeleton","args":{"phases":[
    {"id":"P1","title":"Build","exit_criterion":[{"cmd":"test -f out.txt","expect_exit":0}],"epoch_budget":2},
    {"id":"P2","title":"Verify","exit_criterion":[{"cmd":"npm test --silent","expect_exit":0}],"epoch_budget":1}]}}
