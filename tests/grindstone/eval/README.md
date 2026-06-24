# Real-rig capability harness (`eval`)

This corpus drives the REAL planner and worker against a live rig and asserts
PROPERTIES (bands, not goldens) of what they produce, on the bones contracts:

- `test_eval_boundary.py` -- one real `ScriptPlanner.decide` per boundary; the
  decision must be a well-formed bones decision whose epoch obeys the core rules
  (1..8 tasks, valid mode + tier, concrete non-wildcard implement ownership,
  pairwise-disjoint ownership).
- `test_eval_task.py` -- one real `run_task` (a tiny implement + a tiny research);
  the task must PASS, meaning its handoff validated and the independent critic
  returned a verdict.

The thesis: if the local FLOOR (Qwen) produces conforming output, the cloud
CEILING (Claude/Opus) will too.

## Running it live

The whole corpus is behind the `eval` marker (excluded from the default suite),
and each rig is skipped unless named in `GRINDSTONE_EVAL_RIG` (a comma list), so a
bare run collects-but-skips. Drive a rig live:

```sh
# the local floor (Qwen on :8080 must be up: ~/scripts/llama/start-8080.sh)
GRINDSTONE_EVAL_RIG=local  .venv/bin/python -m pytest tests/grindstone/eval -m eval

# the cloud ceiling (Claude / Opus via `claude -p`)
GRINDSTONE_EVAL_RIG=claude .venv/bin/python -m pytest tests/grindstone/eval -m eval

# both, in one sweep
GRINDSTONE_EVAL_RIG=local,claude .venv/bin/python -m pytest tests/grindstone/eval -m eval
```

Collection is never gated, so `pytest --co -m eval` always lists the full corpus.
