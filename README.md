# Grindstone

**An epoch-based deep-work orchestrator for coding agents.** You hand it a job
spec. A strong cloud planner proposes one small, verifiable epoch at a time;
local workers fan out and grind through the tasks; and a fixed, deterministic
state machine gates every step through disk contracts until the job's exit
criteria actually pass.

> *The model proposes; the state machine disposes.*

![Python](https://img.shields.io/badge/python-%E2%89%A53.10-blue)
![Typed](https://img.shields.io/badge/mypy-strict-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Pydantic](https://img.shields.io/badge/validation-pydantic-e92063)

---

## Why Grindstone

Most agent harnesses let a single model improvise the whole job — plan, edit,
decide it's done — and hope the transcript reflects reality. Grindstone refuses
to trust the model's word for anything that matters:

- **Deterministic gates, not vibes.** A task is done only when a *command exits 0*
  or a *required artifact exists* — never because a model said "done". Every
  worker result crosses a JSON disk contract (`handoff.json`) that is relocated,
  re-validated, and re-checked in the run directory. **stdout is never parsed.**
- **Small, bounded planner turns.** The planner emits exactly one constrained
  tool-call per turn (`propose_skeleton` / `implement` / `review` / `complete_run`
  / …) — schema-validated before it touches the loop. No runaway agent; the loop
  is a fixed, inspectable state machine.
- **Model-agnostic by construction.** Grindstone knows only **three role names** —
  `planner`, `local`, `senior` — each reached through a `models/*.sh` adapter
  behind a file contract. Swap GPT-5.5 for Claude, Qwen for Llama, kimi for
  anything: the orchestrator never changes.
- **Crash-safe and resumable.** Every state transition is an `fsync`'d line in an
  append-only `events.ndjson` journal. Kill a run mid-epoch and `resume` it —
  no work is lost or double-burned. The journal even self-heals a crash-torn tail.
- **Typed at every boundary.** Python, Pydantic on every wire contract,
  `mypy --strict` clean. The schemas in `schemas/` are the single source of truth.
- **Taste, gated.** UI work can be routed to a vision-capable senior model and
  judged by a planner-driven screenshot review that returns a pass/fail disk
  contract — so "make it look good" becomes a checkable gate, not a hope.

## See it work

A two-phase job — *build a greeting module, then document it* — driven to
completion. Live, while it runs, `--watch` paints the run → phase → epoch → task
tree (a pure reader of the event journal, in a separate process):

```
✓ run greet-demo  [completed]  2m31s  · 2/2 phases  · planner calls: 4/96  · last: complete_run
├── ✓ P1 · build  [passed]  1m28s
│   └── ✓ E1 · write greet.py + test  [completed]  1m17s
│       ├── ✓ T1 (implement)  [done]  59s
│       └── ✓ T2 (implement, a2)  [done]  1m16s   ← failed once, retried, passed
└── ✓ P2 · document  [passed]  45s
    └── ✓ E2 · write README.md  [completed]  33s
        └── ✓ T3 (implement)  [done]  32s
```

When the run reaches a terminal state it writes a post-mortem `journal.md`,
rendered from the same event stream — never a second source of truth:

```markdown
# Run greet-demo — completed

- Job: `job.md`
- Duration: 2m31s   ·   Planner calls: 4/96

## P1 · build  [passed]  (1m28s)
- **E1 · write greet.py + test**  [completed]  (1m17s)
    - ✓ T1 (implement) [done]  (59s)
    - ✓ T2 (implement) [done]  (1m16s)

## P2 · document  [passed]  (45s)
- **E2 · write README.md**  [completed]  (33s)
    - ✓ T3 (implement) [done]  (32s)
```

And when a phase carries a **vision-review** gate, a planner-driven model judges
the rendered UI against your criteria and emits a verdict disk contract. A real
failing verdict from the taste gate:

```json
{
  "pass": false,
  "reasons": [
    "Primary button is not full-width on mobile; it floats left of center.",
    "Visual hierarchy is weak — the heading and body copy are the same weight."
  ]
}
```

That `pass: false` fails the phase's exit criterion deterministically — the run
does not advance on a UI the judge rejected.

## How it works

A job flows through a fixed lifecycle, and the state machine — not the model —
decides every transition:

```
job.md
  └─ propose_skeleton ──▶ phases (each: exit_criterion + epoch_budget)
        └─ implement / research / review / artifact ──▶ epochs
              └─ tasks fan out ──▶ workers ──▶ handoff.json (disk contract)
                    └─ deterministic done_when / exit_criterion gates
                          └─ complete_run (re-verifies evidence) ──▶ terminal
```

Each worker writes `handoff.json` in its own throwaway git worktree; the
relocated, re-validated copy in the run dir is the gate. Phases advance only when
their `exit_criterion` checks pass; `complete_run` re-runs the evidence before
the run is allowed to finish. See **[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)**
for the full design — the thesis, the script-backed roles, the run-dir layout,
the taste/vision features, and durability/resume.

## Quickstart

Grindstone runs **from a clone** — no separate package step needed.

```bash
# 1. Clone and install (editable; resolves models/ + schemas/ from the repo root).
git clone <repo-url> grindstone && cd grindstone
pip install -e .                 # or: python3 -m pip install -e .

# 2. Scaffold the per-repo config + gitignore the run dir in your TARGET repo.
python3 -m grindstone init --repo /path/to/target-repo

# 3. Write the ask as a Markdown job spec.
$EDITOR /path/to/target-repo/job.md

# 4. Run it to completion (--watch renders the live TUI; drop it for agents/CI).
python3 -m grindstone run /path/to/target-repo/job.md --repo /path/to/target-repo --watch

# 5. Watch a run's live tree, or resume a killed run — no work is lost or double-burned.
python3 -m grindstone watch  <run-id> --repo /path/to/target-repo
python3 -m grindstone resume <run-id> --repo /path/to/target-repo
```

`run` executes in the **foreground** (use `nohup` / `&` to detach). Exit codes
encode the terminal outcome: **`0`** completed, **`1`** escalated (needs a human),
**`2`** safety-valve stop. Run state lives under `.grindstone/runs/<run-id>/`
inside the target repo (an append-only `events.ndjson` journal, `run_state.json`,
the keyed log `P*/E*/T*/…`, and a post-mortem `journal.md`).

## Compatibility

Grindstone is the orchestrator; the models behind each role are yours to choose.
The reference adapters in `models/*.sh` wire up one working stack — point them at
anything that speaks the file contract.

| Layer            | Reference (what the adapters ship with)            | Swappable with                                  |
| ---------------- | -------------------------------------------------- | ----------------------------------------------- |
| **Runtime**      | Python ≥ 3.10 · Linux & macOS                      | —                                               |
| **Planner role** | `codex` CLI (GPT-5.5, a ChatGPT/OpenAI plan)       | any CLI that emits the decision JSON contract   |
| **Local role**   | Qwen via `llama-server`, driven by the `pi` CLI    | any local/remote LLM server                     |
| **Senior role**  | `opencode` (kimi) with web search — *optional*     | any cloud tier; delete the block for local-only |
| **Vision/taste** | Qwen 3.6 (native VL, `--mmproj`) + `codex` judge   | any vision model behind the screenshot contract |
| **Target repo**  | any `git` repository                               | —                                               |

**macOS note:** the role scripts use GNU `timeout` as a wall-clock backstop and
fall back to `gtimeout` — install it with `brew install coreutils`.

**Bring your own models.** The planner adapter runs whatever your `codex`/ChatGPT
plan serves and the senior tier is optional, so the one real blank for most rigs is
your local model. Swap the model identities without touching code:

| Env var                     | Default                 | What it sets                                          |
| --------------------------- | ----------------------- | ---------------------------------------------------- |
| `GRINDSTONE_LOCAL_PROVIDER` | `local-reviewer`        | the `pi --provider` your agent routes to your endpoint |
| `GRINDSTONE_LOCAL_MODEL`    | `qwen-3-6-27b-dense`    | the local `pi --model`                               |
| `GRINDSTONE_SENIOR_MODEL`   | `opencode-go/kimi-k2.6` | any `opencode -m` target                             |

For anything deeper — a different transport than `pi`/`opencode`/`codex`, extra
flags, your own endpoint — edit the small `models/*.sh` adapter directly. Each one
is the whole grindstone↔model boundary, and the working reference is the clearest
spec of the `handoff.json` contract your replacement must honor (a placeholder
couldn't teach it). `grindstone init` scaffolds `.grindstone/config.yaml`
referencing the scripts by absolute path, with editable `slots` (per-role
concurrency) and `timeout_s` — the orchestrator never needs to know what's behind
them.

## The contracts

The planner emits exactly one constrained tool-call per turn, and every worker
result crosses a disk contract.

- **[`docs/PLANNER_CONTRACT.md`](docs/PLANNER_CONTRACT.md)** — the planner call
  model, the decision tool set, input construction, and the validation pipeline.
- **[`schemas/`](schemas/)** — the wire contracts and single source of truth:
  `epoch_decision.json` (the planner's output), `handoff.json` (the worker's
  result), `vision_verdict.json` (the taste gate). The Pydantic types and
  validators in `grindstone/contracts/` are kept in lockstep with these.

## Security

Grindstone **executes code on your behalf** — it runs planner-chosen shell
commands (a job's `done_when` / `exit_criterion` / `complete_run` evidence checks)
and the request scripts named in the target repo's config. **Run it only on job
specs and target repos you trust**, ideally in a disposable VM or container. See
**[`SECURITY.md`](SECURITY.md)** for the full trust model and what is sandboxed.

## Running the tests

```bash
python3 -m pytest                              # the default unit suite
python3 -m pytest tests/grindstone -m real     # opt-in: live gates (spend real quota)
```

The `-m real` gates exercise live infrastructure (the `codex` planner, the local
GPU workers) and spend real subscription / GPU quota — they are excluded from the
default suite and are opt-in only.

## License

MIT — see [`LICENSE`](LICENSE).
