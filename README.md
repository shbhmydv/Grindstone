# Grindstone

An epoch-based deep-work orchestrator for coding agents. You hand it a job spec.
A strong cloud planner proposes one small, verifiable epoch at a time, local
workers fan out and grind through the tasks, and a fixed state machine gates
every step through disk contracts until the job's exit criteria actually pass.

> *The model proposes; the state machine disposes.*

![Python](https://img.shields.io/badge/python-%E2%89%A53.10-blue)
![Typed](https://img.shields.io/badge/mypy-strict-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Pydantic](https://img.shields.io/badge/validation-pydantic-e92063)

---

## Why Grindstone

Most agent harnesses hand the whole job to one model: it plans the work and makes
the edits, then reports that it finished. You end up trusting the transcript.
Grindstone checks the work directly, and the rest of the design follows from that.

- **Deterministic gates.** A task passes when a command exits 0 or a named
  artifact exists. Every worker result is written to a `handoff.json` file that
  Grindstone relocates into the run directory, re-validates, and re-checks before
  accepting it. stdout is never parsed.
- **Bounded planner turns.** The planner returns one tool call per turn
  (`propose_skeleton`, `implement`, `review`, `complete_run`, and a few others),
  validated against a schema before the loop acts on it. The loop is an ordinary
  state machine you can read top to bottom.
- **Repo-aware navigation.** On a large target repo, the planner and each worker
  get a structural map built fresh with tree-sitter: definitions and references
  ranked by PageRank, rendered to a token budget. The planner sees the whole
  repo's spine; a worker sees the neighborhood of the files its task touches.
  Small repos are skipped, and a map that fails to build never fails the run.
- **Swappable models.** There are three roles: `planner`, `local`, and `senior`.
  Each is a `models/*.sh` script behind a file contract, so you can put GPT-5.5
  where Claude was, or Qwen where Llama was, and the orchestrator stays the same.
- **Resumable.** Every state change is an fsync'd line in an append-only
  `events.ndjson` journal. If a run dies mid-epoch, `resume` picks it back up, and
  the journal repairs a crash-torn tail when it loads.
- **Typed boundaries.** Python, with Pydantic on every wire contract and the whole
  package under `mypy --strict`. The files in `schemas/` define the formats.
- **Taste as a gate.** UI work can route to a vision-capable senior model and be
  judged by a screenshot review that returns a pass/fail file, so "make it look
  good" turns into a check the run can enforce.

## See it work

Here is a two-phase job, building a greeting module and then documenting it, run
to completion. While it runs, `--watch` paints the run → phase → epoch → task tree
from the event journal, in a separate reader process:

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

When a run reaches a terminal state it writes a post-mortem `journal.md` from that
same event stream, so there is never a second source of truth:

```markdown
# Run greet-demo · completed

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

If a phase carries a vision-review gate, a planner-driven model judges the
rendered UI against your criteria and writes a verdict. Here is a real failing one:

```json
{
  "pass": false,
  "reasons": [
    "Primary button is not full-width on mobile; it floats left of center.",
    "Visual hierarchy is weak; the heading and body copy share one weight."
  ]
}
```

That `pass: false` fails the phase's exit criterion, so the run will not advance on
a UI the judge rejected.

## How it works

A job moves through a fixed lifecycle. The state machine decides every transition:

```
job.md
  └─ propose_skeleton ──▶ phases (each: exit_criterion + epoch_budget)
        └─ implement / research / review / artifact ──▶ epochs
              └─ tasks fan out ──▶ workers ──▶ handoff.json (disk contract)
                    └─ deterministic done_when / exit_criterion gates
                          └─ complete_run (re-verifies evidence) ──▶ terminal
```

Each worker writes `handoff.json` in its own throwaway git worktree, and the
relocated, re-validated copy in the run dir is what counts. A phase advances only
once its `exit_criterion` checks pass, and `complete_run` re-runs the evidence
before the run is allowed to finish. The full design lives in
**[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)**: the thesis, the script-backed
roles, the run-dir layout, the taste and vision features, and how durability and
resume work.

## Quickstart

Grindstone runs from a clone, with no separate package step.

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

# 5. Watch a run's live tree, or resume a killed run; nothing is lost or double-burned.
python3 -m grindstone watch  <run-id> --repo /path/to/target-repo
python3 -m grindstone resume <run-id> --repo /path/to/target-repo
```

`run` executes in the foreground; use `nohup` or `&` to detach. The exit code
tells you the terminal outcome: `0` completed, `1` escalated (needs a human), `2`
safety-valve stop. Run state lives under `.grindstone/runs/<run-id>/` inside the
target repo: an append-only `events.ndjson` journal, `run_state.json`, the keyed
log at `P*/E*/T*/…`, and a post-mortem `journal.md`.

## Compatibility

Grindstone is the orchestrator; you choose the models behind each role. The
reference adapters in `models/*.sh` wire up one working stack, but you can point
them at anything that speaks the file contract.

| Layer            | Reference (what the adapters ship with)            | Swappable with                                  |
| ---------------- | -------------------------------------------------- | ----------------------------------------------- |
| **Runtime**      | Python ≥ 3.10 · Linux & macOS                      | n/a                                             |
| **Planner role** | `codex` CLI (GPT-5.5, a ChatGPT/OpenAI plan)       | any CLI that emits the decision JSON contract   |
| **Local role**   | Qwen via `llama-server`, driven by the `pi` CLI    | any local/remote LLM server                     |
| **Senior role**  | `opencode` (kimi) with web search (*optional*)     | any cloud tier; delete the block for local-only |
| **Vision/taste** | Qwen 3.6 (native VL, `--mmproj`) + `codex` judge   | any vision model behind the screenshot contract |
| **Target repo**  | any `git` repository                               | n/a                                             |

macOS note: the role scripts use GNU `timeout` as a wall-clock backstop and fall
back to `gtimeout`, so install it with `brew install coreutils`.

### Bring your own models

The planner adapter runs whatever your `codex`/ChatGPT plan serves, and the senior
tier is optional, so for most rigs the only real blank is your local model. You can
swap the model identities without touching code:

| Env var                     | Default                 | What it sets                                          |
| --------------------------- | ----------------------- | ---------------------------------------------------- |
| `GRINDSTONE_LOCAL_PROVIDER` | `local-reviewer`        | the `pi --provider` your agent routes to your endpoint |
| `GRINDSTONE_LOCAL_MODEL`    | `qwen-3-6-27b-dense`    | the local `pi --model`                               |
| `GRINDSTONE_SENIOR_MODEL`   | `opencode-go/kimi-k2.6` | any `opencode -m` target                             |

For anything deeper, such as a different transport than `pi`/`opencode`/`codex`,
extra flags, or your own endpoint, edit the relevant `models/*.sh` adapter. Each
one is the whole boundary between Grindstone and a model, and the working reference
is the clearest spec of the `handoff.json` contract your replacement has to honor.
`grindstone init` scaffolds `.grindstone/config.yaml` with the scripts referenced
by absolute path, plus editable `slots` (per-role concurrency) and `timeout_s`.

## The contracts

The planner emits one constrained tool-call per turn, and every worker result
crosses a disk contract.

- **[`docs/PLANNER_CONTRACT.md`](docs/PLANNER_CONTRACT.md)** covers the planner
  call model, the decision tool set, input construction, and the validation
  pipeline.
- **[`schemas/`](schemas/)** holds the wire contracts and the source of truth:
  `epoch_decision.json` (the planner's output), `handoff.json` (the worker's
  result), and `vision_verdict.json` (the taste gate). The Pydantic types and
  validators in `grindstone/contracts/` track these files.

## Security

Grindstone executes code on your behalf. It runs planner-chosen shell commands (a
job's `done_when`, `exit_criterion`, and `complete_run` evidence checks) and the
request scripts named in the target repo's config. Run it only on job specs and
target repos you trust, ideally in a disposable VM or container. See
**[`SECURITY.md`](SECURITY.md)** for the full trust model and what is sandboxed.

## Running the tests

```bash
python3 -m pytest                              # the default unit suite
python3 -m pytest tests/grindstone -m real     # opt-in: live gates (spend real quota)
```

The `-m real` gates exercise live infrastructure (the `codex` planner, the local
GPU workers) and spend real subscription or GPU quota, so they are excluded from
the default suite and run only when you ask for them.

## License

MIT. See [`LICENSE`](LICENSE).
