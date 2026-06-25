# Grindstone

An epoch-based deep-work orchestrator for coding agents. You hand it a job spec.
A stateless planner proposes one small epoch of work at a time, workers fan out in
throwaway worktrees and grind through the tasks, a tier-matched critic triages each
result, and a fixed state machine gates every step until the job's own acceptance
command actually passes.

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

- **Two deterministic invariants, nothing more.** Everything else is a model
  judgment, but exactly two things are machine-enforced: each epoch's git diff must
  be a *disjoint-ownership* merge (no two tasks touch the same file), and the job's
  own `done_when` command must exit 0 in a clean checkout, run ONCE at the very end.
  "Done" still means something when every per-epoch check is agentic.
- **Free-form handoffs, never parsed.** A worker writes a free-form `handoff.md`
  report in its own worktree. It is relocated verbatim for the critic to read and is
  never schema-gated; stdout is never parsed either. The only structured wire
  contracts are the planner's decision and the critic's one-line triage verdict.
- **Stateless planner, one decision per boundary.** The planner proposes a single
  typed `Decision` each turn: either an `epoch` of 1 to 8 disjoint tasks, or `end`.
  There are no phases, no skeletons, no exit criteria. After each epoch it writes a
  free-form *baton* (its living plan), re-read fresh next boundary. The only backstop
  is `max_epochs`. The loop is an ordinary state machine you can read top to bottom.
- **Tiered routing.** Each task carries a `mode` (implement / research / review /
  artifact) and a `tier`: `local` (e.g. a Qwen server) for checkable, mechanical
  work, or `senior` (e.g. Claude) for judgment and taste. Optional repo-owned domain
  `skills` are selected per task. The matching critic verdict routes the outcome:
  `PASS` merges onto the epoch's staging branch, `RETRY` is a bounded same-worker
  retry, `ESCALATE` goes back to the planner.
- **Resumable.** Every state change is an fsync'd line in an append-only
  `events.ndjson` journal. If a run dies mid-epoch, `resume` razes the unfinished
  epoch and re-runs it from the durable run branch; the journal repairs a crash-torn
  tail when it loads; a SIGTERM/SIGINT mid-run reaps in-flight workers and exits
  resumable.
- **Typed boundaries.** Python, with Pydantic on every wire contract and the whole
  package under `mypy --strict`. The two files in `schemas/` define the formats.

## See it work

Here is a two-epoch job, building a greeting module and then documenting it. On an
interactive terminal, `grindstone watch` paints the live run -> epoch -> task tree
from the event journal, in a separate reader process. Once a task gets a critic
verdict, a `C` leaf shows the triage:

```
● run greet-demo  [running]  1m48s  · 2/20 epochs
├── ✓ E1 · write greet.py + test  [completed]  1m17s
│   ├── ✓ T1 (implement)  [done]  59s
│   │   └── C  PASS
│   └── ✓ T2 (implement)  [done]  1m16s
│       └── C  PASS
└── ◐ E2 · docs + a polish pass  [started]  30s
    ├── ▶ T3 (implement)  [dispatched]  28s
    └── ▶ T4 (review)  [dispatched]  28s
        └── C  RETRY - tighten the heading hierarchy
```

The view refreshes about once a second so live durations keep ticking. When stdout
is not a TTY (piped, CI, or another agent) or you pass `--once`, `watch` prints a
single static render instead, so non-interactive callers behave predictably.

When a run reaches a terminal state it writes a post-mortem `journal.md` from that
same event stream, so there is never a second source of truth:

```markdown
# Run greet-demo - completed

- Job: `job.md`
- Duration: 1m51s   -   Epochs: 2/20

## E1 - write greet.py + test  [completed]  (1m17s)
    [ok] T1 (implement) [done]  (59s)
    [ok] T2 (implement) [done]  (1m16s)

## E2 - write README.md  [completed]  (32s)
    [ok] T3 (implement) [done]  (30s)
```

## How it works

A job moves through a fixed lifecycle. The stateless planner self-steers one epoch
at a time; the state machine decides every transition:

```
loop (until the planner ends or the max-epochs backstop fires):
  context  = job + keyed-log index + the prior epoch's baton (the planner's living plan)
  decision = planner.decide(context)          # PLAN: ONE typed Decision, self-validated on disk
  if decision is END:
    run the job's done_when ONCE in a clean checkout
    exit 0 -> completed ; non-zero -> a clean partial-end (resumable)
  else (an EPOCH of 1..8 disjoint tasks):
    run the planner-declared setup commands (the one trusted host-mutation seam)
    fan the tasks out (tier-routed, each in its own throwaway worktree):
      worker grinds -> writes a free-form handoff.md report in its CWD
      deterministic gate: an in-scope commit (or the named artifact exists)
      tier-matched critic reads the report + the real diff/artifact -> a lenient verdict
        PASS -> merge-ready ; RETRY -> bounded same-tier retry ; ESCALATE -> the planner
    merge the PASSing tasks onto a staging branch (disjoint-ownership merge)
    planner.close_out(staging tree + per-task outcomes)   # CLOSE-OUT: writes the baton
    atomically finalize: fast-forward the run branch + persist the baton + EpochCompleted
```

Close-out runs BEFORE the run-branch fast-forward, so the single durable commit
point (the fast-forward + the baton write + `EpochCompleted`) already includes the
baton: there is no "integrated-but-not-summarized" limbo, and a close-out crash or
rate-limit is just a raze-and-restart of the same epoch. The full design lives in
**[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)** (the design-of-record): the
thesis, the script-backed roles, the run-dir layout, and how durability and resume
work.

## Quickstart

Grindstone runs from a clone, with no separate package step.

```bash
# 1. Clone and install (editable; resolves models/ + schemas/ from the repo root).
git clone <repo-url> grindstone && cd grindstone
pip install -e .                 # or: python3 -m pip install -e .

# 2. Hand-write the per-repo config in your TARGET repo (there is no init command),
#    and gitignore the run dir there: echo '.grindstone/runs/' >> .gitignore
$EDITOR /path/to/target-repo/.grindstone/config.yaml

# 3. Write the ask as a Markdown job spec.
$EDITOR /path/to/target-repo/job.md

# 4. Run it to completion (foreground; use nohup or & to detach).
python3 -m grindstone run /path/to/target-repo/job.md --repo /path/to/target-repo

# 5. Watch a run's live tree, or resume a killed run; nothing is lost or double-burned.
python3 -m grindstone watch  <run-id> --repo /path/to/target-repo
python3 -m grindstone resume <run-id> --repo /path/to/target-repo
```

The config is a hand-written `<target-repo>/.grindstone/config.yaml`. A minimal one:

```yaml
roles:
  planner:
    rig: claude          # shipped Claude rig; use `rig: codex` for the bundled codex planner
    slots: 1
    timeout_s: 600
  worker:
    rig: claude
    slots: 2
    timeout_s: 1800
  senior:                # optional; omit for a worker-only escalation ladder
    rig: claude
    slots: 2
    timeout_s: 1800
done_when: "python3 -m pytest -q"   # optional final acceptance, run once at the end
max_epochs: 20                       # optional backstop on planner boundaries
```

`run` executes in the foreground. The exit code tells you the terminal outcome:
`0` completed, `1` a clean partial-end (resume it or hand it to a human), `2` a
safety-valve stop. Run state lives under `.grindstone/runs/<run-id>/` inside the
target repo: an append-only, fsync'd `events.ndjson` journal, the keyed log at
`E*/T*/...`, and a post-mortem `journal.md`. Gitignore that runs dir.

## Compatibility

Grindstone is the orchestrator; you choose the models behind each role. The shipped
default rig (`models/claude/`) drives Claude (Opus) through the `claude -p` CLI for
every role, so a fresh clone with Claude Code installed runs with zero setup. You
can point any role at anything that speaks the request-script file contract.

| Layer            | Default rig (`models/claude/`)                     | Swappable with                                  |
| ---------------- | -------------------------------------------------- | ----------------------------------------------- |
| **Runtime**      | Python >= 3.10 - Linux & macOS                     | n/a                                             |
| **Planner role** | Claude (Opus) via `claude -p`, read-only           | any CLI that emits the decision JSON contract   |
| **Worker role**  | Claude (Opus) via `claude -p` in the worktree      | any local/remote LLM server                     |
| **Senior role**  | Claude (Opus) via `claude -p`                       | any cloud tier; omit the block for worker-only  |
| **Target repo**  | any `git` repository                               | n/a                                             |

Two more rigs ship in-tree: `models/codex/` (a `codex exec` planner; select it with
`rig: codex`) and `models/local/` (an all-Qwen rig for local workers). Rig
resolution is explicit-first: a role's `rig:` name searches `[<rig>, claude,
_common]` and never consults `personal`; an implicit default searches `[personal,
claude, _common]`. Your own scripts go in `models/personal/` (gitignored). macOS
note: the role scripts use GNU `timeout` as a wall-clock backstop and fall back to
`gtimeout`, so install it with `brew install coreutils`.

### Bring your own models

The default rig uses one model id (`opus`) for every role; swap it without touching
code via env vars the scripts read:

| Env var                     | Default | What it sets                                |
| --------------------------- | ------- | ------------------------------------------- |
| `GRINDSTONE_PLANNER_MODEL`  | `opus`  | the planner's `claude --model`              |
| `GRINDSTONE_LOCAL_MODEL`    | `opus`  | the worker role's `claude --model`          |
| `GRINDSTONE_SENIOR_MODEL`   | `opus`  | the senior worker's `claude --model`        |

For anything deeper, such as a different transport (your own local server, a
different cloud CLI), extra flags, or your own endpoint, drop a replacement script
into `models/personal/` (gitignored, implicit-default priority) or edit the relevant
`models/claude/` adapter. Each script is the whole boundary between Grindstone and a
model, and a role is wired by either a `rig:` name or an explicit `script:` path in
`config.yaml`, plus its `slots` (per-role concurrency, >= 1) and `timeout_s`.

## The contracts

The planner emits one constrained decision per turn, and the critic returns one
lenient triage verdict per task. Those are the only two structured wire contracts;
the worker's `handoff.md` is deliberately free-form (no schema).

- **[`docs/PLANNER_CONTRACT.md`](docs/PLANNER_CONTRACT.md)** covers the planner call
  model, the decision wire contract, and input construction.
- **[`schemas/`](schemas/)** holds exactly the two contracts: `epoch_decision.json`
  (the planner's output) and `verdict.json` (the critic's triage). The Pydantic
  types in `grindstone/contracts/` are the source of truth these files track.

## Security

Grindstone executes code on your behalf. It runs the planner-declared `setup`
commands, the job's `done_when` acceptance, and the request scripts named in the
target repo's config. A configured `script:` that resolves outside the bundled
`models/` dir is refused unless you opt the trusted repo in. Run it only on job
specs and target repos you trust, ideally in a disposable VM or container. See
**[`SECURITY.md`](SECURITY.md)** for the full trust model and what is sandboxed.

## Running the tests

```bash
python3 -m pytest -q                           # the default unit suite
python3 -m pytest tests/grindstone -m real     # opt-in: live gates (spend real quota)
```

The `-m real` gates exercise live infrastructure (the `codex` planner, the local
GPU workers) and spend real subscription or GPU quota, so they are excluded from the
default suite and run only when you ask for them.

## License

MIT. See [`LICENSE`](LICENSE).
