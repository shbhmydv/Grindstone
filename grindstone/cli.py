"""The ``grindstone`` CLI: ``init`` / ``run`` / ``watch`` / ``resume`` (ARCHITECTURE.md).

argparse, no click dependency. ``init`` scaffolds the per-repo config (S5).
``run`` creates the run dir, assembles the worker ladder + planner transport from
``.grindstone/config.yaml``, each *role* (``planner`` / ``worker`` / ``senior``)
is reached through a request **script** behind the file contract; the CLI never
learns the transport or model behind a role (those live in ``models/``). No
config fails loudly and points at ``grindstone init`` (core ships no rig
defaults). ``run`` drives ``run_grind`` in the FOREGROUND (``nohup`` is the
operator's detach tool). ``resume`` re-enters a killed run; ``watch`` opens the
pure-reader TUI. Exit codes encode the terminal outcome: 0 completed, 1
escalated, 2 failed.

The transport builders are injectable (``planner`` / ``ladder`` overrides on
``main``) so the arg -> wiring path is tested end-to-end through ``main`` with a
mock planner, without spending real codex/pi quota.
"""

from __future__ import annotations

import argparse
import threading
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from grindstone.config import (
    GrindstoneConfig,
    load_config,
    models_script,
    resolve_role_script,
    validate_script_paths,
)
from grindstone.planner import (
    DEFAULT_LOCAL_MAX_TASK_FILES,
    DEFAULT_SENIOR_MAX_TASK_FILES,
    PlannerTransport,
)
from grindstone.run_loop import (
    DEFAULT_MAX_FAILED_EPOCHS_PER_PHASE,
    FinalPolish,
    Ladder,
    RunOutcome,
    resume_grind,
    run_grind,
)
from grindstone.rundir import RunDir, create_run_dir
from grindstone.script_planner import ScriptPlanner
from grindstone.script_polish import ScriptPolisher
from grindstone.script_vision import ScriptVisionReviewer, VisionReviewer
from grindstone.script_worker import ScriptWorker
from grindstone.tui import watch
from grindstone.verify import TaskVerifier, WorkerTaskVerifier

#: Per-run planner-call ceiling when the config does not set one. This is a
#: runaway-spin BACKSTOP, not a quota ration: the codex subscription has ample
#: headroom (measured 2026-06-13: <2%/week consumed across a night of gate
#: runs on the $20 plan), so the cap sits well above any legitimate deep run
#: (~32 calls for a 10-phase skeleton at ~3 calls/phase + bookends) to give
#: complex jobs room, while still ending an infinite revision loop. Gate-5's
#: stuck run burned 34 calls because the valve was test-only and production
#: ran unbounded. The valve ends the run as failed with an explicit reason;
#: owners size it per repo via ``max_planner_calls`` in config.
DEFAULT_MAX_PLANNER_CALLS = 96

#: Wall-clock budget (seconds) for one B3 vision-review gate call when the config
#: does not size it, a single read-only codex look at a screenshot, not a grind.
DEFAULT_VISION_TIMEOUT_S = 600.0

#: The ``.grindstone/config.yaml`` ``init`` scaffolds: one block per *role*
#: (``planner`` / ``worker`` / ``senior``), each NAMING a bundled rig (``rig:``)
#: plus its slots + wall-clock timeout. The CLI never learns the transport or
#: model behind a role, that lives in the rig's ``<role>_request.sh``. ``senior``
#: is scaffolded active; delete it for a local-only ladder. A ``rig:`` name (not
#: an absolute path) keeps the generated config PORTABLE: the role -> script
#: mapping happens at run time via ``resolve_role_script``, so it resolves on any
#: machine, not pinned to one checkout's absolute paths. Loadable as-is by
#: ``config.load_config``.
def _render_config_template(rig: str | None) -> str:
    """The scaffolded config text naming each role's backend by ``rig`` (portable).

    Each role names a bundled rig by NAME (``rig:``), not an absolute script path,
    so a generated config resolves on ANY machine (the rig -> ``<role>_request.sh``
    mapping happens at run time, never pinned to this checkout's paths). The
    ``--rig`` flag selects the name written; with no ``--rig`` each role gets
    ``rig: claude`` (the shipped floor). At run time a role resolves its rig to
    ``<role>_request.sh`` under that rig, falling through to the claude floor when
    the rig lacks the script. The optional vision/polish lines stay commented out
    and name only the bare script (resolved at run time if the block is uncommented).
    """

    rig_name = rig if rig is not None else "claude"
    return f"""\
# Grindstone per-repo config (.grindstone/config.yaml).
# Owner-facing settings for `grindstone run` / `grindstone resume`. Each role
# names a bundled rig (`rig:`); the rig's `<role>_request.sh` owns the transport,
# model identity and GPU. Delete this file to fall back to `grindstone init`.

roles:
  # The planner role: a strong model plans one epoch at a time. slots=1
  # (one planner call in flight); timeout is the per-call wall-clock budget.
  # `rig` names the backend (e.g. claude / codex / local), resolved at run time
  # to that rig's planner_request.sh (claude floor when the rig lacks it).
  planner:
    rig: {rig_name}
    slots: 1
    timeout_s: 600

  # The worker role: the on-rig grinders. slots = the epoch fan-out bound
  # (how many tasks run concurrently on this rig).
  worker:
    rig: {rig_name}
    slots: 2
    timeout_s: 1800

  # The senior worker role: the escalation tier. OPTIONAL, delete this whole
  # block for a local-only ladder.
  senior:
    rig: {rig_name}
    slots: 2
    timeout_s: 3600

# Per-run planner-call ceiling (omit for the built-in default of 96). A
# runaway-spin backstop, not a quota ration, the run ends as failed when
# reached so a stuck planner can never loop unattended. Raise it for very
# large jobs; lower it to fail fast on a tight job.
# max_planner_calls: 96

# The B3 vision-review (taste) gate: a `vision_review` exit-criterion check runs
# this script, which shows a vision model a rendered-UI screenshot + criteria and
# writes a pass/fail verdict the core re-reads. Omit this block to use the bundled
# vision_review.sh (resolved through the rig stack); set it to point the gate at a
# different script.
# vision_review:
#   script: /absolute/path/to/vision_review.sh
#   timeout_s: 600

# The B5 final-polish pass: OFF unless this block is present. After a run's
# complete_run evidence passes, the polisher EDITS the finished repo inline per
# `criteria`; the edits are KEPT only if the SAME evidence still passes (else
# discarded, the original completion stands). It can never bypass the gate or fail
# a completed run. `script` defaults to the bundled codex_polish.sh (resolved
# through the rig stack); `screenshot` (worktree-relative) is optional.
# final_polish:
#   criteria: "tasteful finishing touches; do not change behavior"
#   timeout_s: 900
#   # script: /absolute/path/to/codex_polish.sh
#   # screenshot: ui/home.png
"""

#: The single line `init` ensures in the repo's .gitignore (the run dir + config
#: live under .grindstone/ and must never be committed).
_GITIGNORE_LINE = ".grindstone/"

_EXIT: dict[str, int] = {"completed": 0, "escalated": 1, "failed": 2}


def _default_run_id() -> str:
    """A UTC timestamp slug, e.g. ``20260610T142530Z`` (collision-resistant id)."""

    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


# --- config-driven transport resolution (flag > config > built-in default) -----


def _no_config_exit() -> SystemExit:
    """The shared failure when ``run`` / ``resume`` find no config (no rig defaults)."""

    return SystemExit(
        "no .grindstone/config.yaml found, run `grindstone init --repo <repo>` "
        "first, then edit the role scripts it scaffolds (core ships no rig "
        "defaults)."
    )


def _resolve_planner(
    args: argparse.Namespace, cfg: GrindstoneConfig | None, repo: Path
) -> PlannerTransport:
    """The planner transport: the ``planner`` role's request script.

    The planner is ALWAYS the script (no transport branching, the script owns
    transport + model). No config fails loudly toward ``grindstone init``.
    """

    if cfg is None:
        raise _no_config_exit()
    return ScriptPlanner(
        script=resolve_role_script("planner", cfg.roles.planner),
        stop_script=models_script("stop.sh"),
        repo=repo,
        slots=cfg.roles.planner.slots,
        timeout_s=cfg.roles.planner.timeout_s,
    )


def _resolve_ladder(
    args: argparse.Namespace, cfg: GrindstoneConfig | None, run_dir: RunDir
) -> Ladder:
    """The worker ladder from the role scripts.

    The ``worker`` role is the required first rung (a ``ScriptWorker`` over
    ``worker_request.sh``); the optional ``senior`` role appends a second rung
    (the cloud escalation tier) when present. Each worker logs under
    ``<run-dir>/worker_logs`` (never the run dir's state). With NO config the
    core ships no rig-specific defaults (ARCHITECTURE.md: no model config in core), it
    fails loudly toward ``grindstone init``, whose scaffold carries the editable
    role scripts for the operator to confirm.
    """

    if cfg is None:
        raise _no_config_exit()
    log_root = run_dir.root / "worker_logs"
    stop_script = models_script("stop.sh")
    tiers: list[tuple[str, ScriptWorker]] = [
        (
            "worker",
            ScriptWorker(
                script=resolve_role_script("worker", cfg.roles.worker),
                stop_script=stop_script,
                slots=cfg.roles.worker.slots,
                timeout_s=cfg.roles.worker.timeout_s,
                log_root=log_root,
            ),
        )
    ]
    if cfg.roles.senior is not None:
        senior = cfg.roles.senior
        tiers.append(
            (
                "senior",
                ScriptWorker(
                    script=resolve_role_script("senior", senior),
                    stop_script=stop_script,
                    slots=senior.slots,
                    timeout_s=senior.timeout_s,
                    log_root=log_root,
                ),
            )
        )
    return tiers


def _resolve_concurrency(
    args: argparse.Namespace, cfg: GrindstoneConfig | None
) -> int | None:
    """Fan-out cap = the ``worker`` role's slot count (one in-flight task / slot).

    No config fails loudly toward ``grindstone init`` (run/resume already exit
    in ``_resolve_ladder`` first, but the bound is meaningless without config).
    """

    if cfg is None:
        raise _no_config_exit()
    return cfg.roles.worker.slots


def _resolve_max_planner_calls(cfg: GrindstoneConfig | None) -> int:
    """Planner-call ceiling: config > built-in default, NEVER unbounded.

    ``run_grind``'s ``None`` (= off) stays available as a test seam only; every
    CLI-driven run gets a cap so an unattended revision spin cannot drain the
    planner subscription (gate-5 P0).
    """

    if cfg is not None and cfg.max_planner_calls is not None:
        return cfg.max_planner_calls
    return DEFAULT_MAX_PLANNER_CALLS


def _resolve_failed_epoch_cap(cfg: GrindstoneConfig | None) -> int:
    """The per-phase failed-epoch cap: config > built-in default (Part C).

    After this many failed epochs in one phase the state machine forces a
    halt-to-human regardless of the planner, the deterministic backstop against
    the dogfood spin-loop. ``cfg`` carries its own default (3) when present."""

    if cfg is not None:
        return cfg.max_failed_epochs_per_phase
    return DEFAULT_MAX_FAILED_EPOCHS_PER_PHASE


def _resolve_task_file_bounds(cfg: GrindstoneConfig | None) -> tuple[int, int]:
    """The deterministic size-gate ``(local, senior)`` file-count bounds (Part 4B).

    Config > the built-in defaults (which mirror ``GrindstoneConfig``'s own field
    defaults, so a config-less run and a default config agree)."""

    if cfg is not None:
        return cfg.local_max_task_files, cfg.senior_max_task_files
    return DEFAULT_LOCAL_MAX_TASK_FILES, DEFAULT_SENIOR_MAX_TASK_FILES


def _resolve_vision_reviewer(cfg: GrindstoneConfig | None) -> VisionReviewer | None:
    """The B3 taste-gate reviewer: the config'd ``vision_review`` script, else the
    bundled ``models/vision_review.sh`` (the codex call is swappable via the
    script path). ``None`` only when there is no config at all (run/resume already
    fail loudly earlier); the gate then degrades to a deterministic FAIL on any
    ``vision_review`` check rather than crashing."""

    if cfg is None:
        return None
    if cfg.vision_review is not None:
        return ScriptVisionReviewer(
            script=cfg.vision_review.script, timeout_s=cfg.vision_review.timeout_s
        )
    # No config block: fall back to a bundled vision_review.sh resolved through the
    # rig stack. The shipped default rig carries no taste gate (it is a codex-
    # flavored script that lives in models/personal on a rig that uses it), so when
    # none resolves we return None and the gate degrades to a deterministic FAIL on
    # any vision_review check rather than crashing a run that has no such check.
    try:
        script = models_script("vision_review.sh")
    except FileNotFoundError:
        return None
    return ScriptVisionReviewer(script=script, timeout_s=DEFAULT_VISION_TIMEOUT_S)


def _resolve_verifiers(
    ladder: Ladder, cfg: GrindstoneConfig | None
) -> dict[str, TaskVerifier] | None:
    """The tier-aware per-task verifiers: ``{tier_name: WorkerTaskVerifier}``.

    Each task is verified at the tier that BUILT it (a ``senior`` task by the senior
    verifier, every other by the local ``worker`` verifier), so the critic is at the
    builder's strength, never a weaker model gatekeeping a stronger one. Every ladder
    tier gets its own ``WorkerTaskVerifier`` wrapping that tier's transport; the verifier
    dispatch is a SEPARATE fresh invocation (no inherited builder session), so it stays
    an independent critic. Default ON (``verify_epochs`` true): the pass runs whenever a
    task carries ``criteria``. ``None`` (the pass is disabled for every tier) when the
    config opts out (``verify_epochs: false``) or the ladder is empty; the task loop then
    skips the agentic pass and the deterministic floor alone gates."""

    if cfg is not None and not cfg.verify_epochs:
        return None
    verifiers: dict[str, TaskVerifier] = {
        name: WorkerTaskVerifier(transport) for name, transport in ladder
    }
    return verifiers or None


def _resolve_final_polish(cfg: GrindstoneConfig | None) -> FinalPolish | None:
    """The B5 final-polish wiring: ``None`` unless the config opts in (OFF by
    default, codex never touches a completed run without it). When present, the
    polisher uses the config'd ``script`` else the bundled ``models/codex_polish.sh``
    (the codex call is swappable via the script path, e.g. a stub in tests)."""

    if cfg is None or cfg.final_polish is None:
        return None
    fp = cfg.final_polish
    # `script` omitted -> the bundled codex_polish.sh, resolved through the rig
    # stack. final_polish is opt-in; if no rig supplies the script, models_script
    # raises a clear error (the operator must point `final_polish.script` at one).
    script = fp.script if fp.script is not None else models_script("codex_polish.sh")
    return FinalPolish(
        polisher=ScriptPolisher(script=script, timeout_s=fp.timeout_s),
        criteria=fp.criteria,
        screenshot_rel=fp.screenshot,
    )


def _resolve_run_dir(target: str, repo: str | None) -> RunDir:
    """A run dir from either a directory path or a run id under ``--repo``."""

    path = Path(target)
    if path.is_dir():
        return RunDir(root=path)
    if repo is None:
        raise SystemExit(f"{target!r} is not a run dir; pass --repo to resolve it as a run id")
    return RunDir(root=Path(repo).resolve() / ".grindstone" / "runs" / target)


def _report(outcome: RunOutcome) -> None:
    print(
        f"status={outcome.status} planner_calls={outcome.planner_calls} "
        f"epochs={outcome.epochs_run} branch={outcome.final_branch}"
    )
    if outcome.summary:
        print(f"summary: {outcome.summary}")
    if outcome.reason:
        print(f"reason: {outcome.reason}")


def _cmd_init(args: argparse.Namespace) -> int:
    """Scaffold ``.grindstone/config.yaml`` + ignore the run dir (S5 ruling 2).

    Idempotent: an existing config is left untouched (init is a one-time
    scaffold, never a reset), and ``.grindstone/`` is appended to ``.gitignore``
    only when not already ignored.
    """

    repo = Path(args.repo).resolve()
    cfg_path = repo / ".grindstone" / "config.yaml"
    if cfg_path.exists():
        print(f"config exists, leaving as-is: {cfg_path}")
    else:
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        cfg_path.write_text(_render_config_template(args.rig), encoding="utf-8")
        print(f"wrote {cfg_path}")
    if _ensure_gitignored(repo):
        print(f"added {_GITIGNORE_LINE} to .gitignore")
    return 0


def _ensure_gitignored(repo: Path) -> bool:
    """Append ``.grindstone/`` to the repo's ``.gitignore`` if missing.

    Returns whether the line was added (so init can report it). Match is on the
    stripped line so a pre-existing ignore is never duplicated.
    """

    gitignore = repo / ".gitignore"
    existing = gitignore.read_text(encoding="utf-8") if gitignore.is_file() else ""
    if _GITIGNORE_LINE in {line.strip() for line in existing.splitlines()}:
        return False
    prefix = "" if existing == "" or existing.endswith("\n") else "\n"
    gitignore.write_text(existing + prefix + _GITIGNORE_LINE + "\n", encoding="utf-8")
    return True


def _cmd_run(
    args: argparse.Namespace,
    *,
    planner: PlannerTransport | None,
    ladder: Ladder | None,
) -> int:
    repo = Path(args.repo).resolve()
    cfg = load_config(repo)
    if cfg is not None:
        validate_script_paths(cfg)  # reject an arbitrary-executable repo config (RCE)
    run_id = args.run_id or _default_run_id()
    run_dir = create_run_dir(repo, run_id)
    planner = planner or _resolve_planner(args, cfg, repo)
    ladder = ladder or _resolve_ladder(args, cfg, run_dir)

    def _grind() -> RunOutcome:
        return run_grind(
            run_dir,
            job_path=args.job,
            planner=planner,
            ladder=ladder,
            repo=repo,
            run_id=run_id,
            concurrency=_resolve_concurrency(args, cfg),
            max_planner_calls=_resolve_max_planner_calls(cfg),
            vision_reviewer=_resolve_vision_reviewer(cfg),
            final_polish=_resolve_final_polish(cfg),
            prepare=cfg.prepare if cfg is not None else None,
            floor=cfg.floor if cfg is not None else None,
            verifiers=_resolve_verifiers(ladder, cfg),
            infra_repair=cfg.infra_repair if cfg is not None else None,
            max_failed_epochs_per_phase=_resolve_failed_epoch_cap(cfg),
            local_max_task_files=_resolve_task_file_bounds(cfg)[0],
            senior_max_task_files=_resolve_task_file_bounds(cfg)[1],
        )

    print(f"run {run_id} -> {run_dir.root}")
    outcome = _watched_grind(_grind, run_dir) if args.watch else _grind()
    _report(outcome)
    return _EXIT[outcome.status]


def _watched_grind(grind: Callable[[], RunOutcome], run_dir: RunDir) -> RunOutcome:
    """Run ``grind`` in a background thread while the live TUI renders in the
    foreground, the single human command. The grind owns every write; the TUI is
    the same pure journal reader as ``watch``. We join the grind before returning
    so its workers are reaped cleanly (a Ctrl-C stops the view, then the run still
    finishes, runs are resumable, so abandoning is recoverable, but waiting is
    tidier). The grind's outcome (not the rendered tree) is the return value."""

    holder: dict[str, RunOutcome] = {}
    error: dict[str, BaseException] = {}

    def _worker() -> None:
        try:
            holder["outcome"] = grind()
        except BaseException as exc:  # re-raised in the main thread after join
            error["exc"] = exc

    thread = threading.Thread(target=_worker, name="grind")
    thread.start()
    try:
        watch(run_dir)  # follow=False: returns when the journal goes terminal
    finally:
        thread.join()
    if "exc" in error:
        raise error["exc"]
    return holder["outcome"]


def _cmd_resume(
    args: argparse.Namespace,
    *,
    planner: PlannerTransport | None,
    ladder: Ladder | None,
) -> int:
    repo = Path(args.repo).resolve()
    cfg = load_config(repo)
    if cfg is not None:
        validate_script_paths(cfg)  # reject an arbitrary-executable repo config (RCE)
    run_dir = RunDir(root=repo / ".grindstone" / "runs" / args.run_id)
    planner = planner or _resolve_planner(args, cfg, repo)
    ladder = ladder or _resolve_ladder(args, cfg, run_dir)
    outcome = resume_grind(
        run_dir,
        planner=planner,
        ladder=ladder,
        repo=repo,
        concurrency=_resolve_concurrency(args, cfg),
        max_planner_calls=_resolve_max_planner_calls(cfg),
        vision_reviewer=_resolve_vision_reviewer(cfg),
        final_polish=_resolve_final_polish(cfg),
        prepare=cfg.prepare if cfg is not None else None,
        floor=cfg.floor if cfg is not None else None,
        verifiers=_resolve_verifiers(ladder, cfg),
        infra_repair=cfg.infra_repair if cfg is not None else None,
        max_failed_epochs_per_phase=_resolve_failed_epoch_cap(cfg),
        local_max_task_files=_resolve_task_file_bounds(cfg)[0],
        senior_max_task_files=_resolve_task_file_bounds(cfg)[1],
    )
    _report(outcome)
    return _EXIT[outcome.status]


def _cmd_watch(args: argparse.Namespace) -> int:
    run_dir = _resolve_run_dir(args.target, args.repo)
    tree = watch(run_dir, follow=args.follow)
    if tree is None:
        return 0
    # Mirror the `run` command's exit convention: a watched run that ended
    # escalated/failed surfaces non-zero (a KeyboardInterrupt mid-run -> 0).
    return _EXIT.get(tree.status, 0)


def _add_transport_flags(p: argparse.ArgumentParser) -> None:
    # The CLI no longer knows models / transports / slots, the role scripts and
    # `.grindstone/config.yaml` own them. Only the run-id remains operator-facing.
    p.add_argument("--run-id", help="run id (default: UTC timestamp slug)")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="grindstone", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init", help="scaffold .grindstone/config.yaml + .gitignore")
    init.add_argument("--repo", required=True, help="target repo to initialize")
    init.add_argument(
        "--rig",
        default=None,
        help="bundled preset rig for the planner/worker scripts (e.g. `codex`), "
        "searched before the shipped Claude floor; omit it for the implicit "
        "default, where an operator's models/personal scripts win where present.",
    )

    run = sub.add_parser("run", help="run a job.md to completion in the foreground")
    run.add_argument("job", help="path to the job spec (job.md)")
    run.add_argument("--repo", required=True, help="target repo the run writes into")
    run.add_argument(
        "--watch",
        action="store_true",
        help="render the live TUI while the run grinds (one command for a human; "
        "agents use plain `run`). Exits with the run's status when it finishes.",
    )
    _add_transport_flags(run)

    watch_p = sub.add_parser("watch", help="render a run's live tree (TUI)")
    watch_p.add_argument("target", help="run dir path or run id (with --repo)")
    watch_p.add_argument("--repo", default=None, help="repo to resolve a run id under")
    watch_p.add_argument("--follow", action="store_true", help="keep watching past terminal")

    resume = sub.add_parser("resume", help="re-enter a killed run by id")
    resume.add_argument("run_id", help="the run id to resume")
    resume.add_argument("--repo", required=True, help="target repo the run lives in")
    _add_transport_flags(resume)

    return parser


def main(
    argv: list[str] | None = None,
    *,
    planner: PlannerTransport | None = None,
    ladder: Ladder | None = None,
) -> int:
    """Parse ``argv`` and dispatch. ``planner`` / ``ladder`` override the built-in
    transports (the test seam, like ``run_grind``'s injected ``sleep_fn``)."""

    args = _build_parser().parse_args(argv)
    if args.command == "init":
        return _cmd_init(args)
    if args.command == "run":
        return _cmd_run(args, planner=planner, ladder=ladder)
    if args.command == "resume":
        return _cmd_resume(args, planner=planner, ladder=ladder)
    return _cmd_watch(args)
