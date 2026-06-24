"""``grindstone`` command line: run a job, resume a killed run, watch a run's tree.

A thin front end over the epoch loop (BONES). ``run`` / ``resume`` build the real
planner + backends from the repo config and drive the state machine (``loop.run`` /
``loop.resume``), returning a process exit code; ``watch`` renders a run's journal
from its append-only ``events.ndjson`` (the journal is the single source of truth,
so the watcher needs no live state). ``python -m grindstone --help`` lists the verbs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from grindstone import loop
from grindstone.journal import render_run_dir
from grindstone.rundir import RunDir


def _cmd_run(args: argparse.Namespace) -> int:
    try:
        return loop.run(Path(args.job), Path(args.repo), run_id=args.run_id)
    except (FileNotFoundError, ValueError) as exc:
        print(f"grindstone run: {exc}")
        return 2


def _cmd_resume(args: argparse.Namespace) -> int:
    try:
        return loop.resume(args.run_id, Path(args.repo))
    except (FileNotFoundError, ValueError) as exc:
        print(f"grindstone resume: {exc}")
        return 2


def _resolve_run_dir(target: str, repo: str | None) -> RunDir:
    """A run-dir path, or a run id resolved under ``--repo``."""

    direct = Path(target)
    if direct.is_dir():
        return RunDir(root=direct)
    if repo is None:
        raise FileNotFoundError(
            f"{target!r} is not a run dir; pass a run id with --repo, or a run-dir path"
        )
    return RunDir(root=Path(repo) / ".grindstone" / "runs" / target)


def _cmd_watch(args: argparse.Namespace) -> int:
    try:
        run_dir = _resolve_run_dir(args.target, args.repo)
    except FileNotFoundError as exc:
        print(f"grindstone watch: {exc}")
        return 2
    rendered = render_run_dir(run_dir)
    if rendered is None:
        print(f"grindstone watch: no renderable run at {run_dir.events_path}")
        return 2
    print(rendered, end="")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="grindstone", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run a job.md to a clean terminal")
    run.add_argument("job", help="path to the job spec (job.md)")
    run.add_argument("--repo", required=True, help="target repo the run writes into")
    run.add_argument("--run-id", default=None, help="run id (default: UTC timestamp slug)")
    run.set_defaults(func=_cmd_run)

    resume = sub.add_parser("resume", help="re-enter a killed run by id")
    resume.add_argument("run_id", help="the run id to resume")
    resume.add_argument("--repo", required=True, help="target repo the run lives in")
    resume.set_defaults(func=_cmd_resume)

    watch = sub.add_parser("watch", help="render a run's journal from its events")
    watch.add_argument("target", help="run dir path, or a run id (with --repo)")
    watch.add_argument("--repo", default=None, help="repo to resolve a run id under")
    watch.set_defaults(func=_cmd_watch)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    func = args.func
    result = func(args)
    assert isinstance(result, int)
    return result
