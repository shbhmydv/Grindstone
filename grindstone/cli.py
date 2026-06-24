"""``grindstone`` command line: run a job, resume a killed run, watch a run's tree.

A thin front end over the epoch loop (BONES). ``run`` / ``resume`` drive the
state machine (built in a later part); ``watch`` renders a run's tree from its
append-only ``events.ndjson`` (the journal is the single source of truth, so the
watcher needs no live state). ``python -m grindstone --help`` lists the verbs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from grindstone import loop
from grindstone.events import RunTree, read_events, replay
from grindstone.rundir import RunDir


def _render_tree(tree: RunTree) -> str:
    """One-line-per-node text rendering of a replayed run tree."""

    lines = [f"run {tree.run_id} [{tree.status}] ({tree.job_path})"]
    if tree.last_rate_limit:
        lines.append(f"  rate-limited: {tree.last_rate_limit}")
    for epoch in tree.epochs:
        lines.append(f"  epoch {epoch.id} [{epoch.status}] {epoch.title}")
        for task in epoch.tasks:
            note = f" -- {task.note}" if task.note else ""
            lines.append(f"    {task.id} ({task.mode}) [{task.status}]{note}")
    if tree.end_summary:
        lines.append(f"  ended: {tree.end_summary}")
    return "\n".join(lines)


def _cmd_run(args: argparse.Namespace) -> int:
    try:
        return loop.run(Path(args.job), Path(args.repo), run_id=args.run_id)
    except NotImplementedError as exc:
        print(f"grindstone run: {exc}")
        return 2


def _cmd_resume(args: argparse.Namespace) -> int:
    try:
        return loop.resume(args.run_id, Path(args.repo))
    except NotImplementedError as exc:
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
    events_path = run_dir.events_path
    if not events_path.is_file():
        print(f"grindstone watch: no events at {events_path}")
        return 2
    events = read_events(events_path)
    if not events:
        print("grindstone watch: journal is empty")
        return 2
    print(_render_tree(replay(events)))
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

    watch = sub.add_parser("watch", help="render a run's tree from its journal")
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
