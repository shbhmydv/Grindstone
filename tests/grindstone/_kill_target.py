"""Subprocess entrypoint for the kill-mid-EPOCH signature test (not collected).

Drives ``run_epoch`` with a 2-task implement epoch on a throwaway git repo. T1
runs to DONE fast; T2 blocks at a deterministic sync point (writes a ``ready``
sentinel, then busy-waits on a ``release`` sentinel that never appears). The
parent waits until T1 is DONE *and* T2 is in flight, then SIGKILLs, exercising
ruling 9 (>=1 task DONE, >=1 in flight at kill).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grindstone.mock_worker import MockWorker  # noqa: E402
from grindstone.rundir import create_run_dir  # noqa: E402
from grindstone.worker import WorkerRequest  # noqa: E402
from tests.grindstone.conftest import (  # noqa: E402
    OUT_CONTENT,
    implement_epoch,
    make_toy_task,
    run_one_epoch,
)


class _BlockingWorker:
    def __init__(self, ready: Path, release: Path) -> None:
        self.ready = ready
        self.release = release

    def run(self, request: WorkerRequest) -> None:
        self.ready.write_text("ready", encoding="utf-8")
        # Stay in flight until SIGKILLed (release never appears). Poll on a
        # short sleep, NOT a sched_yield hot-spin, a leaked subprocess must
        # never peg a CPU (the reaper bounds its lifetime, but defence in
        # depth keeps it cheap meanwhile).
        while not self.release.exists():
            time.sleep(0.01)


class _Router:
    def __init__(self, ready: Path, release: Path) -> None:
        self._ok = MockWorker(script=["ok"], artifacts={"f1.txt": OUT_CONTENT})
        self._block = _BlockingWorker(ready, release)

    def run(self, request: WorkerRequest) -> None:
        short = request.task_id.rsplit("/", 1)[-1]
        (self._ok if short == "T1" else self._block).run(request)


def main() -> None:
    repo_root, run_id, ready, release = sys.argv[1:5]
    repo = Path(repo_root)
    run_dir = create_run_dir(repo, run_id)
    t1 = make_toy_task(task_id="T1", out_file="f1.txt", owned=["f1.txt"])
    t2 = make_toy_task(task_id="T2", out_file="f2.txt", owned=["f2.txt"])
    run_one_epoch(
        run_dir,
        args=implement_epoch(t1, t2),
        mode="implement",
        ladder=[("local", _Router(Path(ready), Path(release)))],
        repo=repo,
        concurrency=2,
    )


if __name__ == "__main__":
    main()
