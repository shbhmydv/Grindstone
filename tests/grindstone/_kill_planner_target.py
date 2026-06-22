"""Subprocess entrypoint for the kill-mid-PLANNER-CALL signature test (S3 ruling
6; not collected). Drives ``run_grind`` with a planner that blocks at a file
sentinel on its FIRST call (so the kill lands while a planner call is in flight,
status=awaiting_planner, nothing on disk). The parent SIGKILLs once the sentinel
appears, then resumes, proving the in-flight call is RE-ISSUED, not burned.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grindstone.rundir import create_run_dir  # noqa: E402
from grindstone.run_loop import run_grind  # noqa: E402
from tests.grindstone.conftest import OwnershipWorker, init_git_repo  # noqa: E402


class _BlockingPlanner:
    """Writes a ``ready`` sentinel inside ``plan()``, then busy-waits forever."""

    def __init__(self, ready: Path, release: Path) -> None:
        self.ready = ready
        self.release = release

    def plan(self, prompt: str, *, workdir: Path | None = None) -> str:
        self.ready.write_text("ready", encoding="utf-8")
        while not self.release.exists():
            os.sched_yield()
        return "{}"  # never reached (the parent kills first)


def main() -> None:
    repo_root, run_id, ready, release = sys.argv[1:5]
    repo = init_git_repo(Path(repo_root))
    run_dir = create_run_dir(repo, run_id)
    run_grind(
        run_dir,
        job_path="job.md",
        planner=_BlockingPlanner(Path(ready), Path(release)),
        ladder=[("worker", OwnershipWorker())],
        repo=repo,
    )


if __name__ == "__main__":
    main()
