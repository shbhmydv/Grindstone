"""Subprocess entrypoint for the kill-mid-PHASE-TRANSITION signature test (S4
ruling 6; not collected). Drives ``run_grind`` with a planner that completes
phase P1 (its epoch satisfies P1's exit criterion), lets the core fire
``phase_passed(P1)`` + advance to P2, then BLOCKS at a file sentinel on the
NEXT planner call (now in P2). The parent SIGKILLs once the transition is on
disk, then resumes, proving criteria are re-evaluated idempotently (no
duplicate ``phase_passed`` for the already-passed P1).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grindstone.rundir import create_run_dir  # noqa: E402
from grindstone.run_loop import run_grind  # noqa: E402
from tests.grindstone.conftest import (  # noqa: E402
    OwnershipWorker,
    check_cmd,
    impl_task,
    implement_decision,
    init_git_repo,
    phase_dict,
    skeleton_decision,
)


class _PhasePlanner:
    """Scripted for skeleton + the P1 epoch; the THIRD call (in P2) blocks."""

    def __init__(self, ready: Path, release: Path) -> None:
        self.ready = ready
        self.release = release
        self.calls = 0
        self.script = [
            skeleton_decision(
                phase_dict("P1", title="build", exit_criterion=[check_cmd("test -f f1.txt")]),
                phase_dict("P2", title="verify", exit_criterion=[check_cmd("test -f f2.txt")]),
            ),
            implement_decision(impl_task("T1", "f1.txt")),
        ]

    def plan(self, prompt: str) -> str:
        i = self.calls
        self.calls += 1
        if i < len(self.script):
            return json.dumps(self.script[i])
        # Post-transition (P1 passed, advanced to P2): block until killed.
        self.ready.write_text("ready", encoding="utf-8")
        while not self.release.exists():
            os.sched_yield()
        return "{}"  # never reached


def main() -> None:
    repo_root, run_id, ready, release = sys.argv[1:5]
    repo = init_git_repo(Path(repo_root))
    run_dir = create_run_dir(repo, run_id)
    run_grind(
        run_dir,
        job_path="job.md",
        planner=_PhasePlanner(Path(ready), Path(release)),
        ladder=[("local", OwnershipWorker())],
        repo=repo,
    )


if __name__ == "__main__":
    main()
