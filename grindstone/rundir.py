"""Run-dir layout: ``.grindstone/runs/<run-id>/`` under a target repo.

ARCHITECTURE.md: log keys ARE relative paths under the run dir. This module owns the
directory shape and the traversal guard that keeps every resolved key inside the
run dir. There is no durable state file: resume re-derives the run's position from
the append-only ``events.ndjson`` journal plus the git run-branch tip (BONES), so a
parsed cursor file can never drift from the real boundary.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path

#: Default external base for throwaway git worktrees. ``/tmp`` here is disk-backed
#: ext4 (not tmpfs), so a multi-GB checkout is safe. Operators relocate via the
#: ``GRINDSTONE_WORKTREE_BASE`` env override; tests redirect it into their tmp dir.
_DEFAULT_WORKTREE_BASE = "/tmp/cache/grindstone"

#: Log-key grammar, mirrored from schemas/epoch_decision.json $defs/log_key.
_LOG_KEY_RE = re.compile(r"^[A-Za-z0-9][a-zA-Z0-9._/-]{0,127}$")

#: A per-epoch keyed-log dir (``E1``, ``E2``, ...), the root of the keyed log.
_EPOCH_DIR_RE = re.compile(r"^E[1-9][0-9]*$")


def _contained(root: Path, candidate: Path) -> Path:
    """Resolve ``candidate`` and assert it stays inside ``root`` (real path).

    Catches what the log-key regex cannot: embedded ``..`` segments that pass
    the character class, and symlinks pointing out of the run dir.
    """

    resolved = candidate.resolve()
    root_resolved = root.resolve()
    if resolved != root_resolved and root_resolved not in resolved.parents:
        raise ValueError(f"path escapes run dir: {candidate}")
    return resolved


@dataclass(frozen=True)
class RunDir:
    """Paths for one run; the run id is the directory name under runs/."""

    root: Path

    @property
    def events_path(self) -> Path:
        return self.root / "events.ndjson"

    @property
    def worktrees_root(self) -> Path:
        """External base for this run's throwaway git worktrees, OUTSIDE the repo.

        A worktree nested under the target repo lets a worker that strips its CWD
        back to the repo root write into the MAIN checkout instead of its isolated
        worktree (run 124321Z RCA: a local model wrote ``src/`` into the live repo,
        so its worktree-relative ``done_when`` could never pass). Hosting the
        worktrees on an external base removes the nesting, so the path can no longer
        be stripped to the repo. The model-WRITTEN executor worktrees (task attempts,
        infra-repair, polish) plus the orchestrator scratch + staging trees move out;
        durable run STATE (events, state, handoffs, artifacts, keyed log) stays under
        ``root``, as does the one planner-READ tip (``_planner_tip``), which a
        sandboxed planner rig must reach inside the repo.

        Layout ``<base>/<repo-id>/<run-id>/worktrees`` keeps two repos that share a
        run-id from colliding (``repo-id`` folds the resolved repo path into a short
        hash). The base defaults to ``/tmp/cache/grindstone`` and honors the
        ``GRINDSTONE_WORKTREE_BASE`` override (operator relocation; test isolation).
        """

        repo = self.root.parent.parent.parent
        base = Path(os.environ.get("GRINDSTONE_WORKTREE_BASE", _DEFAULT_WORKTREE_BASE))
        repo_id = f"{repo.name}-{hashlib.sha1(str(repo.resolve()).encode()).hexdigest()[:8]}"
        return base / repo_id / self.root.name / "worktrees"

    @property
    def journal_path(self) -> Path:
        """Human-facing markdown post-mortem, rendered from ``events.ndjson`` at
        terminal. Kept only for the LATEST run (reaped when the next run starts);
        derived, so reaping it loses nothing the events can't re-render."""

        return self.root / "journal.md"

    def log_index(self) -> list[str]:
        """Sorted relative paths of the durable keyed log (ARCHITECTURE.md).

        The log keys a planner may reference as task ``inputs``: every regular
        file under an epoch dir (``E<n>/...``, handoffs, outcomes, relocated
        artifacts). Excludes ``events.ndjson`` / ``journal.md`` / artifact
        scratch, none of which are durable references (the throwaway git
        worktrees live on an external base outside the run dir entirely; see
        ``worktrees_root``).
        """

        out: list[str] = []
        for path in self.root.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(self.root)
            if rel.parts and _EPOCH_DIR_RE.match(rel.parts[0]):
                out.append(rel.as_posix())
        return sorted(out)

    def resolve(self, log_key: str) -> Path:
        """Map a log key to its path, rejecting bad grammar or traversal."""

        if not _LOG_KEY_RE.match(log_key):
            raise ValueError(f"invalid log key: {log_key!r}")
        return _contained(self.root, self.root / log_key)

    def baton_path(self, epoch_index: int) -> Path:
        """The keyed-log path the close-out planner's living BATON is persisted at for
        epoch ``epoch_index`` (``E<n>/baton.md``). The latest completed epoch's baton
        is "current"; history is free (every epoch keeps its own)."""

        return self.resolve(f"E{epoch_index}/baton.md")

    def read_baton(self, epoch_index: int) -> str:
        """The prior epoch's baton text, or ``""`` when absent/unreadable (NEVER
        raises). Seeds the next PLAN's context: epoch 1 reads ``E0/baton.md`` (absent)
        and gets ``""``, the first-epoch signal."""

        try:
            path = self.baton_path(epoch_index)
        except ValueError:
            return ""
        try:
            return path.read_text(encoding="utf-8")
        except OSError:
            return ""

    def artifacts_dir(self, task_key: str) -> Path:
        """Scratch dir for a non-write task; created, guarded for containment."""

        path = _contained(self.root, self.root / "artifacts" / task_key)
        path.mkdir(parents=True, exist_ok=True)
        return path


def create_run_dir(repo_root: Path, run_id: str) -> RunDir:
    """Create ``<repo_root>/.grindstone/runs/<run_id>/``; fail if it exists."""

    root = Path(repo_root) / ".grindstone" / "runs" / run_id
    root.mkdir(parents=True, exist_ok=False)
    return RunDir(root=root)
