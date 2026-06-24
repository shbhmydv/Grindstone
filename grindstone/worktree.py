"""Per-attempt git worktrees + ownership-scoped fast-forward integration.

ARCHITECTURE.md / S2 rulings 4-7. Implement tasks each run in a throwaway worktree
branched from the **epoch base** (the repo tip at epoch dispatch). The worker's
writes land there, the core scope-checks the diff against the task's
``file_ownership`` globs, commits on success (models never run git), and at the
done-predicate the core fast-forward-merges every DONE task's branch, in task
order, into the epoch integration branch. Given a fresh integration branch
started at the epoch base each epoch (the branch name is not run-scoped, so
``loop._integrate`` drops any stale same-named leftover before a fresh
integration), pairwise-disjoint ownership plus the scope check make the merges
commute, so ANY merge conflict is a structural bug and aborts the epoch
(``integration_conflict``), never a retried runtime path.

Pure subprocess git (no GitPython). The scars ported from the v7 pipeline,
worktrees as scratch CWD, force removal + prune of debris, merge in an isolated
worktree so the operator's checkout is never touched, are reimplemented here,
not imported.

Every git op runs with ``cwd`` set to a caller-supplied repo or worktree path;
this module NEVER targets the orchestrator's own checkout.
"""

from __future__ import annotations

import fnmatch
import shutil
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path

#: Serializes the worktree-REGISTRY mutations (``git worktree add`` / ``remove`` /
#: ``prune``). These touch a repo's SHARED ``.git/worktrees`` admin state and the new
#: branch ref, so git is not safe to run them concurrently in one repo, yet the epoch
#: fan-out creates each implement attempt's worktree from a parallel thread. A
#: process-global lock serializes only these sub-millisecond git-admin ops; the slow
#: model dispatch stays concurrent (it runs outside this lock, bounded by the
#: per-backend semaphores). Surfaced by the stochastic convergence E2E: concurrent
#: ``git worktree add`` intermittently exited 128 under contention.
_WORKTREE_LOCK = threading.Lock()

#: Identity for core-authored commits/merges (models never run git, so the
#: committer is always Grindstone itself). Passed per-command so a throwaway
#: repo needs no pre-seeded ``user.*`` config.
_GIT_IDENTITY = (
    "-c",
    "user.name=Grindstone",
    "-c",
    "user.email=grindstone@localhost",
    "-c",
    "commit.gpgsign=false",
)


class GitError(RuntimeError):
    """A git command the core required to succeed exited non-zero."""


@dataclass(frozen=True)
class MergeOutcome:
    """Result of merging one branch into the integration branch."""

    ok: bool
    conflict: str = ""


def _git(
    repo: Path, *args: str, check: bool = True
) -> subprocess.CompletedProcess[str]:
    """Run one git command in ``repo``; raise ``GitError`` on failure if checked.

    Binary-safe decode (``errors='replace'``): a side-effect command may emit
    non-UTF-8 on stderr and must never crash the loop.
    """

    proc = subprocess.run(
        ["git", *args],
        cwd=str(repo),
        capture_output=True,
        text=True,
        errors="replace",
    )
    if check and proc.returncode != 0:
        raise GitError(
            f"git {' '.join(args)} (cwd={repo}) exited {proc.returncode}: "
            f"{(proc.stderr or proc.stdout).strip()}"
        )
    return proc


# --- base + branch helpers -----------------------------------------------------


def resolve_commit(repo: Path, ref: str) -> str:
    """Resolve any commit-ish (branch / sha / HEAD) to a concrete commit sha.

    Epoch chaining (ARCHITECTURE.md ruling 4): the next epoch's base is the previous
    epoch's integration-branch *tip*, pinned to a sha at dispatch so a later
    branch move cannot shift an already-captured base.
    """

    return _git(repo, "rev-parse", ref).stdout.strip()


def head_commit(repo: Path) -> str:
    """The repo tip commit sha, the first epoch's base captured at dispatch."""

    return resolve_commit(repo, "HEAD")


def list_tree(repo: Path, ref: str) -> list[str]:
    """Every tracked file path at ``ref`` (``git ls-tree -r --name-only``).

    The cumulative-state surfacing primitive (S4 ruling 3b): a reference listing
    of the integration tip the planner plans against, names only, never bodies.
    Returns ``[]`` when ``ref`` cannot be resolved (e.g. a not-yet-built branch).
    """

    proc = _git(repo, "ls-tree", "-r", "--name-only", ref, check=False)
    if proc.returncode != 0:
        return []
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def branch_exists(repo: Path, branch: str) -> bool:
    return (
        _git(repo, "show-ref", "--verify", "--quiet", f"refs/heads/{branch}", check=False).returncode
        == 0
    )


def is_ancestor(repo: Path, maybe_ancestor: str, descendant: str) -> bool:
    """True if ``maybe_ancestor`` is already in ``descendant``'s history.

    The structural idempotency check for integration resume: a branch already
    merged into the integration branch is its ancestor, so re-merging is a git
    no-op and may be skipped.
    """

    return (
        _git(repo, "merge-base", "--is-ancestor", maybe_ancestor, descendant, check=False).returncode
        == 0
    )


def delete_branch(repo: Path, branch: str) -> None:
    """Force-delete a branch if it exists (idempotent, zero dead refs)."""

    if branch_exists(repo, branch):
        _git(repo, "branch", "-D", branch, check=False)


def branches_with_prefix(repo: Path, prefix: str) -> list[str]:
    """Short names of every local branch under ``refs/heads/<prefix>*``.

    The resume / rate-limit RAZE primitive uses this to drop ALL transient
    ``grind-wip/`` branches of an incomplete epoch (the durable ``grind/<run-id>``
    run branch is never under that prefix, so it is never touched). Returns ``[]``
    when none match. Every head is listed and filtered in-process (a ref-glob
    ``*`` does not reliably span the slashes of a multi-level wip branch name)."""

    out = _git(
        repo, "for-each-ref", "--format=%(refname:short)", "refs/heads/", check=False
    ).stdout
    return [
        line.strip()
        for line in out.splitlines()
        if line.strip().startswith(prefix)
    ]


def fast_forward_branch(repo: Path, run_branch: str, commit: str) -> None:
    """Advance the persistent RUN branch ``run_branch`` to ``commit`` (create it
    there if it does not yet exist).

    The single ref that survives between epochs: each epoch stages its merges on a
    throwaway ``grind-wip/.../_staging`` branch off the current run tip, then on
    success this advances the run branch to that staging tip. The FIRST epoch
    creates ``run_branch`` at the staging tip; every later epoch is a true
    fast-forward (the staging branch was started off the run tip, so it descends
    from it). ``git branch -f`` covers both create-and-move uniformly; the run
    branch is never checked out in a worktree (the epoch worktrees are pruned
    before this runs), so the force always succeeds; the move is always a forward
    advance (the staging branch was started off the run tip).
    """

    _git(repo, "branch", "-f", run_branch, commit)


# --- worktree lifecycle --------------------------------------------------------


def add_worktree(repo: Path, path: Path, *, branch: str, base: str) -> None:
    """Create a fresh worktree at ``path`` on a new ``branch`` rooted at ``base``.

    The leaf dir must not pre-exist (git creates it); the parent is ensured.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        shutil.rmtree(path)
    with _WORKTREE_LOCK:
        _git(repo, "worktree", "prune")
        _git(repo, "worktree", "add", "-b", branch, str(path), base)


def add_worktree_on(repo: Path, path: Path, *, branch: str) -> None:
    """Create a worktree at ``path`` checking out an EXISTING ``branch``.

    Used for the integration worktree (the integration branch may already carry
    merges from before a kill, so it is checked out at its current tip).
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        shutil.rmtree(path)
    with _WORKTREE_LOCK:
        _git(repo, "worktree", "prune")
        _git(repo, "worktree", "add", str(path), branch)


def add_worktree_detached(repo: Path, path: Path, *, ref: str) -> None:
    """Create a read-only-purpose worktree at ``path`` on a DETACHED checkout of
    ``ref``.

    Used by the check evaluator (phase exit criteria / complete_run evidence):
    it only reads a tree, and `git worktree add <branch>` refuses any branch
    already checked out elsewhere, including the operator's own checkout (E2E
    gate2 P0). Detached HEAD cannot collide by construction.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        shutil.rmtree(path)
    with _WORKTREE_LOCK:
        _git(repo, "worktree", "prune")
        _git(repo, "worktree", "add", "--detach", str(path), ref)


def remove_worktree(repo: Path, path: Path) -> None:
    """Force-remove a worktree + prune its registration (idempotent)."""

    with _WORKTREE_LOCK:
        _git(repo, "worktree", "remove", "--force", str(path), check=False)
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
        _git(repo, "worktree", "prune")


def prune_tree(repo: Path, worktrees_dir: Path) -> None:
    """Delete the whole per-epoch worktrees dir + unregister it (ruling 7).

    Called after integration: the task branches carry the work, so the worktree
    checkouts are pure scratch and leave nothing behind.
    """

    if worktrees_dir.exists():
        shutil.rmtree(worktrees_dir, ignore_errors=True)
    with _WORKTREE_LOCK:
        _git(repo, "worktree", "prune")


def discard_attempt(repo: Path, path: Path, branch: str) -> None:
    """Tear down a failed/burned attempt: worktree removed + branch deleted.

    Ruling 4 zero-dead-artifacts: a rejected or killed attempt leaves nothing.
    """

    remove_worktree(repo, path)
    delete_branch(repo, branch)


# --- commit-on-success + scope check -------------------------------------------


def commit_all(worktree: Path, message: str) -> bool:
    """Stage everything and commit; return whether a commit was created.

    The core commits (models never run git). A zero-diff attempt stages nothing
    and creates no commit (``False``), HEAD stays at base, which integrates as
    a no-op. Identity is supplied per-command so no repo-level config is needed.
    """

    _git(worktree, "add", "-A")
    if _git(worktree, "diff", "--cached", "--quiet", check=False).returncode == 0:
        return False
    _git(worktree, *_GIT_IDENTITY, "commit", "--no-verify", "-q", "-m", message)
    return True


def changed_paths(repo: Path, base: str, head: str = "HEAD") -> list[str]:
    """Paths changed between ``base`` and ``head`` (the committed scope of work)."""

    out = _git(repo, "diff", "--name-only", f"{base}..{head}").stdout
    return [line.strip() for line in out.splitlines() if line.strip()]


def path_in_scope(path: str, ownership: list[str]) -> bool:
    """Does ``path`` match at least one ownership glob?

    fnmatch semantics (case-sensitive), with the explicit ``dir/**`` rule:
    ``dir/**`` matches ``dir`` itself and any file beneath it at any depth.
    Whole-repo ownership (``**`` or ``**/*``) matches every path, root and
    nested alike (fnmatch's ``**/*`` would otherwise miss root-level files).
    """

    for glob in ownership:
        if glob in ("**", "**/*"):
            return True
        if glob.endswith("/**"):
            prefix = glob[:-3]
            if path == prefix or path.startswith(prefix + "/"):
                return True
        elif fnmatch.fnmatchcase(path, glob):
            return True
    return False


def scope_violations(changed: list[str], ownership: list[str]) -> list[str]:
    """Changed paths that fall outside every ownership glob. Empty ownership is
    deny-all: with no globs, every changed path is a violation.

    A worker that installs project dependencies inside its own worktree (BONES: the
    setup seam no longer reaches the task worktrees) writes them under its declared
    ownership or relies on the repo's own ``.gitignore`` keeping them untracked, so
    only genuinely out-of-scope, tracked writes are reported here.
    """

    return [p for p in changed if not path_in_scope(p, ownership)]


# --- integration ---------------------------------------------------------------


def ensure_integration_branch(repo: Path, branch: str, base: str) -> None:
    """Create the integration branch at ``base`` if it does not yet exist.

    A no-op when the branch is already present, which is correct ONLY for a
    resume mid-integration (the branch carries real merged progress). For a FRESH
    integration the caller must drop any same-named leftover first (the branch
    name is not run-scoped, so a prior run's branch would otherwise be reused with
    stale content). ``loop._integrate`` enforces that precondition.
    """

    if not branch_exists(repo, branch):
        _git(repo, "branch", branch, base)


def merge_into(worktree: Path, branch: str) -> MergeOutcome:
    """Merge ``branch`` into the branch checked out in ``worktree``.

    ``--no-ff`` so every task contributes a recorded merge commit. Given a FRESH
    integration branch started at ``base`` each epoch (enforced by
    ``loop._integrate``, which drops any stale same-named branch before a
    fresh integration) plus disjoint ownership + the scope check, a conflict is
    structurally impossible, so it is reported (not retried) and the merge is
    aborted to leave the worktree clean.
    """

    merge = _git(worktree, *_GIT_IDENTITY, "merge", "--no-ff", "--no-edit", branch, check=False)
    if merge.returncode != 0:
        status = _git(worktree, "status", "--short", check=False).stdout.strip()
        _git(worktree, "merge", "--abort", check=False)
        return MergeOutcome(
            ok=False,
            conflict=status or merge.stderr.strip() or merge.stdout.strip() or "merge failed",
        )
    return MergeOutcome(ok=True)
