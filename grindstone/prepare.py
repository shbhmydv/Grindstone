"""Declared dependency materialization for build gates (lockfile-hash cached).

The root cause of a real dogfood failure: a phase ``done_when`` like
``npx tsc --noEmit`` or ``npx expo export`` runs inside a FRESH detached worktree
of the integration tip (``run_loop.evaluate_checks``) that carries only COMMITTED
files. ``node_modules`` is gitignored and never committed, so it is ABSENT there,
``npx`` resolves the wrong stub and exits non-zero DETERMINISTICALLY regardless of
code correctness. Every build gate is then structurally unpassable. The worker's
own worktree, where the agent ran ``npm install``, passes the SAME check, which is
why workers honestly reported success while the phase gate failed.

The fix is general (node / python / c++ / ...), DECLARED in ``prepare:`` config,
never hardcoded: ``cmd`` installs the deps, ``env_dirs`` are the gitignored dirs
it produces (and that checks need), ``cache_key_files`` are the lockfiles whose
content hash keys a snapshot cache. With an unchanged lockfile the cached env_dirs
are restored instead of re-running ``cmd`` (so the same install is reused across
the eval worktree AND every worker worktree).

The cache lives under the repo's ``.grindstone/cache/env/<hash>/`` (gitignored,
shared across runs). Materialization is idempotent and never crashes the run on a
benign miss; a FAILING ``cmd`` raises ``PrepareError`` so the gate reports
``prepare failed`` with captured output, not a silent unpassable gate.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path

from grindstone.config import PrepareConfig

#: Cache root under the repo's gitignored ``.grindstone/``, shared across runs.
_CACHE_REL = Path(".grindstone") / "cache" / "env"


class PrepareError(RuntimeError):
    """The declared ``prepare`` command exited non-zero (deps not installed).

    Surfaced to the check evaluator so the gate reports a real, actionable
    failure ("prepare failed: <cmd>" + captured output) instead of a build check
    that is silently and deterministically unpassable.
    """


def _cache_key(worktree: Path, repo: Path, cache_key_files: list[str]) -> str:
    """Hash the concatenated contents of ``cache_key_files``.

    Each file is read from the worktree first (the tree under test), falling back
    to the repo checkout when absent there; a missing file in BOTH contributes an
    empty body. The relative path is folded into the digest so two files swapping
    contents cannot collide. Stable across processes (sha256 of bytes).
    """

    h = hashlib.sha256()
    for rel in cache_key_files:
        for root in (worktree, repo):
            candidate = root / rel
            if candidate.is_file():
                body = candidate.read_bytes()
                break
        else:
            body = b""
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        h.update(str(len(body)).encode("ascii"))
        h.update(b"\0")
        h.update(body)
        h.update(b"\0")
    return h.hexdigest()


def _copy_tree(src: Path, dst: Path) -> None:
    """Copy ``src`` dir onto ``dst`` (replacing it). Plain copy, correctness first."""

    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst, symlinks=True)


def materialize_env(repo: Path, worktree: Path, prepare: PrepareConfig | None) -> None:
    """Restore the declared dependency dirs into ``worktree`` (cached by lockfile).

    No-op when ``prepare`` is ``None``. Otherwise: compute the lockfile-content
    cache key; on a HIT restore each ``env_dir`` from the cache into the worktree;
    on a MISS run ``cmd`` in the worktree (raising ``PrepareError`` on non-zero
    with captured output), then SNAPSHOT each produced ``env_dir`` into the cache
    under that key for next time. Idempotent: a second call with the same lockfile
    contents reuses the cache and does NOT re-run ``cmd``.

    A cached ``env_dir`` that is missing from the cache snapshot (e.g. a partial
    prior install) simply re-runs ``cmd`` rather than restoring a hole, so a
    benign cache gap can never strand the worktree without its deps.
    """

    if prepare is None:
        return
    key = _cache_key(worktree, repo, prepare.cache_key_files)
    cache_dir = repo / _CACHE_REL / key
    cached = [cache_dir / d for d in prepare.env_dirs]
    if cache_dir.is_dir() and all(c.exists() for c in cached):
        for env_dir, src in zip(prepare.env_dirs, cached):
            _copy_tree(src, worktree / env_dir)
        return

    proc = subprocess.run(
        prepare.cmd,
        shell=True,
        cwd=str(worktree),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip()
        raise PrepareError(
            f"prepare failed: `{prepare.cmd}` (cwd={worktree}) "
            f"exited {proc.returncode}: {tail}"
        )

    # Snapshot each produced env_dir into the cache for next time. A declared
    # env_dir the cmd did not actually create is skipped (not an error: the cmd
    # succeeded, so the build does not need it).
    cache_dir.mkdir(parents=True, exist_ok=True)
    for env_dir in prepare.env_dirs:
        produced = worktree / env_dir
        if produced.exists():
            _copy_tree(produced, cache_dir / env_dir)
