"""Declared dependency materialization (``grindstone.prepare``).

The dogfood root cause: a build ``done_when`` (``npx tsc``) runs in a FRESH
detached worktree of the committed tip, where the gitignored ``node_modules`` is
absent, so the gate is structurally unpassable. ``prepare`` restores the declared
dependency dirs (cached by lockfile hash) before checks/workers run.

These tests use a FAKE prepare cmd (no real npm): a script that creates a marker
dir and appends to a counter file, so we can assert the env_dir appears, the cache
is REUSED on an unchanged lockfile (cmd runs once), a changed lockfile BUSTS the
cache (cmd re-runs), and a failing cmd surfaces a clear error.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from grindstone.config import FloorConfig, PrepareConfig
from grindstone.contracts.models import CmdCheck
from grindstone.prepare import PrepareError, materialize_env
from grindstone.rundir import create_run_dir
from grindstone.run_loop import evaluate_checks

from tests.grindstone.conftest import git, init_git_repo


def _counting_prepare(counter: Path) -> PrepareConfig:
    """A fake prepare: append to ``counter`` (so we can count runs) and create the
    ``node_modules`` env_dir with a marker file. Keyed on ``package-lock.json``."""

    cmd = (
        f"echo run >> {counter} && "
        "mkdir -p node_modules && echo ok > node_modules/marker"
    )
    return PrepareConfig(
        cmd=cmd,
        env_dirs=["node_modules"],
        cache_key_files=["package-lock.json"],
    )


def _write_lock(root: Path, body: str) -> None:
    (root / "package-lock.json").write_text(body, encoding="utf-8")


def test_none_is_noop(tmp_path: Path) -> None:
    """``prepare=None`` materializes nothing (existing behavior unchanged)."""

    repo = tmp_path / "repo"
    repo.mkdir()
    wt = tmp_path / "wt"
    wt.mkdir()
    materialize_env(repo, wt, None)
    assert list(wt.iterdir()) == []


def test_env_dir_materialized_into_worktree(tmp_path: Path) -> None:
    """A first call runs the cmd and the declared env_dir appears in the worktree."""

    repo = tmp_path / "repo"
    wt = tmp_path / "wt"
    repo.mkdir()
    wt.mkdir()
    counter = tmp_path / "counter"
    _write_lock(wt, "v1")
    materialize_env(repo, wt, _counting_prepare(counter))
    assert (wt / "node_modules" / "marker").read_text().strip() == "ok"
    assert counter.read_text().count("run") == 1


def test_cache_reused_on_unchanged_lockfile(tmp_path: Path) -> None:
    """A second materialization with the SAME cache_key_files contents restores
    from cache and does NOT re-run the prepare cmd (counter stays at 1)."""

    repo = tmp_path / "repo"
    repo.mkdir()
    counter = tmp_path / "counter"
    prepare = _counting_prepare(counter)

    wt1 = tmp_path / "wt1"
    wt1.mkdir()
    _write_lock(wt1, "v1")
    materialize_env(repo, wt1, prepare)

    wt2 = tmp_path / "wt2"
    wt2.mkdir()
    _write_lock(wt2, "v1")  # identical lockfile -> cache hit
    materialize_env(repo, wt2, prepare)

    assert (wt2 / "node_modules" / "marker").read_text().strip() == "ok"
    assert counter.read_text().count("run") == 1  # cmd ran exactly once


def test_changed_lockfile_busts_cache(tmp_path: Path) -> None:
    """Changing a cache_key_file's contents busts the cache: the cmd re-runs."""

    repo = tmp_path / "repo"
    repo.mkdir()
    counter = tmp_path / "counter"
    prepare = _counting_prepare(counter)

    wt1 = tmp_path / "wt1"
    wt1.mkdir()
    _write_lock(wt1, "v1")
    materialize_env(repo, wt1, prepare)

    wt2 = tmp_path / "wt2"
    wt2.mkdir()
    _write_lock(wt2, "v2")  # different lockfile -> cache miss -> re-run
    materialize_env(repo, wt2, prepare)

    assert counter.read_text().count("run") == 2


def test_failing_prepare_raises_clear_error(tmp_path: Path) -> None:
    """A non-zero prepare cmd raises PrepareError with the command + captured
    output, the gate reports a real failure rather than a silent unpassable check."""

    repo = tmp_path / "repo"
    wt = tmp_path / "wt"
    repo.mkdir()
    wt.mkdir()
    _write_lock(wt, "v1")
    prepare = PrepareConfig(
        cmd="echo boom-stderr >&2 && exit 3",
        env_dirs=["node_modules"],
        cache_key_files=["package-lock.json"],
    )
    with pytest.raises(PrepareError) as exc:
        materialize_env(repo, wt, prepare)
    msg = str(exc.value)
    assert "prepare failed" in msg
    assert "boom-stderr" in msg


def test_lockfile_read_from_repo_when_absent_in_worktree(tmp_path: Path) -> None:
    """The cache key falls back to the repo checkout when a lockfile is absent in
    the worktree, so two worktrees of the same repo tip share one cache entry."""

    repo = tmp_path / "repo"
    repo.mkdir()
    _write_lock(repo, "v1")  # only in the repo, not the worktree
    counter = tmp_path / "counter"
    prepare = _counting_prepare(counter)

    wt1 = tmp_path / "wt1"
    wt1.mkdir()
    materialize_env(repo, wt1, prepare)
    wt2 = tmp_path / "wt2"
    wt2.mkdir()
    materialize_env(repo, wt2, prepare)
    assert counter.read_text().count("run") == 1  # shared cache via repo fallback


# --- integration through evaluate_checks (the build-gate fix) ------------------


def test_evaluate_checks_materializes_before_cmd_checks(tmp_path: Path) -> None:
    """A cmd check that needs the gitignored env_dir PASSES once prepare restored
    it into the throwaway eval worktree (the structural-unpassable-gate fix)."""

    repo = init_git_repo(tmp_path / "repo")
    # node_modules is gitignored + never committed (the real-world shape).
    (repo / ".gitignore").write_text(
        ".grindstone/\n__pycache__/\nnode_modules/\n", encoding="utf-8"
    )
    _write_lock(repo, "v1")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "lockfile")

    run = create_run_dir(repo, "run-prep")
    counter = tmp_path / "counter"
    prepare = _counting_prepare(counter)
    # The check fails WITHOUT prepare (node_modules absent in the committed tip)
    # and passes WITH it: a marker the prepare step produced.
    check = CmdCheck(cmd="test -f node_modules/marker")

    without = evaluate_checks(
        [check], repo=repo, ref="HEAD", run_dir=run, scratch_name="_eval_no_prep"
    )
    assert without[0][1] is False  # structurally unpassable without deps

    with_prep = evaluate_checks(
        [check],
        repo=repo,
        ref="HEAD",
        run_dir=run,
        scratch_name="_eval_prep",
        prepare=prepare,
    )
    assert with_prep[0][1] is True


def test_evaluate_checks_surfaces_prepare_failure(tmp_path: Path) -> None:
    """A failing prepare makes the cmd checks FAIL with the prepare error in the
    label, NOT a silent unpassable gate (the planner can see why)."""

    repo = init_git_repo(tmp_path / "repo")
    _write_lock(repo, "v1")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "lockfile")
    run = create_run_dir(repo, "run-prep-fail")
    prepare = PrepareConfig(
        cmd="echo nope >&2 && exit 1",
        env_dirs=["node_modules"],
        cache_key_files=["package-lock.json"],
    )
    results = evaluate_checks(
        [CmdCheck(cmd="true")],
        repo=repo,
        ref="HEAD",
        run_dir=run,
        scratch_name="_eval_prep_fail",
        prepare=prepare,
    )
    label, ok = results[0]
    assert ok is False
    assert "prepare failed" in label


# --- the deterministic FLOOR through the gate (gate-rebalance G2) ---------------


def test_floor_checks_run_in_the_gate(tmp_path: Path) -> None:
    """A configured ``floor.checks`` command runs in the eval worktree and its
    pass/fail becomes a gate result, appended after the supplied checks."""

    repo = init_git_repo(tmp_path / "repo")
    (repo / "marker.txt").write_text("ok\n", encoding="utf-8")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "marker")
    run = create_run_dir(repo, "run-floor")
    floor = FloorConfig(checks=["test -f marker.txt"])
    results = evaluate_checks(
        [CmdCheck(cmd="true")],
        repo=repo,
        ref="HEAD",
        run_dir=run,
        scratch_name="_eval_floor_ok",
        floor=floor,
    )
    # The supplied check, then the floor check, both pass.
    assert [ok for _, ok in results] == [True, True]
    assert "test -f marker.txt" in results[1][0]


def test_failing_floor_check_fails_the_gate_with_output(tmp_path: Path) -> None:
    """A FAILING floor check fails the gate exactly like a failed done_when, and
    its captured output is surfaced (the planner can see WHY, not a bare exit)."""

    repo = init_git_repo(tmp_path / "repo")
    run = create_run_dir(repo, "run-floor-fail")
    floor = FloorConfig(checks=["echo floor-stderr >&2 && exit 2"])
    results = evaluate_checks(
        [CmdCheck(cmd="true")],
        repo=repo,
        ref="HEAD",
        run_dir=run,
        scratch_name="_eval_floor_fail",
        floor=floor,
    )
    assert results[0][1] is True  # supplied check passes
    label, ok = results[1]  # floor check fails the gate
    assert ok is False
    assert "exit 2" in label
    assert "floor-stderr" in label  # captured output reaches the planner


def test_floor_runs_after_prepare_so_a_dep_check_passes(tmp_path: Path) -> None:
    """The floor runs AFTER ``prepare`` materializes deps: a floor check that
    needs the gitignored env_dir PASSES because prepare restored it first."""

    repo = init_git_repo(tmp_path / "repo")
    (repo / ".gitignore").write_text(
        ".grindstone/\n__pycache__/\nnode_modules/\n", encoding="utf-8"
    )
    _write_lock(repo, "v1")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "lockfile")
    run = create_run_dir(repo, "run-floor-prep")
    counter = tmp_path / "counter"
    prepare = _counting_prepare(counter)  # produces node_modules/marker
    # The floor check needs node_modules, absent in the committed tip, present
    # only because prepare ran FIRST.
    floor = FloorConfig(checks=["test -f node_modules/marker"])
    results = evaluate_checks(
        [],
        repo=repo,
        ref="HEAD",
        run_dir=run,
        scratch_name="_eval_floor_after_prep",
        prepare=prepare,
        floor=floor,
    )
    assert results[0][1] is True  # the floor check passed thanks to prepare
