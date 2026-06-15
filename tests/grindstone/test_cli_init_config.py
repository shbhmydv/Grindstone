"""Role-split: ``grindstone init`` + config-driven wiring.

``init`` writes a loadable ``.grindstone/config.yaml`` with absolute role-script
paths and idempotently appends ``.grindstone/`` to ``.gitignore``. The
ladder/planner builders resolve each role to its request script (``ScriptWorker``
/ ``ScriptPlanner``); the scaffolded config drives a ``run`` through ``main`` with
an injected mock planner + ladder (no codex/pi quota spent).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from grindstone.cli import (
    DEFAULT_MAX_PLANNER_CALLS,
    _resolve_concurrency,
    _resolve_ladder,
    _resolve_max_planner_calls,
    _resolve_planner,
    main,
)
from grindstone.config import GrindstoneConfig, load_config
from grindstone.mock_planner import MockPlanner
from grindstone.rundir import RunDir
from grindstone.script_planner import ScriptPlanner
from grindstone.script_worker import ScriptWorker

from tests.grindstone.conftest import (
    OwnershipWorker,
    check_cmd,
    complete_decision,
    two_phase_skeleton,
)


def _write_cfg(repo: Path, text: str) -> None:
    cfg_path = repo / ".grindstone" / "config.yaml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(text, encoding="utf-8")


def _flags(**over: object) -> argparse.Namespace:
    base: dict[str, object] = {"run_id": None}
    base.update(over)
    return argparse.Namespace(**base)


def _run_dir(repo: Path) -> RunDir:
    return RunDir(root=repo / ".grindstone" / "runs" / "x")


_LOCAL_ONLY = (
    "roles:\n"
    "  planner: {script: /m/planner.sh, slots: 1, timeout_s: 600}\n"
    "  local: {script: /m/local.sh, slots: 3, timeout_s: 1800}\n"
)


# --- init ----------------------------------------------------------------------


def test_init_writes_loadable_config(tmp_path: Path) -> None:
    assert main(["init", "--repo", str(tmp_path)]) == 0
    cfg_path = tmp_path / ".grindstone" / "config.yaml"
    assert cfg_path.is_file()
    cfg = load_config(tmp_path)  # the scaffold parses to a valid config
    assert isinstance(cfg, GrindstoneConfig)
    # Absolute script paths resolved from the grindstone install root (models/).
    models_dir = Path(__file__).resolve().parents[2] / "models"
    assert cfg.roles.planner.script == models_dir / "planner_request.sh"
    assert cfg.roles.local.script == models_dir / "local_request.sh"
    assert cfg.roles.senior is not None
    assert cfg.roles.senior.script == models_dir / "senior_request.sh"


def test_init_appends_gitignore_idempotently(tmp_path: Path) -> None:
    gitignore = tmp_path / ".gitignore"
    gitignore.write_text("node_modules/\n", encoding="utf-8")
    main(["init", "--repo", str(tmp_path)])
    first = gitignore.read_text(encoding="utf-8")
    assert ".grindstone/" in first
    assert "node_modules/" in first  # pre-existing content preserved
    main(["init", "--repo", str(tmp_path)])  # second init must not duplicate
    assert gitignore.read_text(encoding="utf-8").count(".grindstone/") == 1


def test_init_creates_gitignore_when_absent(tmp_path: Path) -> None:
    main(["init", "--repo", str(tmp_path)])
    assert ".grindstone/" in (tmp_path / ".gitignore").read_text(encoding="utf-8")


def test_init_does_not_clobber_existing_config(tmp_path: Path) -> None:
    _write_cfg(tmp_path, _LOCAL_ONLY)
    code = main(["init", "--repo", str(tmp_path)])
    assert code == 0
    # Existing config is preserved (init is a one-time scaffold, not a reset).
    cfg = load_config(tmp_path)
    assert cfg is not None and cfg.roles.local.slots == 3


# --- config-driven resolution --------------------------------------------------


def test_resolve_ladder_init_config_has_local_and_senior(tmp_path: Path) -> None:
    main(["init", "--repo", str(tmp_path)])
    cfg = load_config(tmp_path)
    ladder = _resolve_ladder(_flags(), cfg, _run_dir(tmp_path))
    assert [tier for tier, _ in ladder] == ["local", "senior"]
    assert all(isinstance(w, ScriptWorker) for _, w in ladder)


def test_resolve_ladder_local_only_when_no_senior(tmp_path: Path) -> None:
    _write_cfg(tmp_path, _LOCAL_ONLY)
    cfg = load_config(tmp_path)
    ladder = _resolve_ladder(_flags(), cfg, _run_dir(tmp_path))
    assert [tier for tier, _ in ladder] == ["local"]
    worker = ladder[0][1]
    assert isinstance(worker, ScriptWorker)
    assert worker.script == Path("/m/local.sh")


def test_resolve_ladder_logs_under_run_dir(tmp_path: Path) -> None:
    _write_cfg(tmp_path, _LOCAL_ONLY)
    cfg = load_config(tmp_path)
    run_dir = _run_dir(tmp_path)
    ladder = _resolve_ladder(_flags(), cfg, run_dir)
    worker = ladder[0][1]
    assert isinstance(worker, ScriptWorker)
    assert worker.log_root == run_dir.root / "worker_logs"


def test_resolve_ladder_no_config_fails_loudly(tmp_path: Path) -> None:
    # No config = no rig defaults in core: fail loudly, point at `grindstone init`.
    with pytest.raises(SystemExit) as exc:
        _resolve_ladder(_flags(), None, _run_dir(tmp_path))
    msg = str(exc.value)
    assert "init" in msg and ".grindstone/config.yaml" in msg


def test_resolve_concurrency_is_local_slots(tmp_path: Path) -> None:
    _write_cfg(tmp_path, _LOCAL_ONLY)
    cfg = load_config(tmp_path)
    assert _resolve_concurrency(_flags(), cfg) == 3  # = roles.local.slots


def test_resolve_concurrency_no_config_fails_loudly() -> None:
    with pytest.raises(SystemExit):
        _resolve_concurrency(_flags(), None)


def test_resolve_max_planner_calls_defaults_on(tmp_path: Path) -> None:
    """Gate-5 P0: an unattended run looped phase revisions for 34 codex calls
    because max_planner_calls was a test-only valve, off in production. The
    CLI now ALWAYS passes a cap: config value if set, else the built-in
    default — never unbounded."""

    main(["init", "--repo", str(tmp_path)])
    cfg = load_config(tmp_path)
    assert _resolve_max_planner_calls(cfg) == DEFAULT_MAX_PLANNER_CALLS
    _write_cfg(tmp_path, _LOCAL_ONLY + "max_planner_calls: 12\n")
    assert _resolve_max_planner_calls(load_config(tmp_path)) == 12


def test_resolve_planner_is_script_planner(tmp_path: Path) -> None:
    main(["init", "--repo", str(tmp_path)])
    cfg = load_config(tmp_path)
    planner = _resolve_planner(_flags(), cfg, tmp_path)
    assert isinstance(planner, ScriptPlanner)
    assert planner.script.name == "planner_request.sh"


def test_resolve_planner_no_config_fails_loudly(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as exc:
        _resolve_planner(_flags(), None, tmp_path)
    assert "init" in str(exc.value)


# --- the scaffolded config drives a run via injected mock planner + ladder -----


def test_init_config_drives_a_run(git_repo: Path) -> None:
    main(["init", "--repo", str(git_repo)])
    job = git_repo / "job.md"
    job.write_text("Build two phases, then complete.\n", encoding="utf-8")
    planner = MockPlanner(script=[two_phase_skeleton(), complete_decision(check_cmd("true"))])
    code = main(
        ["run", str(job), "--repo", str(git_repo), "--run-id", "scaffold-run"],
        planner=planner,
        ladder=[("local", OwnershipWorker())],
    )
    assert code == 0
