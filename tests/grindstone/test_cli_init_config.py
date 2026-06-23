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
    _IGNORE_HEADER,
    _resolve_concurrency,
    _resolve_ladder,
    _resolve_max_planner_calls,
    _resolve_planner,
    ensure_run_incidental_ignores,
    main,
)
from grindstone.config import GrindstoneConfig, load_config, resolve_role_script
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
    "  worker: {script: /m/local.sh, slots: 3, timeout_s: 1800}\n"
)


# --- init ----------------------------------------------------------------------


def _fake_models(root: Path, *, codex: bool = False) -> Path:
    """A hermetic models/ tree (claude floor + optional codex preset, no personal)
    so init's baked paths are deterministic regardless of the operator's gitignored
    models/personal on the dev machine."""

    models = root / "models"
    for sub in ("claude", "codex", "personal", "_common"):
        (models / sub).mkdir(parents=True, exist_ok=True)
    for name in ("planner_request.sh", "worker_request.sh", "senior_request.sh"):
        (models / "claude" / name).write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    (models / "_common" / "stop.sh").write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    if codex:
        (models / "codex" / "planner_request.sh").write_text(
            "#!/usr/bin/env bash\nexit 0\n", encoding="utf-8"
        )
    return models


def test_init_writes_loadable_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    models = _fake_models(tmp_path)
    monkeypatch.setattr("grindstone.config.MODELS_DIR", models)
    assert main(["init", "--repo", str(repo)]) == 0
    cfg_path = repo / ".grindstone" / "config.yaml"
    assert cfg_path.is_file()
    cfg = load_config(repo)  # the scaffold parses to a valid config
    assert isinstance(cfg, GrindstoneConfig)
    # Default rig: each role names `rig: claude` (portable, no absolute paths)...
    assert cfg.roles.planner.rig == "claude" and cfg.roles.planner.script is None
    assert cfg.roles.worker.rig == "claude"
    assert cfg.roles.senior is not None and cfg.roles.senior.rig == "claude"
    # ...and that rig resolves to the bundled models/claude/ scripts at run time.
    assert resolve_role_script("planner", cfg.roles.planner) == (
        models / "claude" / "planner_request.sh"
    ).resolve()
    assert resolve_role_script("worker", cfg.roles.worker) == (
        models / "claude" / "worker_request.sh"
    ).resolve()
    assert resolve_role_script("senior", cfg.roles.senior) == (
        models / "claude" / "senior_request.sh"
    ).resolve()


def test_init_rig_codex_writes_codex_rig_resolves_planner_default_workers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # `init --rig codex` writes `rig: codex` for every role; at run time the
    # planner resolves under codex/, while the workers (no codex/ script) fall
    # through to the claude/ floor.
    repo = tmp_path / "repo"
    repo.mkdir()
    models = _fake_models(tmp_path, codex=True)
    monkeypatch.setattr("grindstone.config.MODELS_DIR", models)
    assert main(["init", "--repo", str(repo), "--rig", "codex"]) == 0
    cfg = load_config(repo)
    assert cfg is not None
    assert cfg.roles.planner.rig == "codex"
    assert cfg.roles.worker.rig == "codex"
    assert cfg.roles.senior is not None and cfg.roles.senior.rig == "codex"
    assert resolve_role_script("planner", cfg.roles.planner) == (
        models / "codex" / "planner_request.sh"
    ).resolve()
    assert resolve_role_script("worker", cfg.roles.worker) == (
        models / "claude" / "worker_request.sh"
    ).resolve()
    assert resolve_role_script("senior", cfg.roles.senior) == (
        models / "claude" / "senior_request.sh"
    ).resolve()


def test_init_appends_gitignore_idempotently(tmp_path: Path) -> None:
    gitignore = tmp_path / ".gitignore"
    gitignore.write_text("node_modules/\n", encoding="utf-8")
    main(["init", "--repo", str(tmp_path)])
    first = gitignore.read_text(encoding="utf-8")
    assert ".grindstone/*" in first
    assert "node_modules/" in first  # pre-existing content preserved
    main(["init", "--repo", str(tmp_path)])  # second init must not duplicate
    # the auto-managed block is rewritten in place, never re-appended.
    assert gitignore.read_text(encoding="utf-8").count(_IGNORE_HEADER) == 1


def test_init_creates_gitignore_when_absent(tmp_path: Path) -> None:
    main(["init", "--repo", str(tmp_path)])
    assert ".grindstone/*" in (tmp_path / ".gitignore").read_text(encoding="utf-8")


# --- run-incidental gitignore bake-step ----------------------------------------


def _ignore(repo: Path) -> str:
    return (repo / ".gitignore").read_text(encoding="utf-8")


def test_bake_adds_managed_block_when_lacking(tmp_path: Path) -> None:
    """A repo whose .gitignore lacks the block gets it, clearly marked, with the
    run-state ignore (+ skills negation) and the ecosystem-standard incidental
    test/build dirs that done_when commands emit."""

    (tmp_path / ".gitignore").write_text("node_modules/\nweb-build/\n", encoding="utf-8")
    assert ensure_run_incidental_ignores(tmp_path) is True
    text = _ignore(tmp_path)
    assert "node_modules/" in text and "web-build/" in text  # user content preserved
    assert _IGNORE_HEADER in text
    for entry in (
        ".grindstone/*",
        "!.grindstone/skills/",
        "test-results/",
        "playwright-report/",
        "coverage/",
        ".pytest_cache/",
    ):
        assert f"\n{entry}\n" in text or text.startswith(entry) or text.endswith(entry + "\n")


def test_bake_is_idempotent(tmp_path: Path) -> None:
    """Re-running never duplicates: the second bake is a byte-for-byte no-op."""

    (tmp_path / ".gitignore").write_text("dist/\n", encoding="utf-8")
    assert ensure_run_incidental_ignores(tmp_path) is True
    once = _ignore(tmp_path)
    assert ensure_run_incidental_ignores(tmp_path) is False  # no change
    assert _ignore(tmp_path) == once
    assert once.count(_IGNORE_HEADER) == 1
    assert once.count("test-results/") == 1
    assert once.count(".grindstone/*") == 1


def test_bake_creates_gitignore_when_absent(tmp_path: Path) -> None:
    """A repo with no .gitignore at all gets one created with just the block."""

    assert ensure_run_incidental_ignores(tmp_path) is True
    text = _ignore(tmp_path)
    assert _IGNORE_HEADER in text
    assert "test-results/" in text and ".grindstone/*" in text


def test_bake_preserves_existing_manual_negation(tmp_path: Path) -> None:
    """An existing manual `.grindstone/*` + `!.grindstone/skills/` negation (as in
    a hand-edited target repo) survives the bake."""

    (tmp_path / ".gitignore").write_text(
        "# grindstone run state\n.grindstone/*\n!.grindstone/skills/\n",
        encoding="utf-8",
    )
    assert ensure_run_incidental_ignores(tmp_path) is True
    text = _ignore(tmp_path)
    assert "!.grindstone/skills/" in text  # skills catalogue still re-included
    assert "test-results/" in text


def test_bake_upgrades_legacy_bare_dir_line(tmp_path: Path) -> None:
    """The legacy single `.grindstone/` line old init wrote (which defeats the
    skills negation) is superseded by the contents-form block, not left behind."""

    (tmp_path / ".gitignore").write_text(".grindstone/\nnode_modules/\n", encoding="utf-8")
    assert ensure_run_incidental_ignores(tmp_path) is True
    lines = [ln.strip() for ln in _ignore(tmp_path).splitlines()]
    assert ".grindstone/" not in lines  # the bare dir form is gone
    assert ".grindstone/*" in lines and "!.grindstone/skills/" in lines
    assert "node_modules/" in lines  # unrelated user content preserved


def test_init_does_not_clobber_existing_config(tmp_path: Path) -> None:
    _write_cfg(tmp_path, _LOCAL_ONLY)
    code = main(["init", "--repo", str(tmp_path)])
    assert code == 0
    # Existing config is preserved (init is a one-time scaffold, not a reset).
    cfg = load_config(tmp_path)
    assert cfg is not None and cfg.roles.worker.slots == 3


# --- config-driven resolution --------------------------------------------------


def test_resolve_ladder_init_config_has_local_and_senior(tmp_path: Path) -> None:
    main(["init", "--repo", str(tmp_path)])
    cfg = load_config(tmp_path)
    ladder = _resolve_ladder(_flags(), cfg, _run_dir(tmp_path))
    assert [tier for tier, _ in ladder] == ["worker", "senior"]
    assert all(isinstance(w, ScriptWorker) for _, w in ladder)


def test_resolve_ladder_local_only_when_no_senior(tmp_path: Path) -> None:
    _write_cfg(tmp_path, _LOCAL_ONLY)
    cfg = load_config(tmp_path)
    ladder = _resolve_ladder(_flags(), cfg, _run_dir(tmp_path))
    assert [tier for tier, _ in ladder] == ["worker"]
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
    assert _resolve_concurrency(_flags(), cfg) == 3  # = roles.worker.slots


def test_resolve_concurrency_no_config_fails_loudly() -> None:
    with pytest.raises(SystemExit):
        _resolve_concurrency(_flags(), None)


def test_resolve_max_planner_calls_defaults_on(tmp_path: Path) -> None:
    """Gate-5 P0: an unattended run looped phase revisions for 34 codex calls
    because max_planner_calls was a test-only valve, off in production. The
    CLI now ALWAYS passes a cap: config value if set, else the built-in
    default, never unbounded."""

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
        ladder=[("worker", OwnershipWorker())],
    )
    assert code == 0
