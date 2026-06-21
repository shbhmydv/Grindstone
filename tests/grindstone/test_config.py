"""Role-split: the typed, frozen ``.grindstone/config.yaml``
loader over the per-role schema.

A minimal Pydantic loader with explicit unknown-key rejection, no schema
framework. Missing config is ``None`` (CLI fails loudly toward init); present
config parses to a frozen typed object; an unknown key, a missing required role
(``planner`` / ``local``), or a bad ``slots`` / ``timeout_s`` is a hard error
(no silent typo-eats-config). ``senior`` is optional = a local-only ladder.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from grindstone.config import (
    ALLOW_REPO_SCRIPTS_ENV,
    MODELS_DIR,
    FloorConfig,
    GrindstoneConfig,
    load_config,
    models_script,
    validate_script_paths,
)

_PLANNER = "  planner: {script: /m/planner.sh, slots: 1, timeout_s: 600}\n"
_LOCAL = "  local: {script: /m/local.sh, slots: 2, timeout_s: 1800}\n"


def _write(repo: Path, text: str) -> None:
    cfg = repo / ".grindstone" / "config.yaml"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(text, encoding="utf-8")


def test_missing_config_is_none(tmp_path: Path) -> None:
    assert load_config(tmp_path) is None


def test_minimal_local_only_config(tmp_path: Path) -> None:
    _write(tmp_path, "roles:\n" + _PLANNER + _LOCAL)
    cfg = load_config(tmp_path)
    assert isinstance(cfg, GrindstoneConfig)
    assert cfg.roles.planner.script == Path("/m/planner.sh")
    assert cfg.roles.planner.slots == 1
    assert cfg.roles.local.script == Path("/m/local.sh")
    assert cfg.roles.local.slots == 2
    assert cfg.roles.local.timeout_s == 1800
    assert cfg.roles.senior is None  # optional: a rig with no cloud tier


def test_senior_tier_present(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "roles:\n"
        + _PLANNER
        + _LOCAL
        + "  senior: {script: /m/senior.sh, slots: 2, timeout_s: 3600}\n",
    )
    cfg = load_config(tmp_path)
    assert cfg is not None and cfg.roles.senior is not None
    assert cfg.roles.senior.script == Path("/m/senior.sh")
    assert cfg.roles.senior.timeout_s == 3600


def test_unknown_key_is_rejected(tmp_path: Path) -> None:
    _write(tmp_path, "roles:\n" + _PLANNER + _LOCAL + "typpo: 3\n")
    with pytest.raises(ValueError):
        load_config(tmp_path)


def test_unknown_role_key_is_rejected(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "roles:\n"
        + _PLANNER
        + _LOCAL
        + "  cloud: {script: /m/c.sh, slots: 1, timeout_s: 1}\n",  # not a role
    )
    with pytest.raises(ValueError):
        load_config(tmp_path)


def test_missing_planner_is_rejected(tmp_path: Path) -> None:
    _write(tmp_path, "roles:\n" + _LOCAL)
    with pytest.raises(ValueError):
        load_config(tmp_path)


def test_missing_local_is_rejected(tmp_path: Path) -> None:
    _write(tmp_path, "roles:\n" + _PLANNER)
    with pytest.raises(ValueError):
        load_config(tmp_path)


def test_missing_required_role_field_is_rejected(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "roles:\n" + _PLANNER + "  local: {script: /m/local.sh, slots: 2}\n",  # no timeout_s
    )
    with pytest.raises(ValueError):
        load_config(tmp_path)


def test_slots_below_one_is_rejected(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "roles:\n" + _PLANNER + "  local: {script: /m/local.sh, slots: 0, timeout_s: 1}\n",
    )
    with pytest.raises(ValueError):
        load_config(tmp_path)


def test_non_positive_timeout_is_rejected(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "roles:\n" + _PLANNER + "  local: {script: /m/local.sh, slots: 1, timeout_s: 0}\n",
    )
    with pytest.raises(ValueError):
        load_config(tmp_path)


def test_config_is_frozen(tmp_path: Path) -> None:
    _write(tmp_path, "roles:\n" + _PLANNER + _LOCAL)
    cfg = load_config(tmp_path)
    assert cfg is not None
    with pytest.raises(Exception):
        cfg.max_planner_calls = 5  # type: ignore[misc]


def test_max_planner_calls_parses(tmp_path: Path) -> None:
    # Gate-5 P0: a stuck run burned 34 codex calls because the safety valve
    # defaulted OFF in production. The owner can now size it per repo.
    _write(tmp_path, "roles:\n" + _PLANNER + _LOCAL + "max_planner_calls: 12\n")
    cfg = load_config(tmp_path)
    assert cfg is not None and cfg.max_planner_calls == 12


def test_max_failed_epochs_per_phase_defaults_and_parses(tmp_path: Path) -> None:
    # The dogfood spin-loop backstop: a deterministic per-phase failed-epoch cap.
    # Defaults to 3 when absent; the owner can size it per repo.
    _write(tmp_path, "roles:\n" + _PLANNER + _LOCAL)
    cfg = load_config(tmp_path)
    assert cfg is not None and cfg.max_failed_epochs_per_phase == 3
    _write(tmp_path, "roles:\n" + _PLANNER + _LOCAL + "max_failed_epochs_per_phase: 5\n")
    cfg2 = load_config(tmp_path)
    assert cfg2 is not None and cfg2.max_failed_epochs_per_phase == 5


def test_max_failed_epochs_per_phase_must_be_positive(tmp_path: Path) -> None:
    _write(tmp_path, "roles:\n" + _PLANNER + _LOCAL + "max_failed_epochs_per_phase: 0\n")
    with pytest.raises(ValueError):
        load_config(tmp_path)


def test_task_file_bounds_default_and_parse(tmp_path: Path) -> None:
    # The deterministic size-gate bounds (Part 4B): tier-aware ceilings on a fresh
    # implement task's file_ownership glob count. Default local=5, senior=12.
    _write(tmp_path, "roles:\n" + _PLANNER + _LOCAL)
    cfg = load_config(tmp_path)
    assert cfg is not None
    assert cfg.local_max_task_files == 5
    assert cfg.senior_max_task_files == 12
    _write(
        tmp_path,
        "roles:\n" + _PLANNER + _LOCAL
        + "local_max_task_files: 3\nsenior_max_task_files: 20\n",
    )
    cfg2 = load_config(tmp_path)
    assert cfg2 is not None
    assert cfg2.local_max_task_files == 3
    assert cfg2.senior_max_task_files == 20


def test_task_file_bounds_must_be_positive(tmp_path: Path) -> None:
    _write(tmp_path, "roles:\n" + _PLANNER + _LOCAL + "local_max_task_files: 0\n")
    with pytest.raises(ValueError):
        load_config(tmp_path)
    _write(tmp_path, "roles:\n" + _PLANNER + _LOCAL + "senior_max_task_files: 0\n")
    with pytest.raises(ValueError):
        load_config(tmp_path)


def test_vision_review_is_optional_and_defaults_none(tmp_path: Path) -> None:
    # B3 taste gate: the vision_review script seam is optional (a rig with no
    # vision tier omits it; the CLI then falls back to the bundled script).
    _write(tmp_path, "roles:\n" + _PLANNER + _LOCAL)
    cfg = load_config(tmp_path)
    assert cfg is not None and cfg.vision_review is None


def test_vision_review_parses_when_present(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "roles:\n"
        + _PLANNER
        + _LOCAL
        + "vision_review: {script: /m/vision.sh, timeout_s: 300}\n",
    )
    cfg = load_config(tmp_path)
    assert cfg is not None and cfg.vision_review is not None
    assert cfg.vision_review.script == Path("/m/vision.sh")
    assert cfg.vision_review.timeout_s == 300


def test_vision_review_non_positive_timeout_is_rejected(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "roles:\n"
        + _PLANNER
        + _LOCAL
        + "vision_review: {script: /m/vision.sh, timeout_s: 0}\n",
    )
    with pytest.raises(ValueError):
        load_config(tmp_path)


def test_empty_config_file_is_rejected_not_silent(tmp_path: Path) -> None:
    # An empty file means no `roles` block: a misconfiguration, not a default.
    _write(tmp_path, "")
    with pytest.raises(ValueError):
        load_config(tmp_path)


def test_final_polish_is_optional_and_defaults_none(tmp_path: Path) -> None:
    # B5 codex polish pass: OFF by default, absent block means no polish runs.
    _write(tmp_path, "roles:\n" + _PLANNER + _LOCAL)
    cfg = load_config(tmp_path)
    assert cfg is not None and cfg.final_polish is None


def test_final_polish_parses_when_present(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "roles:\n"
        + _PLANNER
        + _LOCAL
        + "final_polish: {criteria: tidy the UI, timeout_s: 900, "
        "script: /m/polish.sh, screenshot: ui/home.png}\n",
    )
    cfg = load_config(tmp_path)
    assert cfg is not None and cfg.final_polish is not None
    assert cfg.final_polish.criteria == "tidy the UI"
    assert cfg.final_polish.timeout_s == 900
    assert cfg.final_polish.script == Path("/m/polish.sh")
    assert cfg.final_polish.screenshot == "ui/home.png"


def test_final_polish_script_and_screenshot_default_none(tmp_path: Path) -> None:
    # script omitted -> CLI falls back to the bundled codex_polish.sh; screenshot
    # is optional (a non-visual polish pass needs no image).
    _write(
        tmp_path,
        "roles:\n" + _PLANNER + _LOCAL + "final_polish: {criteria: polish, timeout_s: 600}\n",
    )
    cfg = load_config(tmp_path)
    assert cfg is not None and cfg.final_polish is not None
    assert cfg.final_polish.script is None
    assert cfg.final_polish.screenshot is None


def test_final_polish_non_positive_timeout_is_rejected(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "roles:\n" + _PLANNER + _LOCAL + "final_polish: {criteria: c, timeout_s: 0}\n",
    )
    with pytest.raises(ValueError):
        load_config(tmp_path)


# --- prepare: declared dependency materialization for build gates --------------


def test_prepare_is_optional_and_defaults_none(tmp_path: Path) -> None:
    # Absent block: no dependency materialization, existing runs unchanged.
    _write(tmp_path, "roles:\n" + _PLANNER + _LOCAL)
    cfg = load_config(tmp_path)
    assert cfg is not None and cfg.prepare is None


def test_prepare_parses_when_present(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "roles:\n"
        + _PLANNER
        + _LOCAL
        + "prepare: {cmd: npm ci, env_dirs: [node_modules], "
        "cache_key_files: [package-lock.json]}\n",
    )
    cfg = load_config(tmp_path)
    assert cfg is not None and cfg.prepare is not None
    assert cfg.prepare.cmd == "npm ci"
    assert cfg.prepare.env_dirs == ["node_modules"]
    assert cfg.prepare.cache_key_files == ["package-lock.json"]


def test_prepare_empty_lists_rejected(tmp_path: Path) -> None:
    # An env_dirs / cache_key_files with no entries is a config error, not a
    # silent no-op (the whole point is to restore real dirs keyed on real files).
    _write(
        tmp_path,
        "roles:\n"
        + _PLANNER
        + _LOCAL
        + "prepare: {cmd: npm ci, env_dirs: [], cache_key_files: [x]}\n",
    )
    with pytest.raises(ValueError):
        load_config(tmp_path)


def test_prepare_unknown_key_rejected(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "roles:\n"
        + _PLANNER
        + _LOCAL
        + "prepare: {cmd: x, env_dirs: [a], cache_key_files: [b], bogus: 1}\n",
    )
    with pytest.raises(ValueError):
        load_config(tmp_path)


# --- floor: the deterministic repo-owned verification commands -----------------


def test_floor_is_optional_and_defaults_none(tmp_path: Path) -> None:
    # Absent block: no repo floor commands, only the core invariants apply.
    _write(tmp_path, "roles:\n" + _PLANNER + _LOCAL)
    cfg = load_config(tmp_path)
    assert cfg is not None and cfg.floor is None


def test_floor_parses_when_present(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "roles:\n"
        + _PLANNER
        + _LOCAL
        + 'floor: {checks: ["npx tsc --noEmit", "npm test --silent"]}\n',
    )
    cfg = load_config(tmp_path)
    assert cfg is not None and cfg.floor is not None
    assert cfg.floor.checks == ["npx tsc --noEmit", "npm test --silent"]


def test_floor_empty_checks_allowed(tmp_path: Path) -> None:
    # A fresh project may start with a minimal (empty) floor and grow it; an
    # empty LIST is legal (unlike prepare's non-empty path lists).
    _write(tmp_path, "roles:\n" + _PLANNER + _LOCAL + "floor: {checks: []}\n")
    cfg = load_config(tmp_path)
    assert cfg is not None and cfg.floor is not None
    assert cfg.floor.checks == []


def test_floor_roundtrips(tmp_path: Path) -> None:
    # The frozen model round-trips through dump/validate byte-for-byte.
    cfg = FloorConfig(checks=["npx tsc --noEmit"])
    assert FloorConfig.model_validate(cfg.model_dump()) == cfg


def test_floor_empty_command_string_rejected(tmp_path: Path) -> None:
    # An empty/whitespace COMMAND in the list is a config typo (it would pass
    # trivially), distinct from an empty list, and is rejected.
    _write(tmp_path, "roles:\n" + _PLANNER + _LOCAL + 'floor: {checks: ["", "x"]}\n')
    with pytest.raises(ValueError):
        load_config(tmp_path)


def test_floor_unknown_key_rejected(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "roles:\n" + _PLANNER + _LOCAL + "floor: {checks: [x], bogus: 1}\n",
    )
    with pytest.raises(ValueError):
        load_config(tmp_path)


# --- infra_repair: the auto senior infra-repair policy + host-command guard ----


def test_infra_repair_optional_defaults_none(tmp_path: Path) -> None:
    # Absent block: no auto-repair, an infra fail routes through the ordinary path.
    _write(tmp_path, "roles:\n" + _PLANNER + _LOCAL)
    cfg = load_config(tmp_path)
    assert cfg is not None and cfg.infra_repair is None


def test_infra_repair_parses_with_defaults(tmp_path: Path) -> None:
    # Present-but-bare: attempts defaults to 2, allow_host_commands to EMPTY (the
    # host-command guard is deny-by-default).
    _write(tmp_path, "roles:\n" + _PLANNER + _LOCAL + "infra_repair: {}\n")
    cfg = load_config(tmp_path)
    assert cfg is not None and cfg.infra_repair is not None
    assert cfg.infra_repair.attempts == 2
    assert cfg.infra_repair.allow_host_commands == []


def test_infra_repair_parses_allowlist(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "roles:\n" + _PLANNER + _LOCAL
        + 'infra_repair: {attempts: 3, allow_host_commands: ["apt-get", "brew"]}\n',
    )
    cfg = load_config(tmp_path)
    assert cfg is not None and cfg.infra_repair is not None
    assert cfg.infra_repair.attempts == 3
    assert cfg.infra_repair.allow_host_commands == ["apt-get", "brew"]


def test_infra_repair_attempts_zero_allowed(tmp_path: Path) -> None:
    # attempts=0 disables auto-repair (an infra fail escalates immediately); >= 0.
    _write(tmp_path, "roles:\n" + _PLANNER + _LOCAL + "infra_repair: {attempts: 0}\n")
    cfg = load_config(tmp_path)
    assert cfg is not None and cfg.infra_repair is not None
    assert cfg.infra_repair.attempts == 0


def test_infra_repair_negative_attempts_rejected(tmp_path: Path) -> None:
    _write(tmp_path, "roles:\n" + _PLANNER + _LOCAL + "infra_repair: {attempts: -1}\n")
    with pytest.raises(ValueError):
        load_config(tmp_path)


def test_infra_repair_empty_allowlist_entry_rejected(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "roles:\n" + _PLANNER + _LOCAL
        + 'infra_repair: {allow_host_commands: ["", "apt"]}\n',
    )
    with pytest.raises(ValueError):
        load_config(tmp_path)


def test_infra_repair_unknown_key_rejected(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "roles:\n" + _PLANNER + _LOCAL + "infra_repair: {attempts: 2, bogus: 1}\n",
    )
    with pytest.raises(ValueError):
        load_config(tmp_path)


# --- FIX 6: configured script paths must resolve under the bundled models/ dir --
# A cloned repo's .grindstone/config.yaml is attacker-controlled, and every
# `script:` is Popen'd (final_polish even WRITES), an unconstrained path is RCE.


def _bundled_roles() -> str:
    planner = MODELS_DIR / "default" / "planner_request.sh"
    local = MODELS_DIR / "default" / "local_request.sh"
    return (
        "roles:\n"
        f"  planner: {{script: {planner}, slots: 1, timeout_s: 600}}\n"
        f"  local: {{script: {local}, slots: 2, timeout_s: 1800}}\n"
    )


def test_rce_guard_accepts_every_rig_subdir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The move to default/ + codex/ + override/ subdirs keeps every script UNDER
    # MODELS_DIR, so the RCE guard accepts all three layers unchanged.
    monkeypatch.delenv(ALLOW_REPO_SCRIPTS_ENV, raising=False)
    planner = MODELS_DIR / "codex" / "planner_request.sh"
    local = MODELS_DIR / "override" / "local_request.sh"
    senior = MODELS_DIR / "default" / "senior_request.sh"
    _write(
        tmp_path,
        "roles:\n"
        f"  planner: {{script: {planner}, slots: 1, timeout_s: 600}}\n"
        f"  local: {{script: {local}, slots: 2, timeout_s: 1800}}\n"
        f"  senior: {{script: {senior}, slots: 2, timeout_s: 3600}}\n",
    )
    cfg = load_config(tmp_path)
    assert cfg is not None
    validate_script_paths(cfg)  # default/ + codex/ + override/ -> no raise


# --- models_script resolver: override > preset > default ------------------------


def _fake_models(root: Path, *, default: tuple[str, ...], codex: tuple[str, ...] = (),
                 override: tuple[str, ...] = ()) -> None:
    """Build a fake models/ tree under ``root`` with the named scripts per rig."""

    for sub, names in (("default", default), ("codex", codex), ("override", override)):
        (root / sub).mkdir(parents=True, exist_ok=True)
        for name in names:
            (root / sub / name).write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")


def test_models_script_falls_back_to_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("grindstone.config.MODELS_DIR", tmp_path)
    _fake_models(tmp_path, default=("planner_request.sh",))
    resolved = models_script("planner_request.sh")
    assert resolved == (tmp_path / "default" / "planner_request.sh").resolve()
    assert resolved.is_absolute()


def test_models_script_override_beats_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("grindstone.config.MODELS_DIR", tmp_path)
    _fake_models(tmp_path, default=("local_request.sh",), override=("local_request.sh",))
    assert models_script("local_request.sh") == (
        tmp_path / "override" / "local_request.sh"
    ).resolve()


def test_models_script_rig_inserts_middle_layer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("grindstone.config.MODELS_DIR", tmp_path)
    _fake_models(
        tmp_path,
        default=("planner_request.sh",),
        codex=("planner_request.sh",),
    )
    # rig=codex wins over default...
    assert models_script("planner_request.sh", rig="codex") == (
        tmp_path / "codex" / "planner_request.sh"
    ).resolve()
    # ...but rig=None ignores the codex layer entirely.
    assert models_script("planner_request.sh") == (
        tmp_path / "default" / "planner_request.sh"
    ).resolve()


def test_models_script_override_beats_rig(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("grindstone.config.MODELS_DIR", tmp_path)
    _fake_models(
        tmp_path,
        default=("planner_request.sh",),
        codex=("planner_request.sh",),
        override=("planner_request.sh",),
    )
    assert models_script("planner_request.sh", rig="codex") == (
        tmp_path / "override" / "planner_request.sh"
    ).resolve()


def test_models_script_missing_everywhere_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("grindstone.config.MODELS_DIR", tmp_path)
    _fake_models(tmp_path, default=("planner_request.sh",))
    with pytest.raises(FileNotFoundError) as exc:
        models_script("vision_review.sh")
    assert "vision_review.sh" in str(exc.value)


def test_role_script_outside_models_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(ALLOW_REPO_SCRIPTS_ENV, raising=False)
    _write(tmp_path, "roles:\n" + _PLANNER + _LOCAL)  # /m/*.sh -> outside models/
    cfg = load_config(tmp_path)
    assert cfg is not None
    with pytest.raises(ValueError):
        validate_script_paths(cfg)


def test_bundled_models_scripts_are_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(ALLOW_REPO_SCRIPTS_ENV, raising=False)
    _write(tmp_path, _bundled_roles())
    cfg = load_config(tmp_path)
    assert cfg is not None
    validate_script_paths(cfg)  # under models/ -> no raise


def test_env_opt_in_allows_arbitrary_repo_script(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ALLOW_REPO_SCRIPTS_ENV, "1")
    _write(tmp_path, "roles:\n" + _PLANNER + _LOCAL)  # arbitrary /m/*.sh
    cfg = load_config(tmp_path)
    assert cfg is not None
    validate_script_paths(cfg)  # opt-in trusted-repo escape hatch -> no raise


def test_vision_review_script_outside_models_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(ALLOW_REPO_SCRIPTS_ENV, raising=False)
    _write(tmp_path, _bundled_roles() + "vision_review: {script: /tmp/evil.sh, timeout_s: 600}\n")
    cfg = load_config(tmp_path)
    assert cfg is not None
    with pytest.raises(ValueError):
        validate_script_paths(cfg)


def test_final_polish_script_outside_models_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(ALLOW_REPO_SCRIPTS_ENV, raising=False)
    _write(
        tmp_path,
        _bundled_roles()
        + "final_polish: {criteria: c, timeout_s: 600, script: /tmp/evil.sh}\n",
    )
    cfg = load_config(tmp_path)
    assert cfg is not None
    with pytest.raises(ValueError):
        validate_script_paths(cfg)


def test_final_polish_default_script_is_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # final_polish with NO script -> the CLI uses the bundled default; nothing to
    # validate at this layer (the bundled path is trusted).
    monkeypatch.delenv(ALLOW_REPO_SCRIPTS_ENV, raising=False)
    _write(tmp_path, _bundled_roles() + "final_polish: {criteria: c, timeout_s: 600}\n")
    cfg = load_config(tmp_path)
    assert cfg is not None
    validate_script_paths(cfg)  # no script set -> no raise
