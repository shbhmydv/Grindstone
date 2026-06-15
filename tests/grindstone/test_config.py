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
    GrindstoneConfig,
    load_config,
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


# --- FIX 6: configured script paths must resolve under the bundled models/ dir --
# A cloned repo's .grindstone/config.yaml is attacker-controlled, and every
# `script:` is Popen'd (final_polish even WRITES), an unconstrained path is RCE.


def _bundled_roles() -> str:
    planner = MODELS_DIR / "planner_request.sh"
    local = MODELS_DIR / "local_request.sh"
    return (
        "roles:\n"
        f"  planner: {{script: {planner}, slots: 1, timeout_s: 600}}\n"
        f"  local: {{script: {local}, slots: 2, timeout_s: 1800}}\n"
    )


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
