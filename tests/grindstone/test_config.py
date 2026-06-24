"""Config loading: roles/rig/slots + the epoch backstop, the unknown-key hard
error, and the script-path RCE guard."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from grindstone.config import (
    GrindstoneConfig,
    load_config,
    resolve_role_script,
    validate_script_paths,
)

_MODELS = Path(__file__).resolve().parents[2] / "models"


def _write_config(repo: Path, body: str) -> Path:
    cfg_dir = repo / ".grindstone"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    path = cfg_dir / "config.yaml"
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return path


def test_absent_config_is_none(tmp_path: Path) -> None:
    assert load_config(tmp_path) is None


def test_minimal_rig_config_loads(tmp_path: Path) -> None:
    _write_config(
        tmp_path,
        """
        roles:
          planner: { rig: claude, slots: 1, timeout_s: 600 }
          worker:  { rig: local,  slots: 1, timeout_s: 1800 }
          senior:  { rig: claude, slots: 2, timeout_s: 1200 }
        max_epochs: 12
        """,
    )
    cfg = load_config(tmp_path)
    assert isinstance(cfg, GrindstoneConfig)
    assert cfg.roles.worker.rig == "local"
    assert cfg.roles.worker.slots == 1
    assert cfg.roles.senior is not None and cfg.roles.senior.slots == 2
    assert cfg.max_epochs == 12
    # rig resolves to the bundled per-role script under models/.
    assert resolve_role_script("worker", cfg.roles.worker) == (
        _MODELS / "local" / "worker_request.sh"
    ).resolve()


def test_senior_is_optional(tmp_path: Path) -> None:
    _write_config(
        tmp_path,
        """
        roles:
          planner: { rig: claude, slots: 1, timeout_s: 600 }
          worker:  { rig: local,  slots: 1, timeout_s: 1800 }
        """,
    )
    cfg = load_config(tmp_path)
    assert cfg is not None and cfg.roles.senior is None
    assert cfg.max_epochs is None  # backstop defaults to the CLI built-in


def test_unknown_key_is_a_hard_error(tmp_path: Path) -> None:
    _write_config(
        tmp_path,
        """
        roles:
          planner: { rig: claude, slots: 1, timeout_s: 600 }
          worker:  { rig: local,  slots: 1, timeout_s: 1800 }
        floor: { checks: [pytest] }
        """,
    )
    with pytest.raises(ValueError):
        load_config(tmp_path)


def test_script_xor_rig(tmp_path: Path) -> None:
    _write_config(
        tmp_path,
        """
        roles:
          planner: { rig: claude, script: /x.sh, slots: 1, timeout_s: 600 }
          worker:  { rig: local,  slots: 1, timeout_s: 1800 }
        """,
    )
    with pytest.raises(ValueError):
        load_config(tmp_path)


def test_bad_slots_rejected(tmp_path: Path) -> None:
    _write_config(
        tmp_path,
        """
        roles:
          planner: { rig: claude, slots: 0, timeout_s: 600 }
          worker:  { rig: local,  slots: 1, timeout_s: 1800 }
        """,
    )
    with pytest.raises(ValueError):
        load_config(tmp_path)


def test_validate_script_paths_rejects_outside_models(tmp_path: Path) -> None:
    rogue = tmp_path / "evil.sh"
    rogue.write_text("#!/bin/sh\n", encoding="utf-8")
    _write_config(
        tmp_path,
        f"""
        roles:
          planner: {{ script: {rogue}, slots: 1, timeout_s: 600 }}
          worker:  {{ rig: local, slots: 1, timeout_s: 1800 }}
        """,
    )
    cfg = load_config(tmp_path)
    assert cfg is not None
    with pytest.raises(ValueError):
        validate_script_paths(cfg)


def test_validate_script_paths_allows_bundled(tmp_path: Path) -> None:
    bundled = _MODELS / "claude" / "planner_request.sh"
    _write_config(
        tmp_path,
        f"""
        roles:
          planner: {{ script: {bundled}, slots: 1, timeout_s: 600 }}
          worker:  {{ rig: local, slots: 1, timeout_s: 1800 }}
        """,
    )
    cfg = load_config(tmp_path)
    assert cfg is not None
    validate_script_paths(cfg)  # does not raise
