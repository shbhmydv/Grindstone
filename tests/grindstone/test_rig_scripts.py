"""The rig request scripts as a disk contract (shell-level integration).

These drive the real ``models/<rig>/planner_request.sh`` with a FAKE ``codex`` /
``claude`` on PATH that only records its argv + cwd, so we pin the load-bearing
invocation without a live model. The bug this guards: the codex planner rig used to
run ``-s read-only`` and DISCARD the ``--workdir`` grindstone passes, so it could not
write ``decision.json`` / ``baton.md`` into the throwaway ``_planner_tip`` worktree -
the living baton was dead every epoch (the capture chain caught codex's
read-only-filesystem error as the "baton"). The fix unifies the contract: both planner
rigs grind IN the writable worktree (``codex`` via ``-s workspace-write`` with its cwd
set to that worktree), so the rig-agnostic "write ./baton.md" prompt works for both.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

_CODEX_PLANNER = Path(__file__).resolve().parents[2] / "models" / "codex" / "planner_request.sh"

#: A fake ``codex`` that records the cwd it ran in and its argv, drains stdin (the
#: prompt is piped in), and exits 0 without touching a model.
_FAKE_CODEX = """\
#!/usr/bin/env bash
{ echo "CWD=$(pwd)"; printf 'ARGS=%s\\n' "$*"; } >> "$RIG_LOG"
cat > /dev/null
exit 0
"""


def _install_fake_codex(bindir: Path, log: Path) -> None:
    bindir.mkdir(parents=True, exist_ok=True)
    stub = bindir / "codex"
    stub.write_text(_FAKE_CODEX, encoding="utf-8")
    stub.chmod(0o755)
    log.write_text("", encoding="utf-8")


def _run_codex_planner(
    tmp_path: Path, *, workdir: Path | None
) -> str:
    """Run the codex planner rig with a fake ``codex`` and return the recorded log
    (the cwd + argv the rig invoked ``codex`` with)."""

    repo = tmp_path / "repo"
    repo.mkdir()
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("decide the next epoch\n", encoding="utf-8")
    out = tmp_path / "out.txt"
    handle = tmp_path / "handle"
    bindir = tmp_path / "bin"
    log = tmp_path / "rig.log"
    _install_fake_codex(bindir, log)

    import os

    env = dict(os.environ)
    env["PATH"] = f"{bindir}:{env['PATH']}"
    env["RIG_LOG"] = str(log)

    args = [
        str(_CODEX_PLANNER),
        "--repo", str(repo),
        "--prompt", str(prompt),
        "--out", str(out),
        "--handle-out", str(handle),
        "--timeout", "",
    ]
    if workdir is not None:
        workdir.mkdir(exist_ok=True)
        args += ["--workdir", str(workdir)]
    proc = subprocess.run(args, capture_output=True, text=True, env=env)
    assert proc.returncode == 0, proc.stderr
    return log.read_text(encoding="utf-8")


def test_codex_planner_runs_writable_in_the_passed_workdir(tmp_path: Path) -> None:
    # With --workdir (the in-repo _planner_tip checkout) grindstone always passes, the
    # codex rig must grind IN that worktree with WRITE access, so it can land
    # decision.json + baton.md there like the claude rig - not read-only against the
    # real repo. This is the root-cause fix for the dead baton on a read-only rig.
    workdir = tmp_path / "_planner_tip"
    log = _run_codex_planner(tmp_path, workdir=workdir)
    # writable sandbox, NOT read-only
    assert "workspace-write" in log
    assert "read-only" not in log
    # the workdir is the working root: cwd resolved to it AND it is named as -C
    resolved = str(workdir.resolve())
    assert f"CWD={resolved}" in log
    assert resolved in log.split("ARGS=", 1)[1]  # appears in the codex argv (-C)
    # the --out fallback channel is still wired
    assert str(tmp_path / "out.txt") in log
