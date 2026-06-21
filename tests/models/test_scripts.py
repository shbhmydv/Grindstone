"""Bash-level tests for the ``models/`` boundary scripts.

These drive the entry scripts as real subprocesses with a STUB
``pi``/``codex``/``opencode``/``claude`` prepended onto PATH, so no live model,
GPU, or model binary is ever touched. The scripts are the checked artifact of this
slice; these tests are the gate.

Layout after the rig split: the shipped Claude rig lives in ``models/default/``
(tracked), the codex planner preset in ``models/codex/`` (tracked), and the
operator's personal pi/opencode/codex scripts in ``models/override/`` (GITIGNORED).
Tests for an override script ``skip`` when it is absent (a fresh clone has no
override rig), so the suite stays green everywhere while still covering the
operator's scripts on a rig that ships them.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import pytest

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
DEFAULT_DIR = MODELS_DIR / "default"
CODEX_DIR = MODELS_DIR / "codex"
OVERRIDE_DIR = MODELS_DIR / "override"

# Operator's personal rig (gitignored): pi local, opencode senior, codex gates.
LOCAL = OVERRIDE_DIR / "local_request.sh"
SENIOR = OVERRIDE_DIR / "senior_request.sh"
VISION = OVERRIDE_DIR / "vision_review.sh"
POLISH = OVERRIDE_DIR / "codex_polish.sh"
# Tracked presets: codex planner + generic helpers.
PLANNER = CODEX_DIR / "planner_request.sh"
STOP = DEFAULT_DIR / "stop.sh"
# Shipped default Claude rig (tracked).
DEFAULT_PLANNER = DEFAULT_DIR / "planner_request.sh"
DEFAULT_LOCAL = DEFAULT_DIR / "local_request.sh"
DEFAULT_SENIOR = DEFAULT_DIR / "senior_request.sh"
SCHEMA = Path(__file__).resolve().parents[2] / "schemas" / "vision_verdict.json"


def _require(script: Path) -> None:
    """Skip when ``script`` is absent (the operator's models/override is gitignored,
    so a fresh clone has no personal rig to exercise)."""
    if not script.is_file():
        pytest.skip(f"{script} not present (models/override is gitignored)")


def _make_stub(dir_: Path, name: str, body: str) -> None:
    """Drop an executable shell stub named ``name`` into ``dir_``."""
    p = dir_ / name
    p.write_text("#!/usr/bin/env bash\n" + body, encoding="utf-8")
    p.chmod(0o755)


def _env_with_stub_path(stub_dir: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["PATH"] = f"{stub_dir}{os.pathsep}{env['PATH']}"
    return env


# --- local_request.sh --------------------------------------------------------


def test_local_request_relays_handoff_and_writes_handle(tmp_path: Path) -> None:
    _require(LOCAL)
    stub_dir = tmp_path / "bin"
    stub_dir.mkdir()
    # Stub pi writes handoff.json into its CWD (the worktree) and exits 0.
    _make_stub(
        stub_dir,
        "pi",
        'printf \'{"status":"DONE"}\' > "$PWD/handoff.json"\nexit 0\n',
    )

    worktree = tmp_path / "wt"
    worktree.mkdir()
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("do the thing", encoding="utf-8")
    log_dir = tmp_path / "logs"
    handle = tmp_path / "handle.txt"

    res = subprocess.run(
        [
            "bash", str(LOCAL),
            "--worktree", str(worktree),
            "--prompt", str(prompt),
            "--log-dir", str(log_dir),
            "--handle-out", str(handle),
            "--timeout", "30",
        ],
        env=_env_with_stub_path(stub_dir),
        capture_output=True,
        text=True,
    )

    assert res.returncode == 0, res.stderr
    # handoff landed in the worktree (the only result channel).
    assert (worktree / "handoff.json").read_text() == '{"status":"DONE"}'
    # handle-out holds a numeric pgid.
    pgid = handle.read_text().strip()
    assert pgid.isdigit(), f"handle not numeric: {pgid!r}"
    # logs were teed.
    assert (log_dir / "agent.stdout.log").exists()


def test_local_request_propagates_nonzero_exit(tmp_path: Path) -> None:
    _require(LOCAL)
    stub_dir = tmp_path / "bin"
    stub_dir.mkdir()
    # Stub pi prints a rate-limit reason to stderr and exits non-zero.
    _make_stub(stub_dir, "pi", 'echo "429 rate limit exceeded" >&2\nexit 7\n')

    worktree = tmp_path / "wt"
    worktree.mkdir()
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("x", encoding="utf-8")

    res = subprocess.run(
        [
            "bash", str(LOCAL),
            "--worktree", str(worktree),
            "--prompt", str(prompt),
            "--log-dir", str(tmp_path / "logs"),
            "--handle-out", str(tmp_path / "handle.txt"),
            "--timeout", "30",
        ],
        env=_env_with_stub_path(stub_dir),
        capture_output=True,
        text=True,
    )

    # pi's exit code is propagated EXACTLY (not masked by the script's own error
    # path, the IQ4-era $gpu_index unbound-var bug would have exited 1 here).
    assert res.returncode == 7, res.stderr
    # The reason is forwarded to the caller's stderr (grindstone greps it).
    assert "429" in res.stderr or "rate" in res.stderr.lower()


def test_local_request_missing_arg_errors(tmp_path: Path) -> None:
    _require(LOCAL)
    res = subprocess.run(
        ["bash", str(LOCAL), "--worktree", str(tmp_path)],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 2
    assert "missing required" in res.stderr


def test_local_request_pins_reviewer_subagent_to_local_model(tmp_path: Path) -> None:
    _require(LOCAL)
    # The implement plan spawns a `reviewer` pi-subagent which does NOT inherit
    # our --provider/--model; it reads the nearest `.pi/settings.json` and treats
    # that dir as the project root. The script must pin the reviewer to the SAME
    # local model, else it silently hits the cloud default. The local role is ONE
    # model on :8080 (no GPU keying since the IQ4->Q6 swap).
    stub_dir = tmp_path / "bin"
    stub_dir.mkdir()
    _make_stub(
        stub_dir,
        "pi",
        'printf \'{"status":"DONE"}\' > "$PWD/handoff.json"\nexit 0\n',
    )

    worktree = tmp_path / "wt"
    worktree.mkdir()
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("do the thing", encoding="utf-8")

    res = subprocess.run(
        [
            "bash", str(LOCAL),
            "--worktree", str(worktree),
            "--prompt", str(prompt),
            "--log-dir", str(tmp_path / "logs"),
            "--handle-out", str(tmp_path / "handle.txt"),
            "--timeout", "30",
        ],
        env=_env_with_stub_path(stub_dir),
        capture_output=True,
        text=True,
    )

    assert res.returncode == 0, res.stderr
    settings_path = worktree / ".pi" / "settings.json"
    assert settings_path.exists(), "reviewer-pin .pi/settings.json not written"
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    assert (
        settings["subagents"]["agentOverrides"]["reviewer"]["model"]
        == "local-reviewer/qwen-3-6-27b-dense"
    ), settings


# --- senior_request.sh -------------------------------------------------------


def test_senior_request_relays_handoff_no_gpu(tmp_path: Path) -> None:
    _require(SENIOR)
    stub_dir = tmp_path / "bin"
    stub_dir.mkdir()
    # Senior drives the opencode agent (not pi); it writes handoff.json in its CWD.
    _make_stub(
        stub_dir,
        "opencode",
        'printf \'{"status":"DONE"}\' > "$PWD/handoff.json"\nexit 0\n',
    )

    worktree = tmp_path / "wt"
    worktree.mkdir()
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("senior task", encoding="utf-8")
    handle = tmp_path / "handle.txt"

    res = subprocess.run(
        [
            "bash", str(SENIOR),
            "--worktree", str(worktree),
            "--prompt", str(prompt),
            "--log-dir", str(tmp_path / "logs"),
            "--handle-out", str(handle),
            "--timeout", "30",
        ],
        # NOTE: no GRINDSTONE_GPU_LOCKDIR, senior claims no GPU at all.
        env=_env_with_stub_path(stub_dir),
        capture_output=True,
        text=True,
    )

    assert res.returncode == 0, res.stderr
    assert (worktree / "handoff.json").exists()
    assert handle.read_text().strip().isdigit()


def test_senior_request_runs_opencode_with_exa_and_no_pin(tmp_path: Path) -> None:
    _require(SENIOR)
    # Senior runs the opencode agent with the Exa websearch tool ON, pinned to the
    # cloud model, auto-approving tools. It writes NO .pi pin (opencode owns its own
    # agents), and the per-attempt OPENCODE_DB lives in the LOG dir, never the
    # worktree (so it can't leak into a commit, and concurrent attempts isolate).
    stub_dir = tmp_path / "bin"
    stub_dir.mkdir()
    record = tmp_path / "invocation.txt"
    _make_stub(
        stub_dir,
        "opencode",
        f'{{ echo "ARGV: $*"; echo "EXA=${{OPENCODE_ENABLE_EXA:-}}"; '
        f'echo "DB=${{OPENCODE_DB:-}}"; }} > "{record}"\n'
        'printf \'{"status":"DONE"}\' > "$PWD/handoff.json"\nexit 0\n',
    )

    worktree = tmp_path / "wt"
    worktree.mkdir()
    log_dir = tmp_path / "logs"
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("senior task", encoding="utf-8")

    res = subprocess.run(
        [
            "bash", str(SENIOR),
            "--worktree", str(worktree),
            "--prompt", str(prompt),
            "--log-dir", str(log_dir),
            "--handle-out", str(tmp_path / "handle.txt"),
            "--timeout", "30",
        ],
        env=_env_with_stub_path(stub_dir),
        capture_output=True,
        text=True,
    )

    assert res.returncode == 0, res.stderr
    rec = record.read_text(encoding="utf-8")
    assert "--dangerously-skip-permissions" in rec
    assert "-m opencode-go/glm-5.2" in rec
    assert f"--dir {worktree}" in rec
    assert "EXA=true" in rec
    db_line = rec.split("DB=", 1)[1].splitlines()[0]
    assert str(log_dir) in db_line and str(worktree) not in db_line
    assert not (worktree / ".pi" / "settings.json").exists()


# --- planner_request.sh ------------------------------------------------------


def test_planner_request_writes_out_file(tmp_path: Path) -> None:
    stub_dir = tmp_path / "bin"
    stub_dir.mkdir()
    # Stub codex parses `-o <file>` and writes the final message there.
    _make_stub(
        stub_dir,
        "codex",
        (
            'out=""\n'
            'while [[ $# -gt 0 ]]; do\n'
            '  case "$1" in\n'
            '    -o) out="$2"; shift 2 ;;\n'
            '    *) shift ;;\n'
            '  esac\n'
            'done\n'
            'printf \'{"tool":"emit_epoch"}\' > "$out"\n'
            'exit 0\n'
        ),
    )

    repo = tmp_path / "repo"
    repo.mkdir()
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("plan an epoch", encoding="utf-8")
    out = tmp_path / "decision.txt"
    handle = tmp_path / "handle.txt"

    res = subprocess.run(
        [
            "bash", str(PLANNER),
            "--repo", str(repo),
            "--prompt", str(prompt),
            "--out", str(out),
            "--handle-out", str(handle),
            "--timeout", "30",
        ],
        env=_env_with_stub_path(stub_dir),
        capture_output=True,
        text=True,
    )

    assert res.returncode == 0, res.stderr
    assert out.read_text() == '{"tool":"emit_epoch"}'
    assert handle.read_text().strip().isdigit()


def test_planner_request_propagates_nonzero_exit(tmp_path: Path) -> None:
    stub_dir = tmp_path / "bin"
    stub_dir.mkdir()
    _make_stub(stub_dir, "codex", 'echo "not logged in" >&2\nexit 3\n')

    repo = tmp_path / "repo"
    repo.mkdir()
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("x", encoding="utf-8")

    res = subprocess.run(
        [
            "bash", str(PLANNER),
            "--repo", str(repo),
            "--prompt", str(prompt),
            "--out", str(tmp_path / "out.txt"),
            "--handle-out", str(tmp_path / "handle.txt"),
            "--timeout", "30",
        ],
        env=_env_with_stub_path(stub_dir),
        capture_output=True,
        text=True,
    )

    assert res.returncode != 0
    assert "not logged in" in res.stderr


# --- vision_review.sh --------------------------------------------------------


def _vision_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    """A repo (codex -C root) holding a screenshot, plus a criteria file."""
    repo = tmp_path / "repo"
    (repo / "ui").mkdir(parents=True)
    screenshot = repo / "ui" / "screen.png"
    screenshot.write_bytes(b"\x89PNG\r\n\x1a\n fake")
    criteria = tmp_path / "criteria.txt"
    criteria.write_text("buttons aligned, polished corners", encoding="utf-8")
    return repo, screenshot, criteria, tmp_path / "verdict.json"


def test_vision_review_invokes_codex_with_image_schema_and_prompt_first(tmp_path: Path) -> None:
    _require(VISION)
    # The taste gate calls codex with the screenshot via -i, the verdict schema
    # via --output-schema, and writes the verdict to -o. The PROMPT positional
    # MUST precede -i (codex exec ordering gotcha); the prompt carries the
    # criteria text so we can locate it in argv.
    stub_dir = tmp_path / "bin"
    stub_dir.mkdir()
    record = tmp_path / "argv.txt"
    _make_stub(
        stub_dir,
        "codex",
        (
            'args=("$@")\n'
            # NUL-separated record (args carry newlines: the prompt is multi-line).
            f'printf "%s\\0" "${{args[@]}}" > "{record}"\n'
            'out=""\n'
            'n=0\n'
            'while [[ $n -lt ${#args[@]} ]]; do\n'
            '  if [[ "${args[$n]}" == "-o" ]]; then out="${args[$((n+1))]}"; fi\n'
            '  n=$((n+1))\n'
            'done\n'
            'printf \'{"pass": true, "reasons": []}\' > "$out"\n'
            'exit 0\n'
        ),
    )

    repo, screenshot, criteria, out = _vision_inputs(tmp_path)
    res = subprocess.run(
        [
            "bash", str(VISION),
            "--repo", str(repo),
            "--screenshot", "ui/screen.png",
            "--criteria-file", str(criteria),
            "--schema", str(SCHEMA),
            "--out", str(out),
            "--handle-out", str(tmp_path / "handle.txt"),
            "--timeout", "30",
        ],
        env=_env_with_stub_path(stub_dir),
        capture_output=True,
        text=True,
    )

    assert res.returncode == 0, res.stderr
    # The verdict landed at -o (the disk contract) and parses.
    assert json.loads(out.read_text()) == {"pass": True, "reasons": []}
    assert (tmp_path / "handle.txt").read_text().strip().isdigit()

    parts = record.read_bytes().split(b"\0")
    flat = [p.decode() for p in parts if p]  # drop the trailing empty field
    assert "--output-schema" in flat
    assert str(SCHEMA) in flat
    i_at = flat.index("-i")
    assert flat[i_at + 1].endswith("ui/screen.png")  # screenshot passed to -i
    # The prompt (carrying the criteria text) precedes -i.
    prompt_at = next(n for n, a in enumerate(flat) if "polished corners" in a)
    assert prompt_at < i_at


def test_vision_review_propagates_nonzero_exit(tmp_path: Path) -> None:
    _require(VISION)
    stub_dir = tmp_path / "bin"
    stub_dir.mkdir()
    _make_stub(stub_dir, "codex", 'echo "model overloaded" >&2\nexit 4\n')

    repo, _screenshot, criteria, out = _vision_inputs(tmp_path)
    res = subprocess.run(
        [
            "bash", str(VISION),
            "--repo", str(repo),
            "--screenshot", "ui/screen.png",
            "--criteria-file", str(criteria),
            "--schema", str(SCHEMA),
            "--out", str(out),
            "--handle-out", str(tmp_path / "handle.txt"),
            "--timeout", "30",
        ],
        env=_env_with_stub_path(stub_dir),
        capture_output=True,
        text=True,
    )
    assert res.returncode == 4, res.stderr
    assert "overloaded" in res.stderr


def test_vision_review_rejects_non_image_screenshot(tmp_path: Path) -> None:
    _require(VISION)
    # PNG/JPEG only, a non-image screenshot path is a hard error before codex.
    stub_dir = tmp_path / "bin"
    stub_dir.mkdir()
    _make_stub(stub_dir, "codex", 'exit 0\n')
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "notes.txt").write_text("x", encoding="utf-8")
    criteria = tmp_path / "criteria.txt"
    criteria.write_text("c", encoding="utf-8")
    res = subprocess.run(
        [
            "bash", str(VISION),
            "--repo", str(repo),
            "--screenshot", "notes.txt",
            "--criteria-file", str(criteria),
            "--schema", str(SCHEMA),
            "--out", str(tmp_path / "v.json"),
            "--handle-out", str(tmp_path / "handle.txt"),
            "--timeout", "30",
        ],
        env=_env_with_stub_path(stub_dir),
        capture_output=True,
        text=True,
    )
    assert res.returncode != 0
    assert "png" in res.stderr.lower() or "image" in res.stderr.lower()


def test_vision_review_rejects_traversal_screenshot(tmp_path: Path) -> None:
    _require(VISION)
    # Defense-in-depth at the script boundary: a `..` path segment (or an absolute
    # path) must be rejected (exit 2) BEFORE it is joined onto $repo, so a crafted
    # screenshot can never escape the repo. Mirrors the Python contract (Chunk 2).
    stub_dir = tmp_path / "bin"
    stub_dir.mkdir()
    _make_stub(stub_dir, "codex", "exit 0\n")
    repo = tmp_path / "repo"
    repo.mkdir()
    criteria = tmp_path / "criteria.txt"
    criteria.write_text("c", encoding="utf-8")
    res = subprocess.run(
        [
            "bash", str(VISION),
            "--repo", str(repo),
            "--screenshot", "../escape.png",
            "--criteria-file", str(criteria),
            "--schema", str(SCHEMA),
            "--out", str(tmp_path / "v.json"),
            "--handle-out", str(tmp_path / "handle.txt"),
            "--timeout", "30",
        ],
        env=_env_with_stub_path(stub_dir),
        capture_output=True,
        text=True,
    )
    assert res.returncode == 2
    assert ".." in res.stderr


def test_vision_review_missing_arg_errors(tmp_path: Path) -> None:
    _require(VISION)
    res = subprocess.run(
        ["bash", str(VISION), "--repo", str(tmp_path)],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 2
    assert "missing required" in res.stderr


# --- codex_polish.sh (override) ----------------------------------------------


def _polish_inputs(tmp_path: Path) -> tuple[Path, Path]:
    """A writable worktree codex would edit, plus a criteria file."""
    repo = tmp_path / "wt"
    repo.mkdir()
    (repo / "app.py").write_text("print('hi')\n", encoding="utf-8")
    criteria = tmp_path / "criteria.txt"
    criteria.write_text("tighten the spacing, polish the copy", encoding="utf-8")
    return repo, criteria


def test_polish_invokes_codex_workspace_write_with_prompt_and_criteria(
    tmp_path: Path,
) -> None:
    _require(POLISH)
    # The polish pass runs codex in workspace-write against the worktree, with the
    # criteria woven into the prompt. No --output-schema / -o: codex edits files
    # in place (the gate is the evidence re-run, not a verdict file).
    stub_dir = tmp_path / "bin"
    stub_dir.mkdir()
    record = tmp_path / "argv.txt"
    _make_stub(
        stub_dir,
        "codex",
        (
            'args=("$@")\n'
            f'printf "%s\\0" "${{args[@]}}" > "{record}"\n'
            "exit 0\n"
        ),
    )
    repo, criteria = _polish_inputs(tmp_path)
    res = subprocess.run(
        [
            "bash", str(POLISH),
            "--repo", str(repo),
            "--criteria-file", str(criteria),
            "--handle-out", str(tmp_path / "handle.txt"),
            "--timeout", "30",
        ],
        env=_env_with_stub_path(stub_dir),
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, res.stderr
    assert (tmp_path / "handle.txt").read_text().strip().isdigit()

    parts = record.read_bytes().split(b"\0")
    flat = [p.decode() for p in parts if p]
    assert "exec" in flat
    s_at = flat.index("-s")
    assert flat[s_at + 1] == "workspace-write"
    # `codex exec` is non-interactive (approval=never by default) and REJECTS -a;
    # the script must NOT pass it (regression: real codex 0.130.0 errors on -a).
    assert "-a" not in flat
    # The -C cwd is the writable root, so NO --add-dir is passed (it added nothing).
    assert "--add-dir" not in flat
    # Network access is pinned OFF explicitly (defense-in-depth: workspace-write's
    # default is off but inherited, so a global config could otherwise flip it on).
    assert "--config" in flat
    cfg_at = flat.index("--config")
    assert flat[cfg_at + 1] == "sandbox_workspace_write.network_access=false"
    # The prompt positional carries the criteria text; no verdict schema/-o here.
    assert any("polish the copy" in a for a in flat)
    assert "--output-schema" not in flat and "-o" not in flat


def test_polish_passes_screenshot_after_prompt(tmp_path: Path) -> None:
    _require(POLISH)
    # An optional screenshot is forwarded via -i, AFTER the prompt positional
    # (codex exec arg-ordering gotcha, mirrors vision_review.sh).
    stub_dir = tmp_path / "bin"
    stub_dir.mkdir()
    record = tmp_path / "argv.txt"
    _make_stub(
        stub_dir,
        "codex",
        ('args=("$@")\n' f'printf "%s\\0" "${{args[@]}}" > "{record}"\n' "exit 0\n"),
    )
    repo, criteria = _polish_inputs(tmp_path)
    (repo / "ui").mkdir()
    (repo / "ui" / "shot.png").write_bytes(b"\x89PNG\r\n\x1a\n fake")
    res = subprocess.run(
        [
            "bash", str(POLISH),
            "--repo", str(repo),
            "--criteria-file", str(criteria),
            "--screenshot", "ui/shot.png",
            "--handle-out", str(tmp_path / "handle.txt"),
            "--timeout", "30",
        ],
        env=_env_with_stub_path(stub_dir),
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, res.stderr
    parts = record.read_bytes().split(b"\0")
    flat = [p.decode() for p in parts if p]
    i_at = flat.index("-i")
    assert flat[i_at + 1].endswith("ui/shot.png")
    prompt_at = next(n for n, a in enumerate(flat) if "polish the copy" in a)
    assert prompt_at < i_at


def test_polish_propagates_nonzero_exit(tmp_path: Path) -> None:
    _require(POLISH)
    stub_dir = tmp_path / "bin"
    stub_dir.mkdir()
    _make_stub(stub_dir, "codex", 'echo "model overloaded" >&2\nexit 4\n')
    repo, criteria = _polish_inputs(tmp_path)
    res = subprocess.run(
        [
            "bash", str(POLISH),
            "--repo", str(repo),
            "--criteria-file", str(criteria),
            "--handle-out", str(tmp_path / "handle.txt"),
            "--timeout", "30",
        ],
        env=_env_with_stub_path(stub_dir),
        capture_output=True,
        text=True,
    )
    assert res.returncode == 4, res.stderr
    assert "overloaded" in res.stderr


def test_polish_missing_arg_errors(tmp_path: Path) -> None:
    _require(POLISH)
    res = subprocess.run(
        ["bash", str(POLISH), "--repo", str(tmp_path)],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 2
    assert "missing required" in res.stderr


# --- stop.sh -----------------------------------------------------------------


def test_stop_kills_live_process_group(tmp_path: Path) -> None:
    handle = tmp_path / "handle.txt"
    # setsid makes `sleep` a session+group leader, so its PID == its PGID.
    proc = subprocess.Popen(
        ["setsid", "sleep", "120"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    # Give setsid a moment to establish the new group leader.
    time.sleep(0.3)
    pgid = os.getpgid(proc.pid)
    handle.write_text(f"{pgid}\n", encoding="utf-8")

    res = subprocess.run(
        ["bash", str(STOP), "--handle", str(handle)],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, res.stderr

    # The process group should be gone.
    deadline = time.time() + 5
    while time.time() < deadline:
        if proc.poll() is not None:
            break
        time.sleep(0.1)
    assert proc.poll() is not None, "stop.sh did not kill the process group"
    # Reap to avoid a zombie.
    proc.wait(timeout=5)


def test_stop_is_noop_on_stale_handle(tmp_path: Path) -> None:
    handle = tmp_path / "handle.txt"
    # A pid that is almost certainly not a live process group.
    handle.write_text("999999\n", encoding="utf-8")
    res = subprocess.run(
        ["bash", str(STOP), "--handle", str(handle)],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, res.stderr


def test_stop_is_noop_on_missing_handle(tmp_path: Path) -> None:
    res = subprocess.run(
        ["bash", str(STOP), "--handle", str(tmp_path / "nope.txt")],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, res.stderr


# --- default/ Claude rig (tracked: planner + local + senior) ------------------
# These exercise the shipped default scripts with a STUB `claude` on PATH. They
# run everywhere (the default rig is tracked, unlike models/override).


def test_default_planner_captures_stdout_to_out_file(tmp_path: Path) -> None:
    # The default planner runs `claude -p` read-only and the SCRIPT redirects
    # claude's stdout to --out (the disk contract); grindstone parses --out, never
    # stdout. The stub claude just prints the decision JSON.
    stub_dir = tmp_path / "bin"
    stub_dir.mkdir()
    _make_stub(stub_dir, "claude", 'printf \'{"tool":"emit_epoch"}\'\nexit 0\n')

    repo = tmp_path / "repo"
    repo.mkdir()
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("plan an epoch", encoding="utf-8")
    out = tmp_path / "decision.txt"
    handle = tmp_path / "handle.txt"

    res = subprocess.run(
        [
            "bash", str(DEFAULT_PLANNER),
            "--repo", str(repo),
            "--prompt", str(prompt),
            "--out", str(out),
            "--handle-out", str(handle),
            "--timeout", "30",
        ],
        env=_env_with_stub_path(stub_dir),
        capture_output=True,
        text=True,
    )

    assert res.returncode == 0, res.stderr
    assert out.read_text() == '{"tool":"emit_epoch"}'
    assert handle.read_text().strip().isdigit()


def test_default_planner_is_read_only_no_skip_permissions(tmp_path: Path) -> None:
    # The planner must NOT bypass permissions (read-only): it allowlists only
    # Read/Grep/Glob and never passes --dangerously-skip-permissions, so a headless
    # run cannot edit the repo.
    stub_dir = tmp_path / "bin"
    stub_dir.mkdir()
    record = tmp_path / "argv.txt"
    _make_stub(
        stub_dir,
        "claude",
        f'printf "%s\\0" "$@" > "{record}"\nprintf \'{{"tool":"x"}}\'\nexit 0\n',
    )
    repo = tmp_path / "repo"
    repo.mkdir()
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("plan", encoding="utf-8")
    res = subprocess.run(
        [
            "bash", str(DEFAULT_PLANNER),
            "--repo", str(repo),
            "--prompt", str(prompt),
            "--out", str(tmp_path / "o.txt"),
            "--handle-out", str(tmp_path / "h.txt"),
            "--timeout", "30",
        ],
        env=_env_with_stub_path(stub_dir),
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, res.stderr
    flat = [p.decode() for p in record.read_bytes().split(b"\0") if p]
    assert "--dangerously-skip-permissions" not in flat
    assert "--allowedTools" in flat
    assert "-p" in flat


def test_default_planner_propagates_nonzero_exit(tmp_path: Path) -> None:
    stub_dir = tmp_path / "bin"
    stub_dir.mkdir()
    _make_stub(stub_dir, "claude", 'echo "not logged in" >&2\nexit 3\n')
    repo = tmp_path / "repo"
    repo.mkdir()
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("x", encoding="utf-8")
    res = subprocess.run(
        [
            "bash", str(DEFAULT_PLANNER),
            "--repo", str(repo),
            "--prompt", str(prompt),
            "--out", str(tmp_path / "out.txt"),
            "--handle-out", str(tmp_path / "handle.txt"),
            "--timeout", "30",
        ],
        env=_env_with_stub_path(stub_dir),
        capture_output=True,
        text=True,
    )
    assert res.returncode == 3
    assert "not logged in" in res.stderr


def test_default_planner_missing_arg_errors(tmp_path: Path) -> None:
    res = subprocess.run(
        ["bash", str(DEFAULT_PLANNER), "--repo", str(tmp_path)],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 2
    assert "missing required" in res.stderr


def _run_default_worker(script: Path, tmp_path: Path, claude_body: str) -> subprocess.CompletedProcess[str]:
    stub_dir = tmp_path / "bin"
    stub_dir.mkdir()
    _make_stub(stub_dir, "claude", claude_body)
    worktree = tmp_path / "wt"
    worktree.mkdir()
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("do the thing", encoding="utf-8")
    return subprocess.run(
        [
            "bash", str(script),
            "--worktree", str(worktree),
            "--prompt", str(prompt),
            "--log-dir", str(tmp_path / "logs"),
            "--handle-out", str(tmp_path / "handle.txt"),
            "--timeout", "30",
        ],
        env=_env_with_stub_path(stub_dir),
        capture_output=True,
        text=True,
    )


def test_default_local_relays_handoff_and_writes_handle(tmp_path: Path) -> None:
    # The default local worker runs `claude -p` IN the worktree; the agent writes
    # handoff.json into its CWD (the only result channel).
    res = _run_default_worker(
        DEFAULT_LOCAL,
        tmp_path,
        'printf \'{"status":"DONE"}\' > "$PWD/handoff.json"\nexit 0\n',
    )
    assert res.returncode == 0, res.stderr
    assert (tmp_path / "wt" / "handoff.json").read_text() == '{"status":"DONE"}'
    assert (tmp_path / "handle.txt").read_text().strip().isdigit()
    assert (tmp_path / "logs" / "agent.stdout.log").exists()


def test_default_local_full_permissions_in_worktree(tmp_path: Path) -> None:
    # A worker must be able to edit/exec headlessly: it runs with
    # --dangerously-skip-permissions inside the isolated worktree.
    record = tmp_path / "argv.txt"
    res = _run_default_worker(
        DEFAULT_LOCAL,
        tmp_path,
        f'printf "%s\\0" "$@" > "{record}"\n'
        'printf \'{"status":"DONE"}\' > "$PWD/handoff.json"\nexit 0\n',
    )
    assert res.returncode == 0, res.stderr
    flat = [p.decode() for p in record.read_bytes().split(b"\0") if p]
    assert "--dangerously-skip-permissions" in flat
    assert "-p" in flat


def test_default_local_propagates_nonzero_exit(tmp_path: Path) -> None:
    res = _run_default_worker(
        DEFAULT_LOCAL, tmp_path, 'echo "429 rate limit exceeded" >&2\nexit 7\n'
    )
    assert res.returncode == 7, res.stderr
    assert "429" in res.stderr or "rate" in res.stderr.lower()


def test_default_local_missing_arg_errors(tmp_path: Path) -> None:
    res = subprocess.run(
        ["bash", str(DEFAULT_LOCAL), "--worktree", str(tmp_path)],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 2
    assert "missing required" in res.stderr


def test_default_senior_relays_handoff(tmp_path: Path) -> None:
    res = _run_default_worker(
        DEFAULT_SENIOR,
        tmp_path,
        'printf \'{"status":"DONE"}\' > "$PWD/handoff.json"\nexit 0\n',
    )
    assert res.returncode == 0, res.stderr
    assert (tmp_path / "wt" / "handoff.json").exists()
    assert (tmp_path / "handle.txt").read_text().strip().isdigit()


def test_default_senior_missing_arg_errors(tmp_path: Path) -> None:
    res = subprocess.run(
        ["bash", str(DEFAULT_SENIOR), "--worktree", str(tmp_path)],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 2
    assert "missing required" in res.stderr
