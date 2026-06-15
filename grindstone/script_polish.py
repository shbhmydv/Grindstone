"""Script-backed final polisher: the B5 codex inline-edit pass behind a script.

The B5 polish pass lets codex EDIT a finished, gated repo inline (workspace-write)
for finishing touches. Like every other model boundary, grindstone reaches it
through a request **script** (``models/codex_polish.sh``) and never learns the
transport (codex) behind it: the script runs ``codex exec -s workspace-write`` in
the worktree and edits files IN PLACE, there is NO disk-contract verdict, because
the GATE is the run's evidence re-run (``run_loop._final_polish``), not codex's
word. ``ScriptPolisher`` Popen's the script as a group leader, supervises the
wall-clock, and on timeout delegates the kill to ``stop.sh`` (mirrors
``ScriptVisionReviewer``).

``polish`` returns whether codex ran SUCCESSFULLY (exit 0); a non-zero exit or a
timeout returns ``False``, never an exception. The caller treats a ``False`` (and
any error it raises around the call) as a clean no-op: a failed polish can never
turn a completed run into a failure.
"""

from __future__ import annotations

import os
import signal
import subprocess
from pathlib import Path
from typing import Protocol

#: Slack between the script's own ``--timeout`` (graceful ``timeout --signal=
#: TERM``) and the Python supervisor's SIGKILL, so they never fire at once.
SUPERVISOR_MARGIN_S = 5.0


class Polisher(Protocol):
    """The seam ``run_loop._final_polish`` calls to run one codex polish pass.

    ``worktree`` is a writable checkout of the run's final branch that codex edits
    in place; ``criteria`` is the polish brief; ``screenshot_rel`` is an optional
    worktree-relative image for a visual pass; ``out_dir`` is a per-run scratch dir
    for the criteria file + kill handle. Returns whether codex ran successfully.
    """

    def polish(
        self,
        *,
        worktree: Path,
        criteria: str,
        screenshot_rel: str | None,
        out_dir: Path,
    ) -> bool: ...


class ScriptPolisher:
    """``Polisher`` backed by ``codex_polish.sh`` (workspace-write codex).

    ``script`` is the absolute path to ``codex_polish.sh``; ``stop.sh`` is its
    sibling. ``timeout_s`` is the transport-owned wall-clock supervisor. No model
    identity here, that lives in the script.
    """

    def __init__(self, *, script: Path, timeout_s: float) -> None:
        self.script = Path(script)
        self.timeout_s = timeout_s

    @property
    def _stop_script(self) -> Path:
        return self.script.parent / "stop.sh"

    @property
    def supervise_timeout_s(self) -> float:
        """The Python wall-clock deadline: the script's ``--timeout`` plus margin,
        so the supervisor's SIGKILL never races the script's graceful TERM."""

        return self.timeout_s + SUPERVISOR_MARGIN_S

    def polish(
        self,
        *,
        worktree: Path,
        criteria: str,
        screenshot_rel: str | None,
        out_dir: Path,
    ) -> bool:
        handle_file = out_dir / "handle"
        # Setup + launch can raise OSError (missing/non-executable script, an
        # unwritable out_dir). A failed polish is a clean no-op, so convert it to
        # False rather than letting an OSError escape into the run loop.
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            criteria_file = out_dir / "criteria.txt"
            criteria_file.write_text(criteria, encoding="utf-8")
            argv = [
                str(self.script),
                "--repo",
                str(worktree),
                "--criteria-file",
                str(criteria_file),
                "--handle-out",
                str(handle_file),
                "--timeout",
                str(int(self.timeout_s)),
            ]
            if screenshot_rel is not None:
                argv += ["--screenshot", screenshot_rel]
            proc = subprocess.Popen(
                argv,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
        except OSError:
            return False
        try:
            proc.communicate(timeout=self.supervise_timeout_s)
        except subprocess.TimeoutExpired:
            self._stop(handle_file, proc)
            return False
        return proc.returncode == 0

    def _stop(self, handle_file: Path, proc: "subprocess.Popen[str]") -> None:
        """Delegate the kill to ``stop.sh``; fall back to an in-process group kill."""

        if self._stop_script.is_file():
            try:
                subprocess.run(
                    [str(self._stop_script), "--handle", str(handle_file)],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=15,
                )
            except (OSError, subprocess.SubprocessError):
                self._killpg(proc)
        else:
            self._killpg(proc)
        try:
            proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            self._killpg(proc)
            proc.communicate()

    @staticmethod
    def _killpg(proc: "subprocess.Popen[str]") -> None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
