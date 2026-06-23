"""Script-backed planner transport: the `planner` role behind a file contract.

The role-split boundary: grindstone reaches the planner role
through ``planner_request.sh`` and never learns the transport (codex / claude) or
the model behind it. When the boundary supplies a writable ``--workdir`` (the
planner worktree), a self-validating rig grinds IN it, writing its epoch JSON to
``workdir/decision.json`` after looping ``check_decision.py`` until the core gate
is clean; that file is the disk contract grindstone reads back. A read-only rig
ignores ``--workdir`` and writes its final message to ``--out`` instead. Either
way stdout is never scraped, and the core re-runs ``extract_decision_json`` +
validation as defense in depth (parsing stays in core, ruling 1). ``plan`` Popen's
the script as a group leader, supervises wall-clock, and on timeout delegates the
kill to ``stop.sh``.

``slots`` is the authoritative concurrency bound (typically 1 for the planner):
a ``Semaphore`` acquired around each ``plan``.
"""

from __future__ import annotations

import os
import signal
import subprocess
import tempfile
import threading
from pathlib import Path

from grindstone.check_decision import DECISION_FILE
from grindstone.planner import PlannerHardError
from grindstone.worker import (
    RateLimited,
    SessionLimited,
    TransportError,
    WorkerTimeout,
    is_session_limit,
)


class ScriptPlanner:
    """``PlannerTransport`` backed by ``planner_request.sh`` behind a file contract.

    ``script`` is the absolute path to ``planner_request.sh``; ``stop_script`` is
    the explicit ``stop.sh`` path used to reap the group on timeout (resolved by the
    CLI via ``models_script``, NOT assumed beside the planner script: the planner
    can resolve from a preset dir, e.g. ``codex/``, that ships no ``stop.sh``).
    ``repo`` is the target repo (the planner's read-only working root).
    ``slots`` bounds concurrency; ``timeout_s`` is the transport-owned wall-clock
    supervisor. No model identity: that moved into the script. CLI failures map
    onto the exception family (BOTH stdout and stderr are inspected, the claude CLI
    prints its session limit to STDOUT): session/usage limit → SessionLimited
    (→ hourly park), rate/limit/429/quota → RateLimited (→ short backoff),
    auth/login/401 → PlannerHardError (→ human), other non-zero → TransportError
    (→ transient retry), timeout → WorkerTimeout.
    """

    def __init__(
        self,
        *,
        script: Path,
        stop_script: Path,
        repo: Path,
        slots: int,
        timeout_s: float,
    ) -> None:
        self.script = Path(script)
        self._stop_script = Path(stop_script)
        self.repo = Path(repo)
        self.timeout_s = timeout_s
        self._sem = threading.Semaphore(slots)

    def plan(self, prompt: str, *, workdir: Path | None = None) -> str:
        with self._sem:
            return self._dispatch(prompt, workdir)

    def _dispatch(self, prompt: str, workdir: Path | None) -> str:
        with tempfile.TemporaryDirectory(prefix="grind-planner-") as td:
            tmp = Path(td)
            prompt_file = tmp / "prompt.txt"
            out_file = tmp / "out.txt"
            handle_file = tmp / "handle"
            prompt_file.write_text(prompt, encoding="utf-8")

            # A self-validating rig grinds IN ``workdir`` (the boundary worktree)
            # and writes its decision to ``workdir/decision.json``, the disk
            # contract we read back below. We clear any stale copy so a script
            # that silently writes nothing can never feed us a previous verdict.
            decision_file = workdir / DECISION_FILE if workdir is not None else None
            if decision_file is not None:
                decision_file.unlink(missing_ok=True)

            argv = [
                str(self.script),
                "--repo",
                str(self.repo),
                "--prompt",
                str(prompt_file),
                "--out",
                str(out_file),
                "--handle-out",
                str(handle_file),
                "--timeout",
                str(int(self.timeout_s)),
            ]
            if workdir is not None:
                argv += ["--workdir", str(workdir)]

            proc = subprocess.Popen(
                argv,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
            try:
                stdout, stderr = proc.communicate(timeout=self.timeout_s)
            except subprocess.TimeoutExpired as exc:
                self._stop(handle_file, proc)
                raise WorkerTimeout(
                    f"{self.script.name} timed out after {self.timeout_s}s"
                ) from exc
            if proc.returncode != 0:
                self._raise_for_failure(proc.returncode, stdout, stderr)
            # Disk contract, in priority order. (1) A self-validating rig wrote a
            # gate-clean ``decision.json`` into its worktree, that IS the result
            # (the same file the rig looped ``check_decision.py`` on). (2) Else the
            # rig's final message is ``--out``. (3) Else stdout, so the extractor
            # still has bytes. stdout is never scraped when a file exists.
            if decision_file is not None and decision_file.is_file():
                text = decision_file.read_text(encoding="utf-8")
                if text.strip():
                    return text
            if out_file.is_file():
                text = out_file.read_text(encoding="utf-8")
                if text:
                    return text
            return stdout

    def _raise_for_failure(self, returncode: int, stdout: str, stderr: str) -> None:
        # Inspect BOTH streams: the claude CLI prints "You've hit your session
        # limit" to STDOUT, but historically only stderr was read, so the long
        # session limit fell through to a transient TransportError and burned the
        # retry budget in seconds. ``combined`` covers both.
        combined = f"{stdout or ''}\n{stderr or ''}"
        blob = combined.lower()
        # ORDER MATTERS: the long quota-window SESSION/usage limit is detected
        # FIRST (it resets in hours -> hourly park, never the transient budget);
        # only a plain transient 429 falls to RateLimited (short backoff). "usage
        # limit" is the long kind, so it routes to SessionLimited via is_session_limit.
        snippet = (stderr.strip() or stdout.strip())[:300]
        if is_session_limit(combined):
            raise SessionLimited(
                f"{self.script.name} session-limited: {snippet}"
            )
        if (
            "rate" in blob
            and "limit" in blob
            or "429" in blob
            or "quota" in blob
        ):
            raise RateLimited(
                f"{self.script.name} rate-limited: {snippet}"
            )
        if (
            "401" in blob
            or "unauthor" in blob
            or "not logged in" in blob
            or "please run" in blob
            and "login" in blob
            or "authentication" in blob
        ):
            raise PlannerHardError(
                f"{self.script.name} auth/config failure: {snippet}"
            )
        raise TransportError(
            f"{self.script.name} exited {returncode}: {snippet}"
        )

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
