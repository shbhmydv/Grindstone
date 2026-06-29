"""The real planner rig dispatch: a ``PlannerTransport`` behind the file contract.

The role-split boundary (mirror of ``script_worker.ScriptWorker``): grindstone knows
only the ``planner`` role reached through ``planner_request.sh`` under ``models/``;
the script owns transport, model identity, GPU arbitration, and the killable process
group. ``ScriptPlannerTransport`` writes the prompt to a temp file, ``Popen``s the
script as a group leader with the boundary's ``--workdir`` (the in-repo
``_planner_tip`` checkout) + ``--out`` fallback, supervises wall-clock, and on timeout
delegates the kill to ``stop.sh``. The disk is the result channel: the rig writes its
gate-clean ``decision.json`` into the worktree (and/or its final message to ``--out``);
``dispatch`` returns raw stdout and the core (``ScriptPlanner.decide``) reads the
channels back by priority. stdout is never parsed here.

BONES failure mapping (mirrors the worker): a non-zero exit whose output names a rate /
quota limit raises ``RateLimited`` (the loop parks ~1/hr, node #1); a wall-clock timeout
raises ``PlannerTimeout`` (the loop retries it once, then backs off); every other
non-zero exit raises ``PlannerError``. On the PLAN call the loop retries ALL of these
under a consecutive-failure cap before the clean partial-end; on CLOSE-OUT a non-rate
error routes to the epoch abort node. The CLI wires one ``ScriptPlannerTransport`` from
config in a later part.
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import tempfile
import threading
from pathlib import Path

from grindstone import reaper
from grindstone.planner import (
    PlannerDispatch,
    PlannerError,
    PlannerTimeout,
    RateLimited,
)

#: A non-zero exit whose combined stdout/stderr matches this is a rate / quota
#: refusal (node #1): ``RateLimited`` so the loop parks, not a hard failure.
_RATE_LIMIT_RE = re.compile(r"rate.?limit|429|quota|session limit|usage limit", re.IGNORECASE)


class ScriptPlannerTransport:
    """``PlannerTransport`` backed by ``planner_request.sh`` behind the file contract.

    ``script`` is the absolute ``planner_request.sh``; ``stop_script`` reaps the
    process group on timeout (resolved by the caller, not assumed beside the planner
    script). ``repo`` is the target repo (passed as ``--repo``); ``timeout_s`` is the
    wall-clock supervisor; ``slots`` bounds concurrency (1 for a one-shot planner). No
    model identity here, it lives behind the script.
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

    def dispatch(self, request: PlannerDispatch) -> str:
        with self._sem:
            return self._run(request)

    def _run(self, request: PlannerDispatch) -> str:
        with tempfile.TemporaryDirectory(prefix="grind-planner-") as td:
            tmp = Path(td)
            prompt_file = tmp / "prompt.txt"
            handle_file = tmp / "handle"
            prompt_file.write_text(request.prompt, encoding="utf-8")

            proc = subprocess.Popen(
                [
                    str(self.script),
                    "--repo", str(self.repo),
                    "--prompt", str(prompt_file),
                    "--out", str(request.out_file),
                    "--handle-out", str(handle_file),
                    "--timeout", str(int(self.timeout_s)),
                    "--workdir", str(request.workdir),
                    "--purpose", request.purpose,
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
            # The child leads its own group (pgid == pid); register it so a run-level
            # SIGTERM/SIGINT reaps it instead of orphaning the detached group.
            reaper.register(proc.pid)
            try:
                try:
                    stdout, stderr = proc.communicate(timeout=self.timeout_s)
                except subprocess.TimeoutExpired as exc:
                    self._stop(handle_file, proc)
                    raise PlannerTimeout(
                        f"{self.script.name} timed out after {self.timeout_s}s"
                    ) from exc
                if proc.returncode != 0:
                    combined = f"{stdout or ''}\n{stderr or ''}"
                    snippet = (stderr.strip() or stdout.strip())[:200]
                    if _RATE_LIMIT_RE.search(combined):
                        raise RateLimited(f"{self.script.name} rate-limited: {snippet}")
                    raise PlannerError(
                        f"{self.script.name} exited {proc.returncode}: {snippet}"
                    )
                return stdout
            finally:
                reaper.unregister(proc.pid)

    def _stop(self, handle_file: Path, proc: "subprocess.Popen[str]") -> None:
        """Delegate the kill to ``stop.sh`` (it reads the pgid the planner script
        wrote to ``handle_file``); fall back to an in-process group kill."""

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
