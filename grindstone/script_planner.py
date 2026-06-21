"""Script-backed planner transport: the `planner` role behind a file contract.

The role-split boundary: grindstone reaches the planner role
through ``planner_request.sh`` and never learns the transport (codex) or the
model behind it. The script runs the locked ``codex exec`` invocation against the
target repo (read-only) and writes the agent's final message to ``--out``, a
disk contract, never stdout scraping. grindstone reads ``--out`` and does the
tolerant ``extract_decision_json`` + validation itself (parsing stays in core,
ruling 1). ``plan`` Popen's the script as a group leader, supervises wall-clock,
and on timeout delegates the kill to ``stop.sh``.

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

from grindstone.planner import PlannerHardError
from grindstone.worker import RateLimited, TransportError, WorkerTimeout


class ScriptPlanner:
    """``PlannerTransport`` backed by ``planner_request.sh`` behind a file contract.

    ``script`` is the absolute path to ``planner_request.sh``; ``stop_script`` is
    the explicit ``stop.sh`` path used to reap the group on timeout (resolved by the
    CLI via ``models_script``, NOT assumed beside the planner script: the planner
    can resolve from a preset dir, e.g. ``codex/``, that ships no ``stop.sh``).
    ``repo`` is the target repo (the planner's read-only working root).
    ``slots`` bounds concurrency; ``timeout_s`` is the transport-owned wall-clock
    supervisor. No model identity: that moved into the script. CLI failures map
    onto the exception family: rate/limit/429/quota → RateLimited (→ backoff),
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

    def plan(self, prompt: str) -> str:
        with self._sem:
            return self._dispatch(prompt)

    def _dispatch(self, prompt: str) -> str:
        with tempfile.TemporaryDirectory(prefix="grind-planner-") as td:
            tmp = Path(td)
            prompt_file = tmp / "prompt.txt"
            out_file = tmp / "out.txt"
            handle_file = tmp / "handle"
            prompt_file.write_text(prompt, encoding="utf-8")

            proc = subprocess.Popen(
                [
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
                ],
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
                self._raise_for_failure(proc.returncode, stderr)
            # Disk contract: the final message is the --out file; stdout is the
            # event log, never the result. Fall back to stdout only if the script
            # wrote no file (edge), so the extractor still has bytes.
            if out_file.is_file():
                text = out_file.read_text(encoding="utf-8")
                if text:
                    return text
            return stdout

    def _raise_for_failure(self, returncode: int, stderr: str) -> None:
        blob = (stderr or "").lower()
        if (
            "rate" in blob
            and "limit" in blob
            or "429" in blob
            or "quota" in blob
            or "usage limit" in blob
        ):
            raise RateLimited(
                f"{self.script.name} rate-limited: {stderr.strip()[:300]}"
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
                f"{self.script.name} auth/config failure: {stderr.strip()[:300]}"
            )
        raise TransportError(
            f"{self.script.name} exited {returncode}: {stderr.strip()[:300]}"
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
