"""Script-backed worker transport: a role behind a file-contract subprocess.

The role-split boundary: grindstone knows only a *role* name
(`worker` / `senior`) reached through a script. It never learns the transport
(`pi` / cloud) or the model behind the role, the script owns transport, model
identity, GPU arbitration, the pin file (``.pi/settings.json``) and the killable
process group. On a non-zero exit it inspects BOTH stdout and stderr (the claude
CLI prints its session limit to STDOUT): a session/usage limit -> ``SessionLimited``
(hourly park), a 429 -> ``RateLimited`` (short backoff), else ``TransportError``.
``ScriptWorker`` builds the prompt (orchestration: *what* to ask,
``build_worker_prompt`` in core), Popen's the script as a group leader, supervises
wall-clock, and on timeout delegates the kill to ``stop.sh`` (the v7 kill-group
scar, now in models/). The disk is the only result channel (ARCHITECTURE.md): on success
``run`` returns and the loop reads ``handoff.json`` from the worktree.

``slots`` is the authoritative per-role concurrency bound: a ``Semaphore``
acquired around each ``run`` (it replaces the per-GPU slot router the role config drops).
"""

from __future__ import annotations

import os
import signal
import subprocess
import tempfile
import threading
from pathlib import Path

from grindstone.worker import (
    RateLimited,
    SessionLimited,
    TransportError,
    WorkerRequest,
    WorkerTimeout,
    build_worker_prompt,
    is_session_limit,
)


class ScriptWorker:
    """``WorkerTransport`` backed by a role-request script behind a file contract.

    ``script`` is the absolute path to ``worker_request.sh`` / ``senior_request.sh``;
    ``stop_script`` is the explicit ``stop.sh`` path used to reap the group on
    timeout (resolved by the CLI via ``models_script``, NOT assumed beside the role
    script: a role can resolve from a preset/override dir that ships no ``stop.sh``).
    ``slots`` bounds concurrency, ``timeout_s`` is the transport-owned wall-clock
    supervisor (NOT loop policy, §10), ``log_root`` is where per-attempt log dirs +
    handle files are allocated (never the run dir, §7). No provider/model: those
    moved into the script.
    """

    def __init__(
        self,
        *,
        script: Path,
        stop_script: Path,
        slots: int,
        timeout_s: float,
        log_root: Path,
    ) -> None:
        self.script = Path(script)
        self._stop_script = Path(stop_script)
        self.timeout_s = timeout_s
        self.log_root = Path(log_root)
        self._sem = threading.Semaphore(slots)

    def run(self, request: WorkerRequest) -> None:
        with self._sem:
            self._dispatch(request)

    def _dispatch(self, request: WorkerRequest) -> None:
        prompt = build_worker_prompt(request)
        slug = request.task_id.replace("/", "-")
        attempt_dir = self.log_root / f"{slug}-attempt-{request.attempt}"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        handle_file = attempt_dir / "handle"
        log_dir = attempt_dir / "logs"

        prompt_fd, prompt_name = tempfile.mkstemp(
            prefix="grind-prompt-", suffix=".txt", dir=str(attempt_dir)
        )
        with os.fdopen(prompt_fd, "w", encoding="utf-8") as fh:
            fh.write(prompt)

        proc = subprocess.Popen(
            [
                str(self.script),
                "--worktree",
                str(request.scratch),
                "--prompt",
                prompt_name,
                "--log-dir",
                str(log_dir),
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
            # Inspect BOTH streams: the claude CLI prints its session limit to
            # STDOUT, so a stderr-only check missed it and burned the task retry
            # ladder against a multi-hour limit. ORDER: the long quota-window
            # session/usage limit is detected FIRST (-> SessionLimited -> hourly
            # park, never an attempt burn); only a plain 429 is RateLimited.
            combined = f"{stdout or ''}\n{stderr or ''}"
            blob = combined.lower()
            snippet = (stderr.strip() or stdout.strip())[:200]
            if is_session_limit(combined):
                raise SessionLimited(
                    f"{self.script.name} session-limited: {snippet}"
                )
            if "rate" in blob and "limit" in blob or "429" in blob:
                raise RateLimited(
                    f"{self.script.name} rate-limited: {snippet}"
                )
            raise TransportError(
                f"{self.script.name} exited {proc.returncode}: {snippet}"
            )
        # Success: the loop reads handoff.json from request.scratch. Stdout is
        # never parsed for results (disk contract).

    def _stop(self, handle_file: Path, proc: "subprocess.Popen[str]") -> None:
        """Delegate the kill to ``stop.sh`` (the script owns the group reap).

        ``stop.sh`` reads the pgid the role script wrote to ``handle_file`` and
        SIGTERM→SIGKILLs the whole group. If ``stop.sh`` is missing we fall back
        to a minimal in-process group kill so a hung worker is never orphaned.
        """

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
        # Reap our Popen handle regardless so no zombie lingers.
        try:
            proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            self._killpg(proc)
            proc.communicate()

    @staticmethod
    def _killpg(proc: "subprocess.Popen[str]") -> None:
        """Fallback in-process group kill (SIGKILL) when stop.sh is unavailable."""

        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
