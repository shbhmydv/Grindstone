"""The real rig dispatch: a ``WorkerTransport`` behind the file-contract subprocess.

The role-split boundary (lifted from the v7 pipeline, BONES-simplified): grindstone
knows only a ROLE reached through a ``<role>_request.sh`` under ``models/``; the
script owns transport, model identity, GPU arbitration and the killable process
group. ``ScriptWorker`` builds the prompt (worker OR critic, via ``build_prompt``),
``Popen``s the script as a group leader, supervises wall-clock, and on timeout
delegates the kill to ``stop.sh``. The disk is the only result channel: on success
``run`` returns and ``run_task`` reads ``handoff.json`` / ``verdict.json`` from the
scratch; stdout is never parsed.

BONES drops the v7 SessionLimited node: a non-zero exit whose output names a rate /
quota limit raises ``RateLimited`` (the loop parks ~1/hr, failure model #1); every
other non-zero exit and every timeout raises ``TransportError`` (a task failure that
routes to the planner, failure model #2). ``build_backends`` wires one ScriptWorker
per distinct rig endpoint into the ``Backends`` concurrency map (local = its slots).
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import tempfile
import threading
from pathlib import Path

from grindstone.config import (
    GrindstoneConfig,
    RoleConfig,
    models_script,
    resolve_role_script,
)
from grindstone.worker import (
    Backends,
    RateLimited,
    TransportError,
    WorkerRequest,
    WorkerTransport,
    _Endpoint,
    build_prompt,
)

#: A non-zero exit whose combined stdout/stderr matches this is a rate / quota
#: refusal (failure model #1), not a work failure: ``RateLimited`` so the loop parks.
_RATE_LIMIT_RE = re.compile(r"rate.?limit|429|quota", re.IGNORECASE)


class ScriptWorker:
    """``WorkerTransport`` backed by a role-request script behind the file contract.

    ``script`` is the absolute ``<role>_request.sh``; ``stop_script`` reaps the
    process group on timeout (resolved by the caller, not assumed beside the role
    script). ``timeout_s`` is the wall-clock supervisor; ``log_root`` is where the
    per-dispatch prompt + log dirs are allocated (never the run dir). No model
    identity here, it lives behind the script.
    """

    def __init__(
        self,
        *,
        script: Path,
        stop_script: Path,
        timeout_s: float,
        log_root: Path,
    ) -> None:
        self.script = Path(script)
        self._stop_script = Path(stop_script)
        self.timeout_s = timeout_s
        self.log_root = Path(log_root)

    def run(self, request: WorkerRequest) -> None:
        prompt = build_prompt(request)
        slug = request.task_id.replace("/", "-")
        kind = "critic" if request.critic is not None else "worker"
        attempt_dir = self.log_root / f"{slug}-{kind}"
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
                "--worktree", str(request.scratch),
                "--prompt", prompt_name,
                "--log-dir", str(log_dir),
                "--handle-out", str(handle_file),
                "--timeout", str(int(self.timeout_s)),
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
            raise TransportError(
                f"{self.script.name} timed out after {self.timeout_s}s"
            ) from exc
        if proc.returncode != 0:
            combined = f"{stdout or ''}\n{stderr or ''}"
            snippet = (stderr.strip() or stdout.strip())[:200]
            if _RATE_LIMIT_RE.search(combined):
                raise RateLimited(f"{self.script.name} rate-limited: {snippet}")
            raise TransportError(f"{self.script.name} exited {proc.returncode}: {snippet}")
        # Success: run_task reads the disk artifact from request.scratch.

    def _stop(self, handle_file: Path, proc: "subprocess.Popen[str]") -> None:
        """Delegate the kill to ``stop.sh`` (it reads the pgid the role script wrote
        to ``handle_file`` and SIGTERM->SIGKILLs the group); fall back to an
        in-process group kill so a hung worker is never orphaned."""

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


def build_backends(config: GrindstoneConfig, *, log_root: Path) -> Backends:
    """Wire one ``ScriptWorker`` per DISTINCT rig endpoint into a ``Backends`` map.

    The worker role serves the ``local`` tier; the senior role (when present) serves
    ``senior``, else ``senior`` falls back to the worker endpoint (a rig with no
    cloud tier grinds every tier locally). Two tiers whose resolved scripts are the
    SAME path share ONE endpoint (and thus ONE semaphore: no double-booking of a
    single-slot local GPU). Each endpoint's semaphore is sized by its role's
    ``slots`` (local = 1 keeps the :8080 single slot serial)."""

    roles = config.roles
    endpoints: dict[str, _Endpoint] = {}

    def endpoint_for(role: str, rc: RoleConfig) -> str:
        script = resolve_role_script(role, rc)
        key = str(script)
        if key not in endpoints:
            transport: WorkerTransport = ScriptWorker(
                script=script,
                stop_script=models_script("stop.sh", rig=rc.rig),
                timeout_s=rc.timeout_s,
                log_root=log_root,
            )
            endpoints[key] = _Endpoint(transport, threading.Semaphore(rc.slots))
        return key

    tier_endpoint = {"local": endpoint_for("worker", roles.worker)}
    if roles.senior is not None:
        tier_endpoint["senior"] = endpoint_for("senior", roles.senior)
    else:
        tier_endpoint["senior"] = tier_endpoint["local"]
    return Backends(endpoints, tier_endpoint)
