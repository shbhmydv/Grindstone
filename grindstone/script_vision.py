"""Script-backed vision reviewer: the B3 taste gate behind a file contract.

A ``vision_review`` check (``contracts.models.VisionReviewCheck``) is the codex
VISION REVIEW layered on a visual epoch's deterministic functional floor. Like
every other model boundary, grindstone reaches it through a request **script**
(``models/vision_review.sh``) and never learns the transport (codex) behind it:
the script runs ``codex exec -i <screenshot> --output-schema <verdict> -o <out>``
and writes the verdict to ``--out`` — a DISK CONTRACT grindstone re-reads and
Pydantic-validates (never stdout). ``ScriptVisionReviewer`` Popen's the script as
a group leader, supervises the wall-clock, and on timeout delegates the kill to
``stop.sh`` (mirrors ``ScriptPlanner``). Any failure — non-zero exit, timeout,
missing/invalid verdict — raises ``VisionReviewError``, which the core maps to a
deterministic check FAIL (the gate never crashes the run).
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
from pathlib import Path
from typing import Protocol

from grindstone.contracts.models import VisionVerdict, parse_vision_verdict

#: The verdict schema the script hands codex via ``--output-schema`` (constrains
#: the model's output at the source; grindstone re-validates the returned file).
_VERDICT_SCHEMA = Path(__file__).resolve().parents[1] / "schemas" / "vision_verdict.json"

#: Slack between the script's own ``--timeout`` (a graceful ``timeout --signal=
#: TERM``) and the Python wall-clock supervisor's SIGKILL, so the two never fire
#: simultaneously — the script gets a moment to shut down cleanly first.
SUPERVISOR_MARGIN_S = 5.0


class VisionReviewError(Exception):
    """The vision-review gate could not produce a valid verdict (→ check FAILED)."""


class VisionReviewer(Protocol):
    """The seam ``evaluate_checks`` calls to run one vision_review check.

    ``worktree`` is the eval cwd (the rendered-UI checkout); ``screenshot_rel``
    is the worktree-relative PNG/JPEG a prior cmd check produced; ``out_dir`` is
    a per-check scratch dir under the run dir where the verdict/criteria/handle
    land. Returns the parsed verdict or raises ``VisionReviewError``.
    """

    def review(
        self, *, worktree: Path, screenshot_rel: str, criteria: str, out_dir: Path
    ) -> VisionVerdict: ...


class ScriptVisionReviewer:
    """``VisionReviewer`` backed by ``vision_review.sh`` behind a file contract.

    ``script`` is the absolute path to ``vision_review.sh``; ``stop.sh`` is its
    sibling. ``timeout_s`` is the transport-owned wall-clock supervisor. ``schema``
    defaults to the bundled ``vision_verdict.json``. No model identity here — that
    lives in the script.
    """

    def __init__(
        self, *, script: Path, timeout_s: float, schema: Path | None = None
    ) -> None:
        self.script = Path(script)
        self.timeout_s = timeout_s
        self.schema = Path(schema) if schema is not None else _VERDICT_SCHEMA

    @property
    def _stop_script(self) -> Path:
        return self.script.parent / "stop.sh"

    @property
    def supervise_timeout_s(self) -> float:
        """The Python wall-clock deadline: the script's ``--timeout`` plus margin,
        so the supervisor's SIGKILL never races the script's graceful TERM."""

        return self.timeout_s + SUPERVISOR_MARGIN_S

    def review(
        self, *, worktree: Path, screenshot_rel: str, criteria: str, out_dir: Path
    ) -> VisionVerdict:
        out_file = out_dir / "verdict.json"
        handle_file = out_dir / "handle"
        # Setup + launch can raise OSError (missing/non-executable script, an
        # unwritable out_dir). That is NOT a VisionReviewError, so it would escape
        # _vision_result's catch and crash the run — map it to a deterministic
        # FAIL instead (the gate is "always fail, never crash").
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            criteria_file = out_dir / "criteria.txt"
            criteria_file.write_text(criteria, encoding="utf-8")
            proc = subprocess.Popen(
                [
                    str(self.script),
                    "--repo",
                    str(worktree),
                    "--screenshot",
                    screenshot_rel,
                    "--criteria-file",
                    str(criteria_file),
                    "--schema",
                    str(self.schema),
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
        except OSError as exc:
            raise VisionReviewError(
                f"{self.script.name} could not launch: {type(exc).__name__}: {exc}"
            ) from exc
        try:
            _stdout, stderr = proc.communicate(timeout=self.supervise_timeout_s)
        except subprocess.TimeoutExpired as exc:
            self._stop(handle_file, proc)
            raise VisionReviewError(
                f"{self.script.name} timed out after {self.supervise_timeout_s}s"
            ) from exc
        if proc.returncode != 0:
            raise VisionReviewError(
                f"{self.script.name} exited {proc.returncode}: {stderr.strip()[:300]}"
            )
        if not out_file.is_file():
            raise VisionReviewError(f"{self.script.name} wrote no verdict.json")
        try:
            payload = json.loads(out_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise VisionReviewError(f"verdict.json is not valid JSON: {exc}") from exc
        try:
            return parse_vision_verdict(payload)
        except ValueError as exc:
            raise VisionReviewError(f"verdict.json invalid: {exc}") from exc

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
