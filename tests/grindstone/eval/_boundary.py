"""Replicate ONE real planner boundary the way the run loop does, for the eval.

``run_planner_boundary`` is the eval's single live entry point: given a job spec +
a rig, it constructs the SAME planner input the run loop builds, arms the SAME
self-validation context the run loop arms (``check_decision.write_validator``),
invokes the rig's real ``planner_request.sh`` in self-validate mode, reads back the
decision via the SAME read priority the orchestrator uses, and runs it through the
SAME core gate (``validate_decision``). It returns the validated, typed
``EpochDecision`` (or raises with the gate errors + the raw output for debugging).

The decision READ here mirrors ``grindstone.script_planner.ScriptPlanner._dispatch``
exactly: the decision text is taken from a priority chain, not from
``decision.json`` alone, so the eval accepts precisely what the orchestrator would.
The chain is (1) ``workdir/decision.json`` if it exists and is non-empty after
strip (the self-validating rig's disk contract), (2) else the ``--out`` file if it
exists and is non-empty (the rig's final message), (3) else the script's stdout.
When the winning source is NOT ``decision.json`` the boundary still succeeds, but
emits a ``warnings.warn`` so the self-validate-on-disk reliability gap stays visible
without failing the eval: falling back to ``--out`` means the model skipped the
on-disk grounding loop, a quality signal, not a correctness failure.

The gate CONTEXT here is a faithful copy of ``run_loop._arm_self_validation`` +
``run_loop._plan_boundary_loop``: the exact eight keys ``write_validator`` bakes,
and the exact ``validate_decision`` kwargs. It is DUPLICATED rather than imported
because those run-loop helpers take a live ``_RunStateStore`` / ``RunDir`` /
``_PhaseContext`` (a whole run in flight); the eval has only a job spec and a few
boundary signals, so reconstructing the small context dict here is far cleaner than
standing up a fake run state, and the duplication is pinned by reusing the public
``DEFAULT_*_MAX_TASK_FILES`` constants and the public gate. No production code is
touched.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

from grindstone import check_decision
from grindstone.config import RoleConfig, resolve_role_script
from grindstone.contracts.models import EpochDecision, Phase
from grindstone.domain_skills import load_domain_skill_index
from grindstone.planner import (
    DEFAULT_LOCAL_MAX_TASK_FILES,
    DEFAULT_SENIOR_MAX_TASK_FILES,
    FailedEpochInfo,
    build_planner_input,
    extract_decision_json,
    validate_decision,
)


class BoundaryError(AssertionError):
    """A planner boundary that produced no gate-clean decision.

    Carries the gate errors (or the wiring failure) plus the raw planner output so
    a failing eval can be debugged without a re-run."""


def _current_phase_id(
    skeleton: list[Phase] | None, completed_phase_ids: tuple[str, ...]
) -> str | None:
    """The first not-yet-completed phase id (the boundary's current phase), or None.

    Mirrors the run loop's notion of the active phase for the ``<state>`` tail: the
    earliest skeleton phase not in ``completed_phase_ids``. ``None`` when there is no
    skeleton (first boundary) or every phase is done."""

    if skeleton is None:
        return None
    for phase in skeleton:
        if phase.id not in completed_phase_ids:
            return phase.id
    return None


def run_planner_boundary(
    *,
    job_spec: str,
    rig: str,
    skeleton: list[Phase] | None = None,
    completed_phase_ids: tuple[str, ...] = (),
    failed_epoch: FailedEpochInfo | None = None,
    has_senior: bool = True,
    repo: Path | None = None,
    timeout: float = 900,
) -> EpochDecision:
    """Drive one real planner boundary on ``rig`` and return the validated decision.

    Replicates ``run_loop._plan_boundary``: derive the boundary signals, arm the
    self-validation context, build the planner input, invoke the rig's real planner
    script in self-validate mode (``--workdir``), read back the decision via the same
    priority chain ``ScriptPlanner._dispatch`` uses (decision.json -> --out -> stdout),
    and gate it. Raises ``BoundaryError`` (with gate errors + every captured source)
    when no gate-clean decision is produced."""

    skeleton_exists = skeleton is not None
    failed_epoch_active = failed_epoch is not None
    completed = sorted(completed_phase_ids)
    # The target repo's domain-skill catalogue (empty for the common eval repo with
    # no .grindstone/skills/index.md): the planner sees the index, the gate bounds
    # selection by its names, mirroring run_loop.
    domain_skill_index = load_domain_skill_index(repo) if repo is not None else {}
    known_skill_names = frozenset(domain_skill_index)
    # The gate context, byte-for-byte the set run_loop._arm_self_validation bakes via
    # write_validator (existing_log_keys empty: a fresh eval boundary has no keyed
    # log; phase_escalated False: the eval never drives a budget-exhausted phase).
    context: dict[str, object] = {
        "existing_log_keys": [],
        "completed_phase_ids": completed,
        "skeleton_exists": skeleton_exists,
        "phase_escalated": False,
        "failed_epoch_active": failed_epoch_active,
        "has_senior": has_senior,
        "local_max_task_files": DEFAULT_LOCAL_MAX_TASK_FILES,
        "senior_max_task_files": DEFAULT_SENIOR_MAX_TASK_FILES,
        "known_skill_names": sorted(known_skill_names),
    }

    script = resolve_role_script(
        "planner", RoleConfig(rig=rig, slots=1, timeout_s=timeout)
    )

    with tempfile.TemporaryDirectory(prefix="gs-eval-") as scratch_str:
        scratch = Path(scratch_str)
        workdir = scratch / "workdir"
        workdir.mkdir()
        # The rig script self-validates IN workdir: arm it with the validator + the
        # baked context exactly as run_loop._arm_self_validation does.
        check_decision.write_validator(
            workdir, context=context, grindstone_python=sys.executable
        )

        # A real repo for --repo: the script cd's into it + sets GIT_CEILING_DIRECTORIES.
        # Self-validate mode runs cwd=workdir, so the repo is just a valid git dir.
        repo_dir = repo if repo is not None else _init_temp_repo(scratch / "repo")

        prompt = build_planner_input(
            job=job_spec,
            skeleton=skeleton,
            phase_id=_current_phase_id(skeleton, completed_phase_ids),
            epoch_counter=len(completed),
            log_index=[],
            last_epoch_rows=None,
            reask_errors=[],
            failed_epoch=failed_epoch,
            domain_skill_index=domain_skill_index,
        )
        prompt_file = scratch / "prompt.txt"
        prompt_file.write_text(prompt, encoding="utf-8")
        out_file = scratch / "planner_out.txt"
        handle_file = scratch / "handle.txt"

        proc = subprocess.run(
            [
                str(script),
                "--repo", str(repo_dir),
                "--prompt", str(prompt_file),
                "--out", str(out_file),
                "--handle-out", str(handle_file),
                "--timeout", str(timeout),
                "--workdir", str(workdir),
            ],
            capture_output=True,
            text=True,
            timeout=timeout + 60,
        )

        decision_file = workdir / check_decision.DECISION_FILE
        decision_text, source = _select_decision_text(
            decision_file, out_file, proc.stdout
        )

        gate = validate_decision(
            extract_decision_json(decision_text),
            existing_log_keys=frozenset(),
            completed_phase_ids=frozenset(completed_phase_ids),
            skeleton_exists=skeleton_exists,
            phase_escalated=False,
            failed_epoch_active=failed_epoch_active,
            has_senior=has_senior,
            local_max_task_files=DEFAULT_LOCAL_MAX_TASK_FILES,
            senior_max_task_files=DEFAULT_SENIOR_MAX_TASK_FILES,
            known_skill_names=known_skill_names,
        )
        if gate.decision is None:
            raw_out = (
                out_file.read_text(encoding="utf-8") if out_file.is_file() else ""
            )
            raise BoundaryError(
                f"rig {rig!r} produced no gate-clean decision "
                f"(chosen source {source!r}, script exit {proc.returncode}): "
                f"{gate.errors}\n"
                f"chosen decision text:\n{decision_text}\n"
                f"--out:\n{raw_out}\n"
                f"stdout:\n{proc.stdout}\n"
                f"stderr:\n{proc.stderr}"
            )
        if source != "decision_json":
            warnings.warn(
                f"rig {rig!r} did not self-validate on disk: no gate-clean "
                f"{check_decision.DECISION_FILE} in the workdir; the gate-clean "
                f"decision came from the {source!r} fallback. The orchestrator "
                f"accepts this, but the on-disk self-validate loop was skipped "
                f"(a quality signal, not a correctness failure).",
                stacklevel=2,
            )
        return gate.decision


def _select_decision_text(
    decision_file: Path, out_file: Path, stdout: str
) -> tuple[str, str]:
    """Pick the decision text + its source, mirroring ``ScriptPlanner._dispatch``.

    Priority, exactly as the orchestrator reads it: (1) ``decision_file`` if it
    exists and is non-empty after strip (the self-validating rig's disk contract);
    (2) else ``out_file`` if it exists and is non-empty (the rig's final message);
    (3) else ``stdout``. Returns ``(text, source)`` where ``source`` is one of
    ``"decision_json"``, ``"out_file"``, ``"stdout"``."""

    if decision_file.is_file():
        text = decision_file.read_text(encoding="utf-8")
        if text.strip():
            return text, "decision_json"
    if out_file.is_file():
        text = out_file.read_text(encoding="utf-8")
        if text:
            return text, "out_file"
    return stdout, "stdout"


def _init_temp_repo(path: Path) -> Path:
    """Create a minimal committed git repo so the rig's ``--repo`` resolves.

    The planner script cd's into ``--repo`` and sets ``GIT_CEILING_DIRECTORIES``;
    in self-validate mode the actual planning cwd is the workdir, so the repo only
    needs to be a valid git directory with a born HEAD."""

    path.mkdir(parents=True)
    (path / "README.md").write_text("# eval target repo\n", encoding="utf-8")
    env_git = [
        "git",
        "-c", "user.email=eval@grindstone.test",
        "-c", "user.name=eval",
    ]
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(env_git + ["-C", str(path), "add", "."], check=True)
    subprocess.run(
        env_git + ["-C", str(path), "commit", "-q", "-m", "seed"], check=True
    )
    return path
