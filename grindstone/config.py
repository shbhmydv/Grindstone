"""Per-repo owner config: ``.grindstone/config.yaml`` → frozen typed object.

S5 ruling 3 + role-split. YAML is the owner-facing config
format (project convention); the loader is a small frozen-Pydantic layer with
explicit unknown-key rejection, NOT a schema framework. A typo'd key is a hard
error, never a silently-ignored default, because a config that half-applies is
worse than no config.

The config knows only *roles*, never models or transports: each role
(``planner`` / ``worker`` / ``senior``) names a request **script** behind the
file contract (the script owns transport, model identity, GPU arbitration and
the killable process group, see ``models/``), plus its concurrency ``slots``
and wall-clock ``timeout_s``. ``planner`` + ``worker`` are required; ``senior``
is optional, a rig with no cloud tier runs a local-only escalation ladder.

The CLI consults this loader on ``run`` / ``resume``: present config supplies
the planner + worker ladder wiring; absent config fails loudly toward
``grindstone init`` (core ships no rig-specific defaults).
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    ValidationError,
    field_validator,
    model_validator,
)

#: Repo-relative path of the owner config.
_CONFIG_REL = Path(".grindstone") / "config.yaml"

#: This rig's bundled request-script directory (``models/`` beside the package),
#: the same dir the CLI resolves the bundled defaults from. A configured
#: ``script:`` must live UNDER here unless the operator opts out (below).
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"

#: This rig's bundled operating-skills directory (``skills/`` beside the package),
#: mirrors ``MODELS_DIR``. Grindstone owns these backend-agnostic markdown skills:
#: an operating skill is the thin, deterministically-selected guidance a ROLE
#: (planner / worker / senior) gets for ONE call situation, composed into the
#: prompt as plain text (never a backend flag). They live under
#: ``skills/operating/<role>/<scenario>.md``; ``load_operating_skill`` resolves them.
SKILLS_DIR = Path(__file__).resolve().parents[1] / "skills"

#: The operating-skills subtree under ``SKILLS_DIR`` (room for repo-owned domain
#: skills under a sibling subtree later).
_OPERATING_SUBDIR = "operating"


def load_operating_skill(role: str, scenario: str) -> str:
    """Read the operating skill ``skills/operating/<role>/<scenario>.md``.

    Role-generic on purpose: the planner selects one of its scenarios today, and
    the worker / senior roles reuse the SAME loader for their own scenarios. The
    skill is returned verbatim for the caller to compose into the prompt. Raises
    ``FileNotFoundError`` (naming the resolved path) when the file is missing, a
    selector that names a scenario with no skill file is a build error, never a
    silent empty prompt.
    """

    path = SKILLS_DIR / _OPERATING_SUBDIR / role / f"{scenario}.md"
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise FileNotFoundError(
            f"no operating skill for role={role!r} scenario={scenario!r} at {path}"
        ) from exc

#: The tracked base rig: the shipped Claude (Opus) scripts a fresh cloner runs with
#: zero setup. ``models_script`` always falls back here (the floor).
_CLAUDE_FLOOR = "claude"

#: The operator's personal rig (gitignored): a per-file shadow consulted ONLY for
#: the implicit default (``rig=None``), never for an explicitly selected rig.
_PERSONAL_RIG = "personal"

#: Backend-agnostic shared helpers (``stop.sh``, the sourced ``_*.sh`` prefixes):
#: the lowest fallback under every resolution, so a helper present only here is
#: still found whether a rig is selected or not.
_COMMON_DIR = "_common"


def models_script(name: str, rig: str | None = None) -> Path:
    """Resolve a bundled request script ``name`` to its absolute path.

    Two resolution modes, by whether a rig is EXPLICITLY selected:

    * ``rig`` given (explicit): search ``[rig, "claude", "_common"]`` -- exact,
      reproducible, NEVER shadowed by the gitignored ``personal/``. A test (or a
      ``--rig codex`` run) that asks for rig X gets rig X's script or the shipped
      floor, never the operator's personal script.
    * ``rig`` is ``None`` (implicit default): search ``["personal", "claude",
      "_common"]`` -- the operator's personal rig wins where present, else the
      shipped Claude floor.

    ``_common`` is the shared-helper floor under both modes (``stop.sh`` and the
    sourced prefixes live there). The search list is de-duped, so ``rig="claude"``
    does not double-search. Every candidate sits under ``MODELS_DIR``, so the
    resolved path passes the ``validate_script_paths`` RCE guard unchanged.
    Returns an absolute, resolved path; raises ``FileNotFoundError`` (listing the
    dirs searched) when no layer supplies the script.

    Explicit = exact (reproducible, never shadowed by gitignored ``personal/``);
    implicit = personal-or-shipped; ``_common`` is the shared-helper floor.
    """

    if rig is not None:
        order = [rig, _CLAUDE_FLOOR, _COMMON_DIR]
    else:
        order = [_PERSONAL_RIG, _CLAUDE_FLOOR, _COMMON_DIR]

    searched: list[Path] = []
    seen: set[str] = set()
    for sub in order:
        if sub in seen:
            continue
        seen.add(sub)
        candidate = MODELS_DIR / sub / name
        searched.append(candidate)
        if candidate.is_file():
            return candidate.resolve()
    listing = ", ".join(str(p) for p in searched)
    raise FileNotFoundError(
        f"no bundled models/ script named {name!r} "
        f"(rig={rig!r}); searched: {listing}"
    )

#: Opt-in escape hatch for a trusted repo whose config names its OWN scripts: set
#: ``GRINDSTONE_ALLOW_REPO_SCRIPTS=1`` to skip the under-``models/`` requirement.
ALLOW_REPO_SCRIPTS_ENV = "GRINDSTONE_ALLOW_REPO_SCRIPTS"

_FROZEN = ConfigDict(extra="forbid", frozen=True)


class RoleConfig(BaseModel):
    """One role behind a request script: ``rig`` OR ``script`` + ``slots`` + ``timeout_s``.

    A role is reached through a ``<role>_request.sh`` under ``models/``; name the
    backend two MUTUALLY EXCLUSIVE ways (``resolve_role_script`` does the mapping):

    * ``rig``: a bundled rig NAME (e.g. ``claude`` / ``codex`` / ``local``),
      resolved at run time to the role's ``<role>_request.sh`` under that rig (the
      portable form ``grindstone init`` scaffolds, not pinned to one checkout's
      absolute paths). NAMING a rig lets each role pick a different backend.
    * ``script``: an explicit absolute path to the role's request script (the
      power-user / legacy form). It must resolve under ``models/`` unless the
      operator opts out (``validate_script_paths``).

    Setting BOTH is an error (ambiguous). Setting NEITHER resolves the implicit
    default rig (``rig=None`` -> ``personal/`` then the shipped ``claude/`` floor).
    ``slots`` is the authoritative per-role concurrency bound (>= 1); ``timeout_s``
    is the transport-owned wall-clock supervisor (> 0). No model identity or
    transport, those live behind the script.
    """

    model_config = _FROZEN
    script: Path | None = None
    rig: str | None = None
    slots: int
    timeout_s: float

    @model_validator(mode="after")
    def _script_xor_rig(self) -> RoleConfig:
        if self.script is not None and self.rig is not None:
            raise ValueError(
                "set EITHER script or rig, not both (ambiguous): name a bundled "
                "rig OR an explicit script path, never both"
            )
        return self

    @field_validator("slots")
    @classmethod
    def _slots_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("slots must be >= 1")
        return v

    @field_validator("timeout_s")
    @classmethod
    def _timeout_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("timeout_s must be > 0")
        return v


class RolesConfig(BaseModel):
    """The role set: required ``planner`` + ``worker``, optional ``senior``.

    ``senior`` absent = a rig with no cloud tier → the escalation ladder is
    worker-only (the local rig grinds every tier).
    """

    model_config = _FROZEN
    planner: RoleConfig
    worker: RoleConfig
    senior: RoleConfig | None = None


def resolve_role_script(role: str, rc: RoleConfig) -> Path:
    """Map a role to its request script: the single role->filename mapping.

    ``role`` is one of ``"planner"`` / ``"worker"`` / ``"senior"``. An explicit
    ``rc.script`` wins verbatim (the power-user form); otherwise the role's
    ``<role>_request.sh`` is resolved through ``models_script`` under ``rc.rig``
    (a NAMED rig, or the implicit default rig when ``rc.rig`` is ``None``). This is
    the one place that knows a role's request-script filename, so the CLI never
    spells ``<role>_request.sh`` itself.
    """

    if rc.script is not None:
        return rc.script
    return models_script(f"{role}_request.sh", rig=rc.rig)


class VisionReviewConfig(BaseModel):
    """The B3 taste gate behind a request script: ``script`` + ``timeout_s``.

    The vision reviewer is a deterministic gate (one codex call per
    ``vision_review`` check), not a fan-out worker, so it carries no ``slots``,
    only the script path and the transport-owned wall-clock supervisor (> 0).
    Optional on the whole config: omit it and the CLI falls back to the bundled
    ``models/vision_review.sh``.
    """

    model_config = _FROZEN
    script: Path
    timeout_s: float

    @field_validator("timeout_s")
    @classmethod
    def _timeout_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("timeout_s must be > 0")
        return v


class FinalPolishConfig(BaseModel):
    """The B5 codex inline final-polish pass, OFF unless this block is present.

    After a run's ``complete_run`` evidence passes, codex may EDIT the finished
    repo inline (workspace-write) per ``criteria``; the edits are kept only if the
    SAME evidence still passes (``run_loop._final_polish``). ``script`` is optional
    (omit it and the CLI uses the bundled ``models/codex_polish.sh``); ``timeout_s``
    is the transport-owned wall-clock supervisor (> 0); ``screenshot`` is an
    optional worktree-relative image for a visual polish brief.
    """

    model_config = _FROZEN
    criteria: str
    timeout_s: float
    script: Path | None = None
    screenshot: str | None = None

    @field_validator("timeout_s")
    @classmethod
    def _timeout_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("timeout_s must be > 0")
        return v


class PrepareConfig(BaseModel):
    """Declared dependency materialization for build gates (optional).

    A phase ``done_when`` like ``npx tsc --noEmit`` runs inside a FRESH detached
    worktree of the integration tip that carries only COMMITTED files. The
    gitignored dependency dirs (``node_modules`` / ``.venv`` / ...) are absent
    there, so the check resolves the wrong stub and fails deterministically
    regardless of code correctness, every build gate is structurally unpassable.

    This block declares, per ecosystem and never hardcoded, how to restore those
    dirs into a worktree before checks run: ``cmd`` installs them, ``env_dirs``
    names the repo-relative gitignored dirs it produces (and that checks need),
    and ``cache_key_files`` names the repo-relative lockfiles whose content hash
    keys a snapshot cache (``.grindstone/cache/env/<hash>/``) so an unchanged
    lockfile reuses the install instead of re-running ``cmd``.

    ``None`` on the whole config (the default) = no materialization, existing
    runs are byte-unchanged.
    """

    model_config = _FROZEN
    cmd: str
    env_dirs: list[str]
    cache_key_files: list[str]

    @field_validator("env_dirs", "cache_key_files")
    @classmethod
    def _non_empty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("must list at least one path")
        return v


class FloorConfig(BaseModel):
    """The deterministic FLOOR: the repo's canonical verification commands.

    The first of three verification sources (the gate-rebalance brief): the
    commands the REPO and the CORE own, never the planner. Each ``check`` is a
    shell command re-run in the eval worktree AFTER ``prepare`` materializes deps,
    with the same pass/fail semantics as a ``done_when`` (exit 0 == pass); a
    failing floor check fails the gate exactly like a failed exit criterion, its
    captured output surfaced to the planner. The planner never authors or restates
    the floor, so a brittle planner-invented proxy (token greps, host-tool paths)
    can never gate a structurally-correct build.

    ``checks`` may be EMPTY: a fresh project starts with a minimal floor and grows
    it. ``None`` on the whole config (the default) = no repo floor commands, only
    the core invariants apply, existing runs are byte-unchanged.
    """

    model_config = _FROZEN
    checks: list[str]

    @field_validator("checks")
    @classmethod
    def _checks_non_empty_strings(cls, v: list[str]) -> list[str]:
        # An empty LIST is allowed (a minimal fresh-project floor); an empty
        # COMMAND string in the list is a config typo that would silently pass.
        if any(not c.strip() for c in v):
            raise ValueError("every floor check must be a non-empty command")
        return v


class InfraRepairConfig(BaseModel):
    """The automatic senior infra-repair policy (gate-rebalance G3).

    When a gate check fails for an ENVIRONMENTAL reason (``infra.classify_check_failure``:
    exit 127, a missing tool/dependency, an install failure), the core does NOT
    charge the worker or open a semantic failed epoch. It dispatches a SENIOR
    infra-repair task against a worktree of the failing gate's tip, told to make
    the gate environment satisfiable WITHOUT rewriting application logic, then
    re-runs the gate. ``attempts`` bounds the repair cycles per gate (>= 0; 0
    disables auto-repair, an infra fail then escalates immediately); on exhaustion
    the run escalates to a human naming the unsatisfiable command.

    ``allow_host_commands`` is the HOST-COMMAND GUARD: repo-local / in-worktree
    fixes (``npm install`` a dep landing in package.json, editing config inside the
    repo) are intended to be automatic, while host-level / privileged actions
    (``sudo``, ``apt``, system-wide installs, writes outside the repo) should be
    declined unless their leading token is listed here. The allowlist is CARRIED
    into the repair dispatch and surfaced in the repair prompt (repo-local-only +
    the exact allowlist). It is ADVISORY ONLY: the default senior adapter runs the
    repair agent with ``--dangerously-skip-permissions`` and full Bash, so this list
    is a prompt-level policy the senior is asked to honour, NOT a kernel/sandbox
    boundary it is prevented from crossing. A confused or adversarial senior CAN run
    a host-mutating command despite an empty allowlist. This is bounded by the repair
    running in a throwaway worktree and a false DONE being caught by the post-repair
    gate recheck, but it is not a hard sandbox: only enable ``infra_repair`` on a
    TRUSTED host and a repo you trust. Default EMPTY (nothing host-level allowed).

    ``None`` on the whole config (the default) = no auto-repair; an infra fail is
    surfaced but routes through the ordinary failed-epoch / gate path unchanged.
    """

    model_config = _FROZEN
    attempts: int = 2
    allow_host_commands: list[str] = []

    @field_validator("attempts")
    @classmethod
    def _attempts_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("infra_repair.attempts must be >= 0")
        return v

    @field_validator("allow_host_commands")
    @classmethod
    def _host_commands_non_empty(cls, v: list[str]) -> list[str]:
        if any(not c.strip() for c in v):
            raise ValueError("every allow_host_commands entry must be non-empty")
        return v


class GrindstoneConfig(BaseModel):
    """The whole per-repo config; unknown keys at any level are rejected."""

    model_config = _FROZEN
    roles: RolesConfig
    #: Per-run planner-call ceiling. ``None`` = the CLI's built-in default,
    #: NEVER unbounded (gate-5 P0: a stuck run burned 34 codex calls looping
    #: phase revisions while unattended).
    max_planner_calls: int | None = None
    #: DETERMINISTIC hard cap on consecutive FAILED epochs the planner may
    #: dispose of within ONE phase (each handle_failed_epoch retry/escalate that
    #: fails again counts) before the state machine FORCES a halt-to-human,
    #: regardless of what the planner decides. The backstop against the dogfood
    #: spin-loop (15 near-identical repair epochs against a structurally
    #: unpassable gate). A small int; default 3.
    max_failed_epochs_per_phase: int = 3
    #: DETERMINISTIC tier-aware ceilings on how many ``file_ownership`` globs a
    #: single FRESH implement task may declare (the planner-decomposition size
    #: gate, ``semantics.implement_task_size_violations``). A task over its tier's
    #: bound is rejected back to the planner with the offending task named,
    #: reusing the invalid-decision re-ask path, so an undecomposed "do the whole
    #: app in one task" epoch can never be dispatched. ``local`` is the default
    #: (local-rig) bound; ``senior`` (visual/taste epochs that start on senior)
    #: gets a larger one. Both >= 1.
    local_max_task_files: int = 5
    senior_max_task_files: int = 12
    #: The B3 vision-review script seam. ``None`` = the CLI's bundled default
    #: (``models/vision_review.sh``); set it to point the taste gate at a
    #: different rig's script (or a stub in tests).
    vision_review: VisionReviewConfig | None = None
    #: The B5 final-polish pass. ``None`` (absent block) = OFF, codex never
    #: touches a completed run. Present = the gated inline-edit pass runs after
    #: ``complete_run`` evidence passes (kept only if it re-passes).
    final_polish: FinalPolishConfig | None = None
    #: Declared dependency materialization. ``None`` (absent block) = OFF, eval
    #: + worker worktrees carry only committed files (the prior behavior).
    #: Present = the gitignored dependency dirs are restored (cached) before
    #: checks/workers run, so build gates are not structurally unpassable.
    prepare: PrepareConfig | None = None
    #: The deterministic FLOOR: repo-owned canonical verification commands run
    #: every gate AFTER ``prepare``, never authored by the planner. ``None``
    #: (absent block) = OFF, only the core invariants apply (existing runs are
    #: byte-unchanged). Present = each ``floor.checks`` command is re-run in the
    #: eval worktree as a deterministic check (exit 0 == pass).
    floor: FloorConfig | None = None
    #: The automatic senior infra-repair policy (G3). ``None`` (absent block) =
    #: OFF, an infra-classified gate failure routes through the ordinary path
    #: unchanged. Present = an infra fail auto-dispatches a bounded, host-guarded
    #: senior repair that makes the gate satisfiable, then re-runs the gate.
    infra_repair: InfraRepairConfig | None = None
    #: The end-of-epoch agentic verification pass (G4). Default ON: whenever an epoch
    #: carries natural-language ``criteria`` AND a local tier exists, the core runs
    #: one adversarial verification pass over those criteria after the deterministic
    #: floor clears; an unmet criterion fails the epoch through the failed-epoch
    #: machinery. The pass NEVER runs (and never errors) when an epoch has no criteria
    #: or there is no local tier. Set to ``false`` to disable the pass entirely (the
    #: deterministic floor + planner review epochs still gate). A bool, default True.
    verify_epochs: bool = True

    @field_validator("max_failed_epochs_per_phase")
    @classmethod
    def _failed_cap_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("max_failed_epochs_per_phase must be >= 1")
        return v

    @field_validator("local_max_task_files", "senior_max_task_files")
    @classmethod
    def _task_file_bound_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("max task-file bounds must be >= 1")
        return v


def load_config(repo_root: Path) -> GrindstoneConfig | None:
    """Load + validate ``<repo_root>/.grindstone/config.yaml``, or ``None``.

    ``None`` when the file is absent (CLI uses built-in defaults). A malformed
    document, an unknown key, or a missing required field raises ``ValueError``
    (Pydantic's ``ValidationError`` is a ``ValueError``), never a silent fallback.
    """

    path = Path(repo_root) / _CONFIG_REL
    if not path.is_file():
        return None
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a YAML mapping, got {type(raw).__name__}")
    try:
        return GrindstoneConfig.model_validate(raw)
    except ValidationError as exc:
        raise ValueError(f"{path}: invalid config: {exc}") from exc


def _script_under_models(script: Path) -> bool:
    """Does ``script`` resolve INSIDE the bundled ``models/`` dir? (symlink-safe)."""

    return script.resolve().is_relative_to(MODELS_DIR.resolve())


def validate_script_paths(cfg: GrindstoneConfig) -> None:
    """Reject any configured ``script:`` that is not a bundled ``models/`` script.

    The target repo's ``.grindstone/config.yaml`` is attacker-controlled (a cloned
    repo carries its own), and EVERY configured script, ``roles.*`` /
    ``vision_review`` / ``final_polish``, is ``Popen``'d (final_polish even WRITES
    the worktree). An unconstrained path is arbitrary-code execution, so a script
    must resolve under the rig's bundled ``models/`` dir. The
    ``GRINDSTONE_ALLOW_REPO_SCRIPTS=1`` escape hatch opts a TRUSTED repo back in.
    Raises ``ValueError`` listing every offending path otherwise.
    """

    if os.environ.get(ALLOW_REPO_SCRIPTS_ENV) == "1":
        return
    # Only roles that EXPLICITLY name a ``script:`` are guarded: a rig-derived
    # script (``rig:`` / neither) always resolves under ``models/`` via
    # ``models_script``, so it is inherently safe and has no path to check.
    candidates: list[tuple[str, Path]] = []
    if cfg.roles.planner.script is not None:
        candidates.append(("roles.planner.script", cfg.roles.planner.script))
    if cfg.roles.worker.script is not None:
        candidates.append(("roles.worker.script", cfg.roles.worker.script))
    if cfg.roles.senior is not None and cfg.roles.senior.script is not None:
        candidates.append(("roles.senior.script", cfg.roles.senior.script))
    if cfg.vision_review is not None:
        candidates.append(("vision_review.script", cfg.vision_review.script))
    if cfg.final_polish is not None and cfg.final_polish.script is not None:
        candidates.append(("final_polish.script", cfg.final_polish.script))
    bad = [(name, p) for name, p in candidates if not _script_under_models(p)]
    if bad:
        listing = "; ".join(f"{name}={p}" for name, p in bad)
        raise ValueError(
            f"configured script(s) outside the bundled models/ dir ({MODELS_DIR}): "
            f"{listing}. Every configured script is executed, point them at the "
            f"rig's models/ scripts, or set {ALLOW_REPO_SCRIPTS_ENV}=1 to opt in to "
            f"a trusted repo's own scripts."
        )
