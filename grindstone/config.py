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


class GrindstoneConfig(BaseModel):
    """The whole per-repo config; unknown keys at any level are rejected.

    The bones config knows only ROLES (the rig wiring) and the per-run epoch
    backstop. Phases, floors, vision, infra-repair, polish, dependency
    materialization and the size/verify machinery are gone (BONES): host mutations
    are a planner-declared ``setup`` seam in the decision (the orchestrator runs
    them before the tasks), and per-epoch acceptance is the agentic critic, not a
    deterministic gate.
    """

    model_config = _FROZEN
    roles: RolesConfig
    #: Per-run EPOCH backstop: the max number of planner boundaries (epochs) a run
    #: may take before the state machine FORCES a clean partial-end (BONES failure
    #: model #2, involuntary trigger). ``None`` = the CLI's built-in default, NEVER
    #: unbounded (a stuck planner that spins without progress must be bounded).
    max_epochs: int | None = None

    @field_validator("max_epochs")
    @classmethod
    def _max_epochs_positive(cls, v: int | None) -> int | None:
        if v is not None and v < 1:
            raise ValueError("max_epochs must be >= 1")
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
    repo carries its own), and every configured ``roles.*`` script is ``Popen``'d.
    An unconstrained path is arbitrary-code execution, so a script must resolve
    under the rig's bundled ``models/`` dir. The ``GRINDSTONE_ALLOW_REPO_SCRIPTS=1``
    escape hatch opts a TRUSTED repo back in. Raises ``ValueError`` listing every
    offending path otherwise.
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
    bad = [(name, p) for name, p in candidates if not _script_under_models(p)]
    if bad:
        listing = "; ".join(f"{name}={p}" for name, p in bad)
        raise ValueError(
            f"configured script(s) outside the bundled models/ dir ({MODELS_DIR}): "
            f"{listing}. Every configured script is executed, point them at the "
            f"rig's models/ scripts, or set {ALLOW_REPO_SCRIPTS_ENV}=1 to opt in to "
            f"a trusted repo's own scripts."
        )
