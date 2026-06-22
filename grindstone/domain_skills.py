"""Target-repo-owned DOMAIN skills: a catalogue the TARGET repo carries, the
planner SELECTS per task by judgment, and the core DELIVERS into the worker prompt.

Distinct from the grindstone-OWNED operating skills (``skills/operating/``,
selected deterministically by run state, see ``config.load_operating_skill``):
domain skills live in the TARGET repo, are chosen by the planner's JUDGMENT (a
task's ``skills`` field), and are delivered RETRIEVE-not-concatenate, the worker
prompt carries only the SELECTED skills, never the whole catalogue. An absent
catalogue (no ``.grindstone/skills/`` dir, or no ``index.md``) is a graceful no-op
everywhere: no skills block, no validation restriction.

Layout (target-repo-owned, gitignored-or-committed by the repo, never by grindstone):

    <repo>/.grindstone/skills/<name>.md   one domain skill's content
    <repo>/.grindstone/skills/index.md    the SELECTION INDEX the planner reads

Index format: a Markdown list, ONE ``- <name>: <description>`` line per skill.
``<name>`` is the bare skill name (the matching ``<name>.md`` stem) and must match
``_NAME_RE`` (alphanumeric plus ``._-``, up to 64 chars, mirroring the schema's
``skills`` item cap); ``<description>`` is a one-line "what it is + when to use it".
Lines that do not match the pattern are ignored, so the index may carry a heading
or surrounding prose. A duplicate name keeps the LAST line's description.

Path safety mirrors ``config._script_under_models``: a skill name is resolved
STRICTLY under ``<repo>/.grindstone/skills/`` (pattern-validated, then
``is_relative_to`` checked), so a planner-supplied name can never traverse out of
the catalogue to read an arbitrary on-disk file.
"""

from __future__ import annotations

import re
from pathlib import Path

#: Repo-relative root of the domain-skill catalogue.
DOMAIN_SKILLS_REL = Path(".grindstone") / "skills"

#: The selection index filename inside the catalogue dir.
INDEX_FILENAME = "index.md"

#: A legal domain-skill name: alphanumeric start, then alphanumeric / ``._-``, up
#: to 64 chars total (mirrors the schema's ``skills`` item ``maxLength`` 64). No
#: slash, no ``..``, so a name can never name a subdir or traverse the catalogue.
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")

#: One index line: ``- name: description`` (or ``* name: description``). The name
#: group is pattern-checked separately so a malformed line is simply skipped.
_INDEX_LINE_RE = re.compile(r"^\s*[-*]\s+([^:]+?)\s*:\s*(.+?)\s*$")


def load_domain_skill_index(repo: Path) -> dict[str, str]:
    """Parse ``<repo>/.grindstone/skills/index.md`` into ``{name: description}``.

    Returns an EMPTY dict when the catalogue is absent (no dir / no index file):
    the feature is then a graceful no-op everywhere. Each parsed ``- name: desc``
    line whose name matches ``_NAME_RE`` contributes one entry (a duplicate name
    keeps the last description); non-matching lines are ignored so the file may
    carry a heading or prose. Never raises on a malformed index, an unreadable or
    non-list file simply yields no entries.
    """

    index_path = Path(repo) / DOMAIN_SKILLS_REL / INDEX_FILENAME
    if not index_path.is_file():
        return {}
    try:
        text = index_path.read_text(encoding="utf-8")
    except OSError:
        return {}
    out: dict[str, str] = {}
    for line in text.splitlines():
        match = _INDEX_LINE_RE.match(line)
        if match is None:
            continue
        name, description = match.group(1), match.group(2)
        if _NAME_RE.match(name):
            out[name] = description
    return out


def load_domain_skill(repo: Path, name: str) -> str:
    """Read one domain skill's ``<name>.md`` text from the target repo's catalogue.

    The name is resolved STRICTLY under ``<repo>/.grindstone/skills/``: it must
    match ``_NAME_RE`` (so it carries no slash / ``..`` / absolute prefix) and the
    resolved path must stay inside the catalogue dir (defense in depth), mirroring
    ``config._script_under_models``. Raises ``ValueError`` for an illegal/escaping
    name and ``FileNotFoundError`` (naming the resolved path) when the named skill
    has no file, a planner that selected a skill the repo does not ship is a loud
    failure, never a silent empty block.
    """

    if not _NAME_RE.match(name):
        raise ValueError(
            f"illegal domain skill name {name!r}: a name is alphanumeric plus "
            f"'._-' (<= 64 chars), with no path separator or '..' segment"
        )
    base = (Path(repo) / DOMAIN_SKILLS_REL).resolve()
    candidate = (base / f"{name}.md").resolve()
    if not candidate.is_relative_to(base):
        raise ValueError(f"domain skill name {name!r} escapes the skills dir")
    try:
        return candidate.read_text(encoding="utf-8")
    except OSError as exc:
        raise FileNotFoundError(
            f"no domain skill {name!r} in the target repo at {candidate}"
        ) from exc
