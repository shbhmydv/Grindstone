"""Target-repo domain-skill catalogue: index parse, skill load, path safety.

The catalogue lives in the TARGET repo under ``.grindstone/skills/``: ``index.md``
is the selection index the planner reads (``- name: description`` lines), and each
``<name>.md`` is one skill's text the worker receives. Absent catalogue is a
graceful no-op (empty index, no restriction). Names resolve STRICTLY under the
catalogue dir, no traversal.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from grindstone.domain_skills import (
    DOMAIN_SKILLS_REL,
    load_domain_skill,
    load_domain_skill_index,
)


def _write_catalogue(repo: Path, *, index: str, skills: dict[str, str]) -> None:
    skills_dir = repo / DOMAIN_SKILLS_REL
    skills_dir.mkdir(parents=True, exist_ok=True)
    (skills_dir / "index.md").write_text(index, encoding="utf-8")
    for name, body in skills.items():
        (skills_dir / f"{name}.md").write_text(body, encoding="utf-8")


# --- index parse ---------------------------------------------------------------


def test_index_absent_returns_empty(tmp_path: Path) -> None:
    # No .grindstone/skills/ at all: the feature is a no-op (empty dict).
    assert load_domain_skill_index(tmp_path) == {}


def test_index_dir_without_index_file_returns_empty(tmp_path: Path) -> None:
    (tmp_path / DOMAIN_SKILLS_REL).mkdir(parents=True)
    assert load_domain_skill_index(tmp_path) == {}


def test_index_parses_name_description_lines(tmp_path: Path) -> None:
    index = (
        "# Skills\n"
        "\n"
        "- rn-nav: React Navigation patterns; use when wiring screens/stacks.\n"
        "- rn-a11y: accessibility; use for screen-reader / contrast work.\n"
        "* rn-design: design tokens (asterisk bullet also accepted).\n"
        "some prose that is not a list item and must be ignored\n"
    )
    _write_catalogue(tmp_path, index=index, skills={})
    parsed = load_domain_skill_index(tmp_path)
    assert parsed == {
        "rn-nav": "React Navigation patterns; use when wiring screens/stacks.",
        "rn-a11y": "accessibility; use for screen-reader / contrast work.",
        "rn-design": "design tokens (asterisk bullet also accepted).",
    }


def test_index_ignores_illegal_names_and_keeps_last_duplicate(tmp_path: Path) -> None:
    index = (
        "- ../escape: traversal name, skipped\n"
        "- has space: skipped (space not allowed)\n"
        "- good: first description\n"
        "- good: second description wins\n"
    )
    _write_catalogue(tmp_path, index=index, skills={})
    assert load_domain_skill_index(tmp_path) == {"good": "second description wins"}


# --- skill load + path safety --------------------------------------------------


def test_load_skill_returns_text(tmp_path: Path) -> None:
    _write_catalogue(
        tmp_path, index="- rn-nav: navigation\n", skills={"rn-nav": "NAV SKILL BODY"}
    )
    assert load_domain_skill(tmp_path, "rn-nav") == "NAV SKILL BODY"


def test_load_missing_skill_raises_filenotfound(tmp_path: Path) -> None:
    _write_catalogue(tmp_path, index="- rn-nav: navigation\n", skills={})
    with pytest.raises(FileNotFoundError):
        load_domain_skill(tmp_path, "rn-nav")


@pytest.mark.parametrize(
    "bad",
    ["../secret", "foo/bar", "/etc/passwd", "..", "a/../../b", "name with space"],
)
def test_load_rejects_traversal_and_illegal_names(tmp_path: Path, bad: str) -> None:
    (tmp_path / DOMAIN_SKILLS_REL).mkdir(parents=True)
    with pytest.raises(ValueError):
        load_domain_skill(tmp_path, bad)


def test_load_does_not_escape_to_sibling_file(tmp_path: Path) -> None:
    # A real file outside the catalogue must be unreachable: only <name>.md under
    # the catalogue dir resolves, and only for a pattern-legal bare name.
    (tmp_path / DOMAIN_SKILLS_REL).mkdir(parents=True)
    (tmp_path / "secret.md").write_text("TOP SECRET", encoding="utf-8")
    with pytest.raises((ValueError, FileNotFoundError)):
        load_domain_skill(tmp_path, "../secret")
