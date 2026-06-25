"""The target-repo-owned PLANNER strategy overlay loader.

Mirrors the domain-skill loader's safety contract: never raises, graceful-absent
(``""``), size-bounded (it rides EVERY planner call), fixed filename (no traversal
surface). Single file, no index, no selection.
"""

from __future__ import annotations

from pathlib import Path

from grindstone.strategy_skill import MAX_BYTES, STRATEGY_REL, load_strategy


def _write_strategy(repo: Path, text: str) -> Path:
    path = repo / STRATEGY_REL
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_load_strategy_none_repo_is_empty() -> None:
    assert load_strategy(None) == ""


def test_load_strategy_absent_file_is_empty(tmp_path: Path) -> None:
    # repo exists but ships no strategy.md (the common case) -> graceful no-op.
    assert load_strategy(tmp_path) == ""


def test_load_strategy_reads_present_file_stripped(tmp_path: Path) -> None:
    _write_strategy(tmp_path, "\n  prefer per-screen decomposition; build tokens first\n")
    assert load_strategy(tmp_path) == "prefer per-screen decomposition; build tokens first"


def test_load_strategy_is_size_bounded(tmp_path: Path) -> None:
    # a runaway file cannot blow the planner context: capped at MAX_BYTES.
    _write_strategy(tmp_path, "x" * (MAX_BYTES * 2))
    out = load_strategy(tmp_path)
    assert len(out.encode("utf-8")) <= MAX_BYTES


def test_load_strategy_never_raises_on_a_directory(tmp_path: Path) -> None:
    # strategy.md present but it is a DIR (unreadable as a file) -> "" not a crash.
    target = tmp_path / STRATEGY_REL
    target.mkdir(parents=True)
    assert load_strategy(tmp_path) == ""
