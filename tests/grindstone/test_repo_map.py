"""The target-repo-owned REPO MAP overlay loader.

Mirrors the strategy overlay's safety contract: never raises, graceful-absent (``""``),
size-bounded (it rides EVERY planner and worker boundary), fixed filename (no traversal
surface). Single file, no index, no selection.
"""

from __future__ import annotations

from pathlib import Path

from grindstone.repo_map import MAX_BYTES, REPOMAP_REL, load_repo_map


def _write_repo_map(repo: Path, text: str) -> Path:
    path = repo / REPOMAP_REL
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_load_repo_map_none_repo_is_empty() -> None:
    assert load_repo_map(None) == ""


def test_load_repo_map_absent_file_is_empty(tmp_path: Path) -> None:
    # repo exists but ships no repomap.md (the common case) -> graceful no-op.
    assert load_repo_map(tmp_path) == ""


def test_load_repo_map_reads_present_file_stripped(tmp_path: Path) -> None:
    _write_repo_map(tmp_path, "\n  src/ holds the package; tests/ mirrors it\n")
    assert load_repo_map(tmp_path) == "src/ holds the package; tests/ mirrors it"


def test_load_repo_map_is_size_bounded(tmp_path: Path) -> None:
    # a runaway file cannot blow the context: capped at MAX_BYTES.
    _write_repo_map(tmp_path, "x" * (MAX_BYTES * 2))
    out = load_repo_map(tmp_path)
    assert len(out.encode("utf-8")) <= MAX_BYTES


def test_load_repo_map_never_raises_on_a_directory(tmp_path: Path) -> None:
    # repomap.md present but it is a DIR (unreadable as a file) -> "" not a crash.
    target = tmp_path / REPOMAP_REL
    target.mkdir(parents=True)
    assert load_repo_map(tmp_path) == ""
