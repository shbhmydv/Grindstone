"""S5 ruling 1: the repo-memory READ seam (``load_digest``).

The write-back hook is deliberately absent (ARCHITECTURE.md — organ built after 3-5
real runs). These tests pin the read path only: a missing digest is ``None``
(never an error), a present digest is its text, and an over-cap digest is
truncated with a visible marker rather than failing the run.
"""

from __future__ import annotations

from pathlib import Path

from grindstone.memory import DIGEST_MAX_BYTES, TRUNCATION_MARKER, load_digest


def test_missing_digest_is_none(tmp_path: Path) -> None:
    assert load_digest(tmp_path) is None  # no .grindstone/memory/digest.md


def test_present_digest_is_read(tmp_path: Path) -> None:
    digest = tmp_path / ".grindstone" / "memory" / "digest.md"
    digest.parent.mkdir(parents=True)
    digest.write_text("remember: prefer rg over grep\n", encoding="utf-8")
    assert load_digest(tmp_path) == "remember: prefer rg over grep\n"


def test_empty_digest_file_is_empty_string(tmp_path: Path) -> None:
    digest = tmp_path / ".grindstone" / "memory" / "digest.md"
    digest.parent.mkdir(parents=True)
    digest.write_text("", encoding="utf-8")
    assert load_digest(tmp_path) == ""


def test_over_cap_digest_truncates_with_marker(tmp_path: Path) -> None:
    digest = tmp_path / ".grindstone" / "memory" / "digest.md"
    digest.parent.mkdir(parents=True)
    body = "x" * (DIGEST_MAX_BYTES + 5000)
    digest.write_text(body, encoding="utf-8")
    out = load_digest(tmp_path)
    assert out is not None
    assert out.endswith(TRUNCATION_MARKER)
    assert len(out.encode("utf-8")) <= DIGEST_MAX_BYTES
    # The kept prefix is the head of the original content.
    assert out.startswith("x" * 100)


def test_truncation_never_splits_a_utf8_codepoint(tmp_path: Path) -> None:
    digest = tmp_path / ".grindstone" / "memory" / "digest.md"
    digest.parent.mkdir(parents=True)
    # Multi-byte codepoints straddling the cap must not raise on decode.
    digest.write_text("é" * DIGEST_MAX_BYTES, encoding="utf-8")
    out = load_digest(tmp_path)  # must not raise
    assert out is not None and out.endswith(TRUNCATION_MARKER)
