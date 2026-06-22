"""Hermetic unit tests for ``_select_decision_text`` (NO model call, NOT eval-marked).

These pin the eval boundary's decision-source priority against SYNTHETIC file /
stdout combos, so the chain stays a faithful mirror of
``ScriptPlanner._dispatch`` without needing a live planner call. Runs in the
default suite.
"""

from __future__ import annotations

from pathlib import Path

from tests.grindstone.eval._boundary import _select_decision_text


def test_decision_json_wins_when_present_and_nonempty(tmp_path: Path) -> None:
    decision = tmp_path / "decision.json"
    decision.write_text('{"epoch": 1}', encoding="utf-8")
    out = tmp_path / "out.txt"
    out.write_text("from out", encoding="utf-8")

    text, source = _select_decision_text(decision, out, "from stdout")

    assert source == "decision_json"
    assert text == '{"epoch": 1}'


def test_falls_back_to_out_when_decision_missing(tmp_path: Path) -> None:
    decision = tmp_path / "decision.json"  # never created
    out = tmp_path / "out.txt"
    out.write_text("from out", encoding="utf-8")

    text, source = _select_decision_text(decision, out, "from stdout")

    assert source == "out_file"
    assert text == "from out"


def test_falls_back_to_out_when_decision_blank(tmp_path: Path) -> None:
    decision = tmp_path / "decision.json"
    decision.write_text("   \n\t ", encoding="utf-8")  # whitespace only
    out = tmp_path / "out.txt"
    out.write_text("from out", encoding="utf-8")

    text, source = _select_decision_text(decision, out, "from stdout")

    assert source == "out_file"
    assert text == "from out"


def test_falls_back_to_stdout_when_out_missing(tmp_path: Path) -> None:
    decision = tmp_path / "decision.json"  # never created
    out = tmp_path / "out.txt"  # never created

    text, source = _select_decision_text(decision, out, "from stdout")

    assert source == "stdout"
    assert text == "from stdout"


def test_falls_back_to_stdout_when_out_empty(tmp_path: Path) -> None:
    decision = tmp_path / "decision.json"  # never created
    out = tmp_path / "out.txt"
    out.write_text("", encoding="utf-8")  # empty file

    text, source = _select_decision_text(decision, out, "from stdout")

    assert source == "stdout"
    assert text == "from stdout"


def test_stdout_returned_even_when_empty(tmp_path: Path) -> None:
    decision = tmp_path / "decision.json"  # never created
    out = tmp_path / "out.txt"  # never created

    text, source = _select_decision_text(decision, out, "")

    assert source == "stdout"
    assert text == ""
