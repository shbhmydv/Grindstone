"""The worker-facing handoff validator: ``grindstone.check_handoff`` generates a
self-contained, stdlib-only script a worker drops in its CWD and loops on until
clean. These tests pin the generated script's verdict to the schema (the same
corpus the typed-model equivalence test uses) and cover the context checks the
schema cannot express: exact task_id, the mode citation rule, citation files on
disk, and the byte cap.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from grindstone.check_handoff import generate_check_script
from grindstone.contracts import handoff_schema_errors
from grindstone.contracts.semantics import HANDOFF_MAX_BYTES

CORPUS = Path(__file__).parent / "corpus" / "handoff"


def _run_script(
    tmp_path: Path,
    *,
    task_id: str,
    mode: str,
    handoff: object,
    files: dict[str, str] | None = None,
    repo_root: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    """Materialize the generated script + a handoff (+ cited files) and run it."""

    (tmp_path / "check_handoff.py").write_text(
        generate_check_script(task_id=task_id, mode=mode, repo_root=repo_root),  # type: ignore[arg-type]
        encoding="utf-8",
    )
    if isinstance(handoff, str):
        (tmp_path / "handoff.json").write_text(handoff, encoding="utf-8")
    else:
        (tmp_path / "handoff.json").write_text(json.dumps(handoff), encoding="utf-8")
    for rel, content in (files or {}).items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return subprocess.run(
        [sys.executable, "check_handoff.py"],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
    )


def _cited_files(payload: object) -> dict[str, str]:
    """Create on disk every file the corpus handoff cites (file existence only)."""

    out: dict[str, str] = {}
    if isinstance(payload, dict):
        for cite in payload.get("citations", []):
            if isinstance(cite, dict) and isinstance(cite.get("file"), str):
                out[cite["file"]] = "x\n"
    return out


def _corpus_cases() -> list[object]:
    out = []
    for validity in ("valid", "invalid"):
        for path in sorted((CORPUS / validity).glob("*.json")):
            out.append(pytest.param(path, validity == "valid", id=f"{validity}/{path.stem}"))
    return out


@pytest.mark.parametrize("path,expect_valid", _corpus_cases())
def test_generated_script_agrees_with_schema(
    tmp_path: Path, path: Path, expect_valid: bool
) -> None:
    """The script's accept/reject must match the schema gate across the corpus.

    Parametrized in ``implement`` mode (no citation requirement) with cited
    files pre-created so the only differences left are structural — exactly what
    the script mirrors. Valid fixtures are baked with their own task_id; invalid
    ones with a fixed dispatched id, so ``bad_task_id`` is rejected precisely
    because its id is not the one dispatched.
    """

    payload = json.loads(path.read_text(encoding="utf-8"))
    schema_valid = not handoff_schema_errors(payload)
    assert schema_valid == expect_valid  # corpus sanity, mirrors equivalence test
    expected = payload["task_id"] if expect_valid else "P1/E1/T1"
    proc = _run_script(
        tmp_path,
        task_id=expected,
        mode="implement",
        handoff=payload,
        files=_cited_files(payload),
    )
    assert (proc.returncode == 0) == expect_valid, proc.stdout


def _ok_handoff(task_id: str = "P1/E1/T1", **over: object) -> dict[str, object]:
    base: dict[str, object] = {
        "schema_version": "1",
        "task_id": task_id,
        "status": "DONE",
        "resulting_state": "ok",
        "checks": [{"check": "true", "exit_code": 0}],
        "occupancy": {"compacted": False, "subagent_splits": 0},
    }
    base.update(over)
    return base


def test_research_script_accepts_repo_root_citation(tmp_path: Path) -> None:
    """Gate-5 P0 mirror: research/review/artifact workers cite TARGET-REPO
    files, so the worker-side validator must resolve citations against the
    baked repo root too — or workers self-validate differently than the core
    gate and burn attempts on a disagreement they cannot see."""

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "src.py").write_text("x = 1\n", encoding="utf-8")
    cwd = tmp_path / "scratch"
    cwd.mkdir()
    handoff = _ok_handoff(citations=[{"file": "src.py", "line": 1}])
    proc = _run_script(
        cwd, task_id="P1/E1/T1", mode="research", handoff=handoff, repo_root=repo
    )
    assert proc.returncode == 0, proc.stdout


def test_script_rejects_citation_outside_all_roots(tmp_path: Path) -> None:
    """Containment mirror: an EXISTING file outside both the CWD and the repo
    root is a violation. Previously ``(cwd / "/abs/path")`` discarded the base
    (pathlib semantics), so ANY absolute path that existed on disk passed."""

    handoff = _ok_handoff(citations=[{"file": "/etc/hostname"}])
    proc = _run_script(tmp_path, task_id="P1/E1/T1", mode="implement", handoff=handoff)
    assert proc.returncode == 1
    assert "citation" in proc.stdout


def test_missing_file_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "check_handoff.py").write_text(
        generate_check_script(task_id="P1/E1/T1", mode="implement"), encoding="utf-8"
    )
    proc = subprocess.run(
        [sys.executable, "check_handoff.py"], cwd=str(tmp_path), capture_output=True, text=True
    )
    assert proc.returncode == 1
    assert "handoff.json" in proc.stdout


def test_clean_handoff_exits_zero(tmp_path: Path) -> None:
    proc = _run_script(tmp_path, task_id="P1/E1/T1", mode="implement", handoff=_ok_handoff())
    assert proc.returncode == 0, proc.stdout


def test_task_id_mismatch_is_rejected(tmp_path: Path) -> None:
    proc = _run_script(
        tmp_path, task_id="P1/E1/T1", mode="implement", handoff=_ok_handoff("P1/E1/T2")
    )
    assert proc.returncode == 1
    assert "task_id" in proc.stdout


def test_non_object_json_is_rejected(tmp_path: Path) -> None:
    proc = _run_script(tmp_path, task_id="P1/E1/T1", mode="implement", handoff="[1, 2, 3]")
    assert proc.returncode == 1


@pytest.mark.parametrize("mode", ["research", "review"])
def test_citation_required_mode_rejects_when_empty(tmp_path: Path, mode: str) -> None:
    proc = _run_script(tmp_path, task_id="P1/E1/T1", mode=mode, handoff=_ok_handoff())
    assert proc.returncode == 1
    assert "citation" in proc.stdout


@pytest.mark.parametrize("mode", ["research", "review"])
def test_citation_required_mode_accepts_with_existing_citation(
    tmp_path: Path, mode: str
) -> None:
    handoff = _ok_handoff(citations=[{"file": "notes.md"}])
    proc = _run_script(
        tmp_path, task_id="P1/E1/T1", mode=mode, handoff=handoff, files={"notes.md": "x"}
    )
    assert proc.returncode == 0, proc.stdout


def test_citation_file_missing_on_disk_is_rejected(tmp_path: Path) -> None:
    handoff = _ok_handoff(citations=[{"file": "ghost.py"}])
    proc = _run_script(tmp_path, task_id="P1/E1/T1", mode="implement", handoff=handoff)
    assert proc.returncode == 1
    assert "ghost.py" in proc.stdout


def test_byte_cap_is_enforced(tmp_path: Path) -> None:
    handoff = _ok_handoff(resulting_state="x" * (HANDOFF_MAX_BYTES + 100))
    proc = _run_script(tmp_path, task_id="P1/E1/T1", mode="implement", handoff=handoff)
    assert proc.returncode == 1
    assert str(HANDOFF_MAX_BYTES) in proc.stdout


def test_generated_script_is_stdlib_only() -> None:
    """No grindstone / jsonschema import — target repos have neither."""

    script = generate_check_script(task_id="P1/E1/T1", mode="implement")
    assert "import grindstone" not in script
    assert "from grindstone" not in script
    assert "jsonschema" not in script
    assert str(HANDOFF_MAX_BYTES) in script  # the cap is baked, not imported
