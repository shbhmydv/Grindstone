"""Repo-map: PageRank spine ranking, size gate, never-crash degradation, the
``focus_files`` subtree, and the planner/worker injection seams.

The map is an enhancement that must never crash a run and must never touch the
byte-stable planner head, both are asserted here. A multi-language fixture repo
(Python + TypeScript) above the file-count threshold exercises the real
tree-sitter path; tiny repos and broken files exercise the degrade paths.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from grindstone.contracts.models import ArtifactTask, CmdCheck, ImplementTask
from grindstone.planner import build_planner_input, stable_head
from grindstone.repomap import (
    MIN_FILES_FOR_MAP,
    build_repo_map,
    repo_file_count,
)
from grindstone.worker import WorkerRequest, build_worker_prompt


def _grammars_loadable() -> bool:
    """Whether tree-sitter can actually load a grammar and run a query HERE.

    A fresh install with an ABI/version mismatch between ``tree-sitter`` and
    ``tree-sitter-language-pack`` cannot parse, so the repo-map correctly degrades
    to ``None``. The real-parse assertions below would then be red for an
    environment reason, not a code defect, so they skip instead of failing. The
    pyproject version floors are set to avoid this in practice; this is the net.
    """

    try:
        from tree_sitter import Parser, Query, QueryCursor
        from tree_sitter_language_pack import get_language

        lang = get_language("python")
        tree = Parser(lang).parse(b"def f():\n    return 1\n")
        QueryCursor(Query(lang, "(function_definition) @f")).captures(tree.root_node)
        return True
    except Exception:  # noqa: BLE001 - any failure means grammars are unusable here
        return False


#: Skips the tests that need a real parse when grammars cannot load in this env.
_needs_grammars = pytest.mark.skipif(
    not _grammars_loadable(),
    reason="tree-sitter grammars not loadable here (ABI/version mismatch); the "
    "repo-map degrades to None, so its real-parse assertions cannot run",
)


def _spine_repo(root: Path, *, modules: int = 60) -> None:
    """A repo whose ``util.shared_helper`` is referenced by every module, so it
    is unambiguously the spine; plus a TypeScript file and a second helper."""

    (root / "util.py").write_text(
        "def shared_helper():\n    return 1\n\n\ndef rare_helper():\n    return 2\n",
        encoding="utf-8",
    )
    for i in range(modules):
        (root / f"mod_{i:02d}.py").write_text(
            f"from util import shared_helper\n\n\ndef fn_{i}():\n"
            f"    return shared_helper()\n",
            encoding="utf-8",
        )
    (root / "app.ts").write_text(
        "function tsEntry() { return tsHelper(); }\n"
        "function tsHelper() { return tsEntry(); }\n",
        encoding="utf-8",
    )


def _two_cluster_repo(root: Path) -> None:
    """Two disjoint clusters: cluster A references ``coreA``, B references
    ``coreB``. Used to prove ``focus_files`` collapses the map to a neighborhood."""

    (root / "coreA.py").write_text("def helperA():\n    return 1\n", encoding="utf-8")
    (root / "coreB.py").write_text("def helperB():\n    return 1\n", encoding="utf-8")
    for i in range(28):
        (root / f"a_{i:02d}.py").write_text(
            f"from coreA import helperA\n\n\ndef a{i}():\n    return helperA()\n",
            encoding="utf-8",
        )
        (root / f"b_{i:02d}.py").write_text(
            f"from coreB import helperB\n\n\ndef b{i}():\n    return helperB()\n",
            encoding="utf-8",
        )


def _tokens(text: str) -> int:
    import tiktoken

    return len(tiktoken.get_encoding("cl100k_base").encode(text))


# --- core behaviour ------------------------------------------------------------


@_needs_grammars
def test_spine_files_rank_first_and_within_budget(tmp_path: Path) -> None:
    _spine_repo(tmp_path)
    text = build_repo_map(tmp_path, map_tokens=2000)
    assert text is not None and text.strip()
    # The most-referenced symbol/file is the spine and surfaces first.
    assert "shared_helper" in text
    assert text.splitlines()[0].startswith("util.py")
    # TypeScript is parsed via the JS query alias (no separate TS grammar query).
    assert "app.ts" in text
    # The rendered map respects its token budget.
    assert _tokens(text) <= 2000


def test_size_gate_returns_none_below_threshold(tmp_path: Path) -> None:
    for i in range(MIN_FILES_FOR_MAP - 5):
        (tmp_path / f"f_{i}.py").write_text("def x():\n    return 1\n", encoding="utf-8")
    assert repo_file_count(tmp_path) < MIN_FILES_FOR_MAP
    assert build_repo_map(tmp_path) is None


def test_repo_file_count_skips_build_and_vcs_dirs(tmp_path: Path) -> None:
    (tmp_path / "real.py").write_text("x = 1\n", encoding="utf-8")
    for skip in (".git", "node_modules", "__pycache__", ".grindstone", ".venv"):
        d = tmp_path / skip
        d.mkdir()
        (d / "junk.py").write_text("y = 2\n", encoding="utf-8")
    assert repo_file_count(tmp_path) == 1


@_needs_grammars
def test_broken_file_does_not_raise(tmp_path: Path) -> None:
    _spine_repo(tmp_path)
    (tmp_path / "broken.py").write_text(
        "def (((( this is not valid python @@@ \n class\n", encoding="utf-8"
    )
    (tmp_path / "empty.py").write_text("", encoding="utf-8")
    # Must not raise, and the valid spine still surfaces.
    text = build_repo_map(tmp_path, map_tokens=2000)
    assert text is not None and "shared_helper" in text


@_needs_grammars
def test_read_only_repo_degrades_to_memory_cache(tmp_path: Path) -> None:
    _spine_repo(tmp_path)
    original = stat.S_IMODE(tmp_path.stat().st_mode)
    os.chmod(tmp_path, stat.S_IRUSR | stat.S_IXUSR)  # r-x: cannot create .grindstone
    try:
        text = build_repo_map(tmp_path, map_tokens=2000)
    finally:
        os.chmod(tmp_path, original)
    assert text is not None and "shared_helper" in text
    assert not (tmp_path / ".grindstone").exists()  # nothing written to the repo tree


@_needs_grammars
def test_focus_files_collapse_map_to_neighborhood(tmp_path: Path) -> None:
    _two_cluster_repo(tmp_path)
    whole = build_repo_map(tmp_path, map_tokens=4000)
    assert whole is not None and "coreA" in whole and "coreB" in whole

    focused = build_repo_map(
        tmp_path, map_tokens=4000, focus_files=[Path("a_00.py")]
    )
    assert focused is not None
    # Seeding on a cluster-A file ranks coreA above coreB (the neighborhood).
    assert "coreA" in focused
    a_at = focused.find("coreA")
    b_at = focused.find("coreB")
    assert a_at != -1 and (b_at == -1 or a_at < b_at)


@_needs_grammars
def test_nonexistent_focus_paths_are_dropped(tmp_path: Path) -> None:
    _spine_repo(tmp_path)
    # A brand-new (not-yet-on-disk) file has no graph node; it is simply ignored
    # and the map falls back to whole-repo ranking rather than failing.
    text = build_repo_map(
        tmp_path, map_tokens=2000, focus_files=[Path("does/not/exist.py")]
    )
    assert text is not None and "shared_helper" in text


# --- planner integration -------------------------------------------------------

_JOB = "build the thing"


def _phases() -> list:
    from grindstone.contracts.models import Phase

    return [
        Phase(id="P1", title="A", exit_criterion=[CmdCheck(cmd="true")], epoch_budget=2),
        Phase(id="P2", title="B", exit_criterion=[CmdCheck(cmd="true")], epoch_budget=1),
    ]


def _planner_kwargs() -> dict[str, object]:
    return dict(
        job=_JOB,
        skeleton=_phases(),
        phase_id="P1",
        epoch_counter=1,
        log_index=[],
        last_epoch_rows=None,
        reask_errors=[],
    )


def test_planner_input_has_no_inline_repo_map_block() -> None:
    """The planner's structural map is now delivered BY REFERENCE (a file path in
    the <workspace> manifest), never inlined in the prompt. So even a fully built
    planner input carries NO <repo_map> block and NO inline map text, the prompt no
    longer pays the per-boundary token tax for the whole map. (Was:
    ``test_planner_repo_map_rides_tail_head_unchanged`` / ``..._empty_map_omits_section``,
    which asserted the inline block existed; the inline block + its param are gone.)"""

    out = build_planner_input(**_planner_kwargs())  # type: ignore[arg-type]
    assert "<repo_map>" not in out
    # The renderer + param are removed: build_planner_input no longer accepts repo_map.
    with pytest.raises(TypeError):
        build_planner_input(**_planner_kwargs(), repo_map="util.py:\n  x")  # type: ignore[call-arg]


# --- worker integration --------------------------------------------------------


def _implement_request(repo_map: str | None) -> WorkerRequest:
    task = ImplementTask(
        id="T1",
        goal="edit the widget",
        done_when=[CmdCheck(cmd="true")],
        file_ownership=["src/widget.py"],
    )
    return WorkerRequest(
        task=task,
        task_id="P1/E1/T1",
        inputs={},
        scratch=Path("/tmp/scratch"),
        attempt=1,
        failure_context=[],
        mode="implement",
        repo_map=repo_map,
    )


def _artifact_request(mode: str) -> WorkerRequest:
    task = ArtifactTask(
        id="T1",
        goal="investigate the thing",
        done_when=[CmdCheck(cmd="true")],
        artifact_out="notes.md",
        targets=["src/widget.py"],
    )
    return WorkerRequest(
        task=task,
        task_id="P1/E1/T1",
        inputs={},
        scratch=Path("/tmp/scratch"),
        attempt=1,
        failure_context=[],
        mode=mode,  # type: ignore[arg-type]
    )


def test_worker_prompt_injects_subtree_when_present() -> None:
    prompt = build_worker_prompt(_implement_request("src/widget.py:\n  def render():"))
    assert "<repo_map>" in prompt
    assert "def render():" in prompt


def test_worker_prompt_omits_subtree_when_absent() -> None:
    prompt = build_worker_prompt(_implement_request(None))
    assert "<repo_map>" not in prompt


def test_worker_prompt_carries_convergence_stop_rule() -> None:
    # The worker mirror of planner gate-skepticism (RCA: a senior rat-holed
    # instead of declaring a gate unsatisfiable). A short STOP rule: if the
    # done_when checks can't be met here, write FAILED/PARTIAL and exit, do not
    # loop. Present in every mode (shared block, not mode-specific).
    for req in (
        _implement_request(None),
        _artifact_request("research"),
        _artifact_request("review"),
    ):
        prompt = build_worker_prompt(req)
        assert "<stop_rule>" in prompt
        assert "cannot be satisfied" in prompt
        assert "do NOT loop" in prompt


def test_worker_prompt_carries_seam_scope_note() -> None:
    # Seam clarity (RCA: the senior wasted minutes reverse-engineering git-clean
    # / orchestration-file scope). State plainly: edit only your lane, the core
    # owns git/commit + its bookkeeping files, "working tree clean" is not yours.
    prompt = build_worker_prompt(_implement_request(None))
    assert "<scope>" in prompt
    assert "do NOT git-commit" in prompt
    assert "working tree clean" in prompt


@_needs_grammars
def test_worker_subtree_seeds_implement_on_file_ownership(tmp_path: Path) -> None:
    from grindstone.task_loop import _worker_subtree

    _spine_repo(tmp_path)
    task = ImplementTask(
        id="T1",
        goal="g",
        done_when=[CmdCheck(cmd="true")],
        file_ownership=["mod_0*.py", "util.py"],
    )
    sub = _worker_subtree(tmp_path, task)
    assert sub is not None and "shared_helper" in sub


def test_worker_subtree_none_for_artifact_without_targets(tmp_path: Path) -> None:
    from grindstone.task_loop import _worker_subtree

    _spine_repo(tmp_path)
    task = ArtifactTask(
        id="T1", goal="g", done_when=[CmdCheck(cmd="true")], artifact_out="notes.md"
    )
    assert _worker_subtree(tmp_path, task) is None


def test_worker_subtree_none_below_threshold(tmp_path: Path) -> None:
    from grindstone.task_loop import _worker_subtree

    (tmp_path / "only.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    task = ImplementTask(
        id="T1", goal="g", done_when=[CmdCheck(cmd="true")], file_ownership=["only.py"]
    )
    assert _worker_subtree(tmp_path, task) is None
    assert _worker_subtree(None, task) is None


def test_artifact_request_default_has_no_map() -> None:
    task = ArtifactTask(
        id="T1",
        goal="research the thing",
        done_when=[CmdCheck(cmd="true")],
        artifact_out="notes.md",
    )
    request = WorkerRequest(
        task=task,
        task_id="P1/E1/T1",
        inputs={},
        scratch=Path("/tmp/scratch"),
        attempt=1,
        failure_context=[],
        mode="research",
    )
    assert request.repo_map is None
    assert "<repo_map>" not in build_worker_prompt(request)
