"""B5 codex inline final-polish: an optional, gated post-completion polish pass.

AFTER a run's ``complete_run`` evidence passes, an OPTIONAL pass lets codex EDIT
the finished repo inline (workspace-write) for finishing touches. Safety is the
whole point: codex's edits are KEPT only if the SAME evidence STILL passes against
the polished commit, otherwise discarded, and the original completion stands.
Off by default; a polish pass can never bypass the gate or fail a completed run.

These tests drive the whole pass through ``run_grind`` with a STUB polish script
behind the ``ScriptPolisher`` seam (it edits a file in the worktree directly), so
no real codex is ever called.
"""

from __future__ import annotations

import re
from pathlib import Path

from grindstone import worktree as wt
from grindstone.contracts.models import CmdCheck
from grindstone.events import (
    FinalPolishApplied,
    FinalPolishSkipped,
    JournalWriter,
    read_events,
    replay,
)
from grindstone.mock_planner import MockPlanner
from grindstone.run_loop import FinalPolish, RunState, _final_polish, _RunStateStore, run_grind
from grindstone.script_polish import ScriptPolisher
from grindstone.rundir import RunDir

from tests.grindstone.conftest import (
    check_cmd,
    complete_decision,
    impl_task,
    implement_decision,
    tracked_files,
    two_phase_skeleton,
)


def _ladder() -> list[tuple[str, object]]:
    from tests.grindstone.conftest import OwnershipWorker

    return [("worker", OwnershipWorker())]


def _run_state(run_dir: RunDir) -> RunState:
    return RunState.model_validate_json(run_dir.run_state_path.read_text())


# --- stub polish scripts (edit the worktree in place, no disk-contract verdict) --

#: A stub that finds ``--repo`` (the writable worktree codex would edit), runs an
#: arbitrary ``{body}`` against it, and exits ``$RC``. It never calls codex,
#: ``{body}`` stands in for the model's inline edits.
_STUB = """\
#!/usr/bin/env bash
set -euo pipefail
repo=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo) repo="$2"; shift 2 ;;
    *) shift ;;
  esac
done
{body}
"""

#: Make a benign edit that does NOT break ``test -f f1.txt``: add a new file.
_BENIGN = 'printf polished > "$repo/POLISH.md"\nexit 0\n'
#: Break the evidence: delete the file ``test -f f1.txt`` checks for.
_BREAKS = 'rm -f "$repo/f1.txt"\nexit 0\n'
#: Leave the worktree untouched (codex decided nothing needed polishing).
_NOCHANGE = "exit 0\n"
#: The codex call itself failed (nonzero exit), a clean no-op.
_NONZERO = 'echo "codex polish failed" >&2\nexit 3\n'


def _polish(tmp_path: Path, body: str, *, criteria: str = "tasteful polish") -> FinalPolish:
    script = tmp_path / "polish_stub.sh"
    script.write_text(_STUB.format(body=body), encoding="utf-8")
    script.chmod(0o755)
    return FinalPolish(
        polisher=ScriptPolisher(script=script, timeout_s=30), criteria=criteria
    )


def _completing_planner() -> MockPlanner:
    return MockPlanner(
        script=[
            two_phase_skeleton(),
            implement_decision(impl_task("T1", "f1.txt")),
            complete_decision(check_cmd("test -f f1.txt")),
        ]
    )


# --- disabled (default): completion path is unchanged --------------------------


def test_final_polish_disabled_leaves_completion_unchanged(
    git_repo: Path, run_dir: RunDir
) -> None:
    outcome = run_grind(
        run_dir, job_path="job.md", planner=_completing_planner(),
        ladder=_ladder(), repo=git_repo,
    )
    assert outcome.status == "completed"
    kinds = [e.event for e in read_events(run_dir.events_path)]
    assert "final_polish_applied" not in kinds
    assert "final_polish_skipped" not in kinds
    # No polish worktree was ever created under the run dir.
    assert not (run_dir.root / "worktrees" / "_polish").exists()


# --- enabled + benign edit that keeps evidence green -> APPLIED ----------------


def test_final_polish_applied_when_evidence_still_passes(
    git_repo: Path, run_dir: RunDir
) -> None:
    outcome = run_grind(
        run_dir, job_path="job.md", planner=_completing_planner(),
        ladder=_ladder(), repo=git_repo, final_polish=_polish(run_dir.root, _BENIGN),
    )
    assert outcome.status == "completed"
    events = read_events(run_dir.events_path)
    applied = [e for e in events if isinstance(e, FinalPolishApplied)]
    assert len(applied) == 1
    # FIX 1: the final branch is a real REF materialized at the polish commit
    # (not a bare, dangling sha), and it resolves to the adopted commit.
    state = _run_state(run_dir)
    branch = state.last_integration_branch
    assert branch is not None
    assert wt.resolve_commit(git_repo, branch) == applied[0].commit
    assert outcome.final_branch == branch
    # The polished commit carries BOTH the implemented file and the polish edit,
    # and the evidence (test -f f1.txt) still holds.
    files = set(tracked_files(git_repo, branch))
    assert {"f1.txt", "POLISH.md"} <= files


# --- FIX 1: an adopted polish leaves a reachable ref, not a dangling sha --------


def test_final_polish_applied_materializes_reachable_branch(
    git_repo: Path, run_dir: RunDir
) -> None:
    outcome = run_grind(
        run_dir, job_path="job.md", planner=_completing_planner(),
        ladder=_ladder(), repo=git_repo, final_polish=_polish(run_dir.root, _BENIGN),
    )
    assert outcome.status == "completed"
    applied = [e for e in read_events(run_dir.events_path) if isinstance(e, FinalPolishApplied)]
    assert len(applied) == 1
    branch = _run_state(run_dir).last_integration_branch
    assert branch is not None
    # It is a branch NAME, not a bare 40-hex sha (the gc-prone dangling-commit bug).
    assert not re.fullmatch(r"[0-9a-f]{40}", branch)
    assert wt.branch_exists(git_repo, branch)
    # The ref resolves to the adopted polish commit => the commit is reachable.
    assert wt.resolve_commit(git_repo, branch) == applied[0].commit
    assert outcome.final_branch == branch
    # Checking the branch out yields the polished file.
    assert "POLISH.md" in tracked_files(git_repo, branch)


# --- FIX 7a: the applied event surfaces the polish diff's changed files ---------


def test_final_polish_applied_records_changed_files(
    git_repo: Path, run_dir: RunDir
) -> None:
    run_grind(
        run_dir, job_path="job.md", planner=_completing_planner(),
        ladder=_ladder(), repo=git_repo, final_polish=_polish(run_dir.root, _BENIGN),
    )
    applied = [e for e in read_events(run_dir.events_path) if isinstance(e, FinalPolishApplied)]
    assert len(applied) == 1
    assert "POLISH.md" in applied[0].changed_files


# --- FIX 7b: a re-entered run never re-polishes / stacks a second commit --------


def test_final_polish_is_idempotent_on_reentry(git_repo: Path, run_dir: RunDir) -> None:
    fp = _polish(run_dir.root, _BENIGN)
    run_grind(
        run_dir, job_path="job.md", planner=_completing_planner(),
        ladder=_ladder(), repo=git_repo, final_polish=fp,
    )
    applied1 = [e for e in read_events(run_dir.events_path) if isinstance(e, FinalPolishApplied)]
    assert len(applied1) == 1
    commit1 = applied1[0].commit
    # Re-invoke as a resume between adopt-and-complete would: the journal already
    # records a polish outcome, so it must be a clean no-op (no second commit).
    state = _run_state(run_dir)
    store = _RunStateStore(run_dir, state)
    with JournalWriter(run_dir.events_path) as journal:
        _final_polish(
            journal, store, run_dir, git_repo,
            [CmdCheck(cmd="test -f f1.txt")], fp, None, None, None,
        )
    applied2 = [e for e in read_events(run_dir.events_path) if isinstance(e, FinalPolishApplied)]
    assert len(applied2) == 1  # NOT re-applied
    assert applied2[0].commit == commit1
    branch = state.last_integration_branch
    assert branch is not None
    assert wt.resolve_commit(git_repo, branch) == commit1  # branch did not advance


# --- FIX 2: a missing / bad polish script is a clean skip, never a crash --------


def test_final_polish_missing_script_is_clean_skip(
    git_repo: Path, run_dir: RunDir
) -> None:
    fp = FinalPolish(
        polisher=ScriptPolisher(script=run_dir.root / "nope.sh", timeout_s=30),
        criteria="x",
    )
    outcome = run_grind(
        run_dir, job_path="job.md", planner=_completing_planner(),
        ladder=_ladder(), repo=git_repo, final_polish=fp,
    )
    assert outcome.status == "completed"
    events = read_events(run_dir.events_path)
    assert not [e for e in events if isinstance(e, FinalPolishApplied)]
    assert "run_completed" in [e.event for e in events]
    assert outcome.final_branch is not None
    assert "f1.txt" in tracked_files(git_repo, outcome.final_branch)


def test_polisher_missing_script_returns_false_not_raise(tmp_path: Path) -> None:
    # FIX 2 raw seam: Popen's OSError on a missing script is converted to a clean
    # False (failed polish = no-op), never a propagating OSError.
    p = ScriptPolisher(script=tmp_path / "nope.sh", timeout_s=30)
    out = tmp_path / "out"
    assert p.polish(worktree=tmp_path, criteria="x", screenshot_rel=None, out_dir=out) is False


# --- FIX 4: the polish Python supervisor outlasts the script's --timeout --------


def test_polish_supervisor_timeout_exceeds_script_deadline() -> None:
    p = ScriptPolisher(script=Path("/x/polish.sh"), timeout_s=30)
    assert p.supervise_timeout_s > p.timeout_s
    assert int(p.timeout_s) <= p.supervise_timeout_s


# --- enabled + edit that BREAKS evidence -> DISCARDED --------------------------


def test_final_polish_discarded_when_evidence_regresses(
    git_repo: Path, run_dir: RunDir
) -> None:
    planner = _completing_planner()
    outcome = run_grind(
        run_dir, job_path="job.md", planner=planner,
        ladder=_ladder(), repo=git_repo, final_polish=_polish(run_dir.root, _BREAKS),
    )
    assert outcome.status == "completed"  # the original completion still stands
    events = read_events(run_dir.events_path)
    assert not [e for e in events if isinstance(e, FinalPolishApplied)]
    skipped = [e for e in events if isinstance(e, FinalPolishSkipped)]
    assert len(skipped) == 1 and "f1.txt" in skipped[0].reason
    # Branch unchanged: still carries f1.txt; the broken polish commit is not adopted.
    state = _run_state(run_dir)
    assert state.last_integration_branch is not None
    assert "f1.txt" in tracked_files(git_repo, state.last_integration_branch)


# --- enabled + no change -> no-op SKIPPED --------------------------------------


def test_final_polish_no_change_is_skipped(git_repo: Path, run_dir: RunDir) -> None:
    outcome = run_grind(
        run_dir, job_path="job.md", planner=_completing_planner(),
        ladder=_ladder(), repo=git_repo, final_polish=_polish(run_dir.root, _NOCHANGE),
    )
    assert outcome.status == "completed"
    events = read_events(run_dir.events_path)
    assert not [e for e in events if isinstance(e, FinalPolishApplied)]
    skipped = [e for e in events if isinstance(e, FinalPolishSkipped)]
    assert len(skipped) == 1 and "no change" in skipped[0].reason.lower()


# --- enabled + codex script errors -> clean no-op, run still completed ----------


def test_final_polish_script_error_never_fails_the_run(
    git_repo: Path, run_dir: RunDir
) -> None:
    outcome = run_grind(
        run_dir, job_path="job.md", planner=_completing_planner(),
        ladder=_ladder(), repo=git_repo, final_polish=_polish(run_dir.root, _NONZERO),
    )
    assert outcome.status == "completed"
    events = read_events(run_dir.events_path)
    assert not [e for e in events if isinstance(e, FinalPolishApplied)]
    # A failed polish is a clean no-op; the run completes regardless.
    assert "run_completed" in [e.event for e in events]
    assert outcome.final_branch is not None
    assert "f1.txt" in tracked_files(git_repo, outcome.final_branch)


# --- replay folds the new events -----------------------------------------------


def test_replay_folds_final_polish_event(git_repo: Path, run_dir: RunDir) -> None:
    run_grind(
        run_dir, job_path="job.md", planner=_completing_planner(),
        ladder=_ladder(), repo=git_repo, final_polish=_polish(run_dir.root, _BENIGN),
    )
    tree = replay(read_events(run_dir.events_path))
    assert tree.status == "completed"
    assert tree.final_polish is not None and tree.final_polish.startswith("applied")


def test_replay_folds_skipped_final_polish_event(
    git_repo: Path, run_dir: RunDir
) -> None:
    run_grind(
        run_dir, job_path="job.md", planner=_completing_planner(),
        ladder=_ladder(), repo=git_repo, final_polish=_polish(run_dir.root, _NOCHANGE),
    )
    tree = replay(read_events(run_dir.events_path))
    assert tree.final_polish is not None and tree.final_polish.startswith("skipped")
