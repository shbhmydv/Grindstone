"""The per-task EXECUTION UNIT (``run_task``) + the lenient CRITIC.

Driven entirely through ``MockWorker`` (no real model): a flat script alternates
worker behaviors and critic outcomes (``run_task`` always dispatches worker then,
on a DONE handoff, the tier-matched critic). Covers the BONES control flow: DONE ->
critic PASS -> merge-ready; RETRY-then-PASS; ESCALATE surfaced; retries exhausted
surfaced; BLOCKED skips the critic and routes to the planner; an invalid handoff is
rejected; and worktree isolation (writes never touch the operator checkout).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from grindstone import worktree as wt
from grindstone.contracts.models import Task
from grindstone.mock_worker import MockWorker
from grindstone.rundir import RunDir
from grindstone.worker import (
    Backends,
    RateLimited,
    build_critic_prompt,
    run_task,
)


def _implement(owned: list[str] | None = None) -> Task:
    return Task(
        id="T1", mode="implement", goal="create a.py", file_ownership=owned or ["a.py"]
    )


def _research() -> Task:
    return Task(
        id="T1", mode="research", goal="investigate", artifact_out="P1/E1/T1/r.md"
    )


def _backends(worker: MockWorker) -> Backends:
    return Backends.single(worker)


# --- DONE -> critic PASS -> merge-ready ----------------------------------------


def test_done_pass_is_merge_ready(git_repo: Path, run_dir: RunDir) -> None:
    base = wt.head_commit(git_repo)
    worker = MockWorker(script=["ok", "PASS"], artifacts={"a.py": "print(1)\n"})
    result = run_task(
        _implement(), "P1/E1/T1", run_dir=run_dir, repo=git_repo, base=base,
        backends=_backends(worker),
    )
    assert result.outcome == "passed"
    assert result.branch is not None
    assert result.verdict is not None and result.verdict.outcome == "PASS"
    # The committed branch carries the owned file; the operator checkout is untouched.
    assert "a.py" in wt.changed_paths(git_repo, base, result.branch)
    assert not (git_repo / "a.py").exists()


def test_research_pass_publishes_artifact(git_repo: Path, run_dir: RunDir) -> None:
    base = wt.head_commit(git_repo)
    worker = MockWorker(
        script=["ok", "PASS"], artifacts={"P1/E1/T1/r.md": "# findings\n"}
    )
    result = run_task(
        _research(), "P1/E1/T1", run_dir=run_dir, repo=git_repo, base=base,
        backends=_backends(worker),
    )
    assert result.outcome == "passed"
    assert result.artifact_key == "P1/E1/T1/r.md"
    # The deliverable is relocated into the durable run dir (the keyed log).
    assert run_dir.resolve("P1/E1/T1/r.md").read_text() == "# findings\n"


# --- RETRY then PASS -----------------------------------------------------------


def test_retry_then_pass(git_repo: Path, run_dir: RunDir) -> None:
    base = wt.head_commit(git_repo)
    worker = MockWorker(
        script=["ok", "RETRY", "ok", "PASS"], artifacts={"a.py": "print(1)\n"}
    )
    result = run_task(
        _implement(), "P1/E1/T1", run_dir=run_dir, repo=git_repo, base=base,
        backends=_backends(worker),
    )
    assert result.outcome == "passed"
    assert result.attempts == 2


# --- ESCALATE surfaced ---------------------------------------------------------


def test_escalate_surfaced(git_repo: Path, run_dir: RunDir) -> None:
    base = wt.head_commit(git_repo)
    worker = MockWorker(script=["ok", "ESCALATE"], artifacts={"a.py": "print(1)\n"})
    result = run_task(
        _implement(), "P1/E1/T1", run_dir=run_dir, repo=git_repo, base=base,
        backends=_backends(worker),
    )
    assert result.outcome == "escalated"
    assert result.verdict is not None and result.verdict.outcome == "ESCALATE"
    assert "escalate" in result.reason.lower()


# --- retries exhausted surfaced ------------------------------------------------


def test_retries_exhausted_surfaced(git_repo: Path, run_dir: RunDir) -> None:
    base = wt.head_commit(git_repo)
    worker = MockWorker(
        script=["ok", "RETRY", "ok", "RETRY"], artifacts={"a.py": "print(1)\n"}
    )
    result = run_task(
        _implement(), "P1/E1/T1", run_dir=run_dir, repo=git_repo, base=base,
        backends=_backends(worker),
    )
    assert result.outcome == "escalated"
    assert result.attempts == 2


# --- BLOCKED skips the critic, routes to the planner ---------------------------


def test_blocked_skips_critic(git_repo: Path, run_dir: RunDir) -> None:
    base = wt.head_commit(git_repo)
    # Script has NO critic entry: a BLOCKED handoff must route straight to the
    # planner without dispatching the critic (else the mock would over-run).
    worker = MockWorker(script=["blocked"], artifacts={"a.py": "print(1)\n"})
    result = run_task(
        _implement(), "P1/E1/T1", run_dir=run_dir, repo=git_repo, base=base,
        backends=_backends(worker),
    )
    assert result.outcome == "blocked"
    assert result.handoff is not None and result.handoff.status == "BLOCKED"
    assert result.verdict is None  # the critic never ran


# --- invalid handoff rejected --------------------------------------------------


def test_invalid_handoff_retried_then_pass(git_repo: Path, run_dir: RunDir) -> None:
    base = wt.head_commit(git_repo)
    worker = MockWorker(
        script=["bad_json", "ok", "PASS"], artifacts={"a.py": "print(1)\n"}
    )
    result = run_task(
        _implement(), "P1/E1/T1", run_dir=run_dir, repo=git_repo, base=base,
        backends=_backends(worker),
    )
    assert result.outcome == "passed"
    assert result.attempts == 2


def test_invalid_handoff_exhausts_to_escalate(
    git_repo: Path, run_dir: RunDir
) -> None:
    base = wt.head_commit(git_repo)
    worker = MockWorker(script=["bad_json", "empty"], artifacts={"a.py": "x\n"})
    result = run_task(
        _implement(), "P1/E1/T1", run_dir=run_dir, repo=git_repo, base=base,
        backends=_backends(worker),
    )
    assert result.outcome == "escalated"
    assert result.verdict is None  # no DONE handoff ever reached the critic


# --- rate limit propagates (not a burned attempt) ------------------------------


def test_rate_limit_propagates(git_repo: Path, run_dir: RunDir) -> None:
    base = wt.head_commit(git_repo)
    worker = MockWorker(script=["rate_limit"], artifacts={"a.py": "x\n"})
    with pytest.raises(RateLimited):
        run_task(
            _implement(), "P1/E1/T1", run_dir=run_dir, repo=git_repo, base=base,
            backends=_backends(worker),
        )


# --- worktree isolation --------------------------------------------------------


def test_worktree_isolation_external_base(git_repo: Path, run_dir: RunDir) -> None:
    # The throwaway worktrees live OUTSIDE the repo (the escape lesson): a worker
    # that strips its CWD to the repo root cannot reach the operator checkout.
    repo_resolved = str(git_repo.resolve())
    assert not str(run_dir.worktrees_root.resolve()).startswith(repo_resolved)
    base = wt.head_commit(git_repo)
    worker = MockWorker(script=["ok", "PASS"], artifacts={"a.py": "print(1)\n"})
    run_task(
        _implement(), "P1/E1/T1", run_dir=run_dir, repo=git_repo, base=base,
        backends=_backends(worker),
    )
    # Nothing the worker wrote landed in the live repo working tree.
    assert not (git_repo / "a.py").exists()


# --- the critic prompt encodes the triage --------------------------------------


def test_critic_prompt_encodes_triage() -> None:
    from grindstone.worker import CriticBrief, WorkerRequest

    request = WorkerRequest(
        task=_implement(), task_id="P1/E1/T1", mode="implement", scratch=Path("/x"),
        critic=CriticBrief(goal="create a.py", mode="implement", diff_base="HEAD~1"),
    )
    prompt = build_critic_prompt(request, request.critic)  # type: ignore[arg-type]
    low = prompt.lower()
    # Anchored on the task's own claimed goal, not the critic's taste.
    assert "create a.py" in prompt
    # The lenient bar + the single retry-vs-escalate question.
    assert "good enough to build on" in low
    assert "same worker" in low
    assert "pass" in low and "retry" in low and "escalate" in low
