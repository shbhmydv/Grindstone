"""The per-task EXECUTION UNIT (``run_task``) + the lenient CRITIC.

Driven entirely through ``MockWorker`` (no real model): a flat script alternates
worker behaviors and critic outcomes (``run_task`` always dispatches the worker then,
on a gate-clean attempt, the tier-matched critic). Covers the BONES control flow: a
gate-clean attempt -> critic PASS -> merge-ready; RETRY-then-PASS; ESCALATE surfaced;
retries exhausted surfaced; a worker that REPORTS an environmental blocker -> the
critic ESCALATES (no separate Python BLOCKED gate); an attempt that fails the
deterministic gate (no committed work) is rejected; and worktree isolation (writes
never touch the operator checkout).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from grindstone import worktree as wt
from grindstone.contracts.models import Task
from tests.grindstone.mock_worker import MockWorker
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
        id="T1", mode="research", goal="investigate", artifact_out="E1/T1/r.md"
    )


def _backends(worker: MockWorker) -> Backends:
    return Backends.single(worker)


# --- DONE -> critic PASS -> merge-ready ----------------------------------------


def test_done_pass_is_merge_ready(git_repo: Path, run_dir: RunDir) -> None:
    base = wt.head_commit(git_repo)
    worker = MockWorker(script=["ok", "PASS"], artifacts={"a.py": "print(1)\n"})
    result = run_task(
        _implement(), "E1/T1", run_dir=run_dir, repo=git_repo, base=base,
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
        script=["ok", "PASS"], artifacts={"E1/T1/r.md": "# findings\n"}
    )
    result = run_task(
        _research(), "E1/T1", run_dir=run_dir, repo=git_repo, base=base,
        backends=_backends(worker),
    )
    assert result.outcome == "passed"
    assert result.artifact_key == "E1/T1/r.md"
    # The deliverable is relocated into the durable run dir (the keyed log).
    assert run_dir.resolve("E1/T1/r.md").read_text() == "# findings\n"


# --- RETRY then PASS -----------------------------------------------------------


def test_retry_then_pass(git_repo: Path, run_dir: RunDir) -> None:
    base = wt.head_commit(git_repo)
    worker = MockWorker(
        script=["ok", "RETRY", "ok", "PASS"], artifacts={"a.py": "print(1)\n"}
    )
    result = run_task(
        _implement(), "E1/T1", run_dir=run_dir, repo=git_repo, base=base,
        backends=_backends(worker),
    )
    assert result.outcome == "passed"
    assert result.attempts == 2


# --- ESCALATE surfaced ---------------------------------------------------------


def test_escalate_surfaced(git_repo: Path, run_dir: RunDir) -> None:
    base = wt.head_commit(git_repo)
    worker = MockWorker(script=["ok", "ESCALATE"], artifacts={"a.py": "print(1)\n"})
    result = run_task(
        _implement(), "E1/T1", run_dir=run_dir, repo=git_repo, base=base,
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
        _implement(), "E1/T1", run_dir=run_dir, repo=git_repo, base=base,
        backends=_backends(worker),
    )
    assert result.outcome == "escalated"
    assert result.attempts == 2


# --- a worker-reported environmental blocker -> the critic ESCALATES -----------


def test_blocked_report_escalates(git_repo: Path, run_dir: RunDir) -> None:
    # The worker does some work AND writes a handoff.md reporting a hard blocker; the
    # independent critic reads that report and ESCALATES (no Python BLOCKED gate).
    base = wt.head_commit(git_repo)
    worker = MockWorker(script=["blocked", "ESCALATE"], artifacts={"a.py": "print(1)\n"})
    result = run_task(
        _implement(), "E1/T1", run_dir=run_dir, repo=git_repo, base=base,
        backends=_backends(worker),
    )
    assert result.outcome == "escalated"
    assert result.verdict is not None and result.verdict.outcome == "ESCALATE"
    # The free-form report was relocated for the planner's context.
    assert result.handoff_path is not None and result.handoff_path.is_file()


# --- a failed deterministic gate (no committed work) is rejected ---------------


def test_empty_attempt_retried_then_pass(git_repo: Path, run_dir: RunDir) -> None:
    base = wt.head_commit(git_repo)
    worker = MockWorker(
        script=["empty", "ok", "PASS"], artifacts={"a.py": "print(1)\n"}
    )
    result = run_task(
        _implement(), "E1/T1", run_dir=run_dir, repo=git_repo, base=base,
        backends=_backends(worker),
    )
    assert result.outcome == "passed"
    assert result.attempts == 2


def test_empty_attempts_exhaust_to_escalate(
    git_repo: Path, run_dir: RunDir
) -> None:
    base = wt.head_commit(git_repo)
    worker = MockWorker(script=["empty", "empty"], artifacts={"a.py": "x\n"})
    result = run_task(
        _implement(), "E1/T1", run_dir=run_dir, repo=git_repo, base=base,
        backends=_backends(worker),
    )
    assert result.outcome == "escalated"
    assert result.verdict is None  # no gate-clean attempt ever reached the critic


# --- rate limit propagates (not a burned attempt) ------------------------------


def test_rate_limit_propagates(git_repo: Path, run_dir: RunDir) -> None:
    base = wt.head_commit(git_repo)
    worker = MockWorker(script=["rate_limit"], artifacts={"a.py": "x\n"})
    with pytest.raises(RateLimited):
        run_task(
            _implement(), "E1/T1", run_dir=run_dir, repo=git_repo, base=base,
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
        _implement(), "E1/T1", run_dir=run_dir, repo=git_repo, base=base,
        backends=_backends(worker),
    )
    # Nothing the worker wrote landed in the live repo working tree.
    assert not (git_repo / "a.py").exists()


# --- the critic prompt encodes the triage --------------------------------------


def test_implement_prompt_allows_in_worktree_dep_install() -> None:
    # FIX 3: an implement worker MAY install project deps inside its own worktree if
    # its checks require them (setup no longer carries project-local installs).
    from grindstone.worker import WorkerRequest, build_worker_prompt

    request = WorkerRequest(
        task=_implement(), task_id="E1/T1", mode="implement", scratch=Path("/x"),
    )
    prompt = build_worker_prompt(request).lower()
    assert "install" in prompt and "inside this worktree" in prompt


def test_critic_prompt_encodes_triage() -> None:
    from grindstone.worker import CriticBrief, WorkerRequest

    request = WorkerRequest(
        task=_implement(), task_id="E1/T1", mode="implement", scratch=Path("/x"),
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
