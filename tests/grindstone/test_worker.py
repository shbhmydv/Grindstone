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

import json
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


# --- orchestration-file hygiene: nothing internal lingers in scratch -----------


def test_critic_verdict_relocated_out_of_scratch(
    git_repo: Path, run_dir: RunDir
) -> None:
    # The critic writes verdict.json into scratch; the orchestrator MOVES it to the
    # keyed log (copy + remove). A lingering scratch verdict.json would be swept into
    # the next implement commit and poison the integration base (cross-epoch bug).
    from grindstone.worker import CRITIC_VERDICT_FILENAME, _critic_verdict

    base = wt.head_commit(git_repo)
    scratch = run_dir.artifacts_dir("E1/T1/critic")
    worker = MockWorker(script=["PASS"])
    verdict = _critic_verdict(
        _implement(), "E1/T1", scratch=scratch, base=base, artifact_rel=None,
        handoff_text="", critic_read_root=git_repo, run_dir=run_dir,
        backends=_backends(worker), domain_skills={},
    )
    assert verdict.outcome == "PASS"
    # (a) the scratch original is GONE, (b) the keyed-log dest carries the content.
    assert not (scratch / CRITIC_VERDICT_FILENAME).exists()
    dest = run_dir.resolve(f"E1/T1/{CRITIC_VERDICT_FILENAME}")
    assert dest.is_file()
    assert json.loads(dest.read_text())["outcome"] == "PASS"


def test_relocate_handoff_oversized_is_removed(
    run_dir: RunDir, tmp_path: Path
) -> None:
    # A pathologically-large free-form report is DROPPED (never parsed), but must
    # still be removed from scratch so it cannot be swept into a commit.
    from grindstone.worker import (
        _DISK_READ_MAX_BYTES,
        HANDOFF_FILENAME,
        _relocate_handoff,
    )

    scratch = tmp_path / "scratch"
    scratch.mkdir()
    (scratch / HANDOFF_FILENAME).write_text(
        "x" * (_DISK_READ_MAX_BYTES + 1), encoding="utf-8"
    )
    path, text = _relocate_handoff(scratch, run_dir=run_dir, task_id="E1/T1")
    assert path is None and text == ""
    assert not (scratch / HANDOFF_FILENAME).exists()


def test_relocate_handoff_normal_is_moved(run_dir: RunDir, tmp_path: Path) -> None:
    from grindstone.worker import HANDOFF_FILENAME, _relocate_handoff

    scratch = tmp_path / "scratch"
    scratch.mkdir()
    (scratch / HANDOFF_FILENAME).write_text("# report\n", encoding="utf-8")
    path, text = _relocate_handoff(scratch, run_dir=run_dir, task_id="E1/T1")
    assert path is not None and path.is_file()
    assert text == "# report\n"
    # Moved, not copied: gone from scratch, present at the keyed-log dest.
    assert not (scratch / HANDOFF_FILENAME).exists()
    assert run_dir.resolve(f"E1/T1/{HANDOFF_FILENAME}").read_text() == "# report\n"


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


def _critic_request(skills: dict[str, str]) -> "WorkerRequest":  # noqa: F821
    from grindstone.worker import CriticBrief, WorkerRequest

    return WorkerRequest(
        task=_implement(), task_id="E1/T1", mode="implement", scratch=Path("/x"),
        critic=CriticBrief(goal="create a.py", mode="implement", diff_base="HEAD~1"),
        domain_skills=skills,
    )


def test_critic_prompt_renders_skills_rubric() -> None:
    # The critic IS the "analyse" step: when the task selected domain skills, the
    # critic must verify the work against them as the RUBRIC THE WORK CLAIMS TO MEET,
    # and NOT be lenient on that conformance.
    req = _critic_request({"rn-composition": "Compose screens from primitives."})
    prompt = build_critic_prompt(req, req.critic)  # type: ignore[arg-type]
    low = prompt.lower()
    # The selected skill's name + body are present (retrieve-not-concatenate).
    assert "rn-composition" in prompt
    assert "compose screens from primitives." in low
    # Critic-framed as a rubric, NOT lenient on conformance.
    assert "rubric" in low
    assert "do not be lenient" in low
    # The additive strictness does NOT replace the lenient-router framing.
    assert "good enough to build on" in low


def test_critic_prompt_byte_identical_when_skill_less() -> None:
    # SURGICAL: a skill-less critic keeps today's lenient-router prompt EXACTLY. The
    # rubric block is absent and the lenient framing intact (byte-identical to before).
    skilled = build_critic_prompt(_critic_request({"s": "x"}), _critic_request({"s": "x"}).critic)  # type: ignore[arg-type]
    skill_less = build_critic_prompt(_critic_request({}), _critic_request({}).critic)  # type: ignore[arg-type]
    assert "<rubric>" not in skill_less
    assert "<rubric>" in skilled  # the rubric ONLY appears with skills present
    # The lenient-router framing is preserved verbatim in the skill-less prompt.
    low = skill_less.lower()
    assert "good enough to build on" in low
    assert "not to grade it" in low
    assert "bias to pass when unsure" in low


def test_critic_verdict_threads_domain_skills(git_repo: Path, run_dir: RunDir) -> None:
    # The task's already-loaded domain skills must reach the critic dispatch so the
    # critic can enforce them. Assert the request the transport receives carries them.
    from grindstone.worker import (
        CRITIC_VERDICT_FILENAME,
        WorkerRequest,
        _critic_verdict,
    )

    captured: list[dict[str, str]] = []

    class _Recorder:
        def run(self, request: WorkerRequest) -> None:
            captured.append(dict(request.domain_skills))
            (request.scratch / CRITIC_VERDICT_FILENAME).write_text(
                json.dumps({"outcome": "PASS", "reason": "ok"}), encoding="utf-8"
            )

    base = wt.head_commit(git_repo)
    scratch = run_dir.artifacts_dir("E1/T1/critic")
    skills = {"rn-composition": "Compose screens from primitives."}
    verdict = _critic_verdict(
        _implement(), "E1/T1", scratch=scratch, base=base, artifact_rel=None,
        handoff_text="", critic_read_root=git_repo, run_dir=run_dir,
        backends=Backends.single(_Recorder()), domain_skills=skills,
    )
    assert verdict.outcome == "PASS"
    assert captured == [skills]
