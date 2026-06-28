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
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from grindstone import worktree as wt
from grindstone.contracts.models import Task
from tests.grindstone.mock_worker import MockWorker
from grindstone.rundir import RunDir
from grindstone.worker import (
    Backends,
    CRITIC_VERDICT_FILENAME,
    HANDOFF_FILENAME,
    RateLimited,
    WorkerRequest,
    WorkerTransport,
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
    # A local task on a single-endpoint rig (no distinct senior) runs only its local
    # stage: LOCAL_MAX_ATTEMPTS attempts of RETRY exhaust the ladder -> escalated.
    base = wt.head_commit(git_repo)
    worker = MockWorker(
        script=["ok", "RETRY", "ok", "RETRY", "ok", "RETRY"],
        artifacts={"a.py": "print(1)\n"},
    )
    result = run_task(
        _implement(), "E1/T1", run_dir=run_dir, repo=git_repo, base=base,
        backends=_backends(worker),
    )
    assert result.outcome == "escalated"
    assert result.attempts == 3
    assert "ladder exhausted" in result.reason


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
        _implement(), "E1/T1", tier="local", scratch=scratch, base=base,
        artifact_rel=None, handoff_text="", critic_read_root=git_repo, run_dir=run_dir,
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


def test_worker_prompt_injects_repo_map() -> None:
    from grindstone.worker import WorkerRequest, build_worker_prompt

    repo_map = "src/widget/ is the package; cli.py is the entry point."
    request = WorkerRequest(
        task=_implement(), task_id="E1/T1", mode="implement", scratch=Path("/x"),
        repo_map=repo_map,
    )
    prompt = build_worker_prompt(request)
    assert prompt.count("<repo_map>") == 1
    assert repo_map in prompt


def test_worker_prompt_no_repo_map_is_byte_identical() -> None:
    # the no-repo-map path must be byte-for-byte identical to NOT passing one at all.
    from grindstone.worker import WorkerRequest, build_worker_prompt

    base_req = WorkerRequest(
        task=_implement(), task_id="E1/T1", mode="implement", scratch=Path("/x"),
    )
    empty_req = WorkerRequest(
        task=_implement(), task_id="E1/T1", mode="implement", scratch=Path("/x"),
        repo_map="",
    )
    base = build_worker_prompt(base_req)
    with_empty = build_worker_prompt(empty_req)
    assert with_empty == base
    assert "<repo_map>" not in base


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
        _implement(), "E1/T1", tier="local", scratch=scratch, base=base,
        artifact_rel=None, handoff_text="", critic_read_root=git_repo, run_dir=run_dir,
        backends=Backends.single(_Recorder()), domain_skills=skills,
    )
    assert verdict.outcome == "PASS"
    assert captured == [skills]


# --- the in-epoch tier ladder (local -> senior on a distinct rig) ---------------


@dataclass
class _TierWorker:
    """A loop worker that always routes its critic to ``critic_outcome`` and records
    every (non-critic) grind's task id. Distinct local + senior instances let a test
    prove which tier ``run_task``'s in-epoch ladder dispatched at."""

    critic_outcome: str
    worker_calls: list[str] = field(default_factory=list)

    def run(self, request: "WorkerRequest") -> None:
        if request.critic is not None:
            (request.scratch / CRITIC_VERDICT_FILENAME).write_text(
                json.dumps(
                    {"outcome": self.critic_outcome, "reason": self.critic_outcome}
                ),
                encoding="utf-8",
            )
            return
        self.worker_calls.append(request.task_id)
        for rel in request.task.file_ownership:
            path = request.scratch / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                f"# {request.task_id}\nvalue = {request.task_id!r}\n", encoding="utf-8"
            )
        (request.scratch / HANDOFF_FILENAME).write_text("# handoff\n", encoding="utf-8")


def _two_tier_backends(
    local: "WorkerTransport", senior: "WorkerTransport"
) -> Backends:
    """A Backends with DISTINCT local + senior endpoints (so the in-epoch ladder really
    escalates): each tier dispatches its own transport instance."""

    import threading

    from grindstone.worker import _Endpoint

    endpoints = {
        "local": _Endpoint(local, threading.Semaphore(1)),
        "senior": _Endpoint(senior, threading.Semaphore(1)),
    }
    return Backends(endpoints, {"local": "local", "senior": "senior"})


def test_in_epoch_ladder_escalates_local_to_senior(
    git_repo: Path, run_dir: RunDir
) -> None:
    # A planner-"local" task on a rig with a DISTINCT senior endpoint: the local stage
    # RETRYs until its budget is spent, THEN the ladder re-dispatches at senior on the
    # carried wip within the SAME run_task call, where it PASSES.
    from grindstone.worker import LOCAL_MAX_ATTEMPTS

    base = wt.head_commit(git_repo)
    local = _TierWorker("RETRY")
    senior = _TierWorker("PASS")
    result = run_task(
        _implement(), "E1/T1", run_dir=run_dir, repo=git_repo, base=base,
        backends=_two_tier_backends(local, senior),
    )
    assert result.outcome == "passed"
    assert result.verdict is not None and result.verdict.outcome == "PASS"
    # local exhausted its full budget; senior then ran (on the carried wip) and passed.
    assert len(local.worker_calls) == LOCAL_MAX_ATTEMPTS
    assert len(senior.worker_calls) == 1
    assert result.branch is not None  # the senior attempt's merge-ready wip


def test_in_epoch_ladder_escalates_on_critic_escalate(
    git_repo: Path, run_dir: RunDir
) -> None:
    # A single local ESCALATE breaks straight to the senior stage (no need to burn the
    # whole local budget first); senior passes -> merge-ready.
    base = wt.head_commit(git_repo)
    local = _TierWorker("ESCALATE")
    senior = _TierWorker("PASS")
    result = run_task(
        _implement(), "E1/T1", run_dir=run_dir, repo=git_repo, base=base,
        backends=_two_tier_backends(local, senior),
    )
    assert result.outcome == "passed"
    assert len(local.worker_calls) == 1  # one local attempt, then escalate to senior
    assert len(senior.worker_calls) == 1


def test_senior_task_does_not_escalate_further(
    git_repo: Path, run_dir: RunDir
) -> None:
    # A planner-"senior" task runs ONLY its senior stage: exhausting it escalates to the
    # planner; it never falls back to local (there is no rung below senior).
    from grindstone.worker import SENIOR_MAX_ATTEMPTS

    base = wt.head_commit(git_repo)
    local = _TierWorker("PASS")  # must never run
    senior = _TierWorker("RETRY")
    task = Task(
        id="T1", mode="implement", goal="create a.py", file_ownership=["a.py"],
        tier="senior",
    )
    result = run_task(
        task, "E1/T1", run_dir=run_dir, repo=git_repo, base=base,
        backends=_two_tier_backends(local, senior),
    )
    assert result.outcome == "escalated"
    assert result.attempts == SENIOR_MAX_ATTEMPTS
    assert local.worker_calls == []
    assert len(senior.worker_calls) == SENIOR_MAX_ATTEMPTS


def test_no_distinct_senior_endpoint_does_not_escalate(
    git_repo: Path, run_dir: RunDir
) -> None:
    # On a single-endpoint rig (senior resolves to local), a local task never escalates
    # - escalating would just re-run the SAME model. It runs only its local stage.
    from grindstone.worker import LOCAL_MAX_ATTEMPTS

    base = wt.head_commit(git_repo)
    worker = _TierWorker("RETRY")
    result = run_task(
        _implement(), "E1/T1", run_dir=run_dir, repo=git_repo, base=base,
        backends=Backends.single(worker),
    )
    assert result.outcome == "escalated"
    assert result.attempts == LOCAL_MAX_ATTEMPTS
    assert len(worker.worker_calls) == LOCAL_MAX_ATTEMPTS


# --- the worker prompt: fresh (byte-stable) vs resume (fix-frame) ---------------

#: The exact FRESH implement prompt (captured from the pre-change build_worker_prompt):
#: the no-prior-work path must stay byte-identical so existing workers + the cacheable
#: prefix are untouched.
_FRESH_IMPLEMENT_ORACLE = '''<task id="E1/T1">
create a.py
</task>

<worktree>
You run inside an ISOLATED, throwaway git worktree that is your current working directory
and IS the repository for this task. Create and edit every file with paths RELATIVE to
your CWD; never write to an absolute path and never write outside your CWD. There is no
other repository you may touch - do not go looking for one, this worktree is it. The
orchestrator inspects ONLY this worktree to gate and integrate your work, so anything you
write elsewhere is invisible, discarded, and corrupts the run. If something you depend on
(a module to import, code under test) is NOT present in your worktree, it is owned by
another task that has not merged yet - do NOT create it to unblock yourself: write against
it as if it exists and record the missing dependency in your handoff. Reaching beyond your
own files only fails the gate. You can SEE: read any
image in your worktree or inputs directly (screenshots, mockups, designs) and produce
images where visual proof helps your reviewer - view, do not guess.
</worktree>
<inputs>
  (none)
</inputs>

Make the change inside your file_ownership, then COMMIT it (the orchestrator gates the git diff in this worktree, not your words). If your checks need the project's dependencies, you MAY install them INSIDE THIS WORKTREE as part of your work (setup does not reach here). Run whatever checks you write to convince yourself it works; if the work is visual, render and LOOK at the result.
<file_ownership>
You may create or edit files ONLY within these globs:
  - a.py
Changing ANY other file fails the attempt. (`handoff.md` is an
orchestration file, write it in the CWD as instructed; the orchestrator excludes
it from this rule and from your commit.)
</file_ownership>

<handoff>
When finished, write a SHORT free-form report named exactly `handoff.md` in
your current working directory, for the independent reviewer who reads your work
next. Plain prose (no required schema): what you did, what is DONE, what is still
blocked or unfinished and why, which files you touched, and any grounding /
citations as prose. If a hard ENVIRONMENTAL blocker stopped you (a dependency you
cannot install, a host change you may not make, a decision only a human can take),
SAY SO plainly here so the reviewer can route it onward. This report is for the
reviewer; the orchestrator gates your actual work (the committed diff or the
produced artifact), never this file.
</handoff>
'''


def test_worker_prompt_fresh_is_byte_identical_to_oracle() -> None:
    from grindstone.worker import WorkerRequest, build_worker_prompt

    req = WorkerRequest(
        task=_implement(), task_id="E1/T1", mode="implement", scratch=Path("/x"),
    )
    assert build_worker_prompt(req) == _FRESH_IMPLEMENT_ORACLE


def test_worker_prompt_resume_leads_with_fix_frame() -> None:
    from grindstone.worker import WorkerRequest, build_worker_prompt

    req = WorkerRequest(
        task=_implement(), task_id="E1/T1", mode="implement", scratch=Path("/x"),
        prior_work_present=True,
        failure_context=("critic RETRY: the button has no accessible name",),
    )
    prompt = build_worker_prompt(req)
    low = prompt.lower()
    # The DOMINANT instruction is "resume + fix on the same tree", not "build fresh".
    assert prompt.startswith('<resume id="E1/T1">')
    assert "resuming it on the same working tree" in low
    assert "do not rebuild from scratch" in low
    # The full, untruncated failure reason is prominent.
    assert "the button has no accessible name" in prompt
    # The original goal appears ONLY as marked reference context, not the active command.
    assert "<original_task>" in prompt
    assert "for reference, the original brief" in low
    assert "create a.py" in prompt  # the goal text, as reference
    # No leading active <task> command (that is the fresh shape).
    assert not prompt.startswith("<task")
