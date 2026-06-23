"""Per-task agentic verification: at the BUILDER's tier, folded into the task loop.

After a task clears its deterministic floor (handoff + scope + grounding) the core
runs ONE adversarial verification of THAT task against ITS OWN ``criteria``, at the
tier that built it (a ``senior`` task by the senior verifier, every other by the local
one). A FRESH same-tier critic writes ``verdict.json`` (a re-read disk contract, never
stdout). A ``pass=false`` routes into the task's OWN retry ladder as a CHAINABLE failure
(incremental repair, same worktree), anchored to the verifier's prior verdict so the gap
set CONVERGES (it cannot oscillate). The epoch passes iff every task passes; it fails
only when a task exhausts its ladder, and then handle_failed_epoch re-decomposes ONLY the
failed tasks. The pass can only FAIL a task the deterministic floor already cleared.
"""

from __future__ import annotations

import json
from pathlib import Path

from grindstone.contracts.models import (
    CriterionJudgement,
    EpochVerdict,
    ImplementTask,
    parse_epoch_verdict,
)
from grindstone.events import (
    TaskVerificationFailed,
    TaskVerificationPassed,
    TaskVerificationStarted,
    read_events,
)
from grindstone.mock_planner import MockPlanner
from grindstone.rundir import RunDir
from grindstone.run_loop import RunState, run_grind
from grindstone.verify import VerificationError, WorkerTaskVerifier
from grindstone.worker import (
    VERDICT_FILENAME,
    VerificationBrief,
    WorkerRequest,
    build_verification_prompt,
    build_worker_prompt,
)

from tests.grindstone.conftest import (
    OwnershipWorker,
    artifact_decision,
    check_cmd,
    complete_decision,
    handle_failed_epoch_halt,
    handle_failed_epoch_retry,
    implement_decision,
    phase_dict,
    skeleton_decision,
    two_phase_skeleton,
)


# --- a verdict-writing transport (the fake verifier) ---------------------------


class _VerifierWorker:
    """A transport that writes a fixed ``verdict.json`` for a verification request.

    Only acts on a verification request (asserts the brief is set, records the briefs +
    prior-verdict anchors it saw so a test can assert what reached it). Mirrors the disk
    contract: the verdict is the only output channel, the core re-reads + re-validates it.
    """

    def __init__(self, *, passed: bool, gaps: list[str] | None = None, malformed: bool = False) -> None:
        self._passed = passed
        self._gaps = gaps or []
        self._malformed = malformed
        self.seen_briefs: list[VerificationBrief] = []

    def run(self, request: WorkerRequest) -> None:
        assert request.verification is not None, "verifier got a non-verification request"
        self.seen_briefs.append(request.verification)
        verdict_file = request.scratch / VERDICT_FILENAME
        if self._malformed:
            verdict_file.write_text("{ not json", encoding="utf-8")
            return
        payload = {
            "pass": self._passed,
            "per_criterion": [
                {"criterion": c, "met": self._passed, "evidence": "checked the files"}
                for c in request.verification.criteria
            ],
            "gaps": self._gaps,
        }
        verdict_file.write_text(json.dumps(payload), encoding="utf-8")


class _RecordingPlanner:
    """Wrap a planner, capturing every prompt it is handed (for manifest assertions)."""

    def __init__(self, inner: MockPlanner) -> None:
        self._inner = inner
        self.prompts: list[str] = []

    def plan(self, prompt: str, *, workdir: Path | None = None) -> str:
        self.prompts.append(prompt)
        return self._inner.plan(prompt, workdir=workdir)


def _impl_task_with_criteria(tid: str, fname: str, criteria: list[str]) -> dict[str, object]:
    return {
        "id": tid,
        "goal": f"create {fname}",
        "done_when": [check_cmd(f"test -f {fname}")],
        "criteria": criteria,
        "file_ownership": [fname],
    }


def _skeleton_passing_gate() -> dict[str, object]:
    """P1/P2 whose gate passes once f1.txt exists (so the floor clears cleanly)."""

    pending = [check_cmd("test -f f1.txt")]
    return skeleton_decision(
        phase_dict("P1", title="build", exit_criterion=pending, budget=20),
        phase_dict("P2", title="verify", exit_criterion=pending, budget=20),
    )


def _run_state(run_dir: RunDir) -> RunState:
    return RunState.model_validate_json(run_dir.run_state_path.read_text())


# --- 1. the verdict contract (model + schema + parse helper) -------------------


def test_epoch_verdict_parses_pass() -> None:
    v = parse_epoch_verdict(
        {"pass": True, "per_criterion": [{"criterion": "c", "met": True, "evidence": "e"}], "gaps": []}
    )
    assert isinstance(v, EpochVerdict)
    assert v.passed is True
    assert v.per_criterion[0] == CriterionJudgement(criterion="c", met=True, evidence="e")
    assert v.gaps == []


def test_epoch_verdict_parses_fail_with_gaps() -> None:
    v = parse_epoch_verdict(
        {
            "pass": False,
            "per_criterion": [{"criterion": "Pink ramp", "met": False, "evidence": "absent"}],
            "gaps": ["the Lesson screen never maps the Pink ramp"],
        }
    )
    assert v.passed is False
    assert v.gaps == ["the Lesson screen never maps the Pink ramp"]


def test_epoch_verdict_rejects_malformed() -> None:
    import pytest

    with pytest.raises(ValueError):
        parse_epoch_verdict({"pass": True})
    with pytest.raises(ValueError):
        parse_epoch_verdict({"pass": "yes", "per_criterion": [], "gaps": []})
    with pytest.raises(ValueError):
        parse_epoch_verdict({"pass": True, "per_criterion": [], "gaps": [], "x": 1})


def test_overlong_evidence_is_preserved_in_full() -> None:
    long_evidence = "x" * 20_000
    v = parse_epoch_verdict(
        {
            "pass": True,
            "per_criterion": [{"criterion": "c", "met": True, "evidence": long_evidence}],
            "gaps": [],
        }
    )
    assert v.passed is True
    assert v.per_criterion[0].evidence == long_evidence


def test_overlong_criterion_and_gaps_preserved_in_full() -> None:
    long_criterion = "c" * 4000
    long_gap = "g" * 4000
    v = parse_epoch_verdict(
        {
            "pass": False,
            "per_criterion": [{"criterion": long_criterion, "met": False, "evidence": "e"}],
            "gaps": [long_gap],
        }
    )
    assert v.passed is False
    assert v.per_criterion[0].criterion == long_criterion
    assert v.gaps[0] == long_gap


def test_epoch_verdict_parses_with_digest() -> None:
    v = parse_epoch_verdict(
        {
            "pass": True,
            "per_criterion": [{"criterion": "c", "met": True, "evidence": "e"}],
            "gaps": [],
            "digest": "added lesson.tsx mapping every ramp; theming still TODO",
        }
    )
    assert v.passed is True
    assert v.digest == "added lesson.tsx mapping every ramp; theming still TODO"


def test_epoch_verdict_digest_defaults_empty_when_absent() -> None:
    v = parse_epoch_verdict({"pass": False, "per_criterion": [], "gaps": ["nope"]})
    assert v.digest == ""
    assert v.passed is False


# --- 2. the verification prompt (adversarial skill + convergence anchor) --------


def _impl_task_obj() -> "ImplementTask":
    from grindstone.contracts.models import CmdCheck, ImplementTask

    return ImplementTask(
        id="T1", goal="g", done_when=[CmdCheck(cmd="true")], file_ownership=["**"]
    )


def test_build_verification_prompt_is_adversarial_and_disk_contract(tmp_path: Path) -> None:
    brief = VerificationBrief(
        epoch_goal="map every ramp",
        criteria=["the Lesson screen maps the Pink ramp"],
        artifacts=["T1: created lesson.tsx"],
    )
    req = WorkerRequest(
        task=_impl_task_obj(), task_id="P1/E1/verify", inputs={}, scratch=tmp_path,
        attempt=1, failure_context=[], mode="review", verification=brief,
    )
    prompt = build_verification_prompt(req, brief)
    assert "ADVERSARIAL" in prompt
    assert "the Lesson screen maps the Pink ramp" in prompt
    assert "DEFAULT TO FAIL" in prompt
    assert VERDICT_FILENAME in prompt
    assert "digest" in prompt
    # No prior verdict on the FIRST verification: no convergence-anchor block.
    assert "<prior_verdict>" not in prompt
    # The shared dispatcher routes verification requests to this prompt.
    assert "ADVERSARIAL" in build_worker_prompt(req)


def test_prompt_renders_prior_verdict_anchor_on_reverification(tmp_path: Path) -> None:
    """On a re-verification the prompt carries the verifier's OWN prior verdict as the
    convergence anchor: the passed criteria (regression-check) and the failed criteria +
    gaps (confirm closed), and an explicit instruction that the gap set must SHRINK."""

    prior = parse_epoch_verdict(
        {
            "pass": False,
            "per_criterion": [
                {"criterion": "tokens exported", "met": True, "evidence": "saw export"},
                {"criterion": "Pink ramp mapped", "met": False, "evidence": "absent"},
            ],
            "gaps": ["the Pink ramp is never mapped"],
        }
    )
    brief = VerificationBrief(
        epoch_goal="map every ramp",
        criteria=["tokens exported", "Pink ramp mapped"],
        artifacts=["T1: revised"],
        prior_verdict=prior,
    )
    req = WorkerRequest(
        task=_impl_task_obj(), task_id="P1/E1/verify", inputs={}, scratch=tmp_path,
        attempt=2, failure_context=[], mode="review", verification=brief,
    )
    prompt = build_verification_prompt(req, brief)
    assert "<prior_verdict>" in prompt
    # The previously-PASSED criterion is named under the regression-check list.
    assert "tokens exported" in prompt
    # The previously-FAILED criterion + its gap are named under the confirm-closed list.
    assert "Pink ramp mapped" in prompt
    assert "the Pink ramp is never mapped" in prompt
    # The convergence instruction: the gap set must not grow / must be a subset.
    assert "shrink" in prompt.lower() or "subset" in prompt.lower()


# --- 3. the WorkerTaskVerifier adapter (disk-contract enforcement) -------------


def test_verifier_adapter_rejects_malformed_verdict(tmp_path: Path) -> None:
    import pytest

    verifier = WorkerTaskVerifier(_VerifierWorker(passed=True, malformed=True))
    brief = VerificationBrief(epoch_goal="g", criteria=["c"], artifacts=[])
    with pytest.raises(VerificationError):
        verifier.verify(
            scratch=tmp_path, brief=brief, task_id="P1/E1/T1/verify",
            verdict_dest=tmp_path / "dest" / "verdict.json",
        )


def test_verifier_adapter_fails_when_no_verdict(tmp_path: Path) -> None:
    import pytest

    class _Silent:
        def run(self, request: WorkerRequest) -> None:
            return  # writes no verdict.json

    with pytest.raises(VerificationError):
        WorkerTaskVerifier(_Silent()).verify(
            scratch=tmp_path,
            brief=VerificationBrief(epoch_goal="g", criteria=["c"], artifacts=[]),
            task_id="P1/E1/T1/verify",
            verdict_dest=tmp_path / "dest" / "verdict.json",
        )


def test_verifier_adapter_relocates_full_verdict_by_reference(tmp_path: Path) -> None:
    """The adapter persists the FULL verdict to ``verdict_dest`` (so it survives the
    worktree teardown and reaches the planner BY REFERENCE), preserving long content in
    full, and REMOVES the scratch verdict.json so a chained retry never inherits it."""

    scratch = tmp_path / "wt"
    scratch.mkdir()
    long_evidence = "E" * 20_000
    payload = {
        "pass": True,
        "per_criterion": [{"criterion": "c", "met": True, "evidence": long_evidence}],
        "gaps": [],
        "digest": "D" * 9000,
    }
    (scratch / VERDICT_FILENAME).write_text(json.dumps(payload), encoding="utf-8")

    class _Echo:
        def run(self, request: WorkerRequest) -> None:
            return  # the verdict file is already in place

    dest = tmp_path / "P1" / "E1" / "T1" / "verdict.json"
    verdict = WorkerTaskVerifier(_Echo()).verify(
        scratch=scratch,
        brief=VerificationBrief(epoch_goal="g", criteria=["c"], artifacts=[]),
        task_id="P1/E1/T1/verify",
        verdict_dest=dest,
    )
    assert verdict.per_criterion[0].evidence == long_evidence
    relocated = json.loads(dest.read_text(encoding="utf-8"))
    assert relocated["per_criterion"][0]["evidence"] == long_evidence
    assert relocated["digest"] == "D" * 9000
    # The scratch copy is removed so a chained retry's commit never picks it up.
    assert not (scratch / VERDICT_FILENAME).exists()


def test_verifier_adapter_rejects_oversized_verdict_file(tmp_path: Path) -> None:
    import pytest

    from grindstone.verify import VERDICT_MAX_BYTES

    scratch = tmp_path / "wt"
    scratch.mkdir()
    blob = "x" * (VERDICT_MAX_BYTES + 1024)
    payload = {
        "pass": True,
        "per_criterion": [{"criterion": "c", "met": True, "evidence": blob}],
        "gaps": [],
    }
    (scratch / VERDICT_FILENAME).write_text(json.dumps(payload), encoding="utf-8")

    class _Echo:
        def run(self, request: WorkerRequest) -> None:
            return

    with pytest.raises(VerificationError):
        WorkerTaskVerifier(_Echo()).verify(
            scratch=scratch,
            brief=VerificationBrief(epoch_goal="g", criteria=["c"], artifacts=[]),
            task_id="P1/E1/T1/verify",
            verdict_dest=tmp_path / "dest" / "verdict.json",
        )

    sane = json.dumps(
        {
            "pass": True,
            "per_criterion": [{"criterion": "c", "met": True, "evidence": "y" * 50_000}],
            "gaps": [],
        }
    )
    (scratch / VERDICT_FILENAME).write_text(sane, encoding="utf-8")
    dest = tmp_path / "dest2" / "verdict.json"
    v = WorkerTaskVerifier(_Echo()).verify(
        scratch=scratch,
        brief=VerificationBrief(epoch_goal="g", criteria=["c"], artifacts=[]),
        task_id="P1/E1/T1/verify",
        verdict_dest=dest,
    )
    assert len(v.per_criterion[0].evidence) == 50_000
    assert dest.is_file()


# --- 4. the pass runs per task, AFTER the floor, only-fail ---------------------


def test_task_with_criteria_pass_completes(git_repo: Path, run_dir: RunDir) -> None:
    """A task with criteria + a verifier returning pass=true verifies and the run
    proceeds to completion. The verification pass ran (task started + passed events)."""

    verifier = WorkerTaskVerifier(_VerifierWorker(passed=True))
    planner = MockPlanner(
        script=[
            _skeleton_passing_gate(),
            implement_decision(_impl_task_with_criteria("T1", "f1.txt", ["f1 exists and is right"])),
            complete_decision(check_cmd("test -f f1.txt")),
        ]
    )
    outcome = run_grind(
        run_dir, job_path="job.md", planner=planner,
        ladder=[("worker", OwnershipWorker())], repo=git_repo,
        verifiers={"worker": verifier},
    )
    assert outcome.status == "completed"
    kinds = [e.event for e in read_events(run_dir.events_path)]
    assert "task_verification_started" in kinds
    assert "task_verification_passed" in kinds
    assert "task_verification_failed" not in kinds
    assert "epoch_failed" not in kinds
    fake = verifier._transport  # type: ignore[attr-defined]
    assert isinstance(fake, _VerifierWorker)
    assert fake.seen_briefs[0].criteria == ["f1 exists and is right"]


def test_verdict_persists_to_keyed_log(git_repo: Path, run_dir: RunDir) -> None:
    """The FULL verdict persists to a stable ``P*/E*/T*/verdict.json`` keyed-log path
    (so it shows in the run-dir log index, delivered by reference)."""

    digest = "added f1.txt; theming still a TODO"
    evidence = "read f1.txt " + "Q" * 9000  # long, untruncated

    class _DigestVerifier:
        def run(self, request: WorkerRequest) -> None:
            assert request.verification is not None
            payload = {
                "pass": True,
                "per_criterion": [
                    {"criterion": c, "met": True, "evidence": evidence}
                    for c in request.verification.criteria
                ],
                "gaps": [],
                "digest": digest,
            }
            (request.scratch / VERDICT_FILENAME).write_text(json.dumps(payload), encoding="utf-8")

    verifier = WorkerTaskVerifier(_DigestVerifier())
    planner = MockPlanner(
        script=[
            _skeleton_passing_gate(),
            implement_decision(_impl_task_with_criteria("T1", "f1.txt", ["c"])),
            complete_decision(check_cmd("test -f f1.txt")),
        ]
    )
    outcome = run_grind(
        run_dir, job_path="job.md", planner=planner,
        ladder=[("worker", OwnershipWorker())], repo=git_repo,
        verifiers={"worker": verifier},
    )
    assert outcome.status == "completed"
    verdict_keys = [k for k in run_dir.log_index() if k.endswith("verdict.json")]
    assert verdict_keys, "the verified task must persist a verdict.json log key"
    relocated = json.loads(run_dir.resolve(verdict_keys[0]).read_text(encoding="utf-8"))
    assert relocated["digest"] == digest
    assert relocated["per_criterion"][0]["evidence"] == evidence


def test_no_criteria_skips_the_pass(git_repo: Path, run_dir: RunDir) -> None:
    """A task with NO criteria never runs the verification pass (skipped entirely), even
    with a verifier wired. The verifier is never invoked."""

    fake = _VerifierWorker(passed=False, gaps=["should never be seen"])
    verifier = WorkerTaskVerifier(fake)
    planner = MockPlanner(
        script=[
            _skeleton_passing_gate(),
            implement_decision({"id": "T1", "goal": "create f1.txt",
                                "done_when": [check_cmd("test -f f1.txt")],
                                "file_ownership": ["f1.txt"]}),
            complete_decision(check_cmd("test -f f1.txt")),
        ]
    )
    outcome = run_grind(
        run_dir, job_path="job.md", planner=planner,
        ladder=[("worker", OwnershipWorker())], repo=git_repo,
        verifiers={"worker": verifier},
    )
    assert outcome.status == "completed"
    kinds = [e.event for e in read_events(run_dir.events_path)]
    assert "task_verification_started" not in kinds
    assert fake.seen_briefs == []


def test_no_verifier_skips_the_pass(git_repo: Path, run_dir: RunDir) -> None:
    """With no verifier wired (verify_epochs off / no tier) a task with criteria
    completes without the pass; the deterministic floor still gates."""

    planner = MockPlanner(
        script=[
            _skeleton_passing_gate(),
            implement_decision(_impl_task_with_criteria("T1", "f1.txt", ["c"])),
            complete_decision(check_cmd("test -f f1.txt")),
        ]
    )
    outcome = run_grind(
        run_dir, job_path="job.md", planner=planner,
        ladder=[("worker", OwnershipWorker())], repo=git_repo, verifiers=None,
    )
    assert outcome.status == "completed"
    kinds = [e.event for e in read_events(run_dir.events_path)]
    assert "task_verification_started" not in kinds


def test_pass_runs_after_floor_not_when_task_floor_fails(git_repo: Path, run_dir: RunDir) -> None:
    """The deterministic floor is FIRST: when a task fails its floor (no handoff) the
    verification pass does NOT run for that attempt. The agentic pass can only fire on an
    attempt whose floor cleared."""

    from tests.grindstone.conftest import FailingWorker

    fake = _VerifierWorker(passed=True)  # would pass if asked, must never be asked
    verifier = WorkerTaskVerifier(fake)
    planner = MockPlanner(
        script=[
            two_phase_skeleton(),
            implement_decision(_impl_task_with_criteria("T1", "f1.txt", ["c"])),  # fails (no handoff)
            handle_failed_epoch_halt("the floor failed, not the semantics"),
        ]
    )
    outcome = run_grind(
        run_dir, job_path="job.md", planner=planner,
        ladder=[("worker", FailingWorker())], repo=git_repo,
        verifiers={"worker": verifier}, tier0_attempts=1,
    )
    assert outcome.status == "escalated"
    kinds = [e.event for e in read_events(run_dir.events_path)]
    assert "epoch_failed" in kinds  # the floor-failure path
    assert "task_verification_started" not in kinds
    assert fake.seen_briefs == []


# --- 5. (a) tier routing: senior task -> senior verifier, local -> local --------


def test_senior_task_verified_by_senior_local_task_by_local(git_repo: Path, run_dir: RunDir) -> None:
    """Each task is verified at the tier that BUILT it: a ``senior`` task by the senior
    verifier, a local task by the local one. A ladder with both tiers + distinct
    verifiers proves the routing (each verifier sees only its own task's criteria)."""

    local_v = _VerifierWorker(passed=True)
    senior_v = _VerifierWorker(passed=True)
    epoch: dict[str, object] = {
        "schema_version": "1", "tool": "implement",
        "args": {
            "epoch_title": "mixed", "rationale": "a local slice and a senior slice",
            "tasks": [
                {"id": "T1", "goal": "create f1.txt", "done_when": [check_cmd("test -f f1.txt")],
                 "criteria": ["f1 is mechanical-correct"], "file_ownership": ["f1.txt"]},
                {"id": "T2", "goal": "create f2.txt", "done_when": [check_cmd("test -f f2.txt")],
                 "criteria": ["f2 is tasteful"], "file_ownership": ["f2.txt"], "senior": True},
            ],
        },
    }
    gate = [check_cmd("test -f f1.txt"), check_cmd("test -f f2.txt")]
    planner = MockPlanner(
        script=[
            skeleton_decision(
                phase_dict("P1", title="build", exit_criterion=gate, budget=20),
                phase_dict("P2", title="done", exit_criterion=gate, budget=20),
            ),
            epoch,
            complete_decision(check_cmd("test -f f1.txt")),
        ]
    )
    outcome = run_grind(
        run_dir, job_path="job.md", planner=planner,
        ladder=[("worker", OwnershipWorker()), ("senior", OwnershipWorker())],
        repo=git_repo,
        verifiers={"worker": WorkerTaskVerifier(local_v), "senior": WorkerTaskVerifier(senior_v)},
    )
    assert outcome.status == "completed"
    # The LOCAL verifier saw ONLY the local task's criterion; the SENIOR verifier saw
    # ONLY the senior task's criterion. The builder tier picked the verifier tier.
    assert [b.criteria for b in local_v.seen_briefs] == [["f1 is mechanical-correct"]]
    assert [b.criteria for b in senior_v.seen_briefs] == [["f2 is tasteful"]]


# --- 5. (b)+(c) failed verification -> incremental retry, convergent anchor ------


class _GapThenPass:
    """Verifier: fail the first verdict (gap), pass on every later one. Records the
    prior-verdict anchor it saw on each call so a test can assert convergence."""

    def __init__(self) -> None:
        self.calls = 0
        self.seen_prior: list[EpochVerdict | None] = []

    def run(self, request: WorkerRequest) -> None:
        assert request.verification is not None
        self.calls += 1
        self.seen_prior.append(request.verification.prior_verdict)
        passed = self.calls > 1
        payload = {
            "pass": passed,
            "per_criterion": [
                {"criterion": c, "met": passed, "evidence": "e"}
                for c in request.verification.criteria
            ],
            "gaps": [] if passed else ["the Pink ramp is still unmapped"],
        }
        (request.scratch / VERDICT_FILENAME).write_text(json.dumps(payload), encoding="utf-8")


class _UniqueContentWorker:
    """An OwnershipWorker variant whose content changes each call, so a chained retry
    produces a NON-zero diff vs the epoch base (the floor's zero-diff reject is unrelated
    to the verification routing)."""

    def __init__(self) -> None:
        self._calls = 0

    def run(self, request: WorkerRequest) -> None:
        self._calls += 1
        OwnershipWorker(content=f"v{self._calls}\n").run(request)


def test_failed_verification_triggers_incremental_retry_and_converges(
    git_repo: Path, run_dir: RunDir
) -> None:
    """(b)+(c): a failed per-task verification chains an INCREMENTAL retry (same task,
    inside its OWN ladder, no planner round-trip), and the re-verification receives the
    verifier's PRIOR verdict as the convergence anchor (the first call had none). The
    second pass closes the gap and the run completes."""

    fake = _GapThenPass()
    planner = MockPlanner(
        script=[
            _skeleton_passing_gate(),
            implement_decision(_impl_task_with_criteria("T1", "f1.txt", ["map the Pink ramp"])),
            complete_decision(check_cmd("test -f f1.txt")),
        ]
    )
    outcome = run_grind(
        run_dir, job_path="job.md", planner=planner,
        ladder=[("worker", _UniqueContentWorker())], repo=git_repo,
        verifiers={"worker": WorkerTaskVerifier(fake)},
    )
    assert outcome.status == "completed"
    kinds = [e.event for e in read_events(run_dir.events_path)]
    # The retry was INCREMENTAL (a task_retried inside the same task's ladder), NOT a
    # planner handle_failed_epoch round-trip.
    assert "task_retried" in kinds
    assert "failed_epoch_handled" not in kinds
    assert "epoch_failed" not in kinds
    assert "task_verification_failed" in kinds and "task_verification_passed" in kinds
    # Convergence anchor: the FIRST verification had no prior; the SECOND carried the
    # verifier's prior (failing) verdict, so the gap set is anchored, not re-litigated.
    assert fake.calls == 2
    assert fake.seen_prior[0] is None
    assert fake.seen_prior[1] is not None
    assert fake.seen_prior[1].passed is False
    assert fake.seen_prior[1].gaps == ["the Pink ramp is still unmapped"]


# --- 5. (d)+(e) epoch fails only when a task exhausts its ladder -----------------


def test_unsatisfiable_verification_fails_task_then_epoch_routes_to_planner(
    git_repo: Path, run_dir: RunDir
) -> None:
    """(d)+(e): a verification that ALWAYS fails exhausts the task's ladder, FAILS the
    task (and thus the epoch), and routes through handle_failed_epoch with the gap on the
    failed task's reason. A halt terminates the run. The epoch fails ONLY because a task
    exhausted its ladder (no separate epoch-level re-judge)."""

    gap = "the Pink ramp is never mapped"
    verifier = WorkerTaskVerifier(_VerifierWorker(passed=False, gaps=[gap]))
    planner = _RecordingPlanner(
        MockPlanner(
            script=[
                _skeleton_passing_gate(),
                implement_decision(_impl_task_with_criteria("T1", "f1.txt", ["map the Pink ramp"])),
                handle_failed_epoch_halt("incomplete per the gap"),
            ]
        )
    )
    outcome = run_grind(
        run_dir, job_path="job.md", planner=planner,
        ladder=[("worker", _UniqueContentWorker())], repo=git_repo,
        verifiers={"worker": verifier}, tier0_attempts=2,
    )
    assert outcome.status == "escalated"
    kinds = [e.event for e in read_events(run_dir.events_path)]
    assert "epoch_failed" in kinds  # a task exhausted its ladder -> the epoch failed
    # The gap rode the failed task's reason and reached the planner's failed-epoch block
    # (the <semantic_gaps> block, derived from the verification-failed task).
    failed_prompt = planner.prompts[-1]
    assert gap in failed_prompt
    assert "handle_failed_epoch" in failed_prompt


def test_one_task_fails_verification_other_passes_epoch_fails_only_for_failed(
    git_repo: Path, run_dir: RunDir
) -> None:
    """The epoch passes iff ALL tasks pass: with two tasks where one's verification
    always fails and the other's passes, the epoch FAILS, and the failed-epoch context
    names ONLY the failed task (the passing task stays DONE, never re-dispatched)."""

    epoch: dict[str, object] = {
        "schema_version": "1", "tool": "implement",
        "args": {
            "epoch_title": "two", "rationale": "one passes, one fails verification",
            "tasks": [
                {"id": "T1", "goal": "create f1.txt", "done_when": [check_cmd("test -f f1.txt")],
                 "criteria": ["f1 ok"], "file_ownership": ["f1.txt"]},
                {"id": "T2", "goal": "create f2.txt", "done_when": [check_cmd("test -f f2.txt")],
                 "criteria": ["f2 ok"], "file_ownership": ["f2.txt"]},
            ],
        },
    }
    gate = [check_cmd("test -f f1.txt")]
    planner = _RecordingPlanner(
        MockPlanner(
            script=[
                skeleton_decision(
                    phase_dict("P1", title="build", exit_criterion=gate, budget=20),
                    phase_dict("P2", title="done", exit_criterion=gate, budget=20),
                ),
                epoch,
                handle_failed_epoch_halt("T2 never satisfied its criterion"),
            ]
        )
    )

    class _PerTaskVerifier:
        """Pass T1's criterion, always fail T2's (keyed off the criterion text)."""

        def run(self, request: WorkerRequest) -> None:
            assert request.verification is not None
            passed = "f2 ok" not in request.verification.criteria
            payload = {
                "pass": passed,
                "per_criterion": [
                    {"criterion": c, "met": passed, "evidence": "e"}
                    for c in request.verification.criteria
                ],
                "gaps": [] if passed else ["T2 criterion unmet"],
            }
            (request.scratch / VERDICT_FILENAME).write_text(json.dumps(payload), encoding="utf-8")

    outcome = run_grind(
        run_dir, job_path="job.md", planner=planner,
        ladder=[("worker", _UniqueContentWorker())], repo=git_repo,
        verifiers={"worker": WorkerTaskVerifier(_PerTaskVerifier())}, tier0_attempts=1,
    )
    assert outcome.status == "escalated"
    # The pending failed-epoch context (the planner's input) names ONLY T2 as failed.
    failed_prompt = planner.prompts[-1]
    assert "T2" in failed_prompt
    pending = _run_state(run_dir)  # state cleared at halt, but the prompt proves scoping
    assert pending.pending_failed_epoch is None  # the halt cleared it


# --- relocated artifact_out deliverable reaches the task's verifier --------------


class _ContentArtifactWorker:
    """An artifact-mode worker that writes a fixed body into ``artifact_out`` + a DONE
    handoff in its scratch. The handoff state does NOT echo the body, so a test asserting
    the BODY reached the verifier proves it came from the file (by reference), not the state."""

    def __init__(self, body: str) -> None:
        self._body = body

    def run(self, request: WorkerRequest) -> None:
        from grindstone.contracts.models import ArtifactTask

        task = request.task
        assert isinstance(task, ArtifactTask)
        out = request.scratch / task.artifact_out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(self._body, encoding="utf-8")
        payload = {
            "schema_version": "1",
            "task_id": request.task_id,
            "status": "DONE",
            "what_changed": [{"kind": "artifact", "ref": task.artifact_out}],
            "resulting_state": "produced the artifact",
            "downstream_needs": [],
            "not_done": [],
            "citations": [{"file": task.artifact_out}],
            "checks": [{"check": "artifact", "exit_code": 0}],
            "occupancy": {"compacted": False, "subagent_splits": 0},
        }
        (request.scratch / "handoff.json").write_text(json.dumps(payload), encoding="utf-8")


def _read_artifact_paths(brief: VerificationBrief) -> str:
    bodies: list[str] = []
    for line in brief.artifacts:
        marker = "at this path): "
        if marker in line:
            path = Path(line.rsplit(marker, 1)[1].strip())
            if path.is_file():
                bodies.append(path.read_text(encoding="utf-8"))
    return "\n".join(bodies)


class _ContentVerifier:
    """A verifier that PASSES iff a required marker appears in the artifact FILE the brief
    points at (proving the deliverable's CONTENT reached the verifier BY REFERENCE)."""

    def __init__(self, *, marker: str) -> None:
        self._marker = marker
        self.seen_briefs: list[VerificationBrief] = []

    def run(self, request: WorkerRequest) -> None:
        assert request.verification is not None
        self.seen_briefs.append(request.verification)
        body = _read_artifact_paths(request.verification)
        passed = self._marker in body
        payload = {
            "pass": passed,
            "per_criterion": [
                {"criterion": c, "met": passed, "evidence": "read the artifact"}
                for c in request.verification.criteria
            ],
            "gaps": [] if passed else ["the deliverable does not exist anywhere"],
        }
        (request.scratch / VERDICT_FILENAME).write_text(json.dumps(payload), encoding="utf-8")


def _artifact_task_with_criteria(tid: str, out: str, criteria: list[str]) -> dict[str, object]:
    return {
        "id": tid,
        "goal": f"produce {out}",
        "done_when": [check_cmd("true")],
        "criteria": criteria,
        "artifact_out": out,
    }


def _skeleton_passing_gate_true() -> dict[str, object]:
    pending = [check_cmd("true")]
    return skeleton_decision(
        phase_dict("P1", title="build", exit_criterion=pending, budget=20),
        phase_dict("P2", title="verify", exit_criterion=pending, budget=20),
    )


def test_research_artifact_content_reaches_verifier_and_completes(
    git_repo: Path, run_dir: RunDir
) -> None:
    """A research/artifact task whose deliverable is its ``artifact_out`` file is judged
    against that file's ACTUAL content: the verifier reads the body (by reference) and a
    pass completes."""

    marker = "TOKEN-FLOW-CONTENT-MARKER"
    body = f"# token flow\n\n{marker}\n\nthe mapping is complete.\n"
    verifier = WorkerTaskVerifier(_ContentVerifier(marker=marker))
    out_key = "P1/E1/T1/token_flow_mapping.md"
    planner = MockPlanner(
        script=[
            _skeleton_passing_gate_true(),
            artifact_decision(_artifact_task_with_criteria("T1", out_key, ["the token flow is mapped"])),
            complete_decision(check_cmd("true")),
        ]
    )
    outcome = run_grind(
        run_dir, job_path="job.md", planner=planner,
        ladder=[("worker", _ContentArtifactWorker(body))], repo=git_repo,
        verifiers={"worker": verifier},
    )
    assert outcome.status == "completed"
    kinds = [e.event for e in read_events(run_dir.events_path)]
    assert "task_verification_passed" in kinds
    assert "task_verification_failed" not in kinds
    fake = verifier._transport  # type: ignore[attr-defined]
    assert isinstance(fake, _ContentVerifier)
    brief = fake.seen_briefs[0]
    joined = "\n".join(brief.artifacts)
    # The brief carries the PATH, not the embedded body.
    assert marker not in joined
    assert marker in _read_artifact_paths(brief)


def test_artifact_real_semantic_gap_fails_task(git_repo: Path, run_dir: RunDir) -> None:
    """Reading the artifact does NOT rubber-stamp: a verifier that finds a genuine gap
    fails the task (exhausting its ladder) and the epoch routes to handle_failed_epoch."""

    out_key = "P1/E1/T1/note.md"
    verifier = WorkerTaskVerifier(_ContentVerifier(marker="REQUIRED-BUT-ABSENT"))
    planner = MockPlanner(
        script=[
            _skeleton_passing_gate_true(),
            artifact_decision(_artifact_task_with_criteria("T1", out_key, ["covers every case"])),
            handle_failed_epoch_halt("the deliverable has a real gap"),
        ]
    )
    outcome = run_grind(
        run_dir, job_path="job.md", planner=planner,
        ladder=[("worker", _ContentArtifactWorker("an incomplete note\n"))],
        repo=git_repo, verifiers={"worker": verifier}, tier0_attempts=1,
    )
    assert outcome.status == "escalated"
    kinds = [e.event for e in read_events(run_dir.events_path)]
    assert "task_verification_failed" in kinds
    assert "epoch_failed" in kinds


def test_huge_artifact_is_delivered_by_reference_not_embedded(
    git_repo: Path, run_dir: RunDir
) -> None:
    """A HUGE artifact is delivered to the verifier BY REFERENCE (a path), so the brief
    stays small regardless of size, and the FULL content is on disk for the verifier."""

    out_key = "P1/E1/T1/huge.md"
    huge_marker = "HUGE-BODY-MARKER"
    huge = "Z" * (2 * 1024 * 1024) + huge_marker + "\n"
    captured: list[VerificationBrief] = []

    class _Capture:
        def run(self, request: WorkerRequest) -> None:
            assert request.verification is not None
            captured.append(request.verification)
            payload = {
                "pass": True,
                "per_criterion": [
                    {"criterion": c, "met": True, "evidence": "e"}
                    for c in request.verification.criteria
                ],
                "gaps": [],
            }
            (request.scratch / VERDICT_FILENAME).write_text(json.dumps(payload), encoding="utf-8")

    verifier = WorkerTaskVerifier(_Capture())
    planner = MockPlanner(
        script=[
            _skeleton_passing_gate_true(),
            artifact_decision(_artifact_task_with_criteria("T1", out_key, ["the artifact is produced"])),
            complete_decision(check_cmd("true")),
        ]
    )
    outcome = run_grind(
        run_dir, job_path="job.md", planner=planner,
        ladder=[("worker", _ContentArtifactWorker(huge))], repo=git_repo,
        verifiers={"worker": verifier},
    )
    assert outcome.status == "completed"
    assert captured
    joined = "\n".join(captured[0].artifacts)
    assert len(joined) < 4096  # tiny brief even for a megabyte deliverable
    assert huge_marker not in joined
    body = _read_artifact_paths(captured[0])
    assert huge_marker in body
    assert len(body) > 2 * 1024 * 1024
