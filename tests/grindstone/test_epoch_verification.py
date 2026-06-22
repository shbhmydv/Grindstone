"""G4 end-of-epoch agentic verification pass + semantic-fail routing.

The third gate-rebalance verification source (after the deterministic floor and the
planner's structural checks): natural-language ``criteria`` judged by a local-tier
ADVERSARIAL pass that writes ``verdict.json`` (a re-read disk contract, never stdout).
A pass that fails routes the epoch through the SAME B6 failed-epoch machinery, so the
planner sees the gaps and disposes (retry / escalate_senior / halt). The pass can
only FAIL an epoch the deterministic floor already cleared, never rubber-stamp it.
"""

from __future__ import annotations

import json
from pathlib import Path

from grindstone.config import PrepareConfig
from grindstone.contracts.models import (
    CriterionJudgement,
    EpochVerdict,
    parse_epoch_verdict,
)
from grindstone.events import (
    EpochVerificationFailed,
    EpochVerificationPassed,
    EpochVerificationStarted,
    read_events,
)
from grindstone.mock_planner import MockPlanner
from grindstone.planner import FailedEpochInfo, volatile_tail
from grindstone.rundir import RunDir
from grindstone.run_loop import RunState, resume_grind, run_grind
from grindstone.verify import VerificationError, WorkerEpochVerifier
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


# --- a verdict-writing local transport (the fake verifier) ---------------------


class _VerifierWorker:
    """A local-tier transport that writes a fixed ``verdict.json`` for the G4 pass.

    Only acts on a verification request (asserts the brief is set, records the briefs
    it saw so a test can assert the criteria reached it). Mirrors the disk contract:
    the verdict is the only output channel, the core re-reads + re-validates it.
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

    # Missing required keys.
    with pytest.raises(ValueError):
        parse_epoch_verdict({"pass": True})
    # A stringy pass is rejected (StrictBool).
    with pytest.raises(ValueError):
        parse_epoch_verdict({"pass": "yes", "per_criterion": [], "gaps": []})
    # Unknown key (extra=forbid).
    with pytest.raises(ValueError):
        parse_epoch_verdict({"pass": True, "per_criterion": [], "gaps": [], "x": 1})


def test_overlong_evidence_is_preserved_in_full() -> None:
    """The principle: a verdict is an agent INPUT delivered by reference (persisted on
    disk, the planner reads it), never byte-capped-and-embedded into a prompt. So a
    verbose verifier's ``evidence`` now PARSES and is preserved IN FULL, no truncation,
    no rejection (the old whole-verdict-rejection bug is impossible)."""

    long_evidence = "x" * 20_000
    v = parse_epoch_verdict(
        {
            "pass": True,
            "per_criterion": [
                {"criterion": "c", "met": True, "evidence": long_evidence}
            ],
            "gaps": [],
        }
    )
    assert v.passed is True
    assert v.per_criterion[0].evidence == long_evidence  # full content, untruncated


def test_overlong_criterion_and_gaps_preserved_in_full() -> None:
    """The other free-text strings (``criterion``, ``gaps[]``) are likewise preserved in
    full: no length cap, no truncation marker, the verdict parses cleanly."""

    long_criterion = "c" * 4000
    long_gap = "g" * 4000
    v = parse_epoch_verdict(
        {
            "pass": False,
            "per_criterion": [
                {"criterion": long_criterion, "met": False, "evidence": "e"}
            ],
            "gaps": [long_gap],
        }
    )
    assert v.passed is False
    assert v.per_criterion[0].criterion == long_criterion
    assert v.gaps[0] == long_gap


def test_in_bound_evidence_is_untouched() -> None:
    """A within-bound evidence string passes through verbatim (no spurious marker)."""

    v = parse_epoch_verdict(
        {
            "pass": True,
            "per_criterion": [{"criterion": "c", "met": True, "evidence": "fine"}],
            "gaps": [],
        }
    )
    assert v.per_criterion[0].evidence == "fine"


# --- G10 the steering digest (descriptive, never a gate) -----------------------


def test_epoch_verdict_parses_with_digest() -> None:
    """The verifier emits a free-text ``digest`` alongside the judgement; it is read
    out verbatim and does NOT affect pass/fail (a descriptive steering summary)."""

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
    """An older/malformed verdict with no ``digest`` parses, defaulting to "" (never
    crashes); pass/fail is unchanged."""

    v = parse_epoch_verdict(
        {"pass": False, "per_criterion": [], "gaps": ["nope"]}
    )
    assert v.digest == ""
    assert v.passed is False


def test_overlong_digest_preserved_in_full() -> None:
    """A verbose ``digest`` parses and is preserved in full (it reaches the planner by
    reference via the persisted verdict file, never embedded), no cap, no truncation."""

    long_digest = "d" * 20_000
    v = parse_epoch_verdict(
        {
            "pass": True,
            "per_criterion": [{"criterion": "c", "met": True, "evidence": "e"}],
            "gaps": [],
            "digest": long_digest,
        }
    )
    assert v.passed is True
    assert v.digest == long_digest  # full content, untruncated


# --- 2. the verification prompt (adversarial skill, disk contract) -------------


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
    # G10: the verifier is ALSO asked to emit a descriptive steering digest (a factual
    # summary for the planner deciding the NEXT epoch), distinct from the pass/fail.
    assert "digest" in prompt
    # The shared dispatcher routes verification requests to this prompt.
    assert "ADVERSARIAL" in build_worker_prompt(req)


def _impl_task_obj() -> object:
    from grindstone.contracts.models import CmdCheck, ImplementTask

    return ImplementTask(
        id="T1", goal="g", done_when=[CmdCheck(cmd="true")], file_ownership=["**"]
    )


# --- the WorkerEpochVerifier adapter (disk-contract enforcement) ---------------


def test_verifier_adapter_rejects_malformed_verdict(tmp_path: Path) -> None:
    import pytest

    verifier = WorkerEpochVerifier(_VerifierWorker(passed=True, malformed=True))
    brief = VerificationBrief(epoch_goal="g", criteria=["c"], artifacts=[])
    with pytest.raises(VerificationError):
        verifier.verify(
            worktree=tmp_path, brief=brief, task_id="P1/E1/verify",
            verdict_dest=tmp_path / "dest" / "verdict.json",
        )


def test_verifier_adapter_fails_when_no_verdict(tmp_path: Path) -> None:
    import pytest

    class _Silent:
        def run(self, request: WorkerRequest) -> None:
            return  # writes no verdict.json

    with pytest.raises(VerificationError):
        WorkerEpochVerifier(_Silent()).verify(
            worktree=tmp_path,
            brief=VerificationBrief(epoch_goal="g", criteria=["c"], artifacts=[]),
            task_id="P1/E1/verify",
            verdict_dest=tmp_path / "dest" / "verdict.json",
        )


def test_verifier_adapter_relocates_full_verdict_by_reference(tmp_path: Path) -> None:
    """The adapter persists the FULL verdict (digest + per-criterion evidence + gaps) to
    the stable ``verdict_dest`` so it survives the worktree teardown and reaches the
    planner BY REFERENCE. Long content is preserved in full (no truncation)."""

    worktree = tmp_path / "wt"
    worktree.mkdir()
    long_evidence = "E" * 20_000
    payload = {
        "pass": True,
        "per_criterion": [{"criterion": "c", "met": True, "evidence": long_evidence}],
        "gaps": [],
        "digest": "D" * 9000,
    }
    (worktree / VERDICT_FILENAME).write_text(json.dumps(payload), encoding="utf-8")

    class _Echo:
        def run(self, request: WorkerRequest) -> None:
            return  # the verdict file is already in place

    dest = tmp_path / "P1" / "E1" / "verdict.json"
    verdict = WorkerEpochVerifier(_Echo()).verify(
        worktree=worktree,
        brief=VerificationBrief(epoch_goal="g", criteria=["c"], artifacts=[]),
        task_id="P1/E1/verify",
        verdict_dest=dest,
    )
    assert verdict.per_criterion[0].evidence == long_evidence  # parsed in full
    # The relocated file holds the FULL verdict, untruncated (delivered by reference).
    relocated = json.loads(dest.read_text(encoding="utf-8"))
    assert relocated["per_criterion"][0]["evidence"] == long_evidence
    assert relocated["digest"] == "D" * 9000


def test_verifier_adapter_rejects_oversized_verdict_file(tmp_path: Path) -> None:
    """DoS sanity backstop (item F): a verdict.json above the megabyte-scale guard is
    REJECTED (fail-safe), not truncated; a normal large-but-sane verdict reads fine."""

    import pytest

    from grindstone.verify import VERDICT_MAX_BYTES

    worktree = tmp_path / "wt"
    worktree.mkdir()
    # An absurd file: a valid JSON wrapper around an over-guard blob.
    blob = "x" * (VERDICT_MAX_BYTES + 1024)
    payload = {
        "pass": True,
        "per_criterion": [{"criterion": "c", "met": True, "evidence": blob}],
        "gaps": [],
    }
    (worktree / VERDICT_FILENAME).write_text(json.dumps(payload), encoding="utf-8")

    class _Echo:
        def run(self, request: WorkerRequest) -> None:
            return

    with pytest.raises(VerificationError):
        WorkerEpochVerifier(_Echo()).verify(
            worktree=worktree,
            brief=VerificationBrief(epoch_goal="g", criteria=["c"], artifacts=[]),
            task_id="P1/E1/verify",
            verdict_dest=tmp_path / "dest" / "verdict.json",
        )

    # A sane large verdict (well under the guard) reads fine and relocates.
    sane = json.dumps(
        {
            "pass": True,
            "per_criterion": [{"criterion": "c", "met": True, "evidence": "y" * 50_000}],
            "gaps": [],
        }
    )
    (worktree / VERDICT_FILENAME).write_text(sane, encoding="utf-8")
    dest = tmp_path / "dest2" / "verdict.json"
    v = WorkerEpochVerifier(_Echo()).verify(
        worktree=worktree,
        brief=VerificationBrief(epoch_goal="g", criteria=["c"], artifacts=[]),
        task_id="P1/E1/verify",
        verdict_dest=dest,
    )
    assert len(v.per_criterion[0].evidence) == 50_000
    assert dest.is_file()


# --- 3 + 5. the pass runs end of epoch, AFTER the floor, only-fail -------------


def test_epoch_with_criteria_pass_completes(git_repo: Path, run_dir: RunDir) -> None:
    """An epoch with criteria + a verifier returning pass=true verifies and the run
    proceeds to completion. The verification pass ran (started + passed events)."""

    verifier = WorkerEpochVerifier(_VerifierWorker(passed=True))
    planner = MockPlanner(
        script=[
            _skeleton_passing_gate(),
            implement_decision(_impl_task_with_criteria("T1", "f1.txt", ["f1 exists and is right"])),
            complete_decision(check_cmd("test -f f1.txt")),
        ]
    )
    outcome = run_grind(
        run_dir, job_path="job.md", planner=planner,
        ladder=[("local", OwnershipWorker())], repo=git_repo, verifier=verifier,
    )
    assert outcome.status == "completed"
    kinds = [e.event for e in read_events(run_dir.events_path)]
    assert "epoch_verification_started" in kinds
    assert "epoch_verification_passed" in kinds
    assert "epoch_failed" not in kinds
    # The criteria reached the verifier brief.
    fake = verifier  # WorkerEpochVerifier wraps the _VerifierWorker
    assert isinstance(fake._transport, _VerifierWorker)  # type: ignore[attr-defined]
    assert fake._transport.seen_briefs[0].criteria == ["f1 exists and is right"]  # type: ignore[attr-defined]


def test_verdict_persists_to_keyed_log_and_reaches_planner_by_reference(
    git_repo: Path, run_dir: RunDir
) -> None:
    """Item B: the FULL verdict (digest + per-criterion evidence + gaps) persists to a
    stable keyed-log path and the NEXT planner boundary surfaces that path in the
    <workspace> manifest (delivered by reference, never embedded). The persisted file
    resolves to the full digest + evidence."""

    digest = "added f1.txt mapping every ramp; theming still a TODO"
    evidence = "read f1.txt; the mapping is present " + "Q" * 9000  # long, untruncated

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
            (request.scratch / VERDICT_FILENAME).write_text(
                json.dumps(payload), encoding="utf-8"
            )

    verifier = WorkerEpochVerifier(_DigestVerifier())
    planner = _RecordingPlanner(
        MockPlanner(
            script=[
                _skeleton_passing_gate(),
                implement_decision(_impl_task_with_criteria("T1", "f1.txt", ["c"])),
                complete_decision(check_cmd("test -f f1.txt")),
            ]
        )
    )
    outcome = run_grind(
        run_dir, job_path="job.md", planner=planner,
        ladder=[("local", OwnershipWorker())], repo=git_repo, verifier=verifier,
    )
    assert outcome.status == "completed"
    # The verdict landed at a stable keyed-log path (so it shows in log_index/manifest).
    verdict_keys = [k for k in run_dir.log_index() if k.endswith("verdict.json")]
    assert verdict_keys, "the verified epoch must persist a verdict.json log key"
    verdict_path = run_dir.resolve(verdict_keys[0])
    relocated = json.loads(verdict_path.read_text(encoding="utf-8"))
    # The FULL digest + (long) evidence are preserved on disk, untruncated.
    assert relocated["digest"] == digest
    assert relocated["per_criterion"][0]["evidence"] == evidence
    # The NEXT boundary (complete_run) surfaced that absolute path in the manifest, so
    # the planner reads it by reference. The prompt itself never embeds the digest body.
    final_prompt = planner.prompts[-1]
    assert str(verdict_path) in final_prompt
    assert "<epoch_digest>" not in final_prompt
    assert digest not in final_prompt  # the digest travels by file, not embedded


def test_failing_check_output_reaches_planner_by_path_not_embedded(
    git_repo: Path, run_dir: RunDir
) -> None:
    """Item C: a failing phase-gate command's FULL output is persisted under
    check_output/ and the planner sees its PATH (output_file:) in the manifest + the
    failing-check label, never an embedded truncated tail."""

    # The marker appears ONLY in the command's RUNTIME output, never in the cmd text
    # itself (so finding it in the prompt would prove the output body was embedded). The
    # cmd decodes it at runtime from base64 ("RElTVElOQ1Q=" -> "DISTINCT").
    marker = "DISTINCT"
    failing_gate = [check_cmd("echo RElTVElOQ1Q= | base64 -d; echo; exit 1")]
    planner = _RecordingPlanner(
        MockPlanner(
            script=[
                skeleton_decision(
                    phase_dict("P1", title="build", exit_criterion=failing_gate, budget=1),
                    phase_dict("P2", title="verify", exit_criterion=failing_gate, budget=1),
                ),
                implement_decision({"id": "T1", "goal": "create f1.txt",
                                    "done_when": [check_cmd("test -f f1.txt")],
                                    "file_ownership": ["f1.txt"]}),
                # Budget is 1 and the gate never passes -> phase escalation -> revise/escalate.
                {"schema_version": "1", "tool": "escalate_run",
                 "args": {"reason": "the gate is unsatisfiable"}},
            ]
        )
    )
    outcome = run_grind(
        run_dir, job_path="job.md", planner=planner,
        ladder=[("local", OwnershipWorker())], repo=git_repo,
    )
    assert outcome.status == "escalated"
    # The full failing output was persisted under check_output/ (by reference).
    out_files = list((run_dir.root / "check_output").rglob("*.txt"))
    assert out_files, "the failing cmd check must persist its full output"
    assert any(marker in f.read_text(encoding="utf-8") for f in out_files)
    # The planner's input carried the PATH (output_file:), never an embedded tail with
    # the raw build text inlined into the label.
    joined = "\n".join(planner.prompts)
    assert "output_file:" in joined
    assert marker not in joined  # the output body is by reference, not embedded


def test_no_criteria_skips_the_pass(git_repo: Path, run_dir: RunDir) -> None:
    """An epoch with NO criteria never runs the verification pass (skipped entirely),
    even with a verifier wired. The verifier is never invoked."""

    fake = _VerifierWorker(passed=False, gaps=["should never be seen"])
    verifier = WorkerEpochVerifier(fake)
    planner = MockPlanner(
        script=[
            _skeleton_passing_gate(),
            # impl_task (no criteria field) -> no criteria aggregated
            implement_decision({"id": "T1", "goal": "create f1.txt",
                                "done_when": [check_cmd("test -f f1.txt")],
                                "file_ownership": ["f1.txt"]}),
            complete_decision(check_cmd("test -f f1.txt")),
        ]
    )
    outcome = run_grind(
        run_dir, job_path="job.md", planner=planner,
        ladder=[("local", OwnershipWorker())], repo=git_repo, verifier=verifier,
    )
    assert outcome.status == "completed"
    kinds = [e.event for e in read_events(run_dir.events_path)]
    assert "epoch_verification_started" not in kinds
    assert fake.seen_briefs == []  # the verifier was never invoked


def test_pass_runs_after_floor_not_when_task_floor_fails(git_repo: Path, run_dir: RunDir) -> None:
    """The deterministic floor is FIRST: when a task fails its floor (exhausts the
    ladder) the verification pass does NOT run, the epoch_failed path owns it. The
    agentic pass can only fire on an epoch whose floor already cleared."""

    from tests.grindstone.conftest import FailingWorker

    # A verifier that would PASS if asked, must never be asked (the floor failed).
    fake = _VerifierWorker(passed=True)
    verifier = WorkerEpochVerifier(fake)
    planner = MockPlanner(
        script=[
            two_phase_skeleton(),
            implement_decision(_impl_task_with_criteria("T1", "f1.txt", ["c"])),  # fails (no handoff)
            handle_failed_epoch_halt("the floor failed, not the semantics"),
        ]
    )
    outcome = run_grind(
        run_dir, job_path="job.md", planner=planner,
        ladder=[("local", FailingWorker())], repo=git_repo, verifier=verifier,
        tier0_attempts=1,
    )
    assert outcome.status == "escalated"
    kinds = [e.event for e in read_events(run_dir.events_path)]
    assert "epoch_failed" in kinds  # the floor-failure path, not verification
    assert "epoch_verification_started" not in kinds
    assert fake.seen_briefs == []  # the verifier was never asked to rubber-stamp


# --- 4. semantic-fail routes through the B6 failed-epoch machinery --------------


def test_semantic_fail_opens_failed_epoch_with_gaps(git_repo: Path, run_dir: RunDir) -> None:
    """A verifier returning pass=false with gaps opens a failed epoch routed to
    handle_failed_epoch; the planner sees the gaps; a halt terminates the run."""

    gaps = ["the Lesson screen never maps the Pink ramp"]
    verifier = WorkerEpochVerifier(_VerifierWorker(passed=False, gaps=gaps))
    planner = MockPlanner(
        script=[
            _skeleton_passing_gate(),
            implement_decision(_impl_task_with_criteria("T1", "f1.txt", ["map the Pink ramp"])),
            handle_failed_epoch_halt("incomplete per the gaps"),
        ]
    )
    outcome = run_grind(
        run_dir, job_path="job.md", planner=planner,
        ladder=[("local", OwnershipWorker())], repo=git_repo, verifier=verifier,
    )
    assert outcome.status == "escalated"
    kinds = [e.event for e in read_events(run_dir.events_path)]
    assert "epoch_verification_failed" in kinds
    assert "epoch_failed" not in kinds  # NOT charged as a task/gate failure
    vf = [e for e in read_events(run_dir.events_path) if isinstance(e, EpochVerificationFailed)]
    assert vf and vf[0].gaps == gaps
    # The gaps were persisted into the pending-failed-epoch context (for the planner).
    # (The run halted, so the planner saw the handle_failed_epoch boundary.)


def test_pending_failed_epoch_shows_real_cap_not_at_cap(
    git_repo: Path, run_dir: RunDir
) -> None:
    """A#5: the planner-facing failed-epoch budget must show the TRUE remaining budget
    (e.g. disposed=1/3), not the old false disposed=N/N that always rendered at-cap and
    falsely signaled 'halting now'. With max_failed_epochs_per_phase=3, the FIRST
    disposition boundary renders 1/3."""

    gaps = ["the Pink ramp is never mapped"]
    verifier = WorkerEpochVerifier(_VerifierWorker(passed=False, gaps=gaps))
    planner = _RecordingPlanner(
        MockPlanner(
            script=[
                _skeleton_passing_gate(),
                implement_decision(_impl_task_with_criteria("T1", "f1.txt", ["map the Pink ramp"])),
                handle_failed_epoch_halt("incomplete per the gaps"),
            ]
        )
    )
    outcome = run_grind(
        run_dir, job_path="job.md", planner=planner,
        ladder=[("local", OwnershipWorker())], repo=git_repo, verifier=verifier,
        max_failed_epochs_per_phase=3,
    )
    assert outcome.status == "escalated"
    # The handle_failed_epoch boundary (the last prompt) shows the REAL cap 1/3.
    failed_prompt = planner.prompts[-1]
    assert "disposed=1/3" in failed_prompt
    assert "disposed=1/1" not in failed_prompt  # the old at-cap bug


def test_semantic_fail_gaps_reach_the_planner_input() -> None:
    """The gaps ride the <failed_epoch> block, so the planner's input carries the
    semantic feedback (re-derived from the verdict, not a check label)."""

    info = FailedEpochInfo(
        epoch_id="E1",
        failed_tasks=[],
        failed_checks=[],
        passing_handoffs=[("T1", "wrote lesson.tsx")],
        disposed_count=1,
        cap=3,
        verification_gaps=["the Lesson screen never maps the Pink ramp"],
    )
    tail = volatile_tail(
        phase_id="P1", epoch_counter=1, log_index=[], last_epoch_rows=None,
        reask_errors=[], failed_epoch=info,
    )
    assert "<failed_epoch" in tail
    assert "semantic_gaps" in tail
    assert "the Lesson screen never maps the Pink ramp" in tail
    assert "handle_failed_epoch" in tail


def test_semantic_fail_retry_then_clean_pass_completes(git_repo: Path, run_dir: RunDir) -> None:
    """pass=false (gaps) -> handle_failed_epoch retry re-dispatches the epoch; the
    re-verification now passes and the run completes. Reuses the B6 retry path; the
    retry hint reaches the worker."""

    verifier = WorkerEpochVerifier(_SwapVerifier())  # fails first verdict, passes next
    planner = MockPlanner(
        script=[
            _skeleton_passing_gate(),
            implement_decision(_impl_task_with_criteria("T1", "f1.txt", ["map the Pink ramp"])),
            handle_failed_epoch_retry("now also map the Pink ramp on the Lesson screen"),
            complete_decision(check_cmd("test -f f1.txt")),
        ]
    )
    outcome = run_grind(
        run_dir, job_path="job.md", planner=planner,
        ladder=[("local", _UniqueContentWorker())], repo=git_repo, verifier=verifier,
    )
    assert outcome.status == "completed"
    from grindstone.events import FailedEpochHandled

    handled = [e for e in read_events(run_dir.events_path) if isinstance(e, FailedEpochHandled)]
    assert handled and handled[0].action == "retry"


# --- G15 a prepare/dependency-install failure is a SEMANTIC gap, not infra --------
#
# The epoch's COMMITTED manifest does not install in a clean environment (e.g. a
# package.json peer-dependency conflict makes ``npm install`` fail ERESOLVE). That is
# the worker's work being unbuildable, a DEFECT the planner CAN fix, NOT a verifier-
# tooling failure. So the prepare failure routes to the planner via handle_failed_epoch
# (the FULL prepare output persisted by reference at a keyed-log path), it does NOT
# escalate the whole run to a human dead-end.


def _failing_prepare() -> PrepareConfig:
    """A prepare whose cmd FAILS (so ``materialize_env`` raises PrepareError), with a
    DISTINCT marker only in the runtime output (so finding it in the persisted log
    proves the FULL output, not just the cmd text, reached disk)."""

    return PrepareConfig(
        cmd="echo ERESOLVE-PEER-CONFLICT-MARKER >&2 && exit 1",
        env_dirs=["node_modules"],
        cache_key_files=["package-lock.json"],
    )


def _artifact_gate_script(out_key: str, *, dispose: dict[str, object]) -> list[object]:
    """A skeleton + artifact epoch whose phase gate is ``artifact_exists`` ONLY (no
    CmdCheck), so the gate needs NO eval worktree and does NOT run ``prepare``: the floor
    clears, the epoch reaches the agentic pass, and ONLY there does ``_verify_epoch``
    build its worktree + run the (failing) prepare. ``dispose`` is the planner's
    disposition for the resulting failed epoch."""

    gate = [{"artifact_exists": out_key}]
    return [
        skeleton_decision(
            phase_dict("P1", title="build", exit_criterion=gate, budget=20),
            phase_dict("P2", title="done", exit_criterion=gate, budget=20),
        ),
        artifact_decision(_artifact_task_with_criteria("T1", out_key, ["the work installs"])),
        dispose,
    ]


def test_prepare_failure_routes_to_planner_as_semantic_gap_not_infra(
    git_repo: Path, run_dir: RunDir
) -> None:
    """RED-then-green: an epoch whose committed work does not install (prepare fails)
    routes through handle_failed_epoch (a SEMANTIC gap), NOT an infra dead-end escalation.
    The planner's halt is reached (failed_epoch_handled fires); the run does NOT escalate
    with the verifier-tooling 'could not be prepared/built' message."""

    out_key = "P1/E1/T1/manifest.md"
    verifier = WorkerEpochVerifier(_VerifierWorker(passed=True))  # would pass if asked
    planner = MockPlanner(
        script=_artifact_gate_script(
            out_key,
            # Only consumed if the prepare failure ROUTES as a semantic gap (the fix);
            # under the old infra-escalation this decision is never reached.
            dispose=handle_failed_epoch_halt("the manifest does not install; fix the deps"),
        )
    )
    outcome = run_grind(
        run_dir, job_path="job.md", planner=planner,
        ladder=[("local", _ContentArtifactWorker("a manifest\n"))], repo=git_repo,
        verifier=verifier, prepare=_failing_prepare(),
    )
    assert outcome.status == "escalated"  # the planner's halt, not an infra dead-end
    # The gap routed through the failed-epoch machinery (NOT mis-escalated as infra).
    from grindstone.events import FailedEpochHandled

    handled = [
        e for e in read_events(run_dir.events_path) if isinstance(e, FailedEpochHandled)
    ]
    assert handled and handled[0].action == "halt"
    kinds = [e.event for e in read_events(run_dir.events_path)]
    assert "failed_epoch_handled" in kinds
    # The verifier was NEVER asked (prepare failed before the verdict pass): the gap is
    # the unbuildable manifest, not a verdict.
    assert isinstance(verifier._transport, _VerifierWorker)  # type: ignore[attr-defined]
    assert verifier._transport.seen_briefs == []  # type: ignore[attr-defined]
    # The run did NOT escalate with the verifier-tooling message (the old infra path).
    assert outcome.reason is None or "could not be prepared" not in outcome.reason.lower()
    # The gap routed to the planner is CAPABILITY-NEUTRAL: it tells the planner to
    # resolve the conflict by ALIGNING/correcting versions and explicitly warns against
    # DROPPING dependencies the app needs (so the planner does not lazily delete a
    # required package to make install succeed, silently killing a capability).
    from grindstone.events import EpochVerificationFailed

    vfailed = [
        e for e in read_events(run_dir.events_path)
        if isinstance(e, EpochVerificationFailed)
    ]
    assert vfailed, "the prepare failure must route as a verification gap"
    gap_text = " ".join(vfailed[0].gaps).lower()
    assert "prepare step failed" in gap_text  # still references the failed prepare
    assert "align" in gap_text  # resolve by aligning compatible versions
    # ...and explicitly warns AGAINST removing/dropping needed dependencies.
    assert "without dropping" in gap_text or "not by removing" in gap_text
    assert "drop" in gap_text or "remov" in gap_text


def test_prepare_failure_persists_full_output_to_keyed_log_by_reference(
    git_repo: Path, run_dir: RunDir
) -> None:
    """The FULL prepare failure output is persisted to a stable ``P*/E*/...`` keyed-log
    path (so it shows in log_index/the planner manifest) and the planner-facing gap
    carries that PATH, never the embedded multi-KB output (reference-not-embed)."""

    marker = "ERESOLVE-PEER-CONFLICT-MARKER"
    out_key = "P1/E1/T1/manifest.md"
    verifier = WorkerEpochVerifier(_VerifierWorker(passed=True))
    planner = _RecordingPlanner(
        MockPlanner(
            script=_artifact_gate_script(
                out_key,
                dispose=handle_failed_epoch_halt("the manifest does not install; fix the deps"),
            )
        )
    )
    outcome = run_grind(
        run_dir, job_path="job.md", planner=planner,
        ladder=[("local", _ContentArtifactWorker("a manifest\n"))], repo=git_repo,
        verifier=verifier, prepare=_failing_prepare(),
    )
    assert outcome.status == "escalated"
    # The full prepare failure landed at a P*/E*/... keyed-log path (in the manifest).
    keys = [k for k in run_dir.log_index() if k.endswith("prepare_failure.txt")]
    assert keys, "the prepare failure must persist a keyed-log file"
    persisted = run_dir.resolve(keys[0]).read_text(encoding="utf-8")
    # The FULL captured output (the failing cmd + its runtime marker) is on disk.
    assert "prepare failed" in persisted
    assert marker in persisted
    # The planner saw the failed-epoch boundary; the gap carries the PATH, the body is
    # delivered by reference (the runtime marker is NOT embedded in the prompt).
    failed_prompt = planner.prompts[-1]
    assert keys[0] in failed_prompt or str(run_dir.resolve(keys[0])) in failed_prompt
    assert marker not in failed_prompt  # the output body travels by file, not embedded


def test_unborn_tip_checkout_failure_still_escalates_as_infra(
    git_repo: Path, run_dir: RunDir
) -> None:
    """The OTHER infra path is untouched: when the integration tip cannot be checked out
    (unborn HEAD / unresolvable ref) the verifier environment genuinely cannot be built,
    so the run STILL escalates as infra (the planner cannot fix a missing tip), NOT a
    semantic gap routed to handle_failed_epoch."""

    import grindstone.run_loop as run_loop_mod
    from grindstone import worktree as wt

    # An artifact_exists gate needs no eval worktree, so the floor clears WITHOUT calling
    # add_worktree_detached; the ONLY worktree creation left is _verify_epoch's. Force
    # THAT to fail (unborn/unresolvable tip) so we exercise the genuine verifier-tooling
    # infra path in isolation.
    def _boom_add(repo: Path, worktree: Path, *, ref: str) -> None:
        raise wt.GitError("simulated unborn HEAD: cannot resolve the integration tip")

    out_key = "P1/E1/T1/manifest.md"
    verifier = WorkerEpochVerifier(_VerifierWorker(passed=True))
    planner = MockPlanner(
        script=[
            skeleton_decision(
                phase_dict("P1", title="build", exit_criterion=[{"artifact_exists": out_key}], budget=20),
                phase_dict("P2", title="done", exit_criterion=[{"artifact_exists": out_key}], budget=20),
            ),
            artifact_decision(_artifact_task_with_criteria("T1", out_key, ["c"])),
            # NO handle_failed_epoch: the core must escalate on its own (infra path).
        ]
    )
    orig = run_loop_mod.wt.add_worktree_detached
    run_loop_mod.wt.add_worktree_detached = _boom_add  # type: ignore[assignment]
    try:
        outcome = run_grind(
            run_dir, job_path="job.md", planner=planner,
            ladder=[("local", _ContentArtifactWorker("a manifest\n"))], repo=git_repo,
            verifier=verifier,
        )
    finally:
        run_loop_mod.wt.add_worktree_detached = orig  # type: ignore[assignment]
    assert outcome.status == "escalated"
    kinds = [e.event for e in read_events(run_dir.events_path)]
    # An infra escalation: NOT charged as a semantic gap.
    assert "failed_epoch_handled" not in kinds
    assert "epoch_verification_failed" not in kinds
    assert outcome.reason is not None
    assert "integration tip" in outcome.reason.lower()


def test_persistent_malformed_verdict_escalates_as_infra_not_semantic(
    git_repo: Path, run_dir: RunDir
) -> None:
    """G6 Part B: a STRUCTURALLY-invalid verdict the verifier never produces validly
    (bad JSON on every attempt) is a verification-INFRASTRUCTURE failure, NOT a worker
    semantic gap. After the bounded verifier re-runs are exhausted the core escalates
    the run with a CLEAR message (deterministic floor passed, verification tooling
    could not produce a valid verdict) and does NOT route it through
    handle_failed_epoch (no epoch_verification_failed, the worker is not blamed)."""

    fake = _VerifierWorker(passed=True, malformed=True)
    verifier = WorkerEpochVerifier(fake)
    planner = MockPlanner(
        script=[
            _skeleton_passing_gate(),
            implement_decision(_impl_task_with_criteria("T1", "f1.txt", ["c"])),
            # NO handle_failed_epoch decision: the core must escalate on its own,
            # never reach the planner with a semantic-gap boundary.
        ]
    )
    outcome = run_grind(
        run_dir, job_path="job.md", planner=planner,
        ladder=[("local", OwnershipWorker())], repo=git_repo, verifier=verifier,
    )
    assert outcome.status == "escalated"
    assert outcome.reason is not None
    assert "verification" in outcome.reason.lower()
    assert "could not produce a valid verdict" in outcome.reason.lower()
    kinds = [e.event for e in read_events(run_dir.events_path)]
    # NOT charged to the worker / epoch as a semantic gap.
    assert "epoch_verification_failed" not in kinds
    assert "epoch_failed" not in kinds
    assert "failed_epoch_handled" not in kinds
    # The verifier was re-run the bounded number of times before escalating.
    from grindstone.run_loop import VERIFIER_MAX_ATTEMPTS

    assert len(fake.seen_briefs) == VERIFIER_MAX_ATTEMPTS


def test_malformed_then_valid_verdict_proceeds(git_repo: Path, run_dir: RunDir) -> None:
    """G6 Part B: a verifier that emits bad JSON on its first attempt but a VALID
    (passing) verdict on its re-run proceeds normally; the bounded re-run recovers a
    transient structural glitch instead of charging the epoch."""

    verifier = WorkerEpochVerifier(_MalformedThenValidVerifier())
    planner = MockPlanner(
        script=[
            _skeleton_passing_gate(),
            implement_decision(_impl_task_with_criteria("T1", "f1.txt", ["c"])),
            complete_decision(check_cmd("test -f f1.txt")),
        ]
    )
    outcome = run_grind(
        run_dir, job_path="job.md", planner=planner,
        ladder=[("local", OwnershipWorker())], repo=git_repo, verifier=verifier,
    )
    assert outcome.status == "completed"
    kinds = [e.event for e in read_events(run_dir.events_path)]
    assert "epoch_verification_passed" in kinds
    assert "epoch_verification_failed" not in kinds
    assert "epoch_failed" not in kinds


def test_no_verifier_skips_the_pass(git_repo: Path, run_dir: RunDir) -> None:
    """With no verifier wired (e.g. verify_epochs disabled / no local tier) an epoch
    with criteria completes without the pass, the deterministic floor still gates."""

    planner = MockPlanner(
        script=[
            _skeleton_passing_gate(),
            implement_decision(_impl_task_with_criteria("T1", "f1.txt", ["c"])),
            complete_decision(check_cmd("test -f f1.txt")),
        ]
    )
    outcome = run_grind(
        run_dir, job_path="job.md", planner=planner,
        ladder=[("local", OwnershipWorker())], repo=git_repo, verifier=None,
    )
    assert outcome.status == "completed"
    kinds = [e.event for e in read_events(run_dir.events_path)]
    assert "epoch_verification_started" not in kinds


# --- a local worker that writes UNIQUE content per call (so a retry diffs) ------


class _UniqueContentWorker:
    """An OwnershipWorker variant whose content changes each call, so a re-dispatched
    epoch produces a NON-zero diff (the retry path's commit gate needs a real change;
    rewriting identical bytes is a zero-diff reject, unrelated to the G4 routing)."""

    def __init__(self) -> None:
        self._calls = 0

    def run(self, request: WorkerRequest) -> None:
        self._calls += 1
        OwnershipWorker(content=f"v{self._calls}\n").run(request)


# --- a swap verifier: fail the first verdict, pass on the re-verification -------


class _MalformedThenValidVerifier:
    """Write BAD JSON (a structural failure) on the first call, a well-formed PASSING
    verdict on every later call. Models 'the verifier model emitted garbage once, then
    valid output on the bounded re-run' (G6 Part B)."""

    def __init__(self) -> None:
        self._calls = 0
        self.seen_briefs: list[VerificationBrief] = []

    def run(self, request: WorkerRequest) -> None:
        assert request.verification is not None
        self.seen_briefs.append(request.verification)
        self._calls += 1
        verdict_file = request.scratch / VERDICT_FILENAME
        if self._calls == 1:
            verdict_file.write_text("{ not json", encoding="utf-8")
            return
        payload = {
            "pass": True,
            "per_criterion": [
                {"criterion": c, "met": True, "evidence": "e"}
                for c in request.verification.criteria
            ],
            "gaps": [],
        }
        verdict_file.write_text(json.dumps(payload), encoding="utf-8")


class _SwapVerifier:
    """Write a failing verdict on the first call, a passing one on every later call.
    Models 'the retry closed the gap'."""

    def __init__(self) -> None:
        self._calls = 0

    def run(self, request: WorkerRequest) -> None:
        assert request.verification is not None
        self._calls += 1
        passed = self._calls > 1
        payload = {
            "pass": passed,
            "per_criterion": [
                {"criterion": c, "met": passed, "evidence": "e"}
                for c in request.verification.criteria
            ],
            "gaps": [] if passed else ["the Pink ramp is still unmapped"],
        }
        (request.scratch / VERDICT_FILENAME).write_text(json.dumps(payload), encoding="utf-8")


# --- relocated artifact_out deliverable reaches the verifier (the dogfood bug) --
#
# A RESEARCH/ARTIFACT epoch's deliverable is NOT a committed diff: the core
# relocates the worker's ``artifact_out`` file into the run dir (the same place
# ``artifact_exists`` finds it). The verifier runs in a fresh worktree of the
# committed integration tip, so it sees no such file in the diff. The reference
# contract gives the verifier the relocated artifact PATH (not its embedded
# content): the verifier READS the file off disk, so the deliverable a research
# epoch produces is actually judged, and is NEVER reported "missing" when the
# deterministic ``artifact_exists`` gate just confirmed it present. The content is
# delivered by reference (full body on disk), never byte-capped into the prompt.


class _ContentArtifactWorker:
    """An artifact-mode worker that writes a fixed body into ``artifact_out``.

    Mirrors the disk contract: the worker writes the deliverable + a DONE handoff
    in its scratch CWD; the core relocates the artifact to its log key under the
    run dir. The handoff's ``resulting_state`` deliberately does NOT echo the body,
    so a test asserting the BODY reached the verifier proves the content came from
    the relocated file, not the one-line state."""

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
    """Resolve every ``... at this path: <abs path>`` artifact pointer in the brief and
    read the referenced file, returning the concatenated bodies.

    Models the reference contract: the brief hands the verifier PATHS, not embedded
    content; a real verifier READS the files. The path is the trailing token after the
    ``path): `` marker the core emits for a relocated deliverable."""

    bodies: list[str] = []
    for line in brief.artifacts:
        marker = "at this path): "
        if marker in line:
            path = Path(line.rsplit(marker, 1)[1].strip())
            if path.is_file():
                bodies.append(path.read_text(encoding="utf-8"))
    return "\n".join(bodies)


class _ContentVerifier:
    """A verifier that PASSES iff a required marker appears in the relocated artifact
    FILE the brief points at.

    Proves the relocated artifact's CONTENT reached the verifier BY REFERENCE: the brief
    carries the PATH, the verifier READS the file; if the marker is absent the verdict
    fails with a 'missing' gap, exactly the dogfood contradiction we are fixing (the
    verifier would otherwise never see the deliverable)."""

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


def _artifact_task_with_criteria(
    tid: str, out: str, criteria: list[str]
) -> dict[str, object]:
    return {
        "id": tid,
        "goal": f"produce {out}",
        "done_when": [check_cmd("true")],
        "criteria": criteria,
        "artifact_out": out,
    }


def test_research_artifact_content_reaches_verifier_and_completes(
    git_repo: Path, run_dir: RunDir
) -> None:
    """A research/artifact epoch whose deliverable exists ONLY as a relocated
    ``artifact_out`` file is judged against that file's ACTUAL content: the verifier
    sees the body (not just the one-line handoff state) and a pass completes."""

    marker = "TOKEN-FLOW-CONTENT-MARKER"
    body = f"# token flow\n\n{marker}\n\nthe mapping is complete.\n"
    verifier = WorkerEpochVerifier(_ContentVerifier(marker=marker))
    out_key = "P1/E1/T1/token_flow_mapping.md"
    planner = MockPlanner(
        script=[
            _skeleton_passing_gate(),
            artifact_decision(
                _artifact_task_with_criteria("T1", out_key, ["the token flow is mapped"])
            ),
            complete_decision(check_cmd("true")),
        ]
    )
    outcome = run_grind(
        run_dir, job_path="job.md", planner=planner,
        ladder=[("local", _ContentArtifactWorker(body))], repo=git_repo,
        verifier=verifier,
    )
    assert outcome.status == "completed"
    kinds = [e.event for e in read_events(run_dir.events_path)]
    assert "epoch_verification_passed" in kinds
    assert "epoch_verification_failed" not in kinds
    # The relocated artifact landed where artifact_exists looks, and the verifier
    # received its PATH (by reference) and READ the body (the marker) off disk.
    assert run_dir.find_artifact(out_key) is not None
    fake = verifier
    assert isinstance(fake._transport, _ContentVerifier)  # type: ignore[attr-defined]
    brief = fake._transport.seen_briefs[0]  # type: ignore[attr-defined]
    joined = "\n".join(brief.artifacts)
    # The brief carries the PATH, not the embedded body: the marker is NOT inlined.
    assert marker not in joined
    assert str(run_dir.find_artifact(out_key)) in joined
    # Reading that path yields the full body with the marker (what the verifier saw).
    assert marker in _read_artifact_paths(brief)


def test_artifact_never_reported_missing_when_artifact_exists_passes(
    git_repo: Path, run_dir: RunDir
) -> None:
    """The core contradiction the bug caused: a deliverable the deterministic
    ``artifact_exists`` gate confirms present must NEVER be reported 'missing' by the
    verifier. The verifier passes BECAUSE it can read the relocated content."""

    marker = "PRESENT-MARKER"
    out_key = "P1/E1/T1/research.md"
    verifier = WorkerEpochVerifier(_ContentVerifier(marker=marker))
    # The phase gate's exit criterion is artifact_exists on the produced file: it
    # must PASS, and the verifier must agree (not contradict it with 'missing').
    gate = [{"artifact_exists": out_key}]
    planner = MockPlanner(
        script=[
            skeleton_decision(
                phase_dict("P1", title="research", exit_criterion=gate, budget=20),
                phase_dict("P2", title="done", exit_criterion=gate, budget=20),
            ),
            artifact_decision(
                _artifact_task_with_criteria("T1", out_key, ["the research is grounded"])
            ),
            complete_decision({"artifact_exists": out_key}),
        ]
    )
    outcome = run_grind(
        run_dir, job_path="job.md", planner=planner,
        ladder=[("local", _ContentArtifactWorker(f"body with {marker}\n"))],
        repo=git_repo, verifier=verifier,
    )
    assert outcome.status == "completed"
    # artifact_exists passed (the file is present) AND the verifier saw it: no
    # 'missing' contradiction.
    assert run_dir.find_artifact(out_key) is not None
    kinds = [e.event for e in read_events(run_dir.events_path)]
    assert "epoch_verification_passed" in kinds
    assert "epoch_verification_failed" not in kinds


def test_artifact_real_semantic_gap_still_fails(git_repo: Path, run_dir: RunDir) -> None:
    """Reading the artifact content does NOT rubber-stamp: a verifier that finds a
    genuine gap in the deliverable still FAILS the epoch (routed to handle_failed_epoch).
    The content reaches the verifier; the verdict is honored verbatim."""

    out_key = "P1/E1/T1/note.md"
    # The marker the verifier requires is absent from the body -> a real gap.
    verifier = WorkerEpochVerifier(_ContentVerifier(marker="REQUIRED-BUT-ABSENT"))
    planner = MockPlanner(
        script=[
            _skeleton_passing_gate_true(),
            artifact_decision(
                _artifact_task_with_criteria("T1", out_key, ["covers every case"])
            ),
            handle_failed_epoch_halt("the deliverable has a real gap"),
        ]
    )
    outcome = run_grind(
        run_dir, job_path="job.md", planner=planner,
        ladder=[("local", _ContentArtifactWorker("an incomplete note\n"))],
        repo=git_repo, verifier=verifier,
    )
    assert outcome.status == "escalated"
    kinds = [e.event for e in read_events(run_dir.events_path)]
    assert "epoch_verification_failed" in kinds
    assert "epoch_failed" not in kinds  # a semantic gap, not a floor failure


def _skeleton_passing_gate_true() -> dict[str, object]:
    """A skeleton whose phase gate is a trivially-passing cmd (so the floor clears
    and the epoch reaches the agentic pass without an artifact_exists dependency)."""

    pending = [check_cmd("true")]
    return skeleton_decision(
        phase_dict("P1", title="build", exit_criterion=pending, budget=20),
        phase_dict("P2", title="verify", exit_criterion=pending, budget=20),
    )


def test_huge_artifact_is_delivered_by_reference_not_embedded(
    git_repo: Path, run_dir: RunDir
) -> None:
    """The principle: a HUGE artifact is delivered to the verifier BY REFERENCE (a path),
    so the brief stays small REGARDLESS of the deliverable's size, and the FULL content
    is on disk for the verifier to read (no truncation, no byte cap embedded)."""

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
            (request.scratch / VERDICT_FILENAME).write_text(
                json.dumps(payload), encoding="utf-8"
            )

    verifier = WorkerEpochVerifier(_Capture())
    planner = MockPlanner(
        script=[
            _skeleton_passing_gate_true(),
            artifact_decision(
                _artifact_task_with_criteria("T1", out_key, ["the artifact is produced"])
            ),
            complete_decision(check_cmd("true")),
        ]
    )
    outcome = run_grind(
        run_dir, job_path="job.md", planner=planner,
        ladder=[("local", _ContentArtifactWorker(huge))], repo=git_repo,
        verifier=verifier,
    )
    assert outcome.status == "completed"
    assert captured
    joined = "\n".join(captured[0].artifacts)
    # The brief is TINY even though the deliverable is megabytes: the body is never
    # embedded, only the path is, so no truncation is needed or present.
    assert len(joined) < 4096
    assert huge_marker not in joined
    assert "truncated" not in joined.lower()
    # The FULL body lives on disk at the referenced path (the verifier reads it there).
    body = _read_artifact_paths(captured[0])
    assert huge_marker in body
    assert len(body) > 2 * 1024 * 1024


# --- the resume re-verification guard (the loop-position hole) ------------------
#
# A kill AFTER an epoch persisted `awaiting_planner` (or one finished by resume_epoch)
# but BEFORE its G4 verification reached a terminal event must NOT be treated as
# verified-pass on resume. The DURABLE pending_verification marker + the absence of a
# terminal verification event drives a RE-RUN of the pass before the next boundary.


def _land_recorded_epoch(git_repo: Path, run_dir: RunDir) -> None:
    """Run an implement epoch with criteria to a clean COMPLETION (pass-verifier), then
    roll the run dir back to the exact state a kill leaves: the epoch is recorded
    (`awaiting_planner`, branch + outcome.json on disk, pending_verification set) but
    NO terminal verification event remains in the journal. This is the only way the
    'recorded-but-unverified' cursor arises (verification runs in the same drive
    iteration as the dispatch, so a clean run never persists it)."""

    decision_dict = implement_decision(
        _impl_task_with_criteria("T1", "f1.txt", ["map the Pink ramp"])
    )
    planner = MockPlanner(
        script=[_skeleton_passing_gate(), decision_dict, complete_decision(check_cmd("test -f f1.txt"))]
    )
    outcome = run_grind(
        run_dir, job_path="job.md", planner=planner,
        ladder=[("local", OwnershipWorker())], repo=git_repo,
        verifier=WorkerEpochVerifier(_VerifierWorker(passed=True)),
    )
    assert outcome.status == "completed"  # the epoch + its branch + outcome.json now exist

    # Roll the journal back: drop everything from the first EpochVerificationStarted on
    # (so no terminal verification/completion event survives), as a kill mid-verify would.
    lines = run_dir.events_path.read_text(encoding="utf-8").splitlines()
    kept: list[str] = []
    for line in lines:
        if '"epoch_verification_started"' in line:
            break
        kept.append(line)
    run_dir.events_path.write_text("\n".join(kept) + "\n", encoding="utf-8")

    # Roll run_state back to the post-_record_epoch cursor: awaiting_planner, the epoch's
    # branch retained, and pending_verification set (the durable 'owes the pass' marker).
    state = RunState.model_validate_json(run_dir.run_state_path.read_text())
    rolled = state.model_copy(update={
        "status": "awaiting_planner",
        "terminal_reason": None,
        "pending_decision": None,
        "pending_verification": {"decision": decision_dict, "phase_id": "P1", "epoch_id": "E1"},
    })
    run_dir.run_state_path.write_text(rolled.model_dump_json(), encoding="utf-8")


def test_resume_reverifies_recorded_but_unverified_epoch(
    git_repo: Path, run_dir: RunDir
) -> None:
    """The P0 loop-position repro: an epoch recorded across a kill before verification
    finished is RE-VERIFIED on resume (not silently skipped). A FAIL verdict on resume
    fires EpochVerificationStarted + routes the gap through handle_failed_epoch."""

    _land_recorded_epoch(git_repo, run_dir)
    gaps = ["the Pink ramp is never mapped"]
    planner = MockPlanner(script=[handle_failed_epoch_halt("incomplete per the resume re-verify")])
    outcome = resume_grind(
        run_dir, planner=planner, ladder=[("local", OwnershipWorker())], repo=git_repo,
        verifier=WorkerEpochVerifier(_VerifierWorker(passed=False, gaps=gaps)),
    )
    assert outcome.status == "escalated"
    kinds = [e.event for e in read_events(run_dir.events_path)]
    # Verification ACTUALLY RAN after resume (the whole point), and the gap routed.
    assert "epoch_verification_started" in kinds
    assert "epoch_verification_failed" in kinds
    vf = [e for e in read_events(run_dir.events_path) if isinstance(e, EpochVerificationFailed)]
    assert vf and vf[0].gaps == gaps
    # The marker is consumed: a clean cursor for the next boundary.
    assert RunState.model_validate_json(run_dir.run_state_path.read_text()).pending_verification is None


def test_resume_reverify_pass_proceeds_cleanly(git_repo: Path, run_dir: RunDir) -> None:
    """When the resume re-verification PASSES, the marker clears and the run proceeds to
    completion (no spurious failed epoch). A produced verdict is recorded as a terminal
    event so a SECOND resume reuses it rather than re-running the verifier."""

    _land_recorded_epoch(git_repo, run_dir)
    fake = _VerifierWorker(passed=True)
    planner = MockPlanner(script=[complete_decision(check_cmd("test -f f1.txt"))])
    outcome = resume_grind(
        run_dir, planner=planner, ladder=[("local", OwnershipWorker())], repo=git_repo,
        verifier=WorkerEpochVerifier(fake),
    )
    assert outcome.status == "completed"
    kinds = [e.event for e in read_events(run_dir.events_path)]
    assert "epoch_verification_started" in kinds and "epoch_verification_passed" in kinds
    assert fake.seen_briefs  # the verifier was invoked on resume (the re-run)


def test_resume_no_verifier_skips_pending_verification(git_repo: Path, run_dir: RunDir) -> None:
    """With NO verifier wired (verify_epochs off / no local tier) a resume with a set
    pending_verification marker is a CLEAN skip: no verification runs, no crash, and the
    marker is cleared so the next boundary is not blocked."""

    _land_recorded_epoch(git_repo, run_dir)
    fake = _VerifierWorker(passed=False, gaps=["never asked"])
    planner = MockPlanner(script=[complete_decision(check_cmd("test -f f1.txt"))])
    outcome = resume_grind(
        run_dir, planner=planner, ladder=[("local", OwnershipWorker())], repo=git_repo,
        verifier=None,
    )
    assert outcome.status == "completed"
    kinds = [e.event for e in read_events(run_dir.events_path)]
    assert "epoch_verification_started" not in kinds
    assert fake.seen_briefs == []  # the disabled pass never invoked anything
    assert RunState.model_validate_json(run_dir.run_state_path.read_text()).pending_verification is None
