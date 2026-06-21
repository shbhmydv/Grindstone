"""Planner core (S3): failure classification, backoff schedule, pure input
construction (stable-head byte-identity, volatile-tail content), and the
decision validation pipeline (schema → typed → semantic → position legality).
"""

from __future__ import annotations

import json
from pathlib import Path

from grindstone.contracts.models import Phase
from grindstone.epoch_loop import (
    EpochOutcome,
    IntegrationOutcome,
    TaskResult,
)
from grindstone.planner import (
    BACKOFF_CAP_S,
    MAX_RATE_LIMIT_WAITS,
    SYSTEM_PREAMBLE,
    PlannerHardError,
    RateLimited,
    TransportError,
    WorkspaceInfo,
    WorkerTimeout,
    backoff_delay,
    build_planner_input,
    classify_failure,
    flatten_last_epoch,
    stable_head,
    validate_decision,
    volatile_tail,
)
from grindstone.rundir import create_run_dir

from tests.grindstone.conftest import (
    complete_decision,
    impl_task,
    implement_decision,
    phase_dict,
    two_phase_skeleton,
)


def _phases(*ids: str) -> list[Phase]:
    return [Phase.model_validate(phase_dict(i)) for i in ids]


# --- failure classification (ruling 2) -----------------------------------------


def test_classify_three_way() -> None:
    assert classify_failure(RateLimited("429")) == "rate_limit"
    assert classify_failure(TransportError("5xx")) == "transient"
    assert classify_failure(WorkerTimeout("hang")) == "transient"
    assert classify_failure(PlannerHardError("auth")) == "hard"
    assert classify_failure(RuntimeError("???")) == "hard"
    assert classify_failure(ValueError("config")) == "hard"


def test_backoff_schedule_doubles_and_caps() -> None:
    seq = [backoff_delay(i) for i in range(MAX_RATE_LIMIT_WAITS)]
    assert seq == [30.0, 60.0, 120.0, 240.0, 480.0, 600.0]
    assert backoff_delay(10) == BACKOFF_CAP_S  # saturates at the cap


# --- preamble teaching content (audit fixes #3, #4) ----------------------------


def test_preamble_teaches_sizing_independence_and_decomposition() -> None:
    # Task independence: parallel tasks must not consume each other's outputs.
    assert "MUST NOT consume each other's" in SYSTEM_PREAMBLE
    # The ~90k worker-context sizing contract (ARCHITECTURE.md floor/ceiling).
    assert "90k" in SYSTEM_PREAMBLE
    # epoch_budget meaning explained, not just emitted.
    assert "epoch_budget" in SYSTEM_PREAMBLE
    assert "phase escalation" in SYSTEM_PREAMBLE
    # Conservative top-level decomposition (owner ruling 2026-06-12).
    assert "Decompose CONSERVATIVELY" in SYSTEM_PREAMBLE
    # Verbatim job-spec requirements in goals (owner architectural ruling) + the
    # 1024-char cap escape hatch (quote what fits, reference the rest).
    assert "VERBATIM" in SYSTEM_PREAMBLE
    assert "1024" in SYSTEM_PREAMBLE


def test_preamble_constrains_artifact_mode_done_when() -> None:
    """E2E gate2 finding: the planner attached repo test commands to an artifact
    task's done_when, unsatisfiable by construction (artifact CWD is run-dir
    scratch, not a repo checkout), so the task burned every attempt on a check
    that could never pass. The preamble must scope done_when by mode."""

    assert "artifact itself" in SYSTEM_PREAMBLE
    assert "never repo build/test commands" in SYSTEM_PREAMBLE


def test_preamble_teaches_mode_selection_by_destination() -> None:
    """E2E gates 3+4 (2/2): document deliverables always came out `implement`.
    Both jobs demanded the file IN the repo tree, so implement was right, but
    the preamble never taught the selection rule at all, so jobs wanting
    log-keyed deliverables (analyses, reports) would get worktrees too. The
    rule follows the deliverable's DESTINATION: committed repo files (even
    prose) = implement, the only mode that commits; log-keyed deliverables =
    research/artifact via artifact_out, no worktree."""

    assert "DESTINATION" in SYSTEM_PREAMBLE
    assert "even prose" in SYSTEM_PREAMBLE
    assert "consumed via the keyed log" in SYSTEM_PREAMBLE


def test_preamble_teaches_artifact_publication_and_bare_filename_checks() -> None:
    """Gate-6 RCA companion: the planner must know (a) accepted artifact tasks
    get their artifact_out PUBLISHED to the keyed log, and (b) an
    artifact_exists check may use a bare filename, required at skeleton time,
    when the P*/E*/T*/ placement is unknowable, matching exactly one logged
    artifact."""

    assert "published to the keyed log" in SYSTEM_PREAMBLE
    assert "bare filename" in SYSTEM_PREAMBLE


def test_preamble_carries_valid_implement_example() -> None:
    # Audit fix #4: a worked implement example exists and is valid JSON.
    start = SYSTEM_PREAMBLE.index('{"schema_version":"1","tool":"implement"')
    obj, _ = json.JSONDecoder().raw_decode(SYSTEM_PREAMBLE[start:])
    assert obj["tool"] == "implement"
    tasks = obj["args"]["tasks"]
    # Models disjoint ownership across the epoch + machine-checkable done_when.
    owns = [g for t in tasks for g in t["file_ownership"]]
    assert len(owns) == len(set(owns))  # pairwise-disjoint ownership
    assert all(t["done_when"] for t in tasks)


# --- stable head: byte-identical across a run (ruling 3) -----------------------


_JOB = "Create two text files via independent tasks, then complete."


def test_stable_head_byte_identical_for_same_skeleton() -> None:
    sk = _phases("P1", "P2")
    assert stable_head(_JOB, sk) == stable_head(_JOB, sk)
    # A fresh equal skeleton produces identical bytes (content, not identity).
    assert stable_head(_JOB, sk) == stable_head(_JOB, _phases("P1", "P2"))


def test_stable_head_changes_only_when_skeleton_or_job_changes() -> None:
    base = stable_head(_JOB, _phases("P1", "P2"))
    assert stable_head(_JOB, None) != base
    assert stable_head(_JOB, _phases("P1", "P2", "P3")) != base
    assert stable_head("a different job", _phases("P1", "P2")) != base


def test_stable_head_renders_repo_memory_in_slot() -> None:
    sk = _phases("P1", "P2")
    # Absent digest: the slot is empty and byte-identical to the no-arg form.
    assert stable_head(_JOB, sk, None) == stable_head(_JOB, sk)
    assert "<repo_memory>\n</repo_memory>" in stable_head(_JOB, sk, None)
    # Present digest: it lands inside the slot and shifts the head's bytes
    # (a different repo memory legitimately resets the prefix cache).
    head = stable_head(_JOB, sk, "prefer rg; tests live under tests/")
    assert "<repo_memory>\nprefer rg; tests live under tests/\n</repo_memory>" in head
    assert head != stable_head(_JOB, sk, None)


def test_build_planner_input_threads_repo_memory() -> None:
    sk = _phases("P1", "P2")
    full = build_planner_input(
        job=_JOB, skeleton=sk, phase_id="P1", epoch_counter=0, log_index=[],
        last_epoch_rows=None, reask_errors=[], repo_memory="grindstone fact",
    )
    assert full.startswith(stable_head(_JOB, sk, "grindstone fact"))
    assert "grindstone fact" in full


def test_stable_head_is_independent_of_tail_inputs() -> None:
    sk = _phases("P1", "P2")
    head = stable_head(_JOB, sk)
    a = build_planner_input(
        job=_JOB, skeleton=sk, phase_id="P1", epoch_counter=0, log_index=[],
        last_epoch_rows=None, reask_errors=[],
    )
    b = build_planner_input(
        job=_JOB, skeleton=sk, phase_id="P1", epoch_counter=3, log_index=["P1/E1/T1/handoff.json"],
        last_epoch_rows=[{"task": "T1", "status": "done"}], reask_errors=["bad"],
    )
    assert a.startswith(head) and b.startswith(head)  # head is the shared prefix


# --- volatile tail content -----------------------------------------------------


def test_volatile_tail_carries_state_and_request_not_payloads() -> None:
    tail = volatile_tail(
        phase_id="P1", epoch_counter=2,
        log_index=["P1/E1/T1/handoff.json"],
        last_epoch_rows=[{"task": "T1", "status": "done", "resulting_state": "made f1"}],
        reask_errors=[],
    )
    assert "phase: P1" in tail
    assert "epoch_counter: 2" in tail
    assert "P1/E1/T1/handoff.json" in tail  # a log key reference
    assert "made f1" in tail
    assert "<request>" in tail


def test_volatile_tail_appends_reask_errors() -> None:
    tail = volatile_tail(
        phase_id="P1", epoch_counter=0, log_index=[], last_epoch_rows=None,
        reask_errors=["schema: 'tasks' is required"],
    )
    assert "<errors>" in tail
    assert "schema: 'tasks' is required" in tail


def test_flatten_last_epoch_reads_handoff_refs(tmp_path: Path) -> None:
    run_dir = create_run_dir(tmp_path, "r")
    key = "P1/E1/T1/handoff.json"
    dest = run_dir.resolve(key)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(
        '{"schema_version":"1","task_id":"P1/E1/T1","status":"DONE",'
        '"what_changed":[{"kind":"file","ref":"f1.txt"}],'
        '"resulting_state":"created f1.txt","downstream_needs":["P1/E1/T1/handoff.json"],'
        '"not_done":["theming"],'
        '"checks":[],"occupancy":{"compacted":false,"subagent_splits":0}}',
        encoding="utf-8",
    )
    outcome = EpochOutcome(
        phase_id="P1", epoch_id="E1", status="completed",
        tasks=[
            TaskResult(task_id="T1", fq_task_id="P1/E1/T1", status="done", attempts=1,
                       tier="local", handoff_key=key, failure_reason=None),
            TaskResult(task_id="T2", fq_task_id="P1/E1/T2", status="failed", attempts=4,
                       tier="cloud", handoff_key=None, failure_reason="no handoff.json written"),
        ],
        integration=IntegrationOutcome(status="completed", branch="b", merged=["T1"], conflict=None),
    )
    rows = flatten_last_epoch(run_dir, outcome)
    assert rows[0]["resulting_state"] == "created f1.txt"
    assert rows[0]["downstream_needs"] == ["P1/E1/T1/handoff.json"]
    # G10: each DONE row now ALSO carries what_changed + not_done from its handoff.
    assert rows[0]["what_changed"] == ["file:f1.txt"]
    assert rows[0]["not_done"] == ["theming"]
    assert rows[1]["status"] == "failed"
    assert rows[1]["failure_reason"] == "no handoff.json written"


def test_flatten_last_epoch_preserves_full_what_changed(tmp_path: Path) -> None:
    """The handoff's own schema bounds the worker-written ``what_changed.ref`` /
    ``not_done`` fields legitimately (<=256), so the planner row carries them IN FULL:
    the extra embedding-truncation was a band-aid (silent information loss) and is gone.
    The full handoff is referenceable via the workspace manifest if more is needed."""

    run_dir = create_run_dir(tmp_path, "r")
    key = "P1/E1/T1/handoff.json"
    dest = run_dir.resolve(key)
    dest.parent.mkdir(parents=True, exist_ok=True)
    # The handoff's own 256-char ref bound is the legitimate limit; the row keeps it all.
    long_ref = "x" * 256
    dest.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "task_id": "P1/E1/T1",
                "status": "DONE",
                "what_changed": [{"kind": "file", "ref": long_ref}],
                "resulting_state": "ok",
                "downstream_needs": [],
                "not_done": [],
                "checks": [],
                "occupancy": {"compacted": False, "subagent_splits": 0},
            }
        ),
        encoding="utf-8",
    )
    outcome = EpochOutcome(
        phase_id="P1", epoch_id="E1", status="completed",
        tasks=[
            TaskResult(task_id="T1", fq_task_id="P1/E1/T1", status="done", attempts=1,
                       tier="local", handoff_key=key, failure_reason=None),
        ],
        integration=IntegrationOutcome(status="completed", branch="b", merged=["T1"], conflict=None),
    )
    rows = flatten_last_epoch(run_dir, outcome)
    wc = rows[0]["what_changed"]
    assert isinstance(wc, list)
    assert wc[0] == f"file:{long_ref}"  # full, untruncated, no marker
    assert "...[truncated]" not in wc[0]


# --- the verdict (digest + evidence + gaps) travels by reference, not embedded -


def test_volatile_tail_has_no_epoch_digest_block() -> None:
    """The inlined ``<epoch_digest>`` block is gone: the digest now lives in the
    persisted verdict.json (surfaced in the <workspace> manifest) and the planner reads
    it by reference, so the prompt never embeds it."""

    tail = volatile_tail(
        phase_id="P1", epoch_counter=2, log_index=[], last_epoch_rows=[],
        reask_errors=[],
    )
    assert "<epoch_digest>" not in tail


def test_build_planner_input_has_no_epoch_digest_block() -> None:
    sk = _phases("P1", "P2")
    full = build_planner_input(
        job=_JOB, skeleton=sk, phase_id="P1", epoch_counter=1, log_index=[],
        last_epoch_rows=[], reask_errors=[],
    )
    assert "<epoch_digest>" not in full
    # The stable head stays byte-identical (the verdict travels via the volatile tail's
    # workspace manifest, never the head).
    assert full.startswith(stable_head(_JOB, sk))


# --- the read-capable <workspace> block (planner pull access) ------------------


def test_preamble_invites_reading_the_workspace() -> None:
    """The planner runs as a read-capable agent (codex read-only -C repo / claude
    Read+Grep over the repo). The preamble must INVITE it to grep/read the exposed
    paths for steering, while keeping the JSON-only contract + deterministic-floor
    disposition intact."""

    assert "<workspace>" in SYSTEM_PREAMBLE
    # Read-capability is stated, and that it is for STEERING only (the floor disposes).
    assert "read" in SYSTEM_PREAMBLE and "grep" in SYSTEM_PREAMBLE.lower()
    assert "STEERING" in SYSTEM_PREAMBLE or "steering" in SYSTEM_PREAMBLE
    # The JSON-only output contract is reaffirmed after the reading invitation.
    assert "EXACTLY ONE" in SYSTEM_PREAMBLE


def test_workspace_block_carries_absolute_paths(tmp_path: Path) -> None:
    tip = tmp_path / "tip"
    tip.mkdir()
    run_root = tmp_path / "run"
    run_root.mkdir()
    ws = WorkspaceInfo(
        integration_tip=tip,
        keyed_log_root=run_root,
        manifest=[
            ("P1/E1/T1/handoff.json", run_root / "P1" / "E1" / "T1" / "handoff.json"),
        ],
    )
    tail = volatile_tail(
        phase_id="P1", epoch_counter=1, log_index=[], last_epoch_rows=[],
        reask_errors=[], workspace=ws,
    )
    assert "<workspace>" in tail
    # The two roots are exposed as ABSOLUTE paths the planner may grep/read.
    assert str(tip.resolve()) in tail
    assert str(run_root.resolve()) in tail
    # The per-key manifest resolves each live log key to its absolute path.
    assert "P1/E1/T1/handoff.json" in tail
    assert str((run_root / "P1" / "E1" / "T1" / "handoff.json")) in tail


def test_workspace_block_omitted_cleanly_when_absent() -> None:
    tail = volatile_tail(
        phase_id="P1", epoch_counter=1, log_index=[], last_epoch_rows=[],
        reask_errors=[],
    )
    assert "<workspace>" not in tail


def test_workspace_block_omits_missing_tip_and_empty_manifest(tmp_path: Path) -> None:
    """A run with no integration tip yet (first implement epoch not run) and an
    empty keyed log omits those sub-parts cleanly, no dangling None path."""

    run_root = tmp_path / "run"
    run_root.mkdir()
    ws = WorkspaceInfo(
        integration_tip=None, keyed_log_root=run_root, manifest=[]
    )
    tail = volatile_tail(
        phase_id="P1", epoch_counter=0, log_index=[], last_epoch_rows=[],
        reask_errors=[], workspace=ws,
    )
    assert "<workspace>" in tail  # the keyed-log root alone is still worth exposing
    assert str(run_root.resolve()) in tail
    assert "None" not in tail  # a missing tip never leaks a literal None


def test_workspace_block_surfaces_repo_map_path_by_reference(tmp_path: Path) -> None:
    """The structural repo-map is delivered BY REFERENCE: when present, the
    <workspace> block carries a clearly-labeled ``repo_map`` entry pointing at the
    on-disk map file (a ranked map of the current integration tip), not the inline
    map text. The planner reads that path for structural planning."""

    run_root = tmp_path / "run"
    run_root.mkdir()
    map_file = run_root / "planner_repo_map.txt"
    map_file.write_text("util.py:\n  def shared_helper():\n")
    ws = WorkspaceInfo(
        integration_tip=None,
        keyed_log_root=run_root,
        manifest=[],
        repo_map_path=map_file,
    )
    tail = volatile_tail(
        phase_id="P1", epoch_counter=1, log_index=[], last_epoch_rows=[],
        reask_errors=[], workspace=ws,
    )
    assert "<workspace>" in tail
    assert "repo_map" in tail
    # The absolute path to the on-disk map is surfaced; the map TEXT is not inlined.
    assert str(map_file.resolve()) in tail
    assert "shared_helper" not in tail


def test_workspace_block_omits_repo_map_entry_when_absent(tmp_path: Path) -> None:
    """Below threshold / first epoch the map is None -> no file is written and the
    workspace omits the repo_map entry cleanly (no dangling path)."""

    run_root = tmp_path / "run"
    run_root.mkdir()
    ws = WorkspaceInfo(
        integration_tip=None, keyed_log_root=run_root, manifest=[], repo_map_path=None
    )
    tail = volatile_tail(
        phase_id="P1", epoch_counter=0, log_index=[], last_epoch_rows=[],
        reask_errors=[], workspace=ws,
    )
    assert "<workspace>" in tail
    assert "repo_map" not in tail


def test_preamble_points_planner_at_workspace_repo_map_path() -> None:
    """SYSTEM_PREAMBLE must describe the repo-map as an on-disk file referenced from
    the <workspace> (read it for structural planning), NOT as an inline prompt block.
    The old inline-map wording is gone."""

    assert "repo_map" in SYSTEM_PREAMBLE
    assert "<repo_map>" not in SYSTEM_PREAMBLE


def test_build_planner_input_threads_workspace_in_tail(tmp_path: Path) -> None:
    sk = _phases("P1", "P2")
    run_root = tmp_path / "run"
    run_root.mkdir()
    ws = WorkspaceInfo(integration_tip=tmp_path / "tip", keyed_log_root=run_root, manifest=[])
    full = build_planner_input(
        job=_JOB, skeleton=sk, phase_id="P1", epoch_counter=1, log_index=[],
        last_epoch_rows=[], reask_errors=[], workspace=ws,
    )
    assert "<workspace>" in full
    # The workspace rides the volatile tail; the stable head stays byte-identical.
    assert full.startswith(stable_head(_JOB, sk))


# --- validation pipeline -------------------------------------------------------

_EMPTY: frozenset[str] = frozenset()


def _gate(payload_text: str | None, *, skeleton_exists: bool, log: frozenset[str] = _EMPTY):
    return validate_decision(
        payload_text, existing_log_keys=log, completed_phase_ids=_EMPTY,
        skeleton_exists=skeleton_exists,
    )


def test_gate_accepts_valid_first_skeleton() -> None:
    import json

    res = _gate(json.dumps(two_phase_skeleton()), skeleton_exists=False)
    assert res.errors == []
    assert res.decision is not None and res.decision.tool == "propose_skeleton"


def test_gate_rejects_no_json() -> None:
    res = _gate(None, skeleton_exists=False)
    assert res.decision is None and res.errors


def test_gate_rejects_schema_violation() -> None:
    import json

    res = _gate(json.dumps({"schema_version": "1", "tool": "implement", "args": {}}),
                skeleton_exists=True)
    assert res.decision is None
    assert any("schema" in e for e in res.errors)


def test_gate_rejects_unknown_input_log_key() -> None:
    import json

    dec = implement_decision({**impl_task("T1", "f1.txt"), "inputs": ["P9/E9/T9/ghost.json"]})
    res = _gate(json.dumps(dec), skeleton_exists=True, log=frozenset())
    assert res.decision is None
    assert any("ghost" in e for e in res.errors)


def test_gate_rejects_overlapping_ownership() -> None:
    import json

    dec = implement_decision(impl_task("T1", "shared.txt"), impl_task("T2", "shared.txt"))
    res = _gate(json.dumps(dec), skeleton_exists=True)
    assert res.decision is None
    assert any("overlap" in e for e in res.errors)


def test_gate_accepts_function_call_shorthand() -> None:
    import json

    # The model's common shorthand {<tool>: <args>} (no envelope) is canonicalized.
    shorthand = {"propose_skeleton": two_phase_skeleton()["args"]}
    res = _gate(json.dumps(shorthand), skeleton_exists=False)
    assert res.errors == []
    assert res.decision is not None and res.decision.tool == "propose_skeleton"


def test_gate_supplies_missing_schema_version() -> None:
    import json

    no_version = {"tool": "escalate_run", "args": {"reason": "stuck"}}
    res = _gate(json.dumps(no_version), skeleton_exists=True)
    assert res.errors == []
    assert res.decision is not None and res.decision.tool == "escalate_run"


def test_gate_position_legality() -> None:
    import json

    # propose_skeleton illegal once a skeleton exists.
    res = _gate(json.dumps(two_phase_skeleton()), skeleton_exists=True)
    assert res.decision is None
    assert any("first decision" in e for e in res.errors)
    # a non-skeleton decision illegal before any skeleton exists.
    res2 = _gate(json.dumps(complete_decision({"cmd": "true"})), skeleton_exists=False)
    assert res2.decision is None
    assert any("must be propose_skeleton" in e for e in res2.errors)


# --- size gate (Part 4B): the deterministic decomposition floor ----------------


def _impl_task_n(tid: str, *globs: str) -> dict[str, object]:
    return {
        "id": tid,
        "goal": "g",
        "done_when": [{"cmd": "true"}],
        "file_ownership": list(globs),
    }


def test_gate_rejects_oversized_implement_task_and_names_it() -> None:
    import json

    # Seven owned globs > the default local bound (5): rejected, the offending
    # task id is named so the re-ask tells the planner exactly what to split.
    big = _impl_task_n("T3", "a", "b", "c", "d", "e", "f", "g")
    res = validate_decision(
        json.dumps(implement_decision(big)),
        existing_log_keys=_EMPTY, completed_phase_ids=_EMPTY, skeleton_exists=True,
    )
    assert res.decision is None
    assert any("T3" in e and "too big" in e for e in res.errors)


def test_gate_rejects_whole_repo_ownership_on_fresh_implement() -> None:
    import json

    for glob in ("**", "**/*", "*"):
        res = validate_decision(
            json.dumps(implement_decision(_impl_task_n("T1", glob))),
            existing_log_keys=_EMPTY, completed_phase_ids=_EMPTY, skeleton_exists=True,
        )
        assert res.decision is None, glob
        assert any("T1" in e and "whole-repo" in e for e in res.errors), glob


def test_gate_allows_broad_scope_on_failed_epoch_repair() -> None:
    import json

    # A handle_failed_epoch retry path is exempt: when a failed epoch is awaiting
    # disposition the size gate is skipped (a repair can't predict its files).
    # (Position legality independently constrains the tool here; we assert the
    # size gate itself does not fire on an implement decision in that mode.)
    from grindstone.planner import _size_gate_violations
    from grindstone.contracts.models import parse_decision

    broad = parse_decision(implement_decision(_impl_task_n("T1", "**/*")))
    # Fresh decomposition: rejected.
    assert _size_gate_violations(
        broad, failed_epoch_active=False, has_senior=False,
        local_max_task_files=5, senior_max_task_files=12,
    )
    # Failed-epoch repair: exempt (no violations).
    assert _size_gate_violations(
        broad, failed_epoch_active=True, has_senior=False,
        local_max_task_files=5, senior_max_task_files=12,
    ) == []


def test_gate_size_bound_is_tier_aware_for_visual_epochs() -> None:
    import json

    # Six globs: over the local bound (5) but under the senior bound (12). A
    # visual implement epoch starts on senior, so with a senior tier present it
    # is judged against the senior bound and PASSES; without senior it does not.
    six = _impl_task_n("T1", "a", "b", "c", "d", "e", "f")
    dec = implement_decision(six)
    dec["args"]["visual"] = True  # type: ignore[index]
    payload = json.dumps(dec)
    with_senior = validate_decision(
        payload, existing_log_keys=_EMPTY, completed_phase_ids=_EMPTY,
        skeleton_exists=True, has_senior=True,
        local_max_task_files=5, senior_max_task_files=12,
    )
    assert with_senior.decision is not None  # senior bound (12) -> passes
    no_senior = validate_decision(
        payload, existing_log_keys=_EMPTY, completed_phase_ids=_EMPTY,
        skeleton_exists=True, has_senior=False,
        local_max_task_files=5, senior_max_task_files=12,
    )
    assert no_senior.decision is None  # falls back to local bound (5) -> rejected
    assert any("T1" in e and "too big" in e for e in no_senior.errors)


# --- content-grep forbiddance (gate rebalance G1) ------------------------------


def _impl_task_check(tid: str, fname: str, cmd: str) -> dict[str, object]:
    return {
        "id": tid,
        "goal": f"create {fname}",
        "done_when": [{"cmd": cmd}],
        "file_ownership": [fname],
    }


def test_gate_rejects_content_grep_check_with_steering_message() -> None:
    import json

    for cmd in ('rg -q "Honey|Sky" plan.md', "grep -q DONE out.txt"):
        dec = implement_decision(_impl_task_check("T1", "f.txt", cmd))
        res = validate_decision(
            json.dumps(dec),
            existing_log_keys=_EMPTY, completed_phase_ids=_EMPTY, skeleton_exists=True,
        )
        assert res.decision is None, cmd
        # The rejection names the offending task and steers to `criteria`.
        assert any("T1" in e and "criteria" in e for e in res.errors), cmd


def test_gate_allows_structural_checks_in_done_when() -> None:
    import json

    # Structural checks (existence, type-check, test) remain legal.
    for cmd in ("test -f greeting.txt", "npx tsc --noEmit", "npm test"):
        dec = implement_decision(_impl_task_check("T1", "f.txt", cmd))
        res = validate_decision(
            json.dumps(dec),
            existing_log_keys=_EMPTY, completed_phase_ids=_EMPTY, skeleton_exists=True,
        )
        assert res.decision is not None, cmd


def test_preamble_forbids_content_greps_and_teaches_criteria() -> None:
    # The preamble explicitly forbids content-grep checks and steers the planner
    # to express semantic acceptance as natural-language `criteria` instead.
    assert "CONTENT-GREP" in SYSTEM_PREAMBLE
    assert "`criteria`" in SYSTEM_PREAMBLE
    # The forbiddance names the grep family it rejects.
    assert "grep" in SYSTEM_PREAMBLE
    # checks/done_when are scoped to STRUCTURAL facts.
    assert "STRUCTURAL" in SYSTEM_PREAMBLE
    # The verification floor is owned by repo config + core, not restated.
    assert "FLOOR" in SYSTEM_PREAMBLE
    # The preamble's own examples must not author content-grep checks.
    assert "grep -q" not in SYSTEM_PREAMBLE


def test_preamble_teaches_three_level_skill_split() -> None:
    # Part 4A: the decomposition guidance is split into clearly delineated,
    # per-level sections (phasing / epoch / task), one skill each.
    assert "[LEVEL 1: PHASING]" in SYSTEM_PREAMBLE
    assert "[LEVEL 2: EPOCH]" in SYSTEM_PREAMBLE
    assert "[LEVEL 3: TASK]" in SYSTEM_PREAMBLE
    # The implement-phase baseline-dependencies epoch (committed manifest/lockfile).
    assert "BASELINE DEPENDENCIES" in SYSTEM_PREAMBLE
    assert "lockfile" in SYSTEM_PREAMBLE
    # The size gate is advertised in the task-level guidance.
    assert "SIZE GATE" in SYSTEM_PREAMBLE
