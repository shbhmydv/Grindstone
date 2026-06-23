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
from grindstone.config import load_operating_skill
from grindstone.planner import (
    BACKOFF_CAP_S,
    MAX_RATE_LIMIT_WAITS,
    PLANNER_CORE,
    PLANNER_SCENARIOS,
    FailedEpochInfo,
    PlannerHardError,
    RateLimited,
    TransportError,
    WorkspaceInfo,
    WorkerTimeout,
    backoff_delay,
    build_planner_input,
    classify_failure,
    flatten_last_epoch,
    select_planner_scenario,
    stable_head,
    validate_decision,
    volatile_tail,
)
from grindstone.rundir import create_run_dir


def _plan_skeleton() -> str:
    return load_operating_skill("planner", "plan_skeleton")


def _plan_epoch() -> str:
    return load_operating_skill("planner", "plan_epoch")


def _repair_epoch() -> str:
    return load_operating_skill("planner", "repair_epoch")


def _all_planner_guidance() -> str:
    """The union of the always-on CORE + every selectable scenario skill: the full
    instruction surface the split must preserve (nothing silently dropped)."""

    return PLANNER_CORE + _plan_skeleton() + _plan_epoch() + _repair_epoch()

from tests.grindstone.conftest import (
    complete_decision,
    impl_task,
    implement_decision,
    phase_dict,
    two_phase_skeleton,
)


def _phases(*ids: str) -> list[Phase]:
    return [Phase.model_validate(phase_dict(i)) for i in ids]


def _failed_epoch() -> FailedEpochInfo:
    return FailedEpochInfo(
        epoch_id="E1",
        failed_tasks=[("T1", "no handoff.json written")],
        failed_checks=["cmd `npx tsc` (exit 1)"],
        passing_handoffs=[("T1", "implemented as asked")],
        disposed_count=1,
        cap=3,
    )


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


# --- operating-skill split: CORE + selected scenario (Batch 1) -----------------


def test_select_planner_scenario_maps_state_to_skill() -> None:
    # No skeleton yet -> decompose the job (propose_skeleton is the only legal tool).
    assert (
        select_planner_scenario(skeleton_exists=False, failed_epoch_active=False)
        == "plan_skeleton"
    )
    # A failed epoch awaiting disposition -> the focused repair scenario.
    assert (
        select_planner_scenario(skeleton_exists=True, failed_epoch_active=True)
        == "repair_epoch"
    )
    # Steady state (skeleton exists, nothing failed) -> the default work scenario.
    assert (
        select_planner_scenario(skeleton_exists=True, failed_epoch_active=False)
        == "plan_epoch"
    )
    # PRECEDENCE: the skeleton question is settled FIRST (a failed epoch cannot
    # exist before a skeleton); no-skeleton dominates even if the flag is set.
    assert (
        select_planner_scenario(skeleton_exists=False, failed_epoch_active=True)
        == "plan_skeleton"
    )
    # Every name the selector can return has a loadable skill file.
    for sk in (False, True):
        for fe in (False, True):
            assert (
                select_planner_scenario(skeleton_exists=sk, failed_epoch_active=fe)
                in PLANNER_SCENARIOS
            )


def test_build_planner_input_composes_core_plus_one_scenario() -> None:
    # Sentinels unique to each scenario file (present in exactly one of the three).
    sentinels = {
        "plan_skeleton": "[LEVEL 1: PHASING]",
        "plan_epoch": "[LEVEL 2: EPOCH]",
        "repair_epoch": "GATE SKEPTICISM",
    }
    core_sentinel = "A SCENARIO skill follows"
    cases = {
        "plan_skeleton": dict(skeleton=None, failed_epoch=None),
        "plan_epoch": dict(skeleton=_phases("P1", "P2"), failed_epoch=None),
        "repair_epoch": dict(skeleton=_phases("P1", "P2"), failed_epoch=_failed_epoch()),
    }
    for scenario, state in cases.items():
        prompt = build_planner_input(
            job=_JOB, phase_id="P1", epoch_counter=1, log_index=[],
            last_epoch_rows=None, reask_errors=[],
            **state,  # type: ignore[arg-type]
        )
        # The always-on CORE is present in every composition.
        assert core_sentinel in prompt
        assert f'<scenario name="{scenario}">' in prompt
        # The selected scenario's sentinel is present...
        assert sentinels[scenario] in prompt
        # ...and the OTHER two scenarios' sentinels are NOT.
        for other, sent in sentinels.items():
            if other != scenario:
                assert sent not in prompt


def test_split_preserves_every_load_bearing_instruction() -> None:
    """Content-preservation guard: the monolithic preamble was split into CORE +
    three scenario files. Every load-bearing instruction must survive SOMEWHERE in
    the union, nothing silently dropped by the reorganization."""

    union = _all_planner_guidance()
    for phrase in (
        # task sizing / decomposition (was LEVEL 2/3)
        "MUST NOT consume each other's",
        "90k",
        "epoch_budget",
        "phase escalation",
        "Decompose CONSERVATIVELY",
        "VERBATIM",
        "1024",
        "BASELINE DEPENDENCIES",
        "lockfile",
        "SIZE GATE",
        # mode selection + done_when scoping
        "DESTINATION",
        "even prose",
        "consumed via the keyed log",
        "artifact itself",
        "never repo build/test commands",
        # references / artifact publication (CORE)
        "published to the keyed log",
        "bare filename",
        # structural-checks + content-grep contract (CORE)
        "CONTENT-GREP",
        "STRUCTURAL",
        "FLOOR",
        "`criteria`",
        # failed-epoch repair scenario
        "handle_failed_epoch",
        "GATE SKEPTICISM",
        # phasing scenario
        "[LEVEL 1: PHASING]",
        "[LEVEL 2: EPOCH]",
        "[LEVEL 3: TASK]",
    ):
        assert phrase in union, f"lost instruction: {phrase!r}"
    # The split's own examples must still author no content-grep check.
    assert "grep -q" not in union


def test_core_holds_the_always_on_contract() -> None:
    # Output discipline + the authoritative envelope live in the always-on CORE.
    assert "EXACTLY ONE tool call" in PLANNER_CORE
    assert '{"schema_version":"1","tool":"<one tool name>","args":{ ... }}' in PLANNER_CORE
    # The cross-cutting check contract (structural-only, no content-greps, floor).
    assert "CONTENT-GREP" in PLANNER_CORE
    assert "FLOOR" in PLANNER_CORE
    # Read-capable planning + verdict-by-reference steering are call-invariant.
    assert "READ-CAPABLE PLANNING" in PLANNER_CORE
    assert "verdict.json" in PLANNER_CORE


def test_plan_epoch_skill_carries_valid_implement_example() -> None:
    # The worked implement example moved to the plan_epoch scenario; still valid JSON.
    skill = _plan_epoch()
    start = skill.index('{"schema_version":"1","tool":"implement"')
    obj, _ = json.JSONDecoder().raw_decode(skill[start:])
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
                       tier="worker", handoff_key=key, failure_reason=None),
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
                       tier="worker", handoff_key=key, failure_reason=None),
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

    # Read-capable planning is call-invariant, so it lives in the always-on CORE.
    assert "<workspace>" in PLANNER_CORE
    # Read-capability is stated, and that it is for STEERING only (the floor disposes).
    assert "read" in PLANNER_CORE and "grep" in PLANNER_CORE.lower()
    assert "STEERING" in PLANNER_CORE or "steering" in PLANNER_CORE
    # The JSON-only output contract is reaffirmed after the reading invitation.
    assert "EXACTLY ONE" in PLANNER_CORE


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
    """The CORE must describe the repo-map as an on-disk file referenced from the
    <workspace> (read it for structural planning), NOT as an inline prompt block.
    The old inline-map wording is gone."""

    assert "repo_map" in PLANNER_CORE
    assert "<repo_map>" not in _all_planner_guidance()


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


def test_gate_accepts_phase_complete_once_skeleton_exists() -> None:
    import json

    # phase_complete is a legal steady-state decision once a skeleton exists, and the
    # core gate parses it (the deliverable EXISTENCE grounding runs later, at dispatch).
    dec = {
        "schema_version": "1",
        "tool": "phase_complete",
        "args": {"summary": "phase deliverables built", "deliverables": ["src/a.ts"]},
    }
    res = _gate(json.dumps(dec), skeleton_exists=True)
    assert res.errors == []
    assert res.decision is not None and res.decision.tool == "phase_complete"
    # ...but it is illegal as the FIRST decision (before any skeleton).
    res0 = _gate(json.dumps(dec), skeleton_exists=False)
    assert res0.decision is None
    assert any("must be propose_skeleton" in e for e in res0.errors)


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

    # Seven owned CONCRETE files > the default local bound (5): rejected, the
    # offending task id is named so the re-ask tells the planner what to split.
    big = _impl_task_n("T3", "a.py", "b.py", "c.py", "d.py", "e.py", "f.py", "g.py")
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
        assert any("T1" in e and "broad glob" in e for e in res.errors), glob


def test_gate_rejects_broad_scoped_wildcard_on_fresh_implement() -> None:
    import json

    # The incident: a single SCOPED wildcard (a whole subsystem) used to pass both
    # the whole-repo check and the per-tier cap (one glob, count 1), so every
    # complex epoch became one giant senior task. The gate now rejects ANY
    # wildcard entry (must enumerate concrete files), naming the offending task.
    for glob in ("src/design-system/**", "src/*", "lib/*.ts", "a/b/?.py"):
        res = validate_decision(
            json.dumps(implement_decision(_impl_task_n("T1", glob))),
            existing_log_keys=_EMPTY, completed_phase_ids=_EMPTY, skeleton_exists=True,
        )
        assert res.decision is None, glob
        assert any("T1" in e and "broad glob" in e for e in res.errors), glob


def test_gate_accepts_enumerated_multi_file_implement_task() -> None:
    import json

    # An ENUMERATED slice of concrete files within the local cap passes cleanly.
    enumerated = _impl_task_n(
        "T1", "src/theme.ts", "src/tokens.ts", "src/index.ts"
    )
    res = validate_decision(
        json.dumps(implement_decision(enumerated)),
        existing_log_keys=_EMPTY, completed_phase_ids=_EMPTY, skeleton_exists=True,
    )
    assert res.decision is not None
    assert res.errors == []


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


def _senior_task(tid: str, *files: str) -> dict[str, object]:
    return {**_impl_task_n(tid, *files), "senior": True}


def test_gate_size_bound_is_tier_aware_per_task_senior_flag() -> None:
    import json

    # Six concrete files: over the local bound (5) but under the senior bound (12).
    # A task flagged senior:true is judged against the senior bound, so with a
    # senior tier present it PASSES; without a senior tier it falls back to local
    # and is rejected (its work would actually run on the local rig).
    six = _senior_task("T1", "a.ts", "b.ts", "c.ts", "d.ts", "e.ts", "f.ts")
    payload = json.dumps(implement_decision(six))
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


def test_gate_local_task_uses_local_bound_even_with_senior_sibling() -> None:
    import json

    # Per-TASK bounds: in one epoch a senior task may carry up to 12 files while a
    # local sibling is still held to 5. A 6-file LOCAL task is rejected even though
    # a senior tier exists and a senior sibling is within its larger bound.
    local_big = _impl_task_n("T1", "a.ts", "b.ts", "c.ts", "d.ts", "e.ts", "f.ts")
    senior_ok = _senior_task("T2", "x/1.ts", "x/2.ts", "x/3.ts", "x/4.ts",
                             "x/5.ts", "x/6.ts", "x/7.ts")
    res = validate_decision(
        json.dumps(implement_decision(local_big, senior_ok)),
        existing_log_keys=_EMPTY, completed_phase_ids=_EMPTY,
        skeleton_exists=True, has_senior=True,
        local_max_task_files=5, senior_max_task_files=12,
    )
    assert res.decision is None
    assert any("T1" in e and "too big" in e for e in res.errors)
    assert not any("T2" in e for e in res.errors)


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
    # The content-grep contract is cross-cutting, so it lives in the always-on CORE:
    # it forbids content-grep checks and steers semantic acceptance into `criteria`.
    assert "CONTENT-GREP" in PLANNER_CORE
    assert "`criteria`" in PLANNER_CORE
    # The forbiddance names the grep family it rejects.
    assert "grep" in PLANNER_CORE
    # checks/done_when are scoped to STRUCTURAL facts.
    assert "STRUCTURAL" in PLANNER_CORE
    # The verification floor is owned by repo config + core, not restated.
    assert "FLOOR" in PLANNER_CORE
    # No part of the split (core + every scenario) authors a content-grep check.
    assert "grep -q" not in _all_planner_guidance()


def test_preamble_teaches_three_level_skill_split() -> None:
    # The decomposition guidance is split per level, now across the scenario skills:
    # LEVEL 1 phasing lives in plan_skeleton; LEVELS 2/3 live in plan_epoch.
    assert "[LEVEL 1: PHASING]" in _plan_skeleton()
    assert "[LEVEL 2: EPOCH]" in _plan_epoch()
    assert "[LEVEL 3: TASK]" in _plan_epoch()
    # The implement-phase baseline-dependencies epoch (committed manifest/lockfile).
    assert "BASELINE DEPENDENCIES" in _plan_epoch()
    assert "lockfile" in _plan_epoch()
    # The size gate is advertised in the task-level guidance.
    assert "SIZE GATE" in _plan_epoch()


# --- domain skills: planner index block + skill-name gate ----------------------


def test_domain_skills_block_renders_index_when_present() -> None:
    """A non-empty catalogue index renders a <domain_skills> selection block in the
    volatile tail (name + description), instructing minimal per-task selection."""

    index = {
        "rn-nav": "React Navigation patterns; use when wiring screens.",
        "rn-a11y": "accessibility; screen-reader / contrast work.",
    }
    prompt = build_planner_input(
        job=_JOB, skeleton=_phases("P1", "P2"), phase_id="P1", epoch_counter=1,
        log_index=[], last_epoch_rows=None, reask_errors=[],
        domain_skill_index=index,
    )
    # Sentinel distinct from the plan_epoch scenario nudge (which mentions the tag).
    assert "Domain skills this target repo provides" in prompt
    assert "rn-nav: React Navigation patterns" in prompt
    assert "rn-a11y: accessibility" in prompt


def test_domain_skills_block_absent_when_index_empty() -> None:
    """No catalogue (the common case) renders no block at all, byte-clean no-op."""

    _SENTINEL = "Domain skills this target repo provides"
    prompt = build_planner_input(
        job=_JOB, skeleton=_phases("P1", "P2"), phase_id="P1", epoch_counter=1,
        log_index=[], last_epoch_rows=None, reask_errors=[],
    )
    assert _SENTINEL not in prompt
    # And explicitly passing an empty index is identical (no block).
    empty = build_planner_input(
        job=_JOB, skeleton=_phases("P1", "P2"), phase_id="P1", epoch_counter=1,
        log_index=[], last_epoch_rows=None, reask_errors=[],
        domain_skill_index={},
    )
    assert _SENTINEL not in empty


def _impl_with_skills(tid: str, fname: str, skills: list[str]) -> dict[str, object]:
    return {**impl_task(tid, fname), "skills": skills}


def test_gate_rejects_unknown_skill_name() -> None:
    # A task naming a skill the catalogue does not advertise is rejected, the
    # offending task + skill are named so the re-ask steers the planner.
    dec = implement_decision(_impl_with_skills("T1", "f1.txt", ["ghost-skill"]))
    res = validate_decision(
        json.dumps(dec), existing_log_keys=_EMPTY, completed_phase_ids=_EMPTY,
        skeleton_exists=True, known_skill_names=frozenset({"rn-nav"}),
    )
    assert res.decision is None
    assert any("ghost-skill" in e and "T1" in e for e in res.errors)


def test_gate_accepts_known_skill_name() -> None:
    dec = implement_decision(_impl_with_skills("T1", "f1.txt", ["rn-nav"]))
    res = validate_decision(
        json.dumps(dec), existing_log_keys=_EMPTY, completed_phase_ids=_EMPTY,
        skeleton_exists=True, known_skill_names=frozenset({"rn-nav", "rn-a11y"}),
    )
    assert res.errors == []
    assert res.decision is not None and res.decision.tool == "implement"


def test_gate_no_catalogue_means_skills_must_be_empty() -> None:
    # Default known set is empty (no catalogue): a task may select NO skill...
    ok = validate_decision(
        json.dumps(implement_decision(impl_task("T1", "f1.txt"))),
        existing_log_keys=_EMPTY, completed_phase_ids=_EMPTY, skeleton_exists=True,
    )
    assert ok.errors == [] and ok.decision is not None
    # ...but naming any skill with no catalogue is rejected.
    bad = validate_decision(
        json.dumps(implement_decision(_impl_with_skills("T1", "f1.txt", ["rn-nav"]))),
        existing_log_keys=_EMPTY, completed_phase_ids=_EMPTY, skeleton_exists=True,
    )
    assert bad.decision is None
    assert any("rn-nav" in e for e in bad.errors)
