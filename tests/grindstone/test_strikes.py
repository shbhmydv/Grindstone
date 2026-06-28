"""The per-task STRIKE-LADDER: the deterministic escalation state machine.

Pure-unit coverage of ``grindstone.strikes`` (no loop, no git): the inherited-strike
match across re-decomposition (the trickiest part), the plan-time ladder (BLOCK/park
at strike 2, untouched below - no tier mutation; senior is reached in-epoch now), the
close-time ledger recompute (increment on a carried fail, resolve on a land, supersede
on re-decompose), journal reconstruction (resume-safety), and the carried-nudge
projection.
"""

from __future__ import annotations

from grindstone import strikes
from grindstone.contracts.models import Task
from grindstone.events import (
    EpochCompleted,
    StrikeLedger,
    StrikeLedgerEntry,
    TaskParked,
)


# --- builders ------------------------------------------------------------------


def _impl(tid: str, owned: list[str], *, tier: str = "local") -> Task:
    return Task(
        id=tid, mode="implement", goal="g", file_ownership=owned, tier=tier  # type: ignore[arg-type]
    )


def _research(tid: str, out: str, *, tier: str = "local") -> Task:
    return Task(id=tid, mode="research", goal="g", artifact_out=out, tier=tier)  # type: ignore[arg-type]


def _entry(
    *, ownership: list[str] | None = None, artifact_out: str | None = None,
    mode: str = "implement", strikes_: int = 1, reason: str = "boom",
) -> strikes.StrikeEntry:
    return strikes.StrikeEntry(
        ownership=tuple(ownership or []),
        artifact_out=artifact_out,
        mode=mode,
        strikes=strikes_,
        reason=reason,
    )


# --- inherited_strikes (lineage match) -----------------------------------------


def test_no_struck_lineage_is_zero() -> None:
    assert strikes.inherited_strikes([], _impl("T1", ["a.py"])) == 0


def test_exact_file_inherits() -> None:
    led = [_entry(ownership=["a.py"], strikes_=2)]
    assert strikes.inherited_strikes(led, _impl("T1", ["a.py"])) == 2


def test_redecomposed_child_inherits_parent_strikes() -> None:
    # Parent owned [a.py, b.py] at 1 strike; the planner re-decomposes into a child
    # that owns ONLY a.py. The child must inherit (relabelling cannot dodge the ladder).
    parent = [_entry(ownership=["a.py", "b.py"], strikes_=1)]
    assert strikes.inherited_strikes(parent, _impl("T3", ["a.py"])) == 1


def test_glob_overlap_inherits() -> None:
    parent = [_entry(ownership=["src/**"], strikes_=3)]
    assert strikes.inherited_strikes(parent, _impl("T1", ["src/screen.tsx"])) == 3


def test_disjoint_files_do_not_inherit() -> None:
    led = [_entry(ownership=["a.py"], strikes_=3)]
    assert strikes.inherited_strikes(led, _impl("T1", ["z.py"])) == 0


def test_inherits_max_over_overlaps() -> None:
    led = [
        _entry(ownership=["a.py"], strikes_=1),
        _entry(ownership=["a.py", "b.py"], strikes_=3),
    ]
    assert strikes.inherited_strikes(led, _impl("T1", ["a.py"])) == 3


def test_non_write_matches_by_artifact_out() -> None:
    led = [_entry(artifact_out="E1/r.md", mode="research", strikes_=2)]
    assert strikes.inherited_strikes(led, _research("T1", "E1/r.md")) == 2
    assert strikes.inherited_strikes(led, _research("T1", "E9/other.md")) == 0


def test_implement_and_non_write_never_cross_match() -> None:
    led = [_entry(ownership=["a.py"], strikes_=3)]
    assert strikes.inherited_strikes(led, _research("T1", "a.py")) == 0


# --- apply_ladder (plan-time deterministic transform) --------------------------


def test_below_threshold_untouched() -> None:
    # Strike 0 (fresh) and strike 1 (one whole-ladder failure, the reframe chance) are
    # kept AS PLANNED - the ladder never mutates the tier (senior is reached in-epoch).
    for c in (0, 1):
        led = [_entry(ownership=["a.py"], strikes_=c)] if c else []
        res = strikes.apply_ladder([_impl("T1", ["a.py"])], led)
        assert res.tasks[0].tier == "local"
        assert res.parked == ()


def test_strike_2_blocks_the_task() -> None:
    # The SECOND whole-ladder failure BLOCKS (parks) the lineage: dropped from dispatch.
    led = [_entry(ownership=["a.py"], strikes_=2, reason="still broken")]
    res = strikes.apply_ladder(
        [_impl("T1", ["a.py"]), _impl("T2", ["b.py"])], led
    )
    # T1 removed from the active set; T2 untouched.
    assert [t.id for t in res.tasks] == ["T2"]
    assert len(res.parked) == 1
    assert res.parked[0].task_id == "T1"
    assert res.parked[0].strikes == 2
    assert "still broken" in res.parked[0].reason


def test_strike_1_is_kept_not_blocked() -> None:
    # One strike still gets a re-issue chance (no tier mutation, not parked).
    led = [_entry(ownership=["a.py"], strikes_=1)]
    res = strikes.apply_ladder([_impl("T1", ["a.py"], tier="local")], led)
    assert [t.id for t in res.tasks] == ["T1"]
    assert res.tasks[0].tier == "local"
    assert res.parked == ()


# --- next_ledger (close-time recompute) ----------------------------------------


def test_first_fail_strikes_one() -> None:
    out = strikes.next_ledger([], landed=[], failed=[(_impl("T1", ["a.py"]), "no diff")])
    assert len(out) == 1 and out[0].strikes == 1
    assert out[0].ownership == ("a.py",) and out[0].reason == "no diff"


def test_carried_fail_increments() -> None:
    prior = [_entry(ownership=["a.py"], strikes_=2)]
    out = strikes.next_ledger(prior, landed=[], failed=[(_impl("T8", ["a.py"]), "again")])
    assert len(out) == 1 and out[0].strikes == 3


def test_redecomp_children_both_inherit_then_supersede_parent() -> None:
    # Parent [a,b]@1 re-decomposed into [a] and [b]; BOTH fail. Each inherits 1 -> 2,
    # and the parent entry is superseded (no double-count, no stale lineage).
    prior = [_entry(ownership=["a.py", "b.py"], strikes_=1)]
    out = strikes.next_ledger(
        prior, landed=[],
        failed=[(_impl("T1", ["a.py"]), "x"), (_impl("T2", ["b.py"]), "y")],
    )
    by_owner = {e.ownership: e.strikes for e in out}
    assert by_owner == {("a.py",): 2, ("b.py",): 2}


def test_landed_resolves_lineage() -> None:
    prior = [_entry(ownership=["a.py", "b.py"], strikes_=2)]
    out = strikes.next_ledger(
        prior, landed=[_impl("T1", ["a.py"]), _impl("T2", ["b.py"])], failed=[]
    )
    assert out == []


def test_partial_redecomp_one_lands_one_carries() -> None:
    # Parent [a,b]@1: a.py lands, b.py fails. The b lineage carries to 2; the parent
    # entry is cleared (a resolved, b superseded), so nothing stale remains.
    prior = [_entry(ownership=["a.py", "b.py"], strikes_=1)]
    out = strikes.next_ledger(
        prior, landed=[_impl("T1", ["a.py"])], failed=[(_impl("T2", ["b.py"]), "z")]
    )
    assert len(out) == 1
    assert out[0].ownership == ("b.py",) and out[0].strikes == 2


def test_unrelated_prior_entry_persists() -> None:
    prior = [
        _entry(ownership=["a.py"], strikes_=2),   # a parked, untouched lineage
        _entry(ownership=["b.py"], strikes_=1),
    ]
    out = strikes.next_ledger(
        prior, landed=[_impl("T1", ["b.py"])], failed=[]
    )
    # b resolved; the parked a.py lineage carries forward unchanged.
    assert len(out) == 1 and out[0].ownership == ("a.py",) and out[0].strikes == 2


# --- journal reconstruction (resume-safety) ------------------------------------


def test_reconstruct_takes_last_completed_snapshot() -> None:
    e1 = StrikeLedger(
        seq=1, ts="t", epoch_id="E1",
        entries=[StrikeLedgerEntry(ownership=["a.py"], mode="implement", strikes=1, reason="r1")],
    )
    c1 = EpochCompleted(seq=2, ts="t", epoch_id="E1")
    e2 = StrikeLedger(
        seq=3, ts="t", epoch_id="E2",
        entries=[
            StrikeLedgerEntry(ownership=["a.py"], mode="implement", strikes=2, reason="r2"),
            StrikeLedgerEntry(ownership=["b.py"], mode="implement", strikes=1, reason="r3"),
        ],
    )
    c2 = EpochCompleted(seq=4, ts="t", epoch_id="E2")
    out = strikes.reconstruct_entries([e1, c1, e2, c2])
    assert {e.ownership: e.strikes for e in out} == {("a.py",): 2, ("b.py",): 1}


def test_reconstruct_ignores_orphan_snapshot_of_uncompleted_epoch() -> None:
    # The crash window: a StrikeLedger landed for E2 but E2 never reached its
    # EpochCompleted (the run died between the two emits). Resume must IGNORE the
    # orphan and re-derive from the last COMPLETED epoch, so the re-ground E2 cannot
    # double-count its own strikes.
    e1 = StrikeLedger(
        seq=1, ts="t", epoch_id="E1",
        entries=[StrikeLedgerEntry(ownership=["a.py"], mode="implement", strikes=1, reason="r1")],
    )
    c1 = EpochCompleted(seq=2, ts="t", epoch_id="E1")
    orphan = StrikeLedger(
        seq=3, ts="t", epoch_id="E2",
        entries=[StrikeLedgerEntry(ownership=["a.py"], mode="implement", strikes=2, reason="r2")],
    )
    out = strikes.reconstruct_entries([e1, c1, orphan])
    assert {e.ownership: e.strikes for e in out} == {("a.py",): 1}


def test_reconstruct_empty_when_no_snapshot() -> None:
    assert strikes.reconstruct_entries([]) == []


def test_reconstruct_round_trips_non_write() -> None:
    ev = StrikeLedger(
        seq=1, ts="t", epoch_id="E1",
        entries=[StrikeLedgerEntry(artifact_out="E1/r.md", mode="research", strikes=2, reason="r")],
    )
    done = EpochCompleted(seq=2, ts="t", epoch_id="E1")
    out = strikes.reconstruct_entries([ev, done])
    assert strikes.inherited_strikes(out, _research("T1", "E1/r.md")) == 2


# --- carried-nudge projection + parked summary ---------------------------------


def test_render_carried_flags_block() -> None:
    led = [
        _entry(ownership=["a.py"], strikes_=1),   # one strike: reframe chance, not blocked
        _entry(ownership=["b.py"], strikes_=2),   # two strikes: BLOCKED
    ]
    items = {i.descriptor: i for i in strikes.render_carried(led)}
    assert items["a.py"].parked is False
    assert items["b.py"].parked is True


def test_summarize_parked_from_ledger() -> None:
    led = [
        _entry(ownership=["a.py"], strikes_=2, reason="cannot close a"),
        _entry(ownership=["b.py"], strikes_=1),
    ]
    text = strikes.summarize_parked(led)
    assert "a.py" in text and "cannot close a" in text
    assert "b.py" not in text  # only parked (>=2) lineages surface


def test_summarize_parked_empty_when_none() -> None:
    assert strikes.summarize_parked([_entry(ownership=["a.py"], strikes_=1)]) == ""


def test_parked_events_round_trip() -> None:
    ev = TaskParked(
        seq=1, ts="t", epoch_id="E5", task_id="T1", strikes=2,
        reason="cannot close", descriptor="a.py",
    )
    assert ev.task_id == "T1" and ev.strikes == 2
