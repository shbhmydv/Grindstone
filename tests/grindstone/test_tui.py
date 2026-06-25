"""The watch TUI: the pure events -> RunTree -> rich render is snapshot-tested via
plain-text export (no live terminal, no real wall clock). The live poll loop is
exercised with injected ``sleep`` / ``now_fn``, and ``_cmd_watch``'s TTY-vs-static
routing is unit-tested without ever painting a real terminal.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from grindstone import cli
from grindstone.events import (
    EpochCompleted,
    EpochStarted,
    Event,
    JournalWriter,
    RunCompleted,
    RunEnded,
    RunStarted,
    TaskDispatched,
    TaskDone,
    TaskRef,
    Verdict,
    WorkGateRejected,
    replay,
)
from grindstone.rundir import RunDir, create_run_dir
from grindstone.tui import read_tree, render_plain, render_waiting, watch

TS = "2026-06-10T00:00:00+00:00"


def _journal() -> list[Event]:
    """A 2-epoch run: E1 has a passing task (verdict PASS -> done) and a rejected
    task; E2 is still running with one in-flight task. Run completes."""

    seq = iter(range(100))
    return [
        RunStarted(seq=next(seq), ts=TS, run_id="r1", job_path="job.md", max_epochs=20),
        EpochStarted(
            seq=next(seq), ts=TS, epoch_id="E1", title="build greet",
            tasks=[TaskRef(id="T1", mode="implement"), TaskRef(id="T2", mode="research")],
        ),
        TaskDispatched(seq=next(seq), ts=TS, epoch_id="E1", task_id="T1"),
        TaskDispatched(seq=next(seq), ts=TS, epoch_id="E1", task_id="T2"),
        WorkGateRejected(seq=next(seq), ts=TS, epoch_id="E1", task_id="T2", reason="no diff"),
        Verdict(seq=next(seq), ts=TS, epoch_id="E1", task_id="T1",
                outcome="PASS", reason="looks correct"),
        TaskDone(seq=next(seq), ts="2026-06-10T00:00:59+00:00", epoch_id="E1", task_id="T1"),
        EpochCompleted(seq=next(seq), ts=TS, epoch_id="E1"),
        EpochStarted(
            seq=next(seq), ts=TS, epoch_id="E2", title="doc it",
            tasks=[TaskRef(id="T3", mode="implement")],
        ),
        TaskDispatched(seq=next(seq), ts=TS, epoch_id="E2", task_id="T3"),
        RunCompleted(seq=next(seq), ts="2026-06-10T00:01:30+00:00"),
    ]


def test_render_snapshot_covers_full_tree() -> None:
    out = render_plain(replay(_journal()))
    # Run header: id, status, and the n/max epoch counter (phase-free).
    assert "run r1" in out
    assert "[completed]" in out
    assert "2/20 epochs" in out
    # Epoch nodes with their resolved statuses.
    assert "E1 · build greet" in out and "[completed]" in out
    assert "E2 · doc it" in out and "[started]" in out
    # Task leaves keep (mode) + status.
    assert "T1 (implement)" in out and "[done]" in out
    assert "T2 (research)" in out and "[gate_rejected]" in out
    assert "T3 (implement)" in out and "[dispatched]" in out
    # Status glyphs survive the plain-text export (colour does not).
    assert "✓" in out and "✗" in out
    # T1's duration is shown (done 59s after dispatch).
    assert "59s" in out


def test_render_shows_retained_critic_verdict_node() -> None:
    # The "C" leaf hangs under the verdict-bearing task and SURVIVES `done`
    # (note is cleared on done, the retained verdict is not).
    out = render_plain(replay(_journal()))
    done_task = next(t for t in replay(_journal()).epochs[0].tasks if t.id == "T1")
    assert done_task.status == "done" and done_task.note is None
    assert done_task.verdict == "PASS"
    assert "C  PASS" in out  # the critic leaf is rendered past done
    # A task that never got a verdict has NO C leaf.
    assert out.count("C  ") == 1


def test_render_shows_retry_and_escalate_verdicts() -> None:
    events = [
        RunStarted(seq=0, ts=TS, run_id="r1", job_path="job.md"),
        EpochStarted(seq=1, ts=TS, epoch_id="E1", title="x",
                     tasks=[TaskRef(id="T1", mode="implement"),
                            TaskRef(id="T2", mode="review")]),
        TaskDispatched(seq=2, ts=TS, epoch_id="E1", task_id="T1"),
        Verdict(seq=3, ts=TS, epoch_id="E1", task_id="T1",
                outcome="RETRY", reason="missing test"),
        TaskDispatched(seq=4, ts=TS, epoch_id="E1", task_id="T2"),
        Verdict(seq=5, ts=TS, epoch_id="E1", task_id="T2",
                outcome="ESCALATE", reason="needs taste judgment"),
    ]
    out = render_plain(replay(events))
    assert "[verdict_retry]" in out and "C  RETRY - missing test" in out
    assert "[verdict_escalate]" in out and "C  ESCALATE - needs taste judgment" in out


def test_render_clips_a_long_verdict_reason() -> None:
    reason = "x" * 200
    events = [
        RunStarted(seq=0, ts=TS, run_id="r1", job_path="job.md"),
        EpochStarted(seq=1, ts=TS, epoch_id="E1", title="x",
                     tasks=[TaskRef(id="T1", mode="implement")]),
        TaskDispatched(seq=2, ts=TS, epoch_id="E1", task_id="T1"),
        Verdict(seq=3, ts=TS, epoch_id="E1", task_id="T1", outcome="RETRY", reason=reason),
    ]
    out = render_plain(replay(events), width=200)
    assert "..." in out
    assert ("x" * 200) not in out  # the 200-char reason is truncated


def test_render_reflects_running_and_ended_statuses() -> None:
    running = render_plain(replay(_journal()[:-1]))  # drop RunCompleted
    assert "[running]" in running
    ended = render_plain(replay(
        _journal()[:-1] + [RunEnded(seq=99, ts=TS, summary="rest deferred")]
    ))
    assert "[ended]" in ended


def test_waiting_placeholder_before_run_starts() -> None:
    assert "waiting" in render_waiting().plain.lower()


def test_read_tree_handles_empty_and_missing(tmp_path: Path) -> None:
    rd = RunDir(root=tmp_path / "nope")
    assert read_tree(rd) is None  # missing journal -> None, never raises
    started = create_run_dir(tmp_path, "empty")
    started.events_path.write_text("", encoding="utf-8")
    assert read_tree(started) is None  # empty journal -> None


def test_watch_returns_on_terminal_run_without_sleeping(tmp_path: Path) -> None:
    rd = create_run_dir(tmp_path, "done")
    with JournalWriter(rd.events_path) as j:
        for ev in _journal():
            j.append(ev)

    def _boom(_delay: float) -> None:  # the loop must NOT sleep on a terminal run
        raise AssertionError("watch slept on an already-terminal run")

    tree = watch(rd, sleep=_boom)
    assert tree is not None and tree.status == "completed"


def test_watch_exits_on_ended_run(tmp_path: Path) -> None:
    rd = create_run_dir(tmp_path, "ended")
    events = _journal()[:-1] + [RunEnded(seq=99, ts=TS, summary="partial")]
    with JournalWriter(rd.events_path) as j:
        for ev in events:
            j.append(ev)

    def _boom(_delay: float) -> None:  # a clean partial-end is terminal -> no sleep
        raise AssertionError("watch slept on an ended (terminal) run")

    tree = watch(rd, sleep=_boom)
    assert tree is not None and tree.status == "ended"


def test_watch_heartbeats_while_running(tmp_path: Path) -> None:
    # With NO new journal events, the loop must still repaint each tick so the
    # elapsed clock advances (the "looks frozen" fix); proven by counting now_fn
    # calls across ticks despite a static file, then bailing via KeyboardInterrupt.
    rd = create_run_dir(tmp_path, "live")
    with JournalWriter(rd.events_path) as j:
        for ev in _journal()[:-1]:  # truncate RunCompleted -> still running
            j.append(ev)

    ticks = {"n": 0}

    def _sleep(_delay: float) -> None:
        ticks["n"] += 1
        if ticks["n"] >= 3:
            raise KeyboardInterrupt

    nows: list[str] = []

    def _now() -> str:
        t = f"2026-06-10T00:00:{len(nows):02d}+00:00"
        nows.append(t)
        return t

    tree = watch(rd, sleep=_sleep, now_fn=_now)
    assert tree is not None and tree.status == "running"
    assert len(nows) >= 3  # repainted each tick, not just on a file change


# --- _cmd_watch routing: live on a TTY, static when piped or --once ------------


def _watch_args(target: str, *, once: bool) -> argparse.Namespace:
    return argparse.Namespace(target=target, repo=None, once=once)


def test_cmd_watch_live_when_tty(tmp_path: Path, monkeypatch) -> None:
    rd = create_run_dir(tmp_path, "tty")
    with JournalWriter(rd.events_path) as j:
        for ev in _journal():
            j.append(ev)

    calls: list[RunDir] = []
    monkeypatch.setattr(cli.sys.stdout, "isatty", lambda: True)
    monkeypatch.setattr(cli.tui, "watch", lambda run_dir: calls.append(run_dir))

    rc = cli._cmd_watch(_watch_args(str(rd.root), once=False))
    assert rc == 0
    assert len(calls) == 1  # live path taken on a TTY


def test_cmd_watch_static_when_once(tmp_path: Path, monkeypatch, capsys) -> None:
    rd = create_run_dir(tmp_path, "once")
    with JournalWriter(rd.events_path) as j:
        for ev in _journal():
            j.append(ev)

    # Even on a TTY, --once forces the single static render (no live loop).
    monkeypatch.setattr(cli.sys.stdout, "isatty", lambda: True)
    monkeypatch.setattr(cli.tui, "watch", lambda run_dir: (_ for _ in ()).throw(
        AssertionError("--once must not start the live loop")))

    rc = cli._cmd_watch(_watch_args(str(rd.root), once=True))
    assert rc == 0
    assert "Run r1" in capsys.readouterr().out  # the static journal render


def test_cmd_watch_static_when_not_tty(tmp_path: Path, monkeypatch, capsys) -> None:
    rd = create_run_dir(tmp_path, "piped")
    with JournalWriter(rd.events_path) as j:
        for ev in _journal():
            j.append(ev)

    monkeypatch.setattr(cli.sys.stdout, "isatty", lambda: False)  # piped / CI / agent
    monkeypatch.setattr(cli.tui, "watch", lambda run_dir: (_ for _ in ()).throw(
        AssertionError("piped output must not start the live loop")))

    rc = cli._cmd_watch(_watch_args(str(rd.root), once=False))
    assert rc == 0
    assert "Run r1" in capsys.readouterr().out  # static fallback for non-TTY
