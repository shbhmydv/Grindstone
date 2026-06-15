"""The watch TUI: a PURE reader of ``events.ndjson`` (ARCHITECTURE.md, S4 ruling 4).

Separate process, zero writes, crash-free-to-ignore. The event vocabulary alone
is sufficient to render the full run -> phase -> epoch -> task tree, so this
module only ever calls ``read_events`` + ``replay`` (the S0 fold) and paints the
resulting ``RunTree`` with ``rich``. The render path is pure (events in, a
``rich`` renderable out) and snapshot-tested via plain-text export; the poll
loop's wall clock is injected and kept out of unit tests.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from io import StringIO
from typing import Callable

from rich.console import Console
from rich.live import Live
from rich.text import Text
from rich.tree import Tree

from grindstone.duration import fmt_secs as _fmt_secs
from grindstone.duration import span_secs as _span_secs
from grindstone.events import EpochNode, PhaseNode, RunTree, TaskNode, read_events, replay
from grindstone.rundir import RunDir

SleepFn = Callable[[float], None]
NowFn = Callable[[], str]

#: Terminal run statuses the journal can express (incl. `failed` — the safety
#: valve now emits run_failed, so a capped run closes cleanly instead of hanging).
_TERMINAL = frozenset({"completed", "escalated", "failed"})

#: Note longer than this is truncated in the tree (full reason lives in the log).
_NOTE_MAX = 48


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _with_timing(
    text: Text, started: str | None, ended: str | None, now: str | None
) -> Text:
    """Append a duration suffix: dim for finished spans, cyan for live ones."""

    secs = _span_secs(started, ended, now)
    if secs is not None:
        text.append(f"  {_fmt_secs(secs)}", style="dim" if ended else "cyan")
    return text


def _with_note(text: Text, note: str | None) -> Text:
    if note:
        clipped = note if len(note) <= _NOTE_MAX else note[: _NOTE_MAX - 1] + "…"
        text.append(f"  · {clipped}", style="magenta")
    return text

# Per-level (glyph, rich-style) by status — the only place colour lives.
_RUN_STYLE: dict[str, tuple[str, str]] = {
    "running": ("●", "yellow"),
    "completed": ("✓", "green"),
    "escalated": ("⚠", "red"),
    "failed": ("✗", "red"),
}
_PHASE_STYLE: dict[str, tuple[str, str]] = {
    "pending": ("○", "dim"),
    "started": ("◐", "yellow"),
    "passed": ("✓", "green"),
    "escalated": ("⚠", "red"),
}
_EPOCH_STYLE: dict[str, tuple[str, str]] = {
    "started": ("◐", "yellow"),
    "completed": ("✓", "green"),
}
_TASK_STYLE: dict[str, tuple[str, str]] = {
    "pending": ("○", "dim"),
    "dispatched": ("▶", "cyan"),
    "retried": ("↻", "yellow"),
    "escalated": ("⤴", "magenta"),
    "done": ("✓", "green"),
    "failed": ("✗", "red"),
}


def _styled(style_map: dict[str, tuple[str, str]], status: str, label: str) -> Text:
    glyph, colour = style_map.get(status, ("·", "white"))
    text = Text()
    text.append(f"{glyph} ", style=colour)
    text.append(label)
    text.append(f"  [{status}]", style="dim")
    return text


def _task_label(task: TaskNode, now: str | None) -> Text:
    suffix = f" ({task.mode}, a{task.attempt})" if task.attempt else f" ({task.mode})"
    text = _styled(_TASK_STYLE, task.status, f"{task.id}{suffix}")
    _with_timing(text, task.started_ts, task.ended_ts, now)
    return _with_note(text, task.note)


def _epoch_label(epoch: EpochNode, now: str | None) -> Text:
    text = _styled(_EPOCH_STYLE, epoch.status, f"{epoch.id} · {epoch.title}")
    return _with_timing(text, epoch.started_ts, epoch.ended_ts, now)


def _phase_label(phase: PhaseNode, now: str | None) -> Text:
    text = _styled(_PHASE_STYLE, phase.status, f"{phase.id} · {phase.title}")
    return _with_timing(text, phase.started_ts, phase.ended_ts, now)


def _planner_state(tree: RunTree) -> tuple[str, str] | None:
    """The planner's current standing as ``(label, style)`` — in-flight, last
    failure (e.g. a rate-limit retry), or last decision — or ``None`` if idle."""

    if tree.planner_waiting:
        return "planner: running…", "yellow"
    if tree.last_planner_failure is not None:
        return f"planner: {tree.last_planner_failure} retry", "yellow"
    if tree.last_planner_tool is not None:
        return f"last: {tree.last_planner_tool}", "dim"
    return None


def render_tree(tree: RunTree, *, now: str | None = None) -> Tree:
    """Build the ``rich`` render tree from a replayed ``RunTree`` (pure).

    ``now`` (an ISO-8601 wall clock) drives the elapsed timer on still-running
    nodes; when omitted the journal's own last-event ts is used, so snapshots
    stay deterministic and the renderer never reads the clock itself."""

    ref_now = now or tree.last_ts
    glyph, colour = _RUN_STYLE.get(tree.status, ("·", "white"))
    header = Text()
    header.append(f"{glyph} ", style=colour)
    header.append(f"run {tree.run_id}")
    header.append(f"  [{tree.status}]", style="dim")
    _with_timing(header, tree.started_ts, tree.ended_ts, ref_now)
    if tree.phases:
        passed = sum(1 for p in tree.phases if p.status == "passed")
        done = passed == len(tree.phases)
        header.append(
            f"  · {passed}/{len(tree.phases)} phases", style="green" if done else "white"
        )
    calls = (
        f"{tree.planner_calls}/{tree.planner_cap}"
        if tree.planner_cap is not None
        else str(tree.planner_calls)
    )
    header.append(f"  · planner calls: {calls}", style="cyan")
    state = _planner_state(tree)
    if state is not None:
        header.append(f"  · {state[0]}", style=state[1])
    if tree.escalation_reason:
        clipped = (
            tree.escalation_reason
            if len(tree.escalation_reason) <= _NOTE_MAX
            else tree.escalation_reason[: _NOTE_MAX - 1] + "…"
        )
        header.append(f"  · {clipped}", style="red")
    root = Tree(header)
    for phase in tree.phases:
        phase_node = root.add(_phase_label(phase, ref_now))
        for epoch in phase.epochs:
            epoch_node = phase_node.add(_epoch_label(epoch, ref_now))
            for task in epoch.tasks:
                epoch_node.add(_task_label(task, ref_now))
    return root


def read_tree(run_dir: RunDir) -> RunTree | None:
    """Read + replay the journal into a ``RunTree``; ``None`` if not started yet.

    Crash-free-to-ignore: a partial/empty journal (no ``run_started`` yet) reads
    as ``None`` rather than raising, so the watcher simply shows "waiting".
    """

    path = run_dir.events_path
    if not path.exists():
        return None
    try:
        events = read_events(path)
        return replay(events)
    except ValueError:
        return None


def render_plain(tree: RunTree, *, width: int = 100, now: str | None = None) -> str:
    """Plain-text (style-stripped) export of the render tree — the snapshot seam."""

    console = Console(file=StringIO(), width=width, force_terminal=False, color_system=None)
    console.print(render_tree(tree, now=now))
    out = console.file.getvalue()  # type: ignore[attr-defined]
    assert isinstance(out, str)
    return out


def render_waiting() -> Text:
    """Placeholder shown before the first event lands (no run_started yet) so the
    watcher reads as alive-and-waiting rather than a blank screen."""

    return Text("⏳ waiting for the run to start…", style="dim")


def watch(
    run_dir: RunDir,
    *,
    follow: bool = False,
    console: Console | None = None,
    poll_interval: float = 0.5,
    sleep: SleepFn = time.sleep,
    now_fn: NowFn = _utc_now_iso,
) -> RunTree | None:
    """Poll the journal and live-render the tree until the run is terminal.

    Re-reads the journal on file growth (size/mtime) but RE-RENDERS every tick so
    the elapsed clock on running nodes keeps advancing even when no new event has
    landed (otherwise a multi-minute task looks frozen). Quits cleanly on Ctrl-C
    or — unless ``follow`` — when the run reaches a terminal status (``completed``
    / ``escalated`` / ``failed``). The wall clock is the injected ``now_fn`` and
    the pacing the injected ``sleep`` (both stubbable in tests). Returns the last
    rendered tree (its status drives the CLI exit code)."""

    console = console or Console()
    path = run_dir.events_path
    last_sig: tuple[int, float] | None = None
    tree = read_tree(run_dir)

    def _paint() -> None:  # render the tree, or a waiting placeholder if not started
        renderable = render_tree(tree, now=now_fn()) if tree is not None else render_waiting()
        live.update(renderable, refresh=True)

    with Live(console=console, auto_refresh=False) as live:
        _paint()
        try:
            while True:
                if tree is not None and tree.status in _TERMINAL and not follow:
                    return tree
                sleep(poll_interval)
                sig = (
                    (path.stat().st_size, path.stat().st_mtime) if path.exists() else (0, 0.0)
                )
                if sig != last_sig:
                    last_sig = sig
                    tree = read_tree(run_dir)
                _paint()  # heartbeat: tick the elapsed clock each pass (or keep waiting)
        except KeyboardInterrupt:
            return tree
