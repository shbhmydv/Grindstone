"""The ``watch`` TUI: a PURE reader of ``events.ndjson`` that live-renders the run.

Separate concern from the loop, zero writes, crash-free-to-ignore. The event
stream alone renders the full run -> epoch -> task -> critic-verdict tree, so this
module only ever calls ``read_events`` + ``replay`` (the same fold the journal
post-mortem uses) and paints the resulting ``RunTree`` with ``rich``. The render
path is pure (events in, a ``rich`` renderable out) and snapshot-tested via
plain-text export; the poll loop's wall clock and pacing are injected and kept out
of the render unit tests.

The tree is three deep: each EPOCH is a node, each TASK a child, and once a task
has a critic verdict a single "C" child shows the triage (PASS / RETRY / ESCALATE).
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
from grindstone.events import RunTree, TaskNode, read_events, replay
from grindstone.rundir import RunDir

SleepFn = Callable[[float], None]
NowFn = Callable[[], str]

#: Terminal run statuses (BONES: a run COMPLETES, or ENDS as a clean partial-end).
#: The live loop stops polling and returns once the tree reaches one of these.
_TERMINAL = frozenset({"completed", "ended"})

#: A note (rate-limit / verdict reason) longer than this is clipped in the tree;
#: the full text always lives in the keyed log.
_NOTE_MAX = 80


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clip(text: str, limit: int = _NOTE_MAX) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "..."


def _with_timing(
    text: Text, started: str | None, ended: str | None, now: str | None
) -> Text:
    """Append a duration suffix: dim for finished spans, cyan for live ones."""

    secs = _span_secs(started, ended, now)
    if secs is not None:
        text.append(f"  {_fmt_secs(secs)}", style="dim" if ended else "cyan")
    return text


# Per-level (glyph, rich-style) by status, the only place colour lives. An unmapped
# status falls back to a neutral bullet in ``_styled`` (never a hard error).
_RUN_STYLE: dict[str, tuple[str, str]] = {
    "running": ("●", "yellow"),     # filled circle
    "completed": ("✓", "green"),    # check
    "ended": ("⚠", "yellow"),       # warning (clean partial-end)
}
_EPOCH_STYLE: dict[str, tuple[str, str]] = {
    "started": ("◐", "yellow"),     # half circle (running)
    "completed": ("✓", "green"),    # check
}
_TASK_STYLE: dict[str, tuple[str, str]] = {
    "pending": ("○", "dim"),        # empty circle
    "dispatched": ("▶", "cyan"),    # play (in flight)
    "gate_passed": ("◑", "cyan"),   # half circle (gate ok, awaiting critic)
    "gate_rejected": ("✗", "red"),  # ballot x
    "verdict_pass": ("✓", "green"), # check
    "verdict_retry": ("↻", "yellow"),   # clockwise arrow (retry)
    "verdict_escalate": ("⤴", "magenta"),  # up arrow (escalate)
    "done": ("✓", "green"),         # check
}
#: Critic verdict outcome -> colour for the "C" triage leaf.
_VERDICT_STYLE: dict[str, str] = {
    "PASS": "green",
    "RETRY": "yellow",
    "ESCALATE": "red",
}


def _styled(style_map: dict[str, tuple[str, str]], status: str, label: str) -> Text:
    glyph, colour = style_map.get(status, ("·", "white"))  # middle dot fallback
    text = Text()
    text.append(f"{glyph} ", style=colour)
    text.append(label)
    text.append(f"  [{status}]", style="dim")
    return text


def _task_label(task: TaskNode, now: str | None) -> Text:
    text = _styled(_TASK_STYLE, task.status, f"{task.id} ({task.mode})")
    return _with_timing(text, task.started_ts, task.ended_ts, now)


def _verdict_label(task: TaskNode) -> Text:
    """The "C" critic-triage leaf under a task that has a verdict (PASS/RETRY/ESCALATE).

    Coloured by outcome; the reason (held in ``note`` until ``done`` clears it) is
    appended when still present, clipped, since the full text lives in the log."""

    outcome = task.verdict or ""
    colour = _VERDICT_STYLE.get(outcome, "white")
    text = Text()
    text.append("C  ", style="dim")
    text.append(outcome, style=colour)
    if task.note:
        text.append(f" - {_clip(task.note)}", style=colour)
    return text


def render_tree(tree: RunTree, *, now: str | None = None) -> Tree:
    """Build the ``rich`` render tree from a replayed ``RunTree`` (pure).

    ``now`` (an ISO-8601 wall clock) drives the elapsed timer on still-running
    nodes; when omitted the journal's own last-event ts is used, so snapshots stay
    deterministic and the renderer never reads the clock itself. Shape is
    run -> epoch -> task -> (critic "C" leaf, once a verdict exists)."""

    ref_now = now or tree.last_ts
    glyph, colour = _RUN_STYLE.get(tree.status, ("·", "white"))
    header = Text()
    header.append(f"{glyph} ", style=colour)
    header.append(f"run {tree.run_id}")
    header.append(f"  [{tree.status}]", style="dim")
    _with_timing(header, tree.started_ts, tree.ended_ts, ref_now)
    count = (
        f"{len(tree.epochs)}/{tree.max_epochs}"
        if tree.max_epochs is not None
        else str(len(tree.epochs))
    )
    header.append(f"  · {count} epochs", style="cyan")
    if tree.last_rate_limit:
        header.append(f"  · rate-limited: {_clip(tree.last_rate_limit)}", style="yellow")

    root = Tree(header)
    for epoch in tree.epochs:
        label = _styled(_EPOCH_STYLE, epoch.status, f"{epoch.id} · {epoch.title}")
        _with_timing(label, epoch.started_ts, epoch.ended_ts, ref_now)
        epoch_node = root.add(label)
        for task in epoch.tasks:
            task_node = epoch_node.add(_task_label(task, ref_now))
            if task.verdict is not None:
                task_node.add(_verdict_label(task))
    return root


def read_tree(run_dir: RunDir) -> RunTree | None:
    """Read + replay the journal into a ``RunTree``; ``None`` if not started yet.

    Crash-free-to-ignore: a missing or partial/empty journal (no ``run_started``
    yet) reads as ``None`` rather than raising, so the watcher shows "waiting".
    """

    path = run_dir.events_path
    if not path.exists():
        return None
    try:
        events = read_events(path)
        return replay(events) if events else None
    except (ValueError, KeyError):
        return None


def render_plain(tree: RunTree, *, width: int = 100, now: str | None = None) -> str:
    """Plain-text (style-stripped) export of the render tree, the snapshot seam."""

    console = Console(file=StringIO(), width=width, force_terminal=False, color_system=None)
    console.print(render_tree(tree, now=now))
    out = console.file.getvalue()  # type: ignore[attr-defined]
    assert isinstance(out, str)
    return out


def render_waiting() -> Text:
    """Placeholder shown before the first event lands (no ``run_started`` yet) so the
    watcher reads as alive-and-waiting rather than a blank screen."""

    return Text("waiting for the run to start...", style="dim")


def watch(
    run_dir: RunDir,
    *,
    console: Console | None = None,
    poll_interval: float = 1.0,
    sleep: SleepFn = time.sleep,
    now_fn: NowFn = _utc_now_iso,
) -> RunTree | None:
    """Poll the journal and live-render the tree until the run is terminal.

    Re-reads the journal on file growth (size/mtime) but RE-RENDERS every tick so the
    elapsed clock on running nodes keeps advancing even when no new event has landed
    (otherwise a multi-minute task looks frozen). Quits cleanly on Ctrl-C, or when the
    run reaches a terminal status (``completed`` / ``ended``). The wall clock is the
    injected ``now_fn`` and the pacing the injected ``sleep`` (both stubbable in
    tests). Returns the last rendered tree."""

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
                if tree is not None and tree.status in _TERMINAL:
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
