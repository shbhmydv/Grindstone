"""The per-run journal: a markdown render of ``events.ndjson`` (BONES post-mortem).

A human-facing timeline, "what got done in what epoch", rendered from the
append-only event stream (the same ``replay`` fold ``cli watch`` uses), never a
second source of truth, so it carries no trust/poisoning risk and nothing reads it
back into the loop. BONES is epochs-only: no phases, no floors, no planner-call cap.

Retention: only the LATEST run keeps its ``journal.md``. When a new run starts it
reaps every other run's journal (``reap_sibling_journals``), leaving just the
durable ``events.ndjson`` behind, and because the journal is derived, any old run
stays re-renderable from its retained events. ``write_journal`` renders it once at
the run terminal; ``watch`` renders it on demand from the live event stream.
"""

from __future__ import annotations

from grindstone.duration import fmt_secs, span_secs
from grindstone.events import RunTree, read_events, replay
from grindstone.rundir import RunDir

#: Task-status glyphs (plain ASCII, markdown carries no colour). Unmapped statuses
#: (pending / dispatched / gate_passed) render with the neutral bullet.
_TASK_GLYPH = {
    "done": "[ok]",
    "verdict_pass": "[ok]",
    "verdict_retry": "[retry]",
    "verdict_escalate": "[esc]",
    "gate_rejected": "[x]",
}


def _dur(started: str | None, ended: str | None, ref: str | None) -> str:
    secs = span_secs(started, ended, ref)
    return f"  ({fmt_secs(secs)})" if secs is not None else ""


def render_journal(tree: RunTree) -> str:
    """Render a replayed ``RunTree`` as a markdown post-mortem (pure)."""

    ref = tree.ended_ts or tree.last_ts
    epochs = (
        f"{len(tree.epochs)}/{tree.max_epochs}"
        if tree.max_epochs is not None
        else str(len(tree.epochs))
    )
    run_span = span_secs(tree.started_ts, tree.ended_ts, ref)
    duration = fmt_secs(run_span) if run_span is not None else "n/a"

    lines: list[str] = [
        f"# Run {tree.run_id} - {tree.status}",
        "",
        f"- Job: `{tree.job_path}`",
        f"- Duration: {duration}   -   Epochs: {epochs}",
    ]
    if tree.last_rate_limit:
        lines.append(f"- Rate-limited: {tree.last_rate_limit}")
    lines.append("")

    for epoch in tree.epochs:
        lines.append(
            f"## {epoch.id} - {epoch.title}  [{epoch.status}]"
            f"{_dur(epoch.started_ts, epoch.ended_ts, ref)}"
        )
        for task in epoch.tasks:
            glyph = _TASK_GLYPH.get(task.status, "-")
            note = f"  - {task.note}" if task.note else ""
            lines.append(
                f"    {glyph} {task.id} ({task.mode}) [{task.status}]"
                f"{_dur(task.started_ts, task.ended_ts, ref)}{note}"
            )
        lines.append("")

    if tree.end_summary:
        lines.append(f"Ended: {tree.end_summary}")

    return "\n".join(lines).rstrip() + "\n"


def render_run_dir(run_dir: RunDir) -> str | None:
    """Render the run's journal from its current event stream, or ``None``.

    ``None`` when there is nothing renderable yet (no events file, an empty stream,
    or a stream that never reached ``run_started``), so a watcher or a write never
    strands a stub on a killed-before-start run. Best-effort: a malformed stream
    yields ``None`` rather than raising into a post-mortem path.
    """

    path = run_dir.events_path
    if not path.exists():
        return None
    try:
        events = read_events(path)
        if not events:
            return None
        return render_journal(replay(events))
    except (ValueError, KeyError, OSError):
        return None


def write_journal(run_dir: RunDir) -> None:
    """Render the run's journal from its events and write ``journal.md``.

    A no-op when there is no renderable run yet, so a killed-before-start run never
    strands a stub file. The journal is a DERIVED, non-load-bearing view written
    AFTER the run is durably terminal; it must never raise into that path, so a
    write failure leaves a stale/absent journal, never a crash.
    """

    rendered = render_run_dir(run_dir)
    if rendered is None:
        return
    try:
        run_dir.journal_path.write_text(rendered, encoding="utf-8")
    except OSError:
        return


def reap_sibling_journals(run_dir: RunDir) -> None:
    """Delete ``journal.md`` from every OTHER run under the same ``runs/`` dir.

    Called when a new run starts: only the latest run keeps a rendered journal;
    siblings keep their ``events.ndjson`` (the durable record) and nothing else.
    Best-effort: a missing parent or already-gone journal is fine.
    """

    runs_dir = run_dir.root.parent
    if not runs_dir.is_dir():
        return
    for child in runs_dir.iterdir():
        if child == run_dir.root or not child.is_dir():
            continue
        try:
            RunDir(root=child).journal_path.unlink()
        except FileNotFoundError:
            pass
