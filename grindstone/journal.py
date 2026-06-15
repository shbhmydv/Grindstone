"""The per-run journal: a markdown render of ``events.ndjson`` (ARCHITECTURE.md).

A human-facing post-mortem — "what got done in what phase / epoch / job" —
written once at terminal. It is a DERIVED view of the event stream (the same
``replay`` fold the TUI uses), never a second source of truth, so it carries no
trust/poisoning risk and nothing reads it back into the loop.

Retention: only the LATEST run keeps its ``journal.md``. When a new run starts
it reaps every other run's journal, leaving just the durable ``events.ndjson``
behind — and because the journal is derived, any old run stays re-renderable
from its retained events.
"""

from __future__ import annotations

from grindstone.duration import fmt_secs, span_secs
from grindstone.events import RunTree, read_events, replay
from grindstone.rundir import RunDir

#: Task-status glyphs (plain unicode — markdown carries no colour).
_TASK_GLYPH = {"done": "✓", "failed": "✗", "escalated": "⤴", "retried": "↻"}


def _dur(started: str | None, ended: str | None, ref: str | None) -> str:
    secs = span_secs(started, ended, ref)
    return f"  ({fmt_secs(secs)})" if secs is not None else ""


def render_journal(tree: RunTree) -> str:
    """Render a replayed ``RunTree`` as a markdown post-mortem (pure)."""

    ref = tree.ended_ts or tree.last_ts
    calls = (
        f"{tree.planner_calls}/{tree.planner_cap}"
        if tree.planner_cap is not None
        else str(tree.planner_calls)
    )
    run_span = span_secs(tree.started_ts, tree.ended_ts, ref)
    duration = fmt_secs(run_span) if run_span is not None else "—"

    lines: list[str] = [
        f"# Run {tree.run_id} — {tree.status}",
        "",
        f"- Job: `{tree.job_path}`",
        f"- Duration: {duration}   ·   Planner calls: {calls}",
    ]
    if tree.escalation_reason:
        lines.append(f"- Reason: {tree.escalation_reason}")
    if tree.final_polish:
        lines.append(f"- Final polish: {tree.final_polish}")
    lines.append("")

    for phase in tree.phases:
        lines.append(
            f"## {phase.id} · {phase.title}  [{phase.status}]"
            f"{_dur(phase.started_ts, phase.ended_ts, ref)}"
        )
        for epoch in phase.epochs:
            lines.append(
                f"- **{epoch.id} · {epoch.title}**  [{epoch.status}]"
                f"{_dur(epoch.started_ts, epoch.ended_ts, ref)}"
            )
            for task in epoch.tasks:
                glyph = _TASK_GLYPH.get(task.status, "·")
                note = f"  · {task.note}" if task.note else ""
                lines.append(
                    f"    - {glyph} {task.id} ({task.mode}) [{task.status}]"
                    f"{_dur(task.started_ts, task.ended_ts, ref)}{note}"
                )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_journal(run_dir: RunDir) -> None:
    """Render the run's journal from its events and write ``journal.md``.

    A no-op when there is no renderable run yet (missing/empty/partial journal —
    no ``run_started``), so a killed-before-start run never strands a stub file.
    """

    path = run_dir.events_path
    if not path.exists():
        return
    # Best-effort: the journal is a DERIVED, non-load-bearing view rendered AFTER
    # the run is durably terminal. It must never raise into that path — a KeyError
    # from an inconsistent event stream or an OSError on write must not throw away
    # an already-durable RunOutcome. Any failure -> no/stale journal.md, never a crash.
    try:
        tree = replay(read_events(path))
        run_dir.journal_path.write_text(render_journal(tree), encoding="utf-8")
    except Exception:
        return


def reap_sibling_journals(run_dir: RunDir) -> None:
    """Delete ``journal.md`` from every OTHER run under the same ``runs/`` dir.

    Called when a new run starts: only the latest run keeps a rendered journal;
    siblings keep their ``events.ndjson`` (the durable record) and nothing else.
    Best-effort — a missing parent or already-gone journal is fine.
    """

    runs_dir = run_dir.root.parent
    if not runs_dir.is_dir():
        return
    for child in runs_dir.iterdir():
        if child == run_dir.root or not child.is_dir():
            continue
        journal = RunDir(root=child).journal_path
        try:
            journal.unlink()
        except FileNotFoundError:
            pass
