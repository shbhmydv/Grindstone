"""Elapsed-time formatting for the journal render.

Pure, dependency-free arithmetic over ISO-8601 timestamps so the journal
post-mortem (``journal.py``) can show per-epoch / per-task durations.
"""

from __future__ import annotations

from datetime import datetime


def fmt_secs(secs: float) -> str:
    """Compact human duration: ``45s`` / ``1m23s`` / ``2h05m``."""

    n = max(0, int(secs))
    if n < 60:
        return f"{n}s"
    m, s = divmod(n, 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


def span_secs(started: str | None, ended: str | None, now: str | None) -> float | None:
    """Elapsed seconds between ``started`` and ``ended`` (or ``now`` if still
    running). ``None`` when not started yet or a ts is unparseable (crash-free)."""

    if started is None:
        return None
    end = ended or now
    if end is None:
        return None
    try:
        return (datetime.fromisoformat(end) - datetime.fromisoformat(started)).total_seconds()
    except ValueError:
        return None
