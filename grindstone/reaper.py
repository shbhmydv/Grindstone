"""A process-global registry of live rig subprocess GROUPS + a signal-driven reaper.

Every worker / planner rig subprocess is spawned ``start_new_session=True``, so it
leads its OWN process group (pgid == child pid). On a TIMEOUT the dispatcher kills
its own group; but on a run-level SIGTERM / SIGINT the main process would die and
leave those DETACHED groups orphaned (observed live: a ``timeout``-wrapped local
worker survived a kill). The run installs a handler that calls ``reap_all``, which
SIGTERMs every still-registered group, waits a short grace, then SIGKILLs survivors.

The registry is a module-global guarded by a lock because the epoch fan-out runs the
workers under a ``ThreadPoolExecutor``. ``register`` / ``unregister`` bracket each
``Popen``'s lifetime (the pgid is the child pid). ``reap_all`` is idempotent and
NEVER raises: an already-dead or never-registered group is simply skipped, and the
registry is drained so a second sweep is a no-op. This module touches PROCESSES only,
never git or disk: resume owns all scratch cleanup.
"""

from __future__ import annotations

import os
import signal
import threading
import time
from typing import Callable

#: Guards the live-group set against concurrent fan-out register / unregister.
_LOCK = threading.Lock()

#: pgids of the currently-live rig subprocess groups.
_GROUPS: set[int] = set()

#: Grace between the group SIGTERM sweep and the SIGKILL of any survivor.
_GRACE_S = 2.0


def register(pgid: int) -> None:
    """Record a freshly-spawned group leader's pgid (idempotent)."""

    with _LOCK:
        _GROUPS.add(pgid)


def unregister(pgid: int) -> None:
    """Drop a group whose subprocess has been reaped (idempotent, never raises)."""

    with _LOCK:
        _GROUPS.discard(pgid)


def _snapshot() -> set[int]:
    """A copy of the live-group set taken under the lock."""

    with _LOCK:
        return set(_GROUPS)


def _signal_group(pgid: int, sig: int) -> bool:
    """Send ``sig`` to the whole group; return whether the group still existed.

    A gone group (``ProcessLookupError``) or any other ``OSError`` (e.g. a permission
    quirk on a reused pgid) is swallowed: the reaper must never raise.
    """

    try:
        os.killpg(pgid, sig)
        return True
    except OSError:
        return False


def reap_all(
    *, grace_s: float = _GRACE_S, sleep_fn: Callable[[float], None] = time.sleep
) -> None:
    """SIGTERM every registered group, wait ``grace_s``, then SIGKILL survivors.

    Idempotent and total: the registry is drained at the end, so a re-entrant signal
    or a follow-up call is a clean no-op. Never raises (each group signal is guarded).
    """

    groups = _snapshot()
    alive = [pgid for pgid in groups if _signal_group(pgid, signal.SIGTERM)]
    if alive:
        sleep_fn(grace_s)
        for pgid in alive:
            _signal_group(pgid, signal.SIGKILL)
    with _LOCK:
        for pgid in groups:
            _GROUPS.discard(pgid)
