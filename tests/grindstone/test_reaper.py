"""The live-subprocess-group registry + the signal-driven reaper.

Spawns REAL children in their own sessions (``start_new_session=True``, so each is
its own process-group leader, pgid == pid), registers them, and asserts ``reap_all``
leaves ZERO orphaned groups. Idempotency (already-dead / never-registered groups,
an empty registry) must never raise. Deterministic and fast, so no marker.
"""

from __future__ import annotations

import os
import signal
import subprocess

import pytest

from grindstone import reaper


def _spawn_session_child() -> "subprocess.Popen[bytes]":
    """A real ``sleep`` child that leads its own group (pgid == pid)."""

    return subprocess.Popen(
        ["sleep", "30"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def _group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False


def _teardown(*procs: "subprocess.Popen[bytes]") -> None:
    for proc in procs:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()


def test_reap_all_kills_registered_groups() -> None:
    a = _spawn_session_child()
    b = _spawn_session_child()
    try:
        reaper.register(a.pid)
        reaper.register(b.pid)
        reaper.reap_all(grace_s=0.2)
        # Reap the zombies so killpg(pgid, 0) reflects a truly gone group.
        a.wait()
        b.wait()
        assert not _group_alive(a.pid)
        assert not _group_alive(b.pid)
    finally:
        _teardown(a, b)
        reaper.unregister(a.pid)
        reaper.unregister(b.pid)


def test_reap_all_is_idempotent_and_handles_dead_groups() -> None:
    a = _spawn_session_child()
    reaper.register(a.pid)
    os.killpg(a.pid, signal.SIGKILL)  # already dead before reap_all runs
    a.wait()
    # A registered-but-already-dead group must not raise, and a second sweep is a no-op.
    reaper.reap_all(grace_s=0.2)
    reaper.reap_all(grace_s=0.2)
    assert not _group_alive(a.pid)


def test_unregister_unknown_and_empty_reap_never_raise() -> None:
    reaper.unregister(999_999)  # never registered
    reaper.reap_all(grace_s=0.0)  # empty registry


def test_reap_all_clears_the_registry() -> None:
    a = _spawn_session_child()
    try:
        reaper.register(a.pid)
        reaper.reap_all(grace_s=0.2)
        a.wait()
        # The registry is drained, so a leftover pgid cannot be re-signalled later.
        with pytest.raises(KeyError):
            reaper._snapshot().remove(a.pid)
    finally:
        _teardown(a)
        reaper.unregister(a.pid)
