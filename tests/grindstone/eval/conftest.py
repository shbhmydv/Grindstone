"""Fixtures for the real-rig capability harness (the ``eval`` corpus).

The corpus runs every test on BOTH the floor (``local`` = Qwen via the local rig)
and the ceiling (``claude`` = Opus): if the local floor produces conforming,
correctly-shaped output, the cloud ceiling will too. Each rig is a real endpoint, so
live execution is OPT-IN per rig via ``GRINDSTONE_EVAL_RIG`` (a comma list). A
parametrized rig not named there is SKIPPED (not failed), so:

  * default (unset)                  -> every eval is skipped (collection only)
  * GRINDSTONE_EVAL_RIG=local         -> the floor runs, the ceiling is skipped
  * GRINDSTONE_EVAL_RIG=local,claude  -> both run

Collection is never gated (the skip fires at setup), so ``pytest --co -m eval``
always lists the full corpus. These are slow + need a live endpoint, so the whole
corpus is also behind the ``eval`` marker (excluded from the default suite).
"""

from __future__ import annotations

import os

import pytest

#: The rigs the corpus parametrizes over: the local floor and the cloud ceiling.
EVAL_RIGS: tuple[str, ...] = ("local", "claude")

#: Env var naming which rigs to actually drive live (comma-separated). Absent or
#: empty = none (collection-only); the operator sets it to run the live baseline.
EVAL_RIG_ENV = "GRINDSTONE_EVAL_RIG"


def _enabled_rigs() -> frozenset[str]:
    raw = os.environ.get(EVAL_RIG_ENV, "")
    return frozenset(part.strip() for part in raw.split(",") if part.strip())


@pytest.fixture(params=EVAL_RIGS)
def rig(request: pytest.FixtureRequest) -> str:
    """The rig under test, one per param; skipped unless named in the env var."""

    rig_name: str = request.param
    if rig_name not in _enabled_rigs():
        pytest.skip(
            f"rig {rig_name!r} not enabled; set {EVAL_RIG_ENV} "
            f"(e.g. {EVAL_RIG_ENV}={rig_name}) to drive it live"
        )
    return rig_name
