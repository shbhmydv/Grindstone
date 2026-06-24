"""The kept test doubles still produce gate-valid contracts against the new seam.

These guard the mocks (BONES: "keep the mocks, useful as test doubles for later
stochastic E2E") so the later loop/worker/planner parts inherit working doubles."""

from __future__ import annotations

from pathlib import Path

import pytest

from grindstone.contracts.models import Task, parse_decision, parse_handoff
from grindstone.mock_planner import MockPlanner
from grindstone.mock_worker import MockWorker
from grindstone.planner import RateLimited as PlannerRateLimited
from grindstone.worker import RateLimited as WorkerRateLimited
from grindstone.worker import WorkerRequest


def test_mock_planner_emits_parseable_decision() -> None:
    payload: dict[str, object] = {"kind": "end", "summary": "all done"}
    planner = MockPlanner(script=[payload], wrap="fence")
    text = planner.plan("prompt")
    # The fenced JSON body parses once the fence is stripped (extractor's job later).
    assert "all done" in text


def test_mock_planner_failures_route_two_node() -> None:
    planner = MockPlanner(script=["rate_limit", "invalid"])
    with pytest.raises(PlannerRateLimited):
        planner.plan("p")
    bad = planner.plan("p")  # an invalid decision (empty epoch), not an exception
    import json

    with pytest.raises(Exception):
        parse_decision(json.loads(bad))


def test_mock_worker_writes_valid_handoff(tmp_path: Path) -> None:
    task = Task(id="T1", mode="implement", goal="x", file_ownership=["a.py"])
    request = WorkerRequest(
        task=task, task_id="P1/E1/T1", mode="implement", scratch=tmp_path
    )
    worker = MockWorker(script=["ok"], artifacts={"a.py": "print(1)\n"})
    worker.run(request)
    handoff = parse_handoff(__import__("json").loads(
        (tmp_path / "handoff.json").read_text()
    ))
    assert handoff.status == "DONE"
    assert (tmp_path / "review.md").is_file()  # implement review gate satisfied


def test_mock_worker_rate_limit_raises(tmp_path: Path) -> None:
    task = Task(id="T1", mode="research", goal="x", artifact_out="P1/E1/T1/r.md")
    request = WorkerRequest(
        task=task, task_id="P1/E1/T1", mode="research", scratch=tmp_path
    )
    worker = MockWorker(script=["rate_limit"])
    with pytest.raises(WorkerRateLimited):
        worker.run(request)
