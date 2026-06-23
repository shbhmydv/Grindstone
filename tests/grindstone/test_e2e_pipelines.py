"""Full-run E2E coverage for the multi-mode decomposition patterns the planner
contract nudges toward (PLANNER_CONTRACT §3 "Sequencing by tier of thinking"):

  - research -> artifact: a senior-flagged research/synthesis task on senior, the
    write-up on local, with the research findings flowing forward as a keyed-log input.
  - research -> implement -> review: the heavy-build shape, synthesize (senior),
    build (local, committed), judge (senior) - judgment slices flagged per task.
  - a senior-flagged implement task is built on the senior taste tier.

These drive real runs through ``run_grind`` with a scripted ``MockPlanner`` and a
two-tier ladder, asserting BOTH the tier routing and the cross-epoch handoff,
the seams the dogfood surfaced (artifact publication + senior routing). Each
phase carries an exit criterion that only passes once its epoch's deliverable
exists, so epochs land in the intended phase (a trivially-true criterion would
auto-advance the phase before any work). The epoch counter is global across the
run (E1, E2, E3…); the keyed-log prefix is ``<phase>/<epoch>/<task>``.

Mode routing and citation grounding are unit-tested in ``test_task_loop``; this
file proves they compose end to end.
"""

from __future__ import annotations

import json
from pathlib import Path

from grindstone.contracts.models import ImplementTask
from grindstone.mock_planner import MockPlanner
from grindstone.rundir import RunDir
from grindstone.run_loop import RunState, run_grind
from grindstone.worker import WorkerRequest

from tests.grindstone.conftest import (
    artifact_decision,
    check_cmd,
    complete_decision,
    impl_task,
    implement_decision,
    phase_complete_decision,
    phase_dict,
    research_decision,
    review_decision,
    skeleton_decision,
    tracked_files,
)


class _PipelineWorker:
    """An E2E worker that satisfies any mode's gate with a valid handoff and
    records the ``(task_id, mode)`` it ran (so a test can prove which tier it
    started on). ImplementTask writes its owned file + the review.md gate and
    cites that file (the worktree is the only allowed citation root); a
    research/review/artifact ArtifactTask writes the FULL ``artifact_out`` path it
    was told to produce and cites README.md (resolved against the repo root)."""

    def __init__(self) -> None:
        self.seen: list[tuple[str, str]] = []

    def run(self, request: WorkerRequest) -> None:
        self.seen.append((request.task_id, request.mode))
        task = request.task
        if isinstance(task, ImplementTask):
            (request.scratch / "review.md").write_text("ok\n", encoding="utf-8")
            rel = task.file_ownership[0]
            cite = rel
        else:
            rel = task.artifact_out
            cite = "README.md"
        path = request.scratch / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("content\n", encoding="utf-8")
        payload = {
            "schema_version": "1",
            "task_id": request.task_id,
            "status": "DONE",
            "what_changed": [{"kind": "file", "ref": rel}],
            "resulting_state": f"produced {rel}",
            "downstream_needs": [],
            "not_done": [],
            "citations": [{"file": cite}],
            "checks": [
                {"check": getattr(c, "cmd", "artifact"), "exit_code": 0}
                for c in task.done_when
            ],
            "occupancy": {"compacted": False, "subagent_splits": 0},
        }
        (request.scratch / "handoff.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )


def _two_tier(
    local: _PipelineWorker, senior: _PipelineWorker
) -> list[tuple[str, _PipelineWorker]]:
    return [("worker", local), ("senior", senior)]


def _state(run_dir: RunDir) -> RunState:
    return RunState.model_validate_json(run_dir.run_state_path.read_text())


def _exists(name: str) -> dict[str, object]:
    """A phase exit criterion satisfied only once a keyed artifact of this bare
    name is logged (bare name: the P*/E*/T*/ placement is unknown at skeleton)."""

    return {"artifact_exists": name}


def _artifact_task(
    tid: str,
    out: str,
    *,
    goal: str = "produce it",
    inputs: list[str] | None = None,
    senior: bool = False,
) -> dict[str, object]:
    task: dict[str, object] = {
        "id": tid,
        "goal": goal,
        "done_when": [check_cmd("true")],
        "artifact_out": out,
    }
    if inputs is not None:
        task["inputs"] = inputs
    if senior:
        task["senior"] = True
    return task


def test_research_to_artifact_split_pipeline(git_repo: Path, run_dir: RunDir) -> None:
    """A research task flagged ``senior`` (a synthesis/judgment call) publishes
    findings to the keyed log; a downstream artifact epoch (local) consumes that key
    as an ``input`` and writes the report. Proves the cross-epoch keyed-log handoff
    AND per-task tier routing in one run, and the input only resolves because the
    research artifact was actually published (the dogfood bug would have left the key
    absent and failed the report decision's validation)."""

    local, senior = _PipelineWorker(), _PipelineWorker()
    planner = MockPlanner(
        script=[
            skeleton_decision(
                phase_dict("P1", title="research", exit_criterion=[_exists("findings.md")]),
                phase_dict("P2", title="report", exit_criterion=[_exists("report.md")]),
            ),
            research_decision(
                _artifact_task(
                    "T1", "P1/E1/T1/findings.md",
                    goal="synthesize an approach from the repo", senior=True,
                ),
                title="investigate",
            ),
            phase_complete_decision("findings.md"),  # ends P1 (keyed-log artifact)
            artifact_decision(
                _artifact_task(
                    "T1",
                    "P2/E1/T1/report.md",
                    goal="write the report from the findings",
                    inputs=["P1/E1/T1/findings.md"],
                ),
                title="write report",
            ),
            phase_complete_decision("report.md"),  # ends P2
            complete_decision(check_cmd("true")),
        ]
    )
    outcome = run_grind(
        run_dir,
        job_path="job.md",
        planner=planner,
        ladder=_two_tier(local, senior),
        repo=git_repo,
    )
    assert outcome.status == "completed", outcome
    # the senior-flagged research task started on SENIOR; the report write-up on LOCAL.
    assert ("P1/E1/T1", "research") in senior.seen
    assert ("P2/E1/T1", "artifact") in local.seen
    assert ("P1/E1/T1", "research") not in local.seen
    # both artifacts were published to the keyed log at their full keys.
    assert run_dir.resolve("P1/E1/T1/findings.md").is_file()
    assert run_dir.resolve("P2/E1/T1/report.md").is_file()


def test_research_implement_review_pipeline(git_repo: Path, run_dir: RunDir) -> None:
    """The heavy-build decomposition: a senior-flagged research task synthesizes the
    design (senior), implement (local) builds + commits, a senior-flagged review
    judges (senior). Judgment routes to senior, production to local, and the
    implement task's file lands on the integration branch."""

    local, senior = _PipelineWorker(), _PipelineWorker()
    planner = MockPlanner(
        script=[
            skeleton_decision(
                phase_dict("P1", title="research", exit_criterion=[_exists("design.md")]),
                phase_dict("P2", title="build", exit_criterion=[check_cmd("test -f feature.py")]),
                phase_dict("P3", title="review", exit_criterion=[_exists("verdict.md")]),
            ),
            research_decision(
                _artifact_task(
                    "T1", "P1/E1/T1/design.md",
                    goal="synthesize the design approach", senior=True,
                ),
                title="map",
            ),
            phase_complete_decision("design.md"),  # ends P1 (keyed-log artifact)
            implement_decision(impl_task("T1", "feature.py"), title="build"),
            phase_complete_decision("feature.py"),  # ends P2 (committed file at tip)
            review_decision(
                {
                    "id": "T1",
                    "goal": "judge feature.py taste against intent",
                    "done_when": [check_cmd("true")],
                    "artifact_out": "P3/E1/T1/verdict.md",
                    "targets": ["feature.py"],
                    "senior": True,
                },
                title="review",
            ),
            phase_complete_decision("verdict.md"),  # ends P3
            complete_decision(check_cmd("true")),
        ]
    )
    outcome = run_grind(
        run_dir,
        job_path="job.md",
        planner=planner,
        ladder=_two_tier(local, senior),
        repo=git_repo,
    )
    assert outcome.status == "completed", outcome
    # judgment on senior, production on local.
    assert ("P1/E1/T1", "research") in senior.seen
    assert ("P2/E1/T1", "implement") in local.seen
    assert ("P3/E1/T1", "review") in senior.seen
    # the implement task committed feature.py to the integration branch.
    branch = _state(run_dir).last_integration_branch
    assert branch is not None
    assert "feature.py" in tracked_files(git_repo, branch)


def test_senior_implement_task_builds_on_senior(git_repo: Path, run_dir: RunDir) -> None:
    """Taste routing: an implement task flagged ``senior: true`` (a taste/polish
    slice) is built by the senior tier even though implement normally starts on
    local."""

    local, senior = _PipelineWorker(), _PipelineWorker()
    senior_impl: dict[str, object] = {
        "schema_version": "1",
        "tool": "implement",
        "args": {
            "epoch_title": "polish the UI",
            "rationale": "taste output",
            "tasks": [{**impl_task("T1", "ui.tsx"), "senior": True}],
        },
    }
    planner = MockPlanner(
        script=[
            skeleton_decision(
                phase_dict("P1", title="ui", exit_criterion=[check_cmd("test -f ui.tsx")]),
                phase_dict("P2", title="done", exit_criterion=[check_cmd("true")]),
            ),
            senior_impl,
            complete_decision(check_cmd("true")),
        ]
    )
    outcome = run_grind(
        run_dir,
        job_path="job.md",
        planner=planner,
        ladder=_two_tier(local, senior),
        repo=git_repo,
    )
    assert outcome.status == "completed", outcome
    assert ("P1/E1/T1", "implement") in senior.seen
    assert ("P1/E1/T1", "implement") not in local.seen
