"""Worker prompt shape: the model-facing format ``build_worker_prompt`` emits.

Focused on the reference-not-embed prior-failure feedback: the ``<prior_failures>``
block carries a SHORT summary + PATH per attempt (the worker reads the path for
full detail), never the embedded bulk reason.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from grindstone.config import load_operating_skill
from grindstone.contracts.models import ArtifactTask, CmdCheck, ImplementTask
from grindstone.contracts.semantics import HandoffMode
from grindstone.worker import (
    REVIEW_CHECK_COMMAND,
    REVIEW_FILENAME,
    WORKER_SCENARIOS,
    WorkerRequest,
    build_worker_prompt,
    select_worker_scenario,
)


def _request(failure_context: list[str]) -> WorkerRequest:
    task = ImplementTask(
        id="T1",
        goal="edit the widget",
        done_when=[CmdCheck(cmd="true")],
        file_ownership=["src/widget.py"],
    )
    return WorkerRequest(
        task=task,
        task_id="P1/E1/T1",
        inputs={},
        scratch=Path("/tmp/scratch"),
        attempt=2,
        failure_context=failure_context,
        mode="implement",
    )


def _implement_request() -> WorkerRequest:
    task = ImplementTask(
        id="T1",
        goal="edit the widget",
        done_when=[CmdCheck(cmd="true")],
        file_ownership=["src/widget.py", "src/helpers/*.py"],
    )
    return WorkerRequest(
        task=task,
        task_id="P1/E1/T1",
        inputs={},
        scratch=Path("/tmp/scratch"),
        attempt=1,
        failure_context=[],
        mode="implement",
    )


def _artifact_request(
    mode: HandoffMode, *, targets: list[str] | None = None
) -> WorkerRequest:
    task = ArtifactTask(
        id="T1",
        goal="produce the report",
        done_when=[CmdCheck(cmd="true")],
        artifact_out="report.md",
        targets=targets,
    )
    return WorkerRequest(
        task=task,
        task_id="P1/E1/T1",
        inputs={},
        scratch=Path("/tmp/scratch"),
        attempt=1,
        failure_context=[],
        mode=mode,
    )


# --- operating-skill split: WORKER_CORE + per-mode scenario skill ---------------

#: A phrase unique to each mode's loaded ``.md`` (present in exactly that one
#: scenario, and NOT in WORKER_CORE or the dynamic blocks).
_MODE_SENTINELS = {
    "implement": "CONTRACT FIRST",
    "research": "investigate and report",
    "review": "judge the targets",
    "artifact": "Produce the artifact named above so that",
}


def test_select_worker_scenario_maps_task_and_mode() -> None:
    # An ImplementTask is always the implement scenario, regardless of mode.
    assert select_worker_scenario(_implement_request()) == "implement"
    # A non-write ArtifactTask routes by the epoch mode.
    assert select_worker_scenario(_artifact_request("research")) == "research"
    assert select_worker_scenario(_artifact_request("review")) == "review"
    assert select_worker_scenario(_artifact_request("artifact")) == "artifact"
    # Every name the selector can return has a loadable skill file.
    for req in (
        _implement_request(),
        _artifact_request("research"),
        _artifact_request("review"),
        _artifact_request("artifact"),
    ):
        assert select_worker_scenario(req) in WORKER_SCENARIOS


@pytest.mark.parametrize("mode", ["implement", "research", "review", "artifact"])
def test_build_worker_prompt_composes_core_plus_one_scenario(mode: str) -> None:
    if mode == "implement":
        request = _implement_request()
    else:
        request = _artifact_request(mode)  # type: ignore[arg-type]
    prompt = build_worker_prompt(request)
    # The always-on WORKER_CORE skeleton is present in every composition.
    assert f'<task id="{request.task_id}">' in prompt
    assert "<done_when>" in prompt
    assert "<handoff>" in prompt
    # The selected scenario's sentinel is present...
    assert _MODE_SENTINELS[mode] in prompt
    # ...and the OTHER three modes' sentinels are NOT.
    for other, sentinel in _MODE_SENTINELS.items():
        if other != mode:
            assert sentinel not in prompt


def test_implement_prompt_keeps_dynamic_file_ownership() -> None:
    prompt = build_worker_prompt(_implement_request())
    # The dynamic ownership globs stay in the code-built block (not the .md).
    assert "<file_ownership>" in prompt
    assert "src/widget.py" in prompt
    assert "src/helpers/*.py" in prompt
    # The static discipline is NOT in the dynamic block, it is the loaded skill.
    assert "CONTRACT FIRST" in load_operating_skill("worker", "implement")
    assert "<file_ownership>" not in load_operating_skill("worker", "implement")


def test_review_prompt_keeps_dynamic_targets() -> None:
    prompt = build_worker_prompt(
        _artifact_request("review", targets=["src/a.py", "src/b.py"])
    )
    assert "src/a.py" in prompt
    assert "src/b.py" in prompt
    # The dynamic targets list is NOT baked into the static review skill.
    assert "src/a.py" not in load_operating_skill("worker", "review")


def test_worker_skill_constant_literals_do_not_drift() -> None:
    """Drift guard: the implement skill spells the live constant VALUES (it is
    concatenated raw, never ``.format``-ed), so a constant change must update the
    .md too. The pinned reviewer subagent name is likewise spelled literally."""

    implement = load_operating_skill("worker", "implement")
    assert REVIEW_FILENAME in implement
    assert REVIEW_CHECK_COMMAND in implement
    # The fresh-context review step names the registered ``reviewer`` subagent.
    assert "`reviewer`" in implement


def test_split_preserves_every_load_bearing_worker_instruction() -> None:
    """Content-preservation guard: the four monolithic plan helpers were split into
    WORKER_CORE + four scenario files + the in-code dynamic blocks. Every load-bearing
    instruction must survive SOMEWHERE in the union of the four composed prompts."""

    union = (
        build_worker_prompt(_implement_request())
        + build_worker_prompt(
            _artifact_request("review", targets=["src/a.py"])
        )
        + build_worker_prompt(_artifact_request("research"))
        + build_worker_prompt(_artifact_request("artifact"))
    )
    for phrase in (
        # implement discipline
        "CONTRACT FIRST",
        "WORK IN DEPENDENCY ORDER",
        "VERBATIM SPEC",
        "BAKE BEFORE HANDOFF",
        "fresh-context review",
        "check_handoff.py",
        "false DONE is always caught",
        "Changing ANY other file fails the attempt",
        # research
        "investigate and report",
        "do not modify code",
        "deliverable the planner reads",
        # review
        "judge the targets",
        "explicit verdict",
        "A review handoff with no citations is rejected",
        # artifact
        "every done_when check passes",
        "keep the handoff to references, not payloads",
        # CWD containment (inlined into the three non-implement skills)
        "Your CWD is your entire workspace",
        "there is no repository here",
    ):
        assert phrase in union, f"lost instruction: {phrase!r}"


def test_no_prior_failures_block_when_context_empty() -> None:
    prompt = build_worker_prompt(_request([]))
    assert "<prior_failures>" not in prompt


def test_prior_failures_renders_summary_and_path() -> None:
    entry = (
        "attempt 1: out-of-scope writes: a/x.py, ... and 46 more "
        "[full detail: /run/.grindstone/runs/r/P1/E1/T1/failures/attempt-1.txt]"
    )
    prompt = build_worker_prompt(_request([entry]))
    assert "<prior_failures>" in prompt
    assert entry in prompt
    # The block tells the worker it MAY read the referenced paths for detail.
    block = prompt.split("<prior_failures>", 1)[1].split("</prior_failures>", 1)[0]
    assert "PATH" in block
    assert "may read" in block.lower()


def test_prior_failures_carries_no_embedded_bulk() -> None:
    """Even when the underlying failure was huge, the inline entry is a short
    summary + path: the prompt size is bounded by the entry, not the failure."""

    # A correctly-built entry (what the loop produces) is already short; the prompt
    # must not balloon even across several stacked attempts.
    entries = [
        f"attempt {i}: out-of-scope writes: a/x.py, ... and 4999 more "
        f"[full detail: /run/P1/E1/T1/failures/attempt-{i}.txt]"
        for i in range(1, 4)
    ]
    prompt = build_worker_prompt(_request(entries))
    # No single node_modules-style path list is embedded: the bulk is on disk.
    assert "node_modules" not in prompt
    assert prompt.count("full detail:") == 3
    # The whole prior_failures block stays small (kilobytes, never the megabyte bulk).
    block = prompt.split("<prior_failures>", 1)[1].split("</prior_failures>", 1)[0]
    assert len(block) < 2048


# --- domain skills: selected-only delivery in the worker prompt -----------------


def test_skills_field_round_trips_on_tasks() -> None:
    """The optional ``skills`` field is accepted + preserved on both task shapes."""

    impl = ImplementTask(
        id="T1", goal="g", done_when=[CmdCheck(cmd="true")],
        file_ownership=["a.py"], skills=["rn-nav", "rn-a11y"],
    )
    art = ArtifactTask(
        id="T1", goal="g", done_when=[CmdCheck(cmd="true")],
        artifact_out="r.md", skills=["rn-nav"],
    )
    assert impl.skills == ["rn-nav", "rn-a11y"]
    assert art.skills == ["rn-nav"]
    # Default is an empty list (a task that selected none).
    assert ImplementTask(
        id="T2", goal="g", done_when=[CmdCheck(cmd="true")], file_ownership=["b.py"]
    ).skills == []


def test_worker_prompt_composes_only_selected_domain_skills() -> None:
    """Retrieve-not-concatenate: the prompt carries ONLY the skills in
    request.domain_skills (already the selected subset), each under its named tag."""

    request = WorkerRequest(
        task=ImplementTask(
            id="T1", goal="g", done_when=[CmdCheck(cmd="true")],
            file_ownership=["a.py"], skills=["rn-nav"],
        ),
        task_id="P1/E1/T1", inputs={}, scratch=Path("/tmp/s"),
        attempt=1, failure_context=[], mode="implement",
        domain_skills={"rn-nav": "NAV BODY TEXT"},
    )
    prompt = build_worker_prompt(request)
    assert "<domain_skills>" in prompt
    assert '<skill name="rn-nav">' in prompt
    assert "NAV BODY TEXT" in prompt


def test_worker_prompt_has_no_domain_skills_block_when_empty() -> None:
    request = _implement_request()  # no domain_skills -> empty dict default
    assert "<domain_skills>" not in build_worker_prompt(request)
