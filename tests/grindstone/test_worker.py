"""Worker prompt shape: the model-facing format ``build_worker_prompt`` emits.

Focused on the reference-not-embed prior-failure feedback: the ``<prior_failures>``
block carries a SHORT summary + PATH per attempt (the worker reads the path for
full detail), never the embedded bulk reason.
"""

from __future__ import annotations

from pathlib import Path

from grindstone.contracts.models import CmdCheck, ImplementTask
from grindstone.worker import WorkerRequest, build_worker_prompt


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
