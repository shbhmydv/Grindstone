"""Shared builders for the S2 epoch-loop tests.

Toy tasks, mock workers, a throwaway-git-repo factory, and a per-task routing
worker (so a fan-out test pins deterministic behavior per task under
concurrency) all live here. The only sanctioned randomness is the fuzz test's
seeded RNG; everything here is fixed.

All git ops target the caller-supplied tmp repo, never the Grindstone
checkout.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
from pathlib import Path
from typing import Callable

import pytest

from grindstone.contracts.models import (
    ArtifactEpochArgs,
    ArtifactTask,
    CmdCheck,
    ImplementEpochArgs,
    ImplementTask,
)
from grindstone.epoch_loop import EpochArgs, EpochOutcome, resume_epoch, run_epoch
from grindstone.events import (
    JournalWriter,
    PhaseRef,
    PhaseStarted,
    RunCompleted,
    RunEscalated,
    RunResumed,
    RunStarted,
    SkeletonProposed,
    read_events,
)
from grindstone.mock_worker import MockWorker
from grindstone.rundir import RunDir, create_run_dir
from grindstone.worker import WorkerRequest, WorkerTransport

_TS = "2026-06-10T00:00:00+00:00"


def _emit_run_frame(journal: JournalWriter, args: EpochArgs) -> None:
    """Minimal run/phase frame around an isolated epoch (test-only).

    Production owns this frame in ``run_loop``; the S2 epoch-isolation tests
    synthesize it here so they can still assert on ``run_started`` / replay.
    """

    journal.emit(lambda s: RunStarted(seq=s, ts=_TS, run_id="run", job_path="job.md"))
    journal.emit(
        lambda s: SkeletonProposed(
            seq=s, ts=_TS, phases=[PhaseRef(id="P1", title=args.epoch_title)]
        )
    )
    journal.emit(lambda s: PhaseStarted(seq=s, ts=_TS, phase_id="P1"))


def _close_run_frame(journal: JournalWriter, outcome: EpochOutcome) -> None:
    if outcome.status == "completed":
        journal.emit(lambda s: RunCompleted(seq=s, ts=_TS))
    else:
        journal.emit(lambda s: RunEscalated(seq=s, ts=_TS, reason=outcome.status))


def run_one_epoch(
    run_dir: RunDir,
    *,
    args: EpochArgs,
    mode: str,
    ladder: list[tuple[str, WorkerTransport]],
    repo: Path | None = None,
    **kw: object,
) -> EpochOutcome:
    """Drive a single epoch end-to-end with a synthesized run frame (test helper)."""

    with JournalWriter(run_dir.events_path) as journal:
        _emit_run_frame(journal, args)
        outcome = run_epoch(
            run_dir,
            journal=journal,
            args=args,
            mode=mode,  # type: ignore[arg-type]
            ladder=ladder,
            repo=repo,
            **kw,  # type: ignore[arg-type]
        )
        _close_run_frame(journal, outcome)
    return outcome


def resume_one_epoch(
    run_dir: RunDir,
    *,
    args: EpochArgs,
    mode: str,
    ladder: list[tuple[str, WorkerTransport]],
    repo: Path | None = None,
    **kw: object,
) -> EpochOutcome:
    """Resume a killed epoch with the caller-owned run frame (test helper)."""

    started = next(e for e in read_events(run_dir.events_path) if isinstance(e, RunStarted))
    with JournalWriter(run_dir.events_path) as journal:
        journal.emit(lambda s: RunResumed(seq=s, ts=_TS, run_id=started.run_id))
        outcome = resume_epoch(
            run_dir,
            journal=journal,
            args=args,
            mode=mode,  # type: ignore[arg-type]
            ladder=ladder,
            repo=repo,
            **kw,  # type: ignore[arg-type]
        )
        _close_run_frame(journal, outcome)
    return outcome

def reap_kill_target(proc: "subprocess.Popen[bytes]") -> None:
    """Guarantee a kill-target subprocess can never outlive the test.

    The kill-mid-* signature tests SIGKILL the subprocess as their deliberate
    action, but if the kill point is never reached (e.g. a busy host misses the
    60s deadline) the assert fires first and control skips the kill, leaving
    the subprocess hot-spinning on a ``release`` sentinel that never appears.
    A bare ``proc.wait()`` in ``finally`` then blocks FOREVER on that immortal
    process (observed 2026-06-13: a 5-hour CPU-pegging hang). Always kill if
    still alive, then bound the wait so a wedged reap fails fast instead.
    """

    if proc.poll() is None:
        os.kill(proc.pid, signal.SIGKILL)
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        pass


#: The toy artifact every "ok" attempt produces and the checks grep for.
OUT_FILE = "out.txt"
OUT_CONTENT = "GRINDSTONE\n"


# --- git repo factory ----------------------------------------------------------


def git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=str(repo), capture_output=True, text=True, check=True
    )


def init_git_repo(path: Path) -> Path:
    """Create a throwaway git repo with one base commit + a .grindstone ignore."""

    path.mkdir(parents=True, exist_ok=True)
    git(path, "init", "-q", "-b", "main")
    git(path, "config", "user.email", "toy@grindstone.local")
    git(path, "config", "user.name", "toy")
    # Gitignore the run dir AND Python bytecode: an implement task whose
    # done_when runs `python3 -m pytest` leaves __pycache__/*.pyc, which an
    # un-ignored `git add -A` would stage and the ownership scope check would
    # (correctly) flag as out-of-scope, the S3 Gate-B flakiness. Real Python
    # repos gitignore it; the toy repo mirrors that hygiene.
    (path / ".gitignore").write_text(".grindstone/\n__pycache__/\n", encoding="utf-8")
    (path / "README.md").write_text("base\n", encoding="utf-8")
    git(path, "add", "-A")
    git(path, "commit", "-q", "-m", "base")
    return path


def tracked_files(repo: Path, ref: str) -> list[str]:
    out = git(repo, "ls-tree", "-r", "--name-only", ref).stdout
    return sorted(line.strip() for line in out.splitlines() if line.strip())


# --- task + epoch builders -----------------------------------------------------


def make_toy_task(
    task_id: str = "T1", out_file: str = OUT_FILE, owned: list[str] | None = None
) -> ImplementTask:
    """An implement task: create ``out_file`` containing GRINDSTONE, prove it."""

    return ImplementTask(
        id=task_id,
        goal=f"create {out_file} containing GRINDSTONE",
        done_when=[CmdCheck(cmd=f"test -f {out_file}")],
        criteria=[f"{out_file} contains the token GRINDSTONE"],
        file_ownership=owned if owned is not None else [out_file],
    )


def implement_epoch(*tasks: ImplementTask, title: str = "toy epoch") -> ImplementEpochArgs:
    return ImplementEpochArgs(epoch_title=title, rationale="toy", tasks=list(tasks))


def artifact_epoch(*tasks: ArtifactTask, title: str = "toy artifact epoch") -> ArtifactEpochArgs:
    return ArtifactEpochArgs(epoch_title=title, rationale="toy", tasks=list(tasks))


def run_implement(
    repo: Path,
    run_dir: RunDir,
    tasks: list[ImplementTask],
    ladder: list[tuple[str, WorkerTransport]],
    **kw: object,
) -> EpochOutcome:
    return run_one_epoch(
        run_dir,
        args=implement_epoch(*tasks),
        mode="implement",
        ladder=ladder,
        repo=repo,
        **kw,
    )


def make_ok_worker(content: str = OUT_CONTENT, out_file: str = OUT_FILE) -> MockWorker:
    """A worker that succeeds on the first call, writing the toy artifact."""

    return MockWorker(script=["ok"], artifacts={out_file: content})


# --- handoff helpers (edge-case drivers) ---------------------------------------


def handoff_payload(
    task_id: str = "P1/E1/T1",
    *,
    status: str = "DONE",
    citations: list[dict[str, object]] | None = None,
    out_file: str = OUT_FILE,
) -> dict[str, object]:
    """A schema-valid handoff dict the edge-case workers can mutate freely."""

    return {
        "schema_version": "1",
        "task_id": task_id,
        "status": status,
        "what_changed": [],
        "resulting_state": "edge-case handoff",
        "downstream_needs": [],
        "not_done": [] if status == "DONE" else ["did not finish"],
        "citations": [{"file": out_file}] if citations is None else citations,
        "checks": [{"check": f"test -f {out_file}", "exit_code": 0}],
        "occupancy": {"compacted": False, "subagent_splits": 0},
    }


class HandoffWorker:
    """Writes an arbitrary handoff + scratch files, drives exact disk state."""

    def __init__(
        self, payload: dict[str, object], files: dict[str, str] | None = None
    ) -> None:
        self.payload = payload
        self.files = files if files is not None else {OUT_FILE: OUT_CONTENT}

    def run(self, request: WorkerRequest) -> None:
        if request.mode == "implement":
            # Satisfy the core-appended review gate so each edge-case test
            # exercises ITS failure, not a missing review.md.
            (request.scratch / "review.md").write_text("reviewed\n", encoding="utf-8")
        for rel, content in self.files.items():
            path = request.scratch / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        (request.scratch / "handoff.json").write_text(
            json.dumps(self.payload), encoding="utf-8"
        )


class RoutingWorker:
    """Dispatch each ``run`` to a per-task sub-worker keyed by short task id.

    Fan-out shares ONE ladder worker across all tasks; routing by task id gives
    each task its own stateful sub-worker so behavior stays deterministic under
    concurrency (no shared script consumed out of order).
    """

    def __init__(self, by_task: dict[str, WorkerTransport]) -> None:
        self.by_task = by_task

    def run(self, request: WorkerRequest) -> None:
        short = request.task_id.rsplit("/", 1)[-1]
        self.by_task[short].run(request)


def _first_literal(glob: str) -> str:
    """The literal path part of an ownership/target glob (drops wildcards)."""

    out: list[str] = []
    for ch in glob.rstrip("/"):
        if ch in "*?[":
            break
        out.append(ch)
    return ("".join(out) or "artifact.txt").rstrip("/")


class OwnershipWorker:
    """A multi-epoch worker: each task creates its own owned/target file.

    Derives the file name from the implement task's first ``file_ownership`` glob
    (or an artifact task's first ``targets`` entry), writes it in the scratch
    CWD, and emits a schema- + semantic-valid DONE handoff citing it. Stateless,
    so one instance drives any number of epochs without a consumed script.
    """

    def __init__(self, content: str = "ok\n") -> None:
        self.content = content

    def run(self, request: WorkerRequest) -> None:
        task = request.task
        if request.mode == "implement":
            (request.scratch / "review.md").write_text("reviewed\n", encoding="utf-8")
        if isinstance(task, ImplementTask):
            rel = _first_literal(task.file_ownership[0])
        elif task.targets:
            rel = _first_literal(task.targets[0])
        else:
            rel = _first_literal(task.artifact_out)
        path = request.scratch / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.content, encoding="utf-8")
        payload = {
            "schema_version": "1",
            "task_id": request.task_id,
            "status": "DONE",
            "what_changed": [{"kind": "file", "ref": rel}],
            "resulting_state": f"created {rel}",
            "downstream_needs": [],
            "not_done": [],
            "citations": [{"file": rel}],
            "checks": [
                {"check": c.cmd if hasattr(c, "cmd") else "artifact", "exit_code": 0}
                for c in request.task.done_when
            ],
            "occupancy": {"compacted": False, "subagent_splits": 0},
        }
        (request.scratch / "handoff.json").write_text(json.dumps(payload), encoding="utf-8")


# --- decision builders (mock-planner script payloads) --------------------------


def check_cmd(cmd: str, expect: int = 0) -> dict[str, object]:
    return {"cmd": cmd} if expect == 0 else {"cmd": cmd, "expect_exit": expect}


def phase_dict(
    pid: str,
    *,
    title: str = "phase",
    exit_criterion: list[dict[str, object]] | None = None,
    budget: int = 5,
) -> dict[str, object]:
    return {
        "id": pid,
        "title": title,
        "exit_criterion": exit_criterion or [check_cmd("true")],
        "epoch_budget": budget,
    }


def skeleton_decision(*phases: dict[str, object]) -> dict[str, object]:
    return {"schema_version": "1", "tool": "propose_skeleton", "args": {"phases": list(phases)}}


def two_phase_skeleton() -> dict[str, object]:
    return skeleton_decision(phase_dict("P1", title="build"), phase_dict("P2", title="verify"))


def impl_task(tid: str, fname: str) -> dict[str, object]:
    return {
        "id": tid,
        "goal": f"create {fname}",
        "done_when": [check_cmd(f"test -f {fname}")],
        "file_ownership": [fname],
    }


def implement_decision(
    *tasks: dict[str, object], title: str = "impl", rationale: str = "x"
) -> dict[str, object]:
    return {
        "schema_version": "1",
        "tool": "implement",
        "args": {"epoch_title": title, "rationale": rationale, "tasks": list(tasks)},
    }


def artifact_task(tid: str, out: str | None = None) -> dict[str, object]:
    out = out or f"P1/E1/{tid}/note.md"
    return {"id": tid, "goal": f"produce {out}", "done_when": [check_cmd("true")], "artifact_out": out}


def artifact_decision(
    *tasks: dict[str, object], title: str = "art", rationale: str = "x"
) -> dict[str, object]:
    return {
        "schema_version": "1",
        "tool": "artifact",
        "args": {"epoch_title": title, "rationale": rationale, "tasks": list(tasks)},
    }


def _mode_decision(
    tool: str, *tasks: dict[str, object], title: str, rationale: str = "x"
) -> dict[str, object]:
    return {
        "schema_version": "1",
        "tool": tool,
        "args": {"epoch_title": title, "rationale": rationale, "tasks": list(tasks)},
    }


def research_decision(
    *tasks: dict[str, object], title: str = "research", rationale: str = "x"
) -> dict[str, object]:
    return _mode_decision("research", *tasks, title=title, rationale=rationale)


def review_decision(
    *tasks: dict[str, object], title: str = "review", rationale: str = "x"
) -> dict[str, object]:
    return _mode_decision("review", *tasks, title=title, rationale=rationale)


def complete_decision(*evidence: dict[str, object], summary: str = "all done") -> dict[str, object]:
    return {
        "schema_version": "1",
        "tool": "complete_run",
        "args": {"summary": summary, "evidence": list(evidence)},
    }


def revise_decision(*phases: dict[str, object], reason: str = "rescope") -> dict[str, object]:
    return {"schema_version": "1", "tool": "revise_phases", "args": {"reason": reason, "phases": list(phases)}}


def phase_complete_decision(
    *deliverables: str, summary: str = "phase deliverables are in place"
) -> dict[str, object]:
    return {
        "schema_version": "1",
        "tool": "phase_complete",
        "args": {"summary": summary, "deliverables": list(deliverables)},
    }


def escalate_decision(reason: str = "cannot proceed") -> dict[str, object]:
    return {"schema_version": "1", "tool": "escalate_run", "args": {"reason": reason}}


def handle_failed_epoch_retry(hint: str = "try again", escalate_tier: bool = False) -> dict[str, object]:
    args: dict[str, object] = {"action": "retry", "hint": hint}
    if escalate_tier:
        args["escalate_tier"] = True
    return {"schema_version": "1", "tool": "handle_failed_epoch", "args": args}


def handle_failed_epoch_escalate(diagnosis: str = "needs senior") -> dict[str, object]:
    return {
        "schema_version": "1",
        "tool": "handle_failed_epoch",
        "args": {"action": "escalate_senior", "diagnosis": diagnosis},
    }


def handle_failed_epoch_halt(reason: str = "gate is broken, not the code") -> dict[str, object]:
    return {
        "schema_version": "1",
        "tool": "handle_failed_epoch",
        "args": {"action": "halt", "reason": reason},
    }


class FailingWorker:
    """A worker that NEVER writes a handoff, so every attempt fails the gate.

    Drives a task to FAILED (ladder exhausted) deterministically. Records which
    scratches it saw the planner's failure-context hint in, so a test can assert
    a retry hint reached the worker. Stateless across epochs."""

    def __init__(self) -> None:
        self.seen_failure_contexts: list[list[str]] = []

    def run(self, request: WorkerRequest) -> None:
        self.seen_failure_contexts.append(list(request.failure_context))
        # No handoff.json written -> _collect_handoff raises "no handoff.json".


# --- fixtures ------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_worktree_base(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect the external git-worktree base into the test's own tmp dir.

    Production hosts throwaway worktrees under ``/tmp/cache/grindstone`` (outside any
    target repo, so a worker cannot strip its CWD to the repo root). Tests must not
    collide on that shared dir nor leave debris there, so each test points
    ``GRINDSTONE_WORKTREE_BASE`` at its own ``tmp_path`` (pytest reaps it). The base
    is a SIBLING of the ``tmp_path/repo`` checkout the fixtures build, never nested,
    preserving the very out-of-repo property the relocation guarantees.
    """

    monkeypatch.setenv("GRINDSTONE_WORKTREE_BASE", str(tmp_path / "wt-base"))


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    return init_git_repo(tmp_path / "repo")


@pytest.fixture
def run_dir(git_repo: Path) -> RunDir:
    return create_run_dir(git_repo, "run-1")


@pytest.fixture
def toy_task() -> ImplementTask:
    return make_toy_task()


@pytest.fixture
def ok_worker() -> MockWorker:
    return make_ok_worker()


@pytest.fixture
def make_handoff() -> Callable[..., dict[str, object]]:
    return handoff_payload
