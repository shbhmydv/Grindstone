"""CLI wiring (S4 ruling 5): arg -> run-dir -> run_grind end-to-end through
``main`` with an injected mock planner (no real codex/pi quota), plus exit-code
mapping, watch on a terminal run, resume, and the run-id default slug.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from grindstone import cli
from grindstone.cli import _default_run_id, main
from grindstone.mock_planner import MockPlanner
from grindstone.rundir import RunDir

from tests.grindstone.conftest import (
    OwnershipWorker,
    check_cmd,
    complete_decision,
    escalate_decision,
    two_phase_skeleton,
)


def _ladder() -> list[tuple[str, OwnershipWorker]]:
    return [("worker", OwnershipWorker())]


def _job(repo: Path) -> str:
    # A real run always reads config (the fan-out bound = roles.worker.slots), so
    # scaffold one; planner + ladder are still injected (no codex/pi spent).
    main(["init", "--repo", str(repo)])
    job = repo / "job.md"
    job.write_text("Build two phases, then complete.\n", encoding="utf-8")
    return str(job)


# --- run: arg -> wiring -> exit code -------------------------------------------


def test_run_completes_through_main(git_repo: Path) -> None:
    planner = MockPlanner(script=[two_phase_skeleton(), complete_decision(check_cmd("true"))])
    code = main(
        ["run", _job(git_repo), "--repo", str(git_repo), "--run-id", "cli-run"],
        planner=planner,
        ladder=_ladder(),
    )
    assert code == 0  # completed
    # The run dir was created under the target repo and carries terminal state.
    run_dir = RunDir(root=git_repo / ".grindstone" / "runs" / "cli-run")
    assert run_dir.run_state_path.exists()
    assert '"status": "completed"' in run_dir.run_state_path.read_text()


def test_run_watch_runs_and_renders_then_exits(git_repo: Path) -> None:
    # `run --watch` is the single human-facing command: it grinds in a background
    # thread AND renders the live TUI in the foreground, returning the run's exit
    # code when it reaches terminal. Same wiring as plain `run`, plus the render.
    planner = MockPlanner(script=[two_phase_skeleton(), complete_decision(check_cmd("true"))])
    code = main(
        ["run", _job(git_repo), "--repo", str(git_repo), "--run-id", "wd", "--watch"],
        planner=planner,
        ladder=_ladder(),
    )
    assert code == 0  # completed end-to-end through the watched path
    run_dir = RunDir(root=git_repo / ".grindstone" / "runs" / "wd")
    assert '"status": "completed"' in run_dir.run_state_path.read_text()


def test_run_writes_journal_then_next_run_reaps_it(git_repo: Path) -> None:
    # A completed run leaves a markdown journal.md; starting a SECOND run reaps
    # the first run's journal (only the latest keeps one) but never its events.
    runs = git_repo / ".grindstone" / "runs"
    main(
        ["run", _job(git_repo), "--repo", str(git_repo), "--run-id", "j1"],
        planner=MockPlanner(script=[two_phase_skeleton(), complete_decision(check_cmd("true"))]),
        ladder=_ladder(),
    )
    j1 = RunDir(root=runs / "j1")
    assert j1.journal_path.exists()
    assert "# Run j1" in j1.journal_path.read_text(encoding="utf-8")

    main(
        ["run", _job(git_repo), "--repo", str(git_repo), "--run-id", "j2"],
        planner=MockPlanner(script=[two_phase_skeleton(), complete_decision(check_cmd("true"))]),
        ladder=_ladder(),
    )
    j2 = RunDir(root=runs / "j2")
    # The previous run's journal is reaped; its durable events.ndjson survives.
    assert not j1.journal_path.exists()
    assert j1.events_path.exists()
    # The latest run keeps its journal.
    assert j2.journal_path.exists()


def test_run_escalated_exit_code(git_repo: Path) -> None:
    planner = MockPlanner(script=[two_phase_skeleton(), escalate_decision("cannot proceed")])
    code = main(
        ["run", _job(git_repo), "--repo", str(git_repo), "--run-id", "esc"],
        planner=planner,
        ladder=_ladder(),
    )
    assert code == 1  # escalated


def test_run_valve_failed_exit_code_and_watch(
    git_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The production safety valve (planner-call cap) trips before the run can
    # complete -> durable status "failed" -> exit 2. The run_failed event closes
    # the journal, so a subsequent `watch` renders + exits 2 instead of hanging.
    # Pin the config-derived cap to 1: the skeleton call lands, the next planner
    # call trips the valve (the CLI passes this resolved cap explicitly).
    planner = MockPlanner(script=[two_phase_skeleton()])  # never reaches complete_run
    monkeypatch.setattr(cli, "_resolve_max_planner_calls", lambda cfg: 1)
    code = main(
        ["run", _job(git_repo), "--repo", str(git_repo), "--run-id", "v"],
        planner=planner,
        ladder=_ladder(),
    )
    assert code == 2  # failed (valve)
    assert main(["watch", "v", "--repo", str(git_repo)]) == 2


# --- FIX 6: run/resume reject a repo config naming a script outside models/ -----


def test_run_rejects_repo_script_outside_models(
    git_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("GRINDSTONE_ALLOW_REPO_SCRIPTS", raising=False)
    main(["init", "--repo", str(git_repo)])
    # Tamper: a cloned repo points the roles at arbitrary executables (RCE).
    cfg = git_repo / ".grindstone" / "config.yaml"
    cfg.write_text(
        "roles:\n"
        "  planner: {script: /tmp/evil.sh, slots: 1, timeout_s: 600}\n"
        "  worker: {script: /tmp/eviltoo.sh, slots: 2, timeout_s: 1800}\n",
        encoding="utf-8",
    )
    job = git_repo / "job.md"
    job.write_text("x\n", encoding="utf-8")
    # The validator fires BEFORE any grind/Popen (even with injected transports).
    with pytest.raises(ValueError):
        main(
            ["run", str(job), "--repo", str(git_repo), "--run-id", "rce"],
            planner=MockPlanner(script=[]),
            ladder=_ladder(),
        )


# --- watch: terminal run renders and exits -------------------------------------


def test_watch_terminal_run_exits_zero(git_repo: Path) -> None:
    planner = MockPlanner(script=[two_phase_skeleton(), complete_decision(check_cmd("true"))])
    main(
        ["run", _job(git_repo), "--repo", str(git_repo), "--run-id", "w"],
        planner=planner,
        ladder=_ladder(),
    )
    # Resolve by run id under --repo.
    assert main(["watch", "w", "--repo", str(git_repo)]) == 0
    # Resolve by explicit run-dir path too.
    run_dir = git_repo / ".grindstone" / "runs" / "w"
    assert main(["watch", str(run_dir)]) == 0


# --- resume: the G4 verifier is wired exactly like `run` (P0 repro) ------------


def test_resume_passes_verifier_like_run(
    git_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """P0: `grindstone resume` must build + pass the G4 verifier to ``resume_grind``
    exactly as `run` does for ``run_grind``. The scaffolded config has ``verify_epochs``
    on + a local tier, so a NON-None verifier must reach ``resume_grind``; before the
    fix it passed ``None`` and silently lost semantic verification for the whole
    remainder of any resumed run."""

    main(["init", "--repo", str(git_repo)])
    job = git_repo / "job.md"
    job.write_text("x\n", encoding="utf-8")
    main(
        ["run", str(job), "--repo", str(git_repo), "--run-id", "rv"],
        planner=MockPlanner(script=[two_phase_skeleton(), complete_decision(check_cmd("true"))]),
        ladder=_ladder(),
    )

    captured: dict[str, object] = {}
    real = cli.resume_grind

    def _spy(run_dir: object, **kwargs: object) -> object:
        captured.update(kwargs)
        return real(run_dir, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(cli, "resume_grind", _spy)
    main(["resume", "rv", "--repo", str(git_repo)], planner=MockPlanner(script=[]), ladder=_ladder())
    assert captured["verifiers"] is not None


def test_resume_no_verifier_when_verify_epochs_off(
    git_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With ``verify_epochs: false`` in the config the resume path stays verifier-less
    (a clean skip, the no-verifier case must keep behaving as today)."""

    main(["init", "--repo", str(git_repo)])
    cfg = git_repo / ".grindstone" / "config.yaml"
    cfg.write_text(cfg.read_text(encoding="utf-8") + "\nverify_epochs: false\n", encoding="utf-8")
    job = git_repo / "job.md"
    job.write_text("x\n", encoding="utf-8")
    main(
        ["run", str(job), "--repo", str(git_repo), "--run-id", "rnv"],
        planner=MockPlanner(script=[two_phase_skeleton(), complete_decision(check_cmd("true"))]),
        ladder=_ladder(),
    )

    captured: dict[str, object] = {}
    real = cli.resume_grind

    def _spy(run_dir: object, **kwargs: object) -> object:
        captured.update(kwargs)
        return real(run_dir, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(cli, "resume_grind", _spy)
    main(["resume", "rnv", "--repo", str(git_repo)], planner=MockPlanner(script=[]), ladder=_ladder())
    assert captured["verifiers"] is None


# --- resume: terminal run is idempotent through main ---------------------------


def test_resume_terminal_run_through_main(git_repo: Path) -> None:
    planner = MockPlanner(script=[two_phase_skeleton(), complete_decision(check_cmd("true"))])
    main(
        ["run", _job(git_repo), "--repo", str(git_repo), "--run-id", "r"],
        planner=planner,
        ladder=_ladder(),
    )
    code = main(
        ["resume", "r", "--repo", str(git_repo)],
        planner=MockPlanner(script=[]),
        ladder=_ladder(),
    )
    assert code == 0


# --- run id slug ---------------------------------------------------------------


def test_default_run_id_is_utc_slug() -> None:
    assert re.fullmatch(r"\d{8}T\d{6}Z", _default_run_id())


def test_run_uses_default_run_id_when_unset(git_repo: Path) -> None:
    planner = MockPlanner(script=[two_phase_skeleton(), complete_decision(check_cmd("true"))])
    code = main(
        ["run", _job(git_repo), "--repo", str(git_repo)],
        planner=planner,
        ladder=_ladder(),
    )
    assert code == 0
    runs = list((git_repo / ".grindstone" / "runs").iterdir())
    assert len(runs) == 1 and re.fullmatch(r"\d{8}T\d{6}Z", runs[0].name)
