"""B3 vision-review taste gate: the third ``Check`` variant + its disk contract.

A ``visual`` epoch's gate adds a codex VISION REVIEW on top of the deterministic
functional floor: codex looks at a rendered-UI screenshot + criteria and writes a
``pass``/``reasons`` verdict to ``verdict.json`` (a disk contract grindstone
re-reads + Pydantic-validates, never stdout). These tests drive the whole gate
through ``evaluate_checks`` with a STUB vision-review script behind the
``ScriptVisionReviewer`` config seam, so no real codex is ever called.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from grindstone.contracts import decision_schema_errors, parse_decision
from grindstone.contracts.models import (
    VisionReviewCheck,
    VisionReviewSpec,
    parse_vision_verdict,
)
from grindstone.rundir import create_run_dir
from grindstone.run_loop import evaluate_checks
from grindstone.script_vision import ScriptVisionReviewer, VisionReviewError

# --- stub vision-review scripts (honour the --out disk contract) ---------------

#: A stub that writes a canned verdict to ``--out`` and exits ``$RC`` (default 0).
#: It parses only ``--out`` (the disk contract), the screenshot/criteria/schema
#: it ignores, standing in for codex without ever calling it.
_STUB = """\
set -euo pipefail
out=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --out) out="$2"; shift 2 ;;
    *) shift ;;
  esac
done
{body}
"""


def _reviewer(tmp_path: Path, body: str) -> ScriptVisionReviewer:
    script = tmp_path / "vision_stub.sh"
    script.write_text("#!/usr/bin/env bash\n" + _STUB.format(body=body), encoding="utf-8")
    script.chmod(0o755)
    return ScriptVisionReviewer(script=script, timeout_s=30)


_PASS = 'printf \'{"pass": true, "reasons": []}\' > "$out"\nexit 0\n'
_FAIL = (
    'printf \'{"pass": false, "reasons": ["button misaligned", "low contrast"]}\' > "$out"\n'
    "exit 0\n"
)
_BAD_JSON = 'printf \'not json at all\' > "$out"\nexit 0\n'
_NONZERO = 'echo "codex vision call failed" >&2\nexit 5\n'


def _vision_check(screenshot: str = "ui/screen.png") -> VisionReviewCheck:
    return VisionReviewCheck(
        vision_review=VisionReviewSpec(screenshot=screenshot, criteria="polished, aligned UI")
    )


def _put_screenshot(root: Path, rel: str = "ui/screen.png") -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"\x89PNG\r\n\x1a\n fake png bytes")


# --- VisionVerdict: the parsed disk contract -----------------------------------


def test_vision_verdict_parses_pass() -> None:
    v = parse_vision_verdict({"pass": True, "reasons": []})
    assert v.passed is True and v.reasons == []


def test_vision_verdict_parses_fail_with_reasons() -> None:
    v = parse_vision_verdict({"pass": False, "reasons": ["misaligned"]})
    assert v.passed is False and v.reasons == ["misaligned"]


def test_vision_verdict_rejects_non_bool_pass() -> None:
    # StrictBool mirrors the schema's boolean type: "yes"/1 are not a verdict.
    with pytest.raises(ValidationError):
        parse_vision_verdict({"pass": "yes", "reasons": []})


def test_vision_verdict_rejects_extra_key() -> None:
    with pytest.raises(ValidationError):
        parse_vision_verdict({"pass": True, "reasons": [], "confidence": 0.9})


def test_vision_verdict_rejects_missing_reasons() -> None:
    with pytest.raises(ValidationError):
        parse_vision_verdict({"pass": True})


# --- the new Check variant: schema + typed model agree -------------------------


def _skeleton_with_vision(extra: dict[str, object] | None = None) -> dict[str, object]:
    vr: dict[str, object] = {"screenshot": "ui/screen.png", "criteria": "polished"}
    if extra:
        vr.update(extra)
    return {
        "schema_version": "1",
        "tool": "propose_skeleton",
        "args": {
            "phases": [
                {"id": "P1", "title": "build", "exit_criterion": [{"cmd": "true"}], "epoch_budget": 2},
                {
                    "id": "P2",
                    "title": "review the look",
                    "exit_criterion": [
                        {"cmd": "make screenshot"},
                        {"vision_review": vr},
                    ],
                    "epoch_budget": 1,
                },
            ]
        },
    }


def test_check_union_accepts_vision_review_both_layers() -> None:
    payload = _skeleton_with_vision()
    assert not decision_schema_errors(payload)
    decision = parse_decision(payload)
    # The third Check variant parsed into a typed VisionReviewCheck.
    p2_checks = decision.args.phases[1].exit_criterion  # type: ignore[union-attr]
    assert isinstance(p2_checks[1], VisionReviewCheck)
    assert p2_checks[1].vision_review.screenshot == "ui/screen.png"


def test_check_union_vision_review_forbids_extra_key_both_layers() -> None:
    payload = _skeleton_with_vision(extra={"weight": 3})
    assert decision_schema_errors(payload)  # additionalProperties: false
    with pytest.raises(ValidationError):
        parse_decision(payload)


def test_existing_checks_still_parse_backward_compat() -> None:
    # A skeleton using only the pre-B3 cmd/artifact_exists variants is unchanged.
    payload = {
        "schema_version": "1",
        "tool": "propose_skeleton",
        "args": {
            "phases": [
                {"id": "P1", "title": "b", "exit_criterion": [{"cmd": "pytest -q"}], "epoch_budget": 2},
                {"id": "P2", "title": "h", "exit_criterion": [{"artifact_exists": "notes.md"}], "epoch_budget": 1},
            ]
        },
    }
    assert not decision_schema_errors(payload)
    parse_decision(payload)


# --- evaluate_checks integration: the gate through the config seam --------------


def test_vision_review_passes_when_verdict_pass_true(tmp_path: Path) -> None:
    run = create_run_dir(tmp_path, "r")
    _put_screenshot(run.root)
    results = evaluate_checks(
        [_vision_check()],
        repo=None,
        ref=None,
        run_dir=run,
        scratch_name="eval",
        vision_reviewer=_reviewer(tmp_path, _PASS),
    )
    assert [ok for _, ok in results] == [True]


def test_vision_review_fails_and_surfaces_reasons(tmp_path: Path) -> None:
    run = create_run_dir(tmp_path, "r")
    _put_screenshot(run.root)
    results = evaluate_checks(
        [_vision_check()],
        repo=None,
        ref=None,
        run_dir=run,
        scratch_name="eval",
        vision_reviewer=_reviewer(tmp_path, _FAIL),
    )
    (label, ok) = results[0]
    assert ok is False
    assert "button misaligned" in label and "low contrast" in label


def test_vision_review_fails_when_screenshot_absent(tmp_path: Path) -> None:
    # No screenshot produced: deterministic fail, the script is never invoked.
    run = create_run_dir(tmp_path, "r")
    results = evaluate_checks(
        [_vision_check()],
        repo=None,
        ref=None,
        run_dir=run,
        scratch_name="eval",
        vision_reviewer=_reviewer(tmp_path, _PASS),
    )
    (label, ok) = results[0]
    assert ok is False and "missing" in label.lower()


def test_vision_review_fails_when_script_exits_nonzero(tmp_path: Path) -> None:
    run = create_run_dir(tmp_path, "r")
    _put_screenshot(run.root)
    results = evaluate_checks(
        [_vision_check()],
        repo=None,
        ref=None,
        run_dir=run,
        scratch_name="eval",
        vision_reviewer=_reviewer(tmp_path, _NONZERO),
    )
    assert results[0][1] is False


def test_vision_review_fails_when_verdict_malformed(tmp_path: Path) -> None:
    run = create_run_dir(tmp_path, "r")
    _put_screenshot(run.root)
    results = evaluate_checks(
        [_vision_check()],
        repo=None,
        ref=None,
        run_dir=run,
        scratch_name="eval",
        vision_reviewer=_reviewer(tmp_path, _BAD_JSON),
    )
    assert results[0][1] is False


def test_vision_review_fails_with_no_reviewer_configured(tmp_path: Path) -> None:
    # A vision_review check but no reviewer wired: deterministic fail, never crash.
    run = create_run_dir(tmp_path, "r")
    _put_screenshot(run.root)
    results = evaluate_checks(
        [_vision_check()],
        repo=None,
        ref=None,
        run_dir=run,
        scratch_name="eval",
        vision_reviewer=None,
    )
    assert results[0][1] is False


def test_prior_cmd_produces_screenshot_then_vision_reviews_it(tmp_path: Path) -> None:
    # The designed shape: a cmd check renders the screenshot into the eval cwd,
    # the following vision_review check judges it, both in order, one cwd.
    run = create_run_dir(tmp_path, "r")
    from grindstone.contracts.models import CmdCheck

    results = evaluate_checks(
        [
            CmdCheck(cmd="mkdir -p ui && printf x > ui/screen.png"),
            _vision_check(),
        ],
        repo=None,
        ref=None,
        run_dir=run,
        scratch_name="eval",
        vision_reviewer=_reviewer(tmp_path, _PASS),
    )
    assert [ok for _, ok in results] == [True, True]


def test_vision_review_fails_when_script_missing(tmp_path: Path) -> None:
    # FIX 2: a non-existent script path makes Popen raise OSError; the reviewer
    # must convert it to a deterministic check FAIL (never crash the run).
    run = create_run_dir(tmp_path, "r")
    _put_screenshot(run.root)
    reviewer = ScriptVisionReviewer(script=tmp_path / "does_not_exist.sh", timeout_s=30)
    results = evaluate_checks(
        [_vision_check()], repo=None, ref=None, run_dir=run,
        scratch_name="eval", vision_reviewer=reviewer,
    )
    assert results[0][1] is False


def test_vision_review_fails_when_script_not_executable(tmp_path: Path) -> None:
    # FIX 2: a present but non-executable script -> PermissionError in Popen ->
    # VisionReviewError -> deterministic FAIL.
    run = create_run_dir(tmp_path, "r")
    _put_screenshot(run.root)
    script = tmp_path / "noexec.sh"
    script.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    script.chmod(0o644)  # readable but NOT executable
    reviewer = ScriptVisionReviewer(script=script, timeout_s=30)
    results = evaluate_checks(
        [_vision_check()], repo=None, ref=None, run_dir=run,
        scratch_name="eval", vision_reviewer=reviewer,
    )
    assert results[0][1] is False


def test_script_vision_reviewer_maps_oserror_to_visionerror(tmp_path: Path) -> None:
    # FIX 2 raw seam: a launch OSError surfaces as VisionReviewError, not a bare
    # OSError that would escape evaluate_checks' VisionReviewError-only catch.
    run = create_run_dir(tmp_path, "r")
    _put_screenshot(run.root)
    reviewer = ScriptVisionReviewer(script=tmp_path / "nope.sh", timeout_s=30)
    with pytest.raises(VisionReviewError):
        reviewer.review(
            worktree=run.root, screenshot_rel="ui/screen.png",
            criteria="polished", out_dir=run.root / "vision" / "x",
        )


# --- FIX 3: screenshot path-traversal is rejected at the contract boundary ------


@pytest.mark.parametrize(
    "bad", ["../escape.png", "/etc/x.png", "a/../../b.png", "ui/../../etc/p.png", "../../secret"]
)
def test_vision_screenshot_rejects_traversal_both_layers(bad: str) -> None:
    # A planner-supplied screenshot that escapes the eval worktree (a `..` segment
    # or an absolute path) is rejected by BOTH validation layers.
    payload = _skeleton_with_vision(extra={"screenshot": bad})
    assert decision_schema_errors(payload)
    with pytest.raises(ValidationError):
        parse_decision(payload)


@pytest.mark.parametrize("good", ["ui/screen.png", "a/b/c.png", "screen.png", "out/img.PNG", "v1.2/shot.png"])
def test_vision_screenshot_accepts_normal_relative_path(good: str) -> None:
    # A normal nested relative path (dots inside a filename are fine) still parses.
    assert VisionReviewSpec(screenshot=good, criteria="x").screenshot == good


def test_vision_review_fails_when_screenshot_symlink_escapes(tmp_path: Path) -> None:
    # FIX 3 runtime guard: a screenshot that passes the lexical pattern but
    # RESOLVES (via symlink) outside the eval worktree is a deterministic FAIL,
    # the off-worktree image is never handed to codex.
    run = create_run_dir(tmp_path, "r")
    secret = tmp_path / "secret.png"
    secret.write_bytes(b"PNG-secret-bytes")
    (run.root / "ui").mkdir(parents=True, exist_ok=True)
    (run.root / "ui" / "screen.png").symlink_to(secret)
    results = evaluate_checks(
        [_vision_check()], repo=None, ref=None, run_dir=run,
        scratch_name="eval", vision_reviewer=_reviewer(tmp_path, _PASS),
    )
    assert results[0][1] is False


# --- FIX 4: Python supervisor wall-clock outlasts the script's own deadline -----


def test_vision_supervisor_timeout_exceeds_script_deadline() -> None:
    reviewer = ScriptVisionReviewer(script=Path("/x/vision.sh"), timeout_s=30)
    # The script receives int(timeout_s) as --timeout; the Python kill must fire
    # strictly LATER so the script's graceful TERM is never pre-empted by SIGKILL.
    assert reviewer.supervise_timeout_s > reviewer.timeout_s
    assert int(reviewer.timeout_s) <= reviewer.supervise_timeout_s


def test_script_vision_reviewer_raises_on_nonzero(tmp_path: Path) -> None:
    # The reviewer surfaces a script failure as VisionReviewError (evaluate_checks
    # catches it into a deterministic fail; here we assert the raw seam).
    run = create_run_dir(tmp_path, "r")
    _put_screenshot(run.root)
    reviewer = _reviewer(tmp_path, _NONZERO)
    with pytest.raises(VisionReviewError):
        reviewer.review(
            worktree=run.root,
            screenshot_rel="ui/screen.png",
            criteria="polished",
            out_dir=run.root / "vision" / "x",
        )
