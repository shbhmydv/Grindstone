"""Infra-failure classification: environmental fault vs genuine assertion fail.

The gate-rebalance brief (G3) splits a failed deterministic check into two
classes, two automatic responses:

  - a *semantic* fail (a real assertion failed: the code is wrong) routes through
    the failed-epoch disposition; and
  - an *infra* fail (the command could not RUN for an environmental reason: a
    missing launcher/binary, or a broken package install) is, at the PHASE GATE
    (``run_loop._maybe_repair_infra``), routed to an automatic senior infra-repair
    instead of a worker charge. At the TASK level (``task_loop._run_one_check``)
    the same verdict only ANNOTATES the failure label (``[infra: ...]``): the task
    still fails and the worker is still charged; auto-repair happens only at the
    gate. The "no worker charge" guarantee is therefore scoped to the gate caller.

This module is the SINGLE source of truth for that distinction, a pure function
used by BOTH the task-loop done_when re-run and the gate evaluator, so the two
can never drift.

The classifier is deliberately CONSERVATIVE, and intentionally NARROW. The danger
is asymmetric: an infra fault mis-read as a code bug merely charges the worker
(safe, recoverable), but a CODE BUG mis-read as infra auto-hands it to the senior
to "repair the environment", masking the real defect (the dogfood scapegoat
failure). So we only flip the verdict on UNAMBIGUOUS launch/environment signatures:
exit 127, ``command not found``, a missing/bad interpreter, ``ENOENT`` for a
binary, and package-MANAGER install failures (npm/pip/cargo install errors, no
matching distribution, etc.). We deliberately do NOT match bare ``ImportError`` /
``ModuleNotFoundError`` / ``cannot find module``: those are the most common
signatures of a GENUINE code bug (a worker renames or deletes a module and a test
fails with ``ModuleNotFoundError`` dumped into a pytest traceback on stdout, which
this classifier scans). A plain ``exit 1`` carrying ordinary test output is NOT
infra; only the narrow signatures below qualify.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: ``exit 127`` is the shell's canonical "command not found": the named binary
#: does not exist on PATH. Unambiguously environmental (a missing gate tool, the
#: ripgrep-style host-tool need), never a real assertion failure.
_NOT_FOUND_EXIT = 127

#: Narrow, unambiguous environmental signatures. Each pattern is anchored to a
#: phrasing a runtime emits when it CANNOT START the work, not when the work ran
#: and an assertion failed. Matched case-insensitively against combined output.
_INFRA_PATTERNS: tuple[tuple[str, str], ...] = (
    # Shell / OS: the launcher or interpreter itself is absent (cannot START).
    (r"command not found", "command not found"),
    (r":\s*not found", "command not found"),
    # A missing/bad interpreter: `/usr/bin/env: 'node': No such file or directory`
    # (env reports the missing interpreter by name) and the exec-error phrasings.
    (r"/usr/bin/env:.*no such file or directory", "missing interpreter"),
    (r"bad interpreter", "bad interpreter"),
    (r"\bENOENT\b", "missing path/binary (ENOENT)"),
    # Package-MANAGER install failures (npm / pip / cargo install errors). NOTE: we
    # deliberately do NOT match bare ImportError / ModuleNotFoundError / "cannot
    # find module": those are the dominant signatures of a genuine code bug (a
    # renamed/deleted module surfacing in a test traceback) and masking that as
    # infra is the dangerous misclassification this rule guards against.
    (r"npm ERR!", "npm install/run error"),
    (r"npm install.*(?:failed|error)", "npm install error"),
    (r"pip install.*(?:failed|error)", "pip install error"),
    (r"No matching distribution found", "pip could not install a dependency"),
    (r"Could not find a version that satisfies", "pip dependency resolution failed"),
    (r"error: could not compile|error: no matching package", "cargo build/install error"),
)

_COMPILED: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(pat, re.IGNORECASE), reason) for pat, reason in _INFRA_PATTERNS
)


@dataclass(frozen=True)
class InfraClassification:
    """The verdict on one failed check: environmental or genuine.

    ``is_infra`` True means the command could not run for an environmental reason
    (route to senior infra-repair); ``reason`` is a short human-readable label
    naming the signature that matched (or, when not infra, why it does not count).
    """

    is_infra: bool
    reason: str


def classify_check_failure(
    *, returncode: int, stdout: str, stderr: str
) -> InfraClassification:
    """Classify a FAILED check as infra (environmental) or semantic (genuine).

    The rule, in one sentence: a failed check is INFRA iff its exit code is 127 OR
    its combined output matches one of the narrow LAUNCH/ENVIRONMENT signatures
    (command not found, a missing/bad interpreter, ``ENOENT`` for a binary, or a
    package-MANAGER install failure); anything else (a plain ``exit 1`` with
    ordinary test output, OR a bare ``ImportError`` / ``ModuleNotFoundError`` in a
    traceback) is semantic. Conservative by construction: only the listed
    signatures flip the verdict, so a real assertion failure (including the very
    common module-import failure of a code bug) is never mistaken for infra.

    The caller is responsible for only invoking this on an ALREADY-failed check
    (a passing check is never environmental); a 0 return code here still reports
    not-infra so the function is total.
    """

    if returncode == _NOT_FOUND_EXIT:
        return InfraClassification(True, f"exit {_NOT_FOUND_EXIT} (command not found)")
    combined = "\n".join(p for p in (stdout, stderr) if p)
    for pattern, reason in _COMPILED:
        if pattern.search(combined):
            return InfraClassification(True, reason)
    return InfraClassification(False, "no environmental signature (treated as semantic)")
