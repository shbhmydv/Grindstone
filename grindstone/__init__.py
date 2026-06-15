"""Grindstone — epoch-based deep-work orchestrator (v2 rebuild).

Built up the rung ladder (ARCHITECTURE.md): S0 the contract layer (typed boundary
models, runtime JSON-Schema gate, semantic validators, journal vocabulary,
run-dir layout); S1 the single-task state machine + transports; S2 one epoch —
fan-out, per-attempt worktrees with ownership scope checks, fast-forward
integration, the done-predicate, and kill-mid-epoch resume; S3 the multi-epoch
run loop — the stateless planner (``codex exec`` / mock) behind a transport
interface, pure stable-head/volatile-tail input construction, the decision
gate (schema → typed → semantic), 3-way failure classification with injected
backoff, epoch chaining, ``complete_run`` evidence re-runs, and
kill-mid-planner-call resume (in-flight calls re-issued, not burned). The
run/phase frame is now owned by the loop, not synthesized in the epoch.
"""
