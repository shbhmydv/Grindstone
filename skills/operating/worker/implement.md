<implement_plan>
Work this plan in order. You implement everything yourself; subagents are for
the review step only. A saturated context silently drops shared-contract
details, the discipline below is what keeps a large solo build coherent.
  1. CONTRACT FIRST. Identify the shared pieces every other file depends on,
     constants, interface signatures, exception types, schemas. Implement them
     first, completely, and verify they import/load cleanly before moving on.
  2. WORK IN DEPENDENCY ORDER, ANCHORED TO THE CONTRACT. Before each unit ask:
     "must this agree with the internals of another unit?" Wherever two units
     must agree on something the contract files do not fully fix, pin that
     convention explicitly and apply it identically in both places yourself.
     Run the relevant done_when checks as each unit lands.
  3. VERBATIM SPEC. Before each unit, re-read its authoritative spec in the
     task goal and inputs end to end. Never work from a paraphrase or from
     memory, paraphrase silently drops requirements. If your context was
     compacted, recover by re-reading the spec and the files on disk.
  4. BAKE BEFORE HANDOFF, mandatory, all of (a)-(c) BEFORE handoff.json:
     (a) run EVERY done_when check yourself and fix every failure you see
         (exception: `python3 check_handoff.py` validates handoff.json itself,
         it cannot pass yet; you satisfy it in step 5);
     (b) re-read the full task goal once more and audit the seams, implement
         anything no earlier step clearly covered;
     (c) get ONE fresh-context review: spawn the registered `reviewer`
         subagent with the goal, the done_when checks and a summary of what
         you changed; its findings must be written to `review.md` in
         this directory (non-empty, `test -s review.md` is one of your
         checks). ACT on what it finds.
  5. Write handoff.json LAST, as its own final step, only after the bake, then
     run `python3 check_handoff.py` and fix violations until it exits 0.
If after honest effort the checks cannot pass, write a truthful FAILED or
PARTIAL handoff with `not_done` and `downstream_needs` filled in, the planner
re-plans from that. Never claim DONE on failing checks: every check is re-run
by the orchestrator and a false DONE is always caught.

THE TASK IS NOT DONE UNTIL handoff.json EXISTS ON DISK AND check_handoff.py
EXITS 0. A prose completion summary is NOT a handoff: narrating that you
finished, however thorough, accomplishes nothing the orchestrator can read. It
parses only the handoff.json file in this directory. Stopping after the prose,
without the write, is a silent failure that wastes a whole worker subprocess.

STOP-CHECK, apply it before you end this turn: "Does handoff.json exist and
does check_handoff.py exit 0?" If the answer is no, you are not done. Write the
handoff and run the check NOW, then re-ask. Do not end your turn until both are
true (or you have written a truthful FAILED/PARTIAL handoff that itself exits 0).
</implement_plan>
