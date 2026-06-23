<review_plan>
This is a review task: judge the targets; do not modify them.
Your CWD is your entire workspace: read your resolved inputs, write your
artifact and handoff here. Never cd above it, never read or modify the
surrounding repository, and do not run git, there is no repository here.
The specific targets under review are listed in the <review_targets> block below.
  1. Examine each target against the question in the goal.
  2. Write your findings AND an explicit verdict into the artifact named above.
  3. Ground every finding: the handoff's `citations` MUST contain at least one
     real file/line. A review handoff with no citations is rejected.
  4. BAKE BEFORE HANDOFF: before you write a DONE handoff, run EVERY done_when
     check yourself and confirm the artifact exists and carries an explicit
     verdict, then re-read the goal once more and check every target was judged.
     Only hand off DONE when all checks pass and at least one citation is
     present; every check is re-run by the orchestrator and a false DONE is
     always caught. If after honest effort the checks cannot pass, write a
     truthful FAILED or PARTIAL handoff with `not_done` filled in instead.
</review_plan>
