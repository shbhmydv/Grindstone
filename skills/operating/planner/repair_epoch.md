SCENARIO: repair_epoch. The prior epoch FAILED the gate and is awaiting
disposition. The ONLY legal decision now is handle_failed_epoch (the `<failed_epoch>`
block in the state below carries the failed tasks, the failing checks WITH their
captured output, and the worker handoffs that claimed pass). revise_phases is NOT
for this, the phase STRUCTURE is unchanged.

- When an epoch FAILS (a task exhausted its retry ladder, and/or the phase gate
  kept failing), the next call is CONSTRAINED to handle_failed_epoch: choose
  exactly one action for THAT epoch, retry (with a `hint`, optionally
  `escalate_tier:true` to start on senior), escalate_senior (with a `diagnosis`),
  or halt (with a `reason`, stops the run for a human). The input carries the
  failed checks WITH their captured command output and the worker handoffs that
  claimed pass, read them before deciding.
- GATE SKEPTICISM: if the workers repeatedly report an HONEST pass (their
  done_when passed in their scratch) while the PHASE GATE keeps failing the SAME
  way, SUSPECT the gate or the environment, not the code: a missing dependency, a
  wrong verification context, a check that cannot run where the gate runs it. The
  captured check output is your evidence. In that situation prefer halt (with a
  reason naming the suspected env/gate problem) over ordering yet another
  identical code repair, repeated identical repairs against a structurally
  unpassable gate is the failure mode this decision exists to stop.
