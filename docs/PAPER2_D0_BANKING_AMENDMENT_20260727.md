# Paper Two D0 Banking Amendment

**Date:** 2026-07-27  
**Registered interpretation:** `not_recoverable_at_pilot_scale`

D0 is complete and final for its registered recipe: binary teacher-disagreement targets, 4,000 steps, and one seed. Its negative does not close adaptive depth as a mechanism. The label was not aligned with deployment utility. The pretraining floor showed that only about one in six 7B-rejected positions was recoverable at depth 2, while every rejection was labeled continue and accepted-position harm was absent from the target.

The post-D0 mixed natural-plus-rehearsal run also erased the small descriptive T1 preservation cost. T1-family retention reached `1005/1024`, exactly the full-block non-halting reference, with perfect continue/stop and exact depth selection. This is not a retroactive T1 pass; it shows that the earlier 3.3-to-3.7-point cost was transient under continued mixed training.

The accepted-position guardrail protected loop-1 weights (`99.31%`, pass) but not the deployed adaptive policy. Relative to the plain drafter, the adaptive policy lost a net 4,928 accepted positions. D1 must therefore guard deployed-policy agreement on baseline-accepted positions, not only loop-1 behavior.

The teacher-shift reading remains the registered same-depth branch: the 7B and 14B demand curves both had median and peak depth 2 on their own rejection populations, before and after training. A drafter-side ceiling remains a viable alternative explanation. The supervision signal also contains measured teacher disagreement: `16.567%` of 7B rejections have a loop-1 drafter token endorsed by the 14B teacher. D1 headroom must not be priced from raw forced-depth-4 recovery alone.

No disagreement-target extension, lambda change, or threshold sweep is authorized. The next authorized action is the read-only causal allocation audit in `PAPER2_D1_CAUSAL_ALLOCATION_AUDIT_SPEC_20260727.md`. No D1 training is authorized until a new preregistration locks.
