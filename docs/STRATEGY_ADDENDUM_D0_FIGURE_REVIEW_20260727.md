# Strategy Addendum to the D0 Launch Handoff — Figure Review
Date: 2026-07-27. Amends: STRATEGY_TO_CODING_AGENT_D0_LAUNCH_20260727.md (Drive 1t6wq1h-cemPFlk3NOT9qFn5lWumzDUYR), section 5 only. All other sections, authorizations, and the launch order stand unchanged.

Provenance: the strategy review and launch handoff were written from the feasibility handoff's tables; the figure (paper2_d0_router_handoff_20260727.svg) was reviewed afterward. It confirms the tabulated values and exposes one datum the tables omitted — the full 14B per-depth curve, whose depth-1 value is approximately 16.5 percent agreement on the 7B-rejected positions.

## 1. New interpretive datum: teacher disagreement inside the supervision target

Roughly one in six positions where the 7B corrects the drafter carries a loop-1 drafter token the 14B endorses. This is a direct measurement of teacher disagreement within the agreement target, and it turns the registered "agreement is not correctness" caveat from an abstract disclaimer into a quantified bound. File it as a descriptive receipt: the share of 7B rejections whose depth-1 drafter token matches the cached 14B greedy (exact figure from the caches, approximately 16.5 percent per the plotted curve). Manuscript use: one sentence bounding the teacher-noise share of the supervision signal.

## 2. Section 5 job revision: population definition for the 14B demand distribution

The original section 5 named the 7B-rejected set as the population and the 14B's own rejected set as optional. The figure shows this must be flipped. On the 7B-rejected set, the 14B's approximately 16.5 percent depth-1 agreement would pour depth-1 mass into the first-correct distribution and the median would measure teacher overlap, not depth demand. Revised specification:

1. Primary population for the 14B demand curve: the 14B-rejected set (positions where the drafter's loop-1 token disagrees with the cached 14B greedy), where depth-1 agreement is zero by construction — the clean mirror of the 7B analysis.
2. The 7B-rejected crossover analysis (14B first-correct distribution on 7B rejections) is retained as a descriptive teacher-overlap statistic, not the demand curve.
3. The teacher-disagreement receipt of section 1 above is computed in the same pass.
4. The floor-layer teacher-shift comparison of the launch handoff's section 4 therefore reads: 7B median first-correct depth on 7B rejections (known, 2) versus 14B median first-correct depth on 14B rejections, each teacher measured against its own rejected population. The trained-model layer inherits the same population convention.

All of this remains read-only post-processing on existing caches and floor predictions. No GPU, no change to training targets, gates, guardrails, bands, or the launch order.