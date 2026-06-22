# Stage 5 Programmatic Depth Assessment - stage5_programmatic_depth_assessment_20260622_140220

- Status: `programmatic_depth_no_lift`
- Passed: `False`
- Source summary: `outputs/stage5/stage5_arc_agi_next_action_20260622_175531_plan_programmatic_depth_repair/summary.json`
- Checkpoint: `outputs/stage5/stage5_arc_agi_next_action_20260622_175531_plan_programmatic_depth_repair/phase1/phase1_step_50.pt`
- Loss delta: `0.009893999999999986`
- Loop error delta: `-0.021142999999999912`
- Target loop mean: `2.1708333333333334`
- Routing repair diagnostic: `routing_repair_answer_prior_drift.md`
- ARC-Easy base vs routing-repair: `87/128` -> `84/128`
- Main failure mode: answer-prior drift on base-confident direct questions.
  - Base-confident direct proxy delta: `-12`
  - Ambiguous proxy delta: `+8`
  - Mean margin delta: `-1.5559766787882836`
  - Max candidate-base prediction count shift: `23`
  - Candidate prediction shift: `A +23`, `B -16`, `C -20`, `D +13`
- Next step: Do not extend this constructed pass; return to ARC-routing failure analysis. The next repair should preserve base-confident direct predictions more explicitly instead of adding more generic depth supervision.
