# Paper Two D0 Router Feasibility Battery

Date: 2026-07-27. Diagnostic, read-only, and confined to the locked calibration partition.

## Question

Measure the value of depth routing before training D0: first the exact oracle ceiling over forced depths 1 through 6, then whether that choice can be predicted from information available before loop 1 or accumulated after each loop.

## R0: oracle ceiling, CPU-only

Use the landed private floor rows without rerunning inference. Report every fixed-depth curve, the first-correct-depth distribution, transient recovery and harm relative to loop 1, the exact any-depth oracle, and the compute-constrained oracle frontier. Unmatched positions select loop 1 in the compute-minimizing oracle and remain classified as unrecovered within the tested budget.

Teacher-derived KL, rank, entropy, teacher probability, and rejection-run signals are evaluated only as a diagnostic upper bound. They are not deployable router features. Token-change behavior after each loop is reported as the cheapest sequential observable. The untouched evaluation partition remains untouched.

## R1: deployable probes, conditional on R0

If R0 materially exceeds the best fixed policy, collect a resumable read-only feature cache on the same calibration rows. A pre-loop probe receives prelude-state features only. Sequential probes receive current loop confidence, control-token margin, recurrent state, state update, and output change available by that loop. Teacher logits, teacher identity, and future-loop features are prohibited.

R0 authorized R1: primary-target oracle agreement was 20.79 percent versus 14.47 percent at the best fixed depth, an absolute uplift of 6.32 points, while the first-correct oracle used 1.28 loops on average. R1 uses a fixed seeded 128-dimensional orthogonal projection of the Prelude and recurrent states. The pre-loop probe compares projected Prelude plus structural features against a structural-only baseline. At each loop, the sequential probe compares projected-state plus scalar features against scalar-only features. Labels use teacher outcomes, but teacher features are excluded from every input. The feature cache is private on Drive and resumable by source row.

Source rows, not token positions, define deterministic train, validation, and test groups. Report AUROC and calibration on discordant decisions plus compute-versus-agreement frontiers against random allocation, best fixed depth, and the oracle.

Diagnostic interpretation bands fixed before R1: viable deployable signal requires held-out AUROC at least 0.60 and at least one percentage point over random extra-loop allocation at two of the 25, 50, and 75 percent budgets, with positive source-row bootstrap lower bounds. Strong requires AUROC at least 0.70 and two points at two budgets. A teacher-feature-only result is leakage-only, not a usable router.

## Boundaries

No model weights change. No D0 evaluation rows are read. No D0 training launches until the target-policy implementation is reconciled with the landed floor receipt and this battery is reviewed.
