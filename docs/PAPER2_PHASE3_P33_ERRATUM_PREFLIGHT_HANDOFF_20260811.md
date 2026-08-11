# Phase 3 P3.3 Erratum Preflight Handoff

Date: 2026-08-11

## 0. Executive reading

Erratum e1 is implemented. The CPU-only data preflight reran at commit
`3197999b460887b018874bed6f26e8b964ca808a`, and every new cohort assertion
passed. The 12,288-row negative audit slice now makes gate precision, false
positive rate, and collateral chi estimable without touching the training
negatives. No optimizer was constructed and no training step ran.

The P3.3 training core is implemented and tested, but launch remains blocked by
one contract ambiguity that should not be resolved in code: the ratified Tier-S
rule is defined on paired augmented-versus-base task correctness on a 1,024-row
capability panel, while no registered inference adapter specifies how the
four-horizon Phase-3 sidecar is applied during those task evaluations. Replacing
that panel silently with token-retention positions would change the estimator.

## 1. Erratum implementation

The implementation binds the four e1 corrections:

1. Migration identity is evaluated with the gate ceiling disabled. The ceiling
   is then activated at `gamma = 0.02`; any initialization binding is telemetry,
   not a failure.
2. `c = 0.15` is audit-only. Forced-open `pi_dir` uses
   `0.15 * capped_RMS_reference`, and its oracle denominator uses the identical
   magnitude. No `c` factor was added to the train-time bridge equation.
3. The negative audit slice contains 12,288 confident-agreement positions,
   selected immediately after the frozen 103,563-row training-negative cohort.
   It is disjoint from the 4,096 positives and excluded from training.
4. The run constants are fixed at 1,000 updates and 20 looks, one every 50
   updates. The inherited optimizer is AdamW, learning rate `3e-4`, betas
   `(0.9, 0.999)`, weight decay `0.01`, batch size 128, and 100 warmup updates.
   The inherited A2 configuration has no gradient clipping.

The trainable set remains bridge, per-position gate, and control state only.
The loss set remains aim cosine loss, inverse-frequency gate BCE, and
preservation KL at weight 1.0. Gate BCE uses the unclamped sigmoid probability;
otherwise the operating ceiling would erase the positive-class gradient.

## 2. CPU preflight results

The durable Drive receipt is:

`/content/drive/MyDrive/recurrent-qwen-svgd-artifacts/stage5/stage5_paper2_phase3_guardrail_p33_prep_20260810/receipts/summary.json`

Key data results:

| Item | Result |
|---|---:|
| Coverage-index positions | 727,876 |
| Training positives | 34,521 |
| Training negatives | 103,563 |
| Positive audit rows | 4,096 |
| Negative audit rows | 12,288 |
| Negative-to-positive audit ratio | 3.0 |
| Realized negative confidence cut | 0.9937900901 |
| Position-zero labels | ignored only |

Hashes:

| Artifact | SHA-256 |
|---|---|
| Coverage index | `cb50bee824abdc28638e07138d5079073b35a9fc6c528ef7a5be5a101e91e73b` |
| Staged labels | `dbc5e488b93a1da2fb90854b171a2e52d146be463166222813cf5ea3102f7461` |
| Positive audit file | `6dea4392c8c9e19d4903cd86ab70c7c922389c03e0be5f25615e2756c80f0579` |
| Negative audit file | `8aed17517c1931a0b7ce4f4ae1feb4c50affd3edf5219621c39215180305791a` |
| Fixed random projection | `a87f93e81631a308e8d6bd18b4dbec0d8306de93f84a07328842573835c7ebe1` |
| Canonical projection | `53215130a75c929def4bcc4b81ba9187d90a7b6fe2b1502391518d45c09476e3` |

All receipt assertions are true: both audit sizes, cohort disjointness,
training exclusion, 3:1 ratio, ignored position zero, zero optimizer steps, and
no CONFIRM contact.

## 3. Guardrail receipt retained

The existing 20-look calibration remains hash-stable and was reused:

- empirical discordance: `0.3263157895`;
- adjacent-look autocorrelation: `0.7387033141`;
- Tier-S panel: 1,024 rows, one-sided alpha `0.005`;
- simulated familywise null stop probability: `0.00002`;
- catastrophic drop with at least 99% detection: `0.085`;
- Tier-W null warning probability: `0.00441`.

These operating characteristics describe the registered paired task-correctness
estimator. They cannot be transferred automatically to a token-level proxy.

## 4. Remaining blocker: Tier-S deployment semantics

P3.1 supplies base and 14B correctness on ARC-Easy, MMLU, and Tier-1 rows for
calibration. It does not supply an augmented Phase-3 evaluation path. The
current sidecar code consumes four cached speculative horizons and returns a
writeback at one token position. The lock does not specify how to construct
those horizons and apply the writeback during autoregressive benchmark scoring
or generation. No existing P3.1 evaluator performs that integration.

Without a ruling, either implementation choice changes the experiment:

### Path A - token-level P3.3 tripwire (recommended)

Freeze a 1,024-position panel from the new held-out confident-agreement audit
cohort. Score paired top-1 retention for augmented versus base at every look,
recalibrate Tier-S and Tier-W on this exact estimator, and scope the result as a
P3.3 collateral tripwire only. Restore task-level capability scoring in P3.4,
where a deployment path is already required for answer-distillation claims.

This path preserves P3.3's role as the cheap aim falsifier, uses data already
sealed and disjoint, and avoids inventing inference semantics inside a safety
check. It does not support a general-capability-preservation claim.

### Path B - task-level sidecar adapter

Specify and preregister the complete inference graph for each generated token:
how four speculative horizons are built, when the sidecar writes, whether the
write is repeated during generation, and how direct-answer and code readers
consume the result. Then run the registered 1,024-row capability panel at every
look under that path.

This is the stronger guardrail but is a new architecture/evaluation decision,
not a mechanical completion. It adds substantial evaluation cost and can make
the guardrail rather than aim learning the dominant P3.3 workload.

## 5. Requested strategy ruling

Choose Path A or Path B and bind the estimator before optimizer construction.
If Path A is selected, also authorize recalibration on the fixed 1,024-position
panel. If Path B is selected, provide the four inference semantics listed above
and the exact 1,024-row panel membership rule.

## 6. Current state and next action

- Erratum implementation: complete.
- Focused tests: 13 passed locally; the staging/core subset passed again on the
  Colab CPU VM before receipt generation.
- Negative-audit receipt: complete and Drive-backed.
- Colab sessions: none active after receipt completion.
- P3.3 optimizer: not constructed.
- P3.3 training: not started.
- CONFIRM/EVAL-E: untouched.

After the Tier-S ruling, the coding lane can finalize the machine lock, add the
resumable two-seed launcher, run the five pre-run assertions, and launch P3.3.
