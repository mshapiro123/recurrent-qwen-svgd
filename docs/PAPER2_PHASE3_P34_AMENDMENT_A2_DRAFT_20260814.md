# P3.4 Amendment a2 Draft - Dynamic Objective-Share Recovery

Date: 2026-08-14. Status: **draft for Mark and strategy review; training disabled**.

Governing strategy response: `STRATEGY_P34_RESULT_RESPONSE_20260814.md`, Drive `16lVxjVNEPUurmZ-TLN1hIh94QFXf13Go`.

CPU autopsy receipt: `outputs/stage5/stage5_paper2_phase3_p34_a2_autopsy_20260814/summary.json`, SHA-256 `25836439b34bebd83fd63286a1876cf974305f324479a0687c75e15e5037b1d4`.

This amendment changes only the objective-share recovery mechanism, rung-specific share targets, continuation checkpoints, and confirmation-planning inputs authorized by the strategy response. It does not change the model, trainable surface, data, losses, sampled-depth estimator, task inference graph, task panel, ordinary thresholds, registered 4,000-step endpoint, or any sealed partition.

## 1. Reading carried forward

The P3.4 campaign is banked as `POSITIVE_SIGNAL_WITH_OBJECTIVE_CONTROLLER_STOP`, ledger status `exploratory_positive_interrupted`. It is neither null nor confirmation. CONFIRM and EVAL-E remain unspent.

The CPU task audit adds an attribution correction. Relative to the actual i1 training starts, not the frozen base, the main endpoints gained five pooled DEV rows in each seed. Their target-half gains were two and three rows. Several base-relative battery patterns were inherited before P3.4: the same two Tier-1 additions were already wrong at both i1 starts, and P3.4 itself improved GSM8K by four rows in both main seeds.

## 2. Continuation checkpoints

The latest evaluation checkpoint whose trailing share window passed every registered share bound is the only eligible continuation point.

| Condition | Continue from | Checkpoint SHA-256 | Disposition |
|---|---:|---|---|
| main seed 0 | step 400 | `56dfa30d19166dfd3a788e2e6f68e0613f366e55601b5d690b087e1a3edb9230` | eligible |
| main seed 1 | step 1,000 | `2ff122cdc1d3c3208c9eb367345f360a31676f0f821c311ed98f6cc690c8e66f` | eligible |
| slot seed 0 | none | none | shelved; restart under a future lock if reopened |

Both main runs retain their original registered endpoint at step 4,000 and resume the original evaluation cadence. Optimizer and scheduler state come from the pinned checkpoint. No endpoint or best-look selection occurs here; checkpoint choice is determined solely by the pre-existing share contract.

## 3. Two-timescale controller

### 3.1 Objective-share controller

At each non-overlapping 100-step share window, use the existing post-clip gradient-share estimator. For every active loss `i`, update its scalar weight in log space:

```text
delta_i = clip(0.5 * log(target_share_i / observed_share_i), -0.5, 0.5)
weight_i <- weight_i * exp(delta_i)
```

Normalize all weights after the update so `weight_KL = 1`. The update is applied only after the window's observed shares, classification, and same-window counterfactual shares have been written. The controller cannot change model state, optimizer state, the active data sample, or the annealing rung.

The existing share rule remains the health contract: the first failed window is observed, two consecutive failed windows trigger the registered Tier-W rung demotion and strategy flag, and four consecutive failed windows stop. The log-weight update occurs at every completed share window, including a passing window, because it regulates toward a target allocation rather than merely reacting after a floor fails.

### 3.2 Capability and causal controller

The annealing rung remains controlled only by its registered task and causal evidence. Tier-W can still demote one rung. Rung changes no longer alter objective weights or serve as the objective-share repair. At most one rung transition may occur at a window, as before.

## 4. Rung-specific targets

The binding preservation targets are the median observed preservation shares among the two main arms at each rung. The shelved slot arm is excluded from these main-arm calibration values because it has an additional objective and different allocation geometry.

| Rung | KL | Aim | CE | Gate | Preserve |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.545464 | 0.233770 | 0.155847 | 0.046754 | 0.018165 |
| 1 | 0.495229 | 0.212241 | 0.141494 | 0.042448 | 0.108587 |

After fixing the rung's preservation target, the remaining mass is distributed in the registered primary-floor ratio `0.35:0.15:0.10:0.03`. These are controller targets, not replacements for the standing hard floors and preservation ceiling.

The all-condition descriptive preservation medians were 0.018591 at rung 0 and 0.116605 at rung 1. They are not binding because the slot arm is shelved.

## 5. Autopsy result and exact preflight

The cached campaign receipts persist post-clip shares but not per-loss gradient bundles. Consequently:

- the CPU autopsy can invert the landed shares into local unit masses while holding the observed group-clip scale fixed;
- it cannot recompute how a different weight vector would change group clipping;
- it cannot perform the optional PCGrad comparison on exact cached gradients.

Under this explicitly local replay, the dynamic controller's maximum miss streak is one window for main seed 0 and two windows for main seed 1, so neither reaches the registered four-window stop. A joint, one-vector-per-rung hard-floor linear program across both main seeds is algebraically feasible only by driving the preservation weight to approximately `1e-8`. That degenerate solution is rejected as a practical controller. The nondegenerate geometric target-fit vectors pass 10/12 rung-0 windows and 13/17 rung-1 windows, so they also fail to provide one stable static policy. The evidence therefore favors dynamic regulation over a single static vector.

Before constructing an optimizer for either resumed run, the launcher must execute an exact gradient-bundle preflight at its pinned continuation checkpoint under the unchanged sampled-depth estimator. It must:

1. recompute every active per-loss gradient bundle with the existing group-clip implementation;
2. initialize weights by applying the bound controller to the exact observed shares;
3. recompute post-clip shares under those proposed weights;
4. assert all standing floors and the preservation ceiling;
5. write both the before and after bundles, shares, clip scales, weights, hashes, and maximum arithmetic residual.

Failure is a pre-optimizer stop and returns to strategy. It does not authorize changing controller constants or targets. This preflight resolves the CPU replay's fixed-clip limitation without spending a training step.

## 6. DEV telemetry repair

The cached task rows contain predictions and correctness but omit the per-position gate and realized writeback ratio. Therefore the authorized CPU audit cannot answer whether gates open differentially by battery or whether GSM8K regressions co-locate with writes.

The resumed DEV evaluator must add score-preserving telemetry fields for gate activation and realized writeback magnitude to each existing row receipt. This is instrumentation only: it cannot alter generation, scoring, controller decisions, checkpoint selection, or thresholds. It is reported by battery and separately for fixes, regressions, and unchanged rows.

The two shared Tier-1 errors are already identified:

- `base_capability_addition_00`: `13 + 7`, expected `20`, sidecar answer `10`;
- `base_capability_addition_02`: `15 + 13`, expected `28`, sidecar answer `38`.

Both were wrong at the i1 start in both lineages and stayed wrong throughout P3.4. They are inherited inference-path errors, not P3.4 training regressions.

## 7. Confirmation planning, without spending CONFIRM

The sealed CONFIRM membership contains 1,502 pooled rows and 926 target-group rows. Using the mean DEV endpoint discordance from the two main seeds and an exact one-sided paired sign-test planning model at alpha `0.05`, the minimum single-seed effects giving at least 80% power are:

| Accounting | Mean DEV discordance | Minimum CONFIRM net rows | Accuracy points | Projected gap closed |
|---|---:|---:|---:|---:|
| pooled, 1,502 rows | 0.09375 | 31 | 2.064 | 7.24% |
| target group, 926 rows | 0.15332 | 31 | 3.348 | 10.92% |

At effects of 0.6, 0.8, and 1.1 accuracy points, projected pooled power is only 0.166, 0.236, and 0.368; target-group power is 0.103, 0.134, and 0.191. The current sealed panel is therefore underpowered at the effect size priced by the mechanism.

The seed-specific row-delta correlations are 0.917 pooled and 0.930 on the target group. A planning-only normal approximation to the average two-seed effect reduces the required effect only to 1.923 pooled points or 3.146 target-group points. Naively treating seeds as independent rows is prohibited.

For confirmation eligibility after the repaired run, bind all of the following:

1. each main seed's registered step-4,000 endpoint is positive versus its frozen base;
2. each main seed's target-half endpoint delta is non-negative;
3. the mean two-seed pooled DEV endpoint is at least `22/1024` net rows (2.148 points), the integer DEV analogue of the single-seed 80%-power requirement;
4. any P3.6 joint-seed analysis must first lock a cluster-level estimator and simulate its operating characteristics under the observed cross-seed dependence.

Meeting these conditions authorizes drafting P3.6; it does not itself confirm the claim. If the repaired campaign remains in the mechanism-priced 0.6-to-1.1-point range, the program reports that CONFIRM is underpowered and does not spend the seal under an improvised estimator.

## 8. Unchanged safeguards and boundaries

All catastrophe tripwires carry forward unchanged: frozen-lineage integrity, non-finite loss, sealed-data contact, task collapse, collateral, and resume identity. The original data, optimizer family, learning rate, sampled-depth lottery, rehearsal, model path, trainable set, gate ceilings, task controller, and look schedule are unchanged.

No CONFIRM or EVAL-E contact is allowed. No slot continuation is allowed. No code-review cleanup enters the checkpoint-defining branch. No claim of a better model, confirmation, general code capability, or general knowledge improvement follows from DEV.

## 9. Ratification and execution state

This draft is not executable authorization. The machine-readable companion must retain:

```json
{
  "locked_before_resumed_training": false,
  "training_authorized": false,
  "mark_ratified": false
}
```

After Mark and strategy approve this amendment, the coding agent may mark it locked, implement the controller and added telemetry under tests, run both exact pre-optimizer preflights, and then resume the two main seeds. Any material mismatch between this draft, the governing strategy response, and the implementation stops before optimizer construction.

**Ratification line:** I approve P3.4 amendment a2 as written, including the two pinned continuation checkpoints, dynamic log-weight controller, rung-specific targets, exact pre-optimizer calibration, score-preserving DEV telemetry, and confirmation-planning criterion.
