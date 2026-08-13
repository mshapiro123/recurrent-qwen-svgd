# P3.4 Sampled-Depth Estimator Amendment Request

Date: 2026-08-13. Audience: strategy and research review. Status: pre-optimizer mismatch measured; training held; explicit amendment required.

## 0. Executive result

The P3.4 campaign did not begin scientific training. Its pre-optimizer assertion exposed a mismatch between the calibration estimator and the registered training estimator. The locked scalar weights were solved at full depth four. The campaign samples depths one through four with probabilities `0.1, 0.2, 0.3, 0.4`. Under that actual depth mixture, seed 0 main missed the aim and gate floors, and seed 0 slot missed aim, gate, and slot. Seed 1 main happened to pass, but it was calibrated on a different estimator and therefore cannot serve as the matched exception.

The exact repair is already measured. Re-solving the static weights on the same fixed 256-row calibration population, with the registered sampled-depth mixture and all clipping rules unchanged, recovers the intended target shares to numerical precision for all three arms. No CONFIRM or EVAL-E row was touched. The read-only diagnostic mode took zero optimizer steps.

Requested ruling: ratify replacement of all three scalar vectors with the sampled-depth solves below. This changes only the calibration estimator and its derived weights. It does not change data, depth probabilities, targets, losses, optimizer, schedule, controller, thresholds, seeds, or arm definitions.

## 1. Why the launch stopped

B6 binds per-loss inequalities on the trailing training-stream estimator. The executed lock also binds a depth lottery proportional to depth. The original calibration evaluator, however, called the four-loop loss graph without varying `steps`, so it solved the weights at depth four only. The runner correctly performed a pre-optimizer check over the registered depth lottery and rejected seed 0.

This is a lock-construction error, not a model failure. The first seed-0 attempt failed on a keyword-only controller call and the second reached the intended share tripwire. Both stopped before optimizer construction. Seed 1 produced a durable step-zero checkpoint before the mismatch was understood across arms, but was stopped before any optimizer update. It must restart from the original i1 endpoint after amendment; its stale step-zero resume will be parked, not reused.

## 2. Experimental design of the repair measurement

- Population: the unchanged, hash-pinned 256-row B6 batch, balanced 128 code and 128 general with the registered 1:3 positive-negative ratio.
- Depths: forced reads at `k = 1, 2, 3, 4`, aggregated with masses `0.1, 0.2, 0.3, 0.4`.
- Trainable surface, losses, combined group clips, and target shares: unchanged from the executed lock.
- Solver: the existing exact iterative post-clip solver. Each depth bundle is counted `k` times, implementing the integer-equivalent `k / 10` mixture without approximation.
- Arms: main seed 0, main seed 1, and slot seed 0.
- Mode: read-only preflight. No optimizer construction in the diagnostic jobs, zero optimizer steps, no sealed evaluation contact.

## 3. Results

### 3.1 Old locked weights under the actual depth mixture

| Arm | KL | Aim | CE | Gate | Slot | Preserve | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| Main seed 0 | 61.13% | 10.97% | 11.61% | 2.77% | - | 13.52% | Aim and gate below floor |
| Main seed 1 | 45.99% | 20.23% | 11.09% | 3.61% | - | 19.08% | Passes floors, wrong calibration estimator |
| Slot seed 0 | 54.79% | 9.84% | 10.40% | 2.48% | 8.44% | 14.04% | Aim, gate, and slot below floor |

The asymmetry is endpoint geometry, not evidence that the old estimator is acceptable for seed 1. Keeping seed 1 while changing seed 0 would make the replication use different calibration definitions.

Figure: `docs/figures/p34_sampled_depth_preflight_20260813.svg` (SHA-256 `ae33a17bc33be7155d898f8f1f8c6b835ad3349ade77e8f8c809b0d1893625eb`; PNG SHA-256 `d64fda6e4c91900725d18ae2cbe118ae8ec6f0afa805e7eacdfd049f04d43926`). The blue bars are the already registered targets; the red bars are the shares the old weights produce under the actual training-depth mixture.

### 3.2 Proposed KL-normalized scalar weights

| Arm | KL | Aim | CE | Gate | Slot | Preserve |
|---|---:|---:|---:|---:|---:|---:|
| Main seed 0 | 1 | 0.12926775 | 0.05205299 | 0.000432747 | - | 74.37863 |
| Main seed 1 | 1 | 0.02055845 | 0.03924744 | 0.0000672940 | - | 7.15361 |
| Slot seed 0 | 1 | 0.12926775 | 0.05205299 | 0.000432747 | 0.03204862 | 86.18476 |

The solved main shares are exactly KL 41.667%, aim 17.857%, CE 11.905%, gate 3.571%, preserve 25%. The solved slot shares are KL 35.959%, aim 15.411%, CE 10.274%, gate 3.082%, slot 10.274%, preserve 25%. Maximum error across the three solves is `8.33e-17`.

## 4. Interpretation

### Supported

- The pre-optimizer tripwire caught a genuine objective-allocation error before scientific training.
- The mismatch is fully explained by the omitted depth mixture.
- A deterministic, no-search correction exists and exactly restores the already registered targets.
- The repair applies coherently to all arms and preserves the causal comparison.

### Not supported

- No P3.4 task-improvement or gap-closed result exists yet.
- Seed 1's old-weight floor pass does not validate the old calibration estimator.
- The corrected step-zero solve does not guarantee shares remain in bounds during training; the ratified 2/4 trailing-window contract remains necessary.

## 5. Requested amendment and relaunch sequence

1. Replace all three weight vectors in the machine lock with the sampled-depth-mixture solves.
2. Add this consolidated receipt and its three source receipts to the lock by SHA.
3. Retain the original full-depth receipts as superseded calibration archaeology, not delete them.
4. Park the stale seed-1 step-zero resume under an explicit pre-amendment name.
5. Rerun all three pre-optimizer assertions. Each must pass before optimizer construction.
6. Launch main seeds 0 and 1 concurrently, then slot seed 0 as soon as Colab's two-session assignment limit frees a slot.
7. Keep every other executed-lock constant unchanged.

## 6. Receipt map

- Consolidated receipt: `outputs/stage5/stage5_paper2_phase3_p34_sampled_depth_preflight_20260813/summary.json`, SHA-256 `e6d681223b10fbad89682eb837fa0fbc2c61aced6bda7ac98731fedf01d5468f`.
- Main seed 0 source: `outputs/stage5/stage5_paper2_phase3_p34_sampled_depth_preflight_20260813/receipts/main_seed_0.json`, SHA-256 `39d7c7e7cd7676508bc5df415a60286d9eb67ac27678a67b3d17a9b67e35e762`.
- Main seed 1 source: `outputs/stage5/stage5_paper2_phase3_p34_sampled_depth_preflight_20260813/receipts/main_seed_1.json`, SHA-256 `c4d1f9d136abc2a64214d3502c3d985e5c3349f56851d8c37140e9454ab897d2`.
- Slot seed 0 source: `outputs/stage5/stage5_paper2_phase3_p34_sampled_depth_preflight_20260813/receipts/slot_seed_0.json`, SHA-256 `70562b1588e2cbee8130fb1e0a4897462a268c8519d7ce05bc0e03a3701d2df4`.
- Private durable release: `p34-campaign-20260813` in `mshapiro123/recurrent-qwen-svgd-runtime-private`.
- Measurement commits: `33d24501` and `2ef34843`.

## 7. Plain-language summary

The safety check caught that we had tuned the balance of the training signals while always running four internal steps, even though the real experiment randomly runs one, two, three, or four. That changed how much learning pressure each objective would receive. The correction is not a new hypothesis or a sweep: recompute the same balance over the actual mix of depths. We have done that for all three runs without training anything. Once strategy ratifies the corrected numbers, the experiment can restart cleanly from its original checkpoints.
