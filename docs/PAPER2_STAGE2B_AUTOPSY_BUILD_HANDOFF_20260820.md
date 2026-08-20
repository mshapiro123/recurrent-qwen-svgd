# Paper Two Stage 2B-A Autopsy Build Handoff

Date: 2026-08-20. Status: build and score-blind prelock complete; blocked on a narrow strategy amendment because the requested intermediate checkpoints were not retained. No model has been loaded by this program, no optimizer has been constructed, and CONFIRM and EVAL-E remain sealed.

## 1. Purpose

The registered Stage 2B-D campaign stopped at step 1,000 after both seeds failed the 20-percent task-signal kill gate. The stop is banked. This follow-up is the strategy-authorized, score-only autopsy that distinguishes three successor-relevant explanations without reopening the failed recipe:

- H-A, attractor formation or task-signal loss despite improvement on the training objective.
- H-B, excessive write magnitude at the registered amplitude.
- H-C, damage attributable to the constitutive innovation rather than the inherited residual flow or state carry.

The router-geometry proposal supplied after the stop is successor-design context. It is not tested here because Stage 2B-D left the router and loop-scoped LoRA dormant during M2. It should be reconsidered only after the autopsy identifies which interface needs replacement.

## 2. Governing records

- Strategy stop analysis: Drive `1jeWnmBZHxOgaQw_Dnf9NqOYSrxmpSfoX`, 12,680 bytes, SHA-256 `58f2420d7728df66cf8cc5ee5b1142d92840c975cd7bc37db881d613f9f9e791`.
- Registered stop handoff: Drive `1WoHNkiI6xK8r6BohgeYZyj0ouORHO0HN`, SHA-256 `bfeb74e91b59e89c798009bb3551b4a4593fb2e4073324ce7b35910ecd31d312`.
- Source Stage 2B lock canonical SHA-256: `30a97e175200d3a58bc0cc0c200acec301d3a4f4cd662466d4c3491b9f816597`.
- Build commit: `d446b3b9` on `codex/stage2b-launch`.
- Draft machine lock: `training/paper2_stage2b_autopsy_lock.json`.
- Prelock receipt: `docs/receipts/paper2_stage2b_autopsy_prelock_20260820.json`, 11,939 bytes, SHA-256 `4c01812ecd545c076ae540b3e3bdbce5e314933157ea5e6a8aa7d0ef1c95d732`.
- Strategy amendment request: Drive `1eA0WGyOkaIfeLuMUfG9y6f4KTkCCfNNE`, 4,358 bytes, SHA-256 `d69963b60a63964ca1c3c81fc2f1b4b9da5b67adb1a6fd685240ae7dd25f12ce`.

## 3. Five diagnostic arms

1. Onset trajectory on one deterministic, proportionally battery-stratified 256-row DEV-2 subsample at steps 0, 20, 60, 100, 200, 300, 500, 700, and 1,000.
2. Amplitude response at initialization and step-1,000 EMA for gamma 0, 0.01, 0.02, and 0.05, with both comparators and a full-logit zero-write identity gate.
3. Component attribution at step 1,000: standard, constitutive innovation off, fresh scratch state each loop, and inherited residual-flow update off.
4. Attractor diagnostics: K1-versus-K4 margin correlation, raw and centered cross-question recurrent-state cosine, loop-direction cosine, and a K1-K4 generative DEV-1 sweep.
5. Objective-task divergence: per-loop held-out CE, forward KL, and monotonicity at initialization and step 1,000, interpreted alongside the logged training trajectory.

Every arm is evaluation-only. The sealed partitions are absent from the runner's input graph.

## 4. Implementation contracts

- Diagnostic behavior is opt-in through `stage2b_diagnostic_mode`; the default training path remains `standard`.
- `zero_write` executes the diagnostic state machinery but returns the recurrent hidden state unchanged and records a zero gate and zero write.
- The zero-write gate compares both generated predictions and the complete loop-logit tensor between initialization and step-1,000 states. Any difference stops the run before diagnostic interpretation.
- `constitutive_off` disables only the constitutive innovation.
- `fresh_state_each_loop` resets the carried scratch state to its initial value at every re-entry.
- `inherited_flow_off` disables the inherited `SharedResidualFlow` state update while preserving final bridge execution.
- The signed lock is revalidated immediately before model contact. The runner cannot proceed with an unsigned status or any open field.
- A100 execution is pinned to one NVIDIA A100-SXM4-40GB session, bfloat16, SDPA, and both seeds sequentially.

## 5. Verification

- Python compilation passed for the evaluator, orchestrator, lock helpers, model changes, and launchers.
- Targeted and regression validation: 112 tests passed across Stage 2B adjudication, runtime, campaign, task inference, recurrent wrapper, bootstrap parity, and the new autopsy paths.
- `git diff --check` passed.
- The unrelated `.verify_dev2_v2/` and `.verify_dev2_v3/` directories were not staged or modified.

## 6. Prelock result

The CPU prelock completed and exited 2 as registered. It froze the 256-row DEV-2 subsample at SHA-256 `d48d5d7c0ad9d6b9a6267e50fb0d0c753aa5ff2a7bc883ab00508c6bd8d84bc4` with selection-receipt SHA-256 `19a9df5ced8ce6f52ac87c32c3251eb67f63eb450f3fd9c01601ae09236463b1`.

The checkpoint inventory found both deterministic step-0 constructions and both step-1,000 EMA endpoints with their registered hashes. It found none of the 14 requested intermediate cells: steps 20, 60, 100, 200, 300, 500, and 700 for either seed. Source inspection and the receipt agree that `resume.pt` was atomically overwritten every 20 steps and named snapshots were emitted only at registered looks. This is not a path or transport error.

The prelock confirms `model_loaded=false`, `optimizer_constructed=false`, `optimizer_steps=0`, `confirm_scored=false`, and `eval_e_scored=false`. The CPU session was released after the receipt was copied and verified; no Colab sessions remain active.

## 7. Open lock fields

The score pass must not launch until all three fields are closed:

1. A strategy amendment for the unavailable onset checkpoints.
2. Strategy confirmation that `inherited_flow_off` is the intended nontrivial reading of the requested inherited-bridge ablation.
3. Mark's signature after the completed lock is reviewed.

## 8. Checkpoint finding

The source runner wrote `resume.pt` every 20 steps by atomic replacement and emitted named EMA checkpoints only at registered looks. The `RETAIN_STEPS` declaration was not used by the save path. The prelock has now confirmed the resulting loss of intermediate model states with a durable inventory table.

No missing state will be substituted, reconstructed by training replay, or approximated from logs under the present authority. The endpoint, amplitude, component, attractor, and objective-task arms remain technically runnable, but they do not run piecemeal without the amendment requested in `docs/STRATEGY_REQUEST_STAGE2B_AUTOPSY_ONSET_AMENDMENT_20260820.md`.

## 9. Decision mapping

- H-B magnitude: a lower gamma outperforms the matched initialization comparator; prioritize margin-indexed radius control and a trainable gate temperature.
- H-C constitutive: disabling the constitutive innovation restores materially more capability than disabling the inherited residual flow; replace the constructor with gated additive innovation.
- H-A attractor: task-margin correlation collapses or question states converge while held-out CE/KL improves or remains flat; require an explicit task-preservation anchor in every successor.
- Composite outcomes remain composite. The autopsy is not forced to name one exclusive cause.

## 10. Immediate sequence

1. Obtain the two narrow strategy rulings in the amendment request.
2. Fill the machine lock from the ratified values and obtain Mark's signature.
3. Run both seeds in one A100-40GB session.
4. Validate row and aggregate receipts, prepare figures where they clarify the mechanism, publish the final handoff, and release all paid compute.
