# Paper Two Stage 2B-A Autopsy Build Handoff

Date: 2026-08-20. Status: build complete; score-blind prelock inventory pending Drive authorization. No model has been loaded by this program, no optimizer has been constructed, and CONFIRM and EVAL-E remain sealed.

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

## 6. Open lock fields

The score pass must not launch until all four fields are closed:

1. The frozen DEV-2 subsample SHA-256 from the CPU prelock.
2. Exact per-seed SHA-256 values for every requested onset checkpoint.
3. Strategy confirmation that `inherited_flow_off` is the intended nontrivial reading of the requested inherited-bridge ablation.
4. Mark's signature after the completed lock is reviewed.

## 7. Known checkpoint risk

The source runner wrote `resume.pt` every 20 steps by atomic replacement and emitted named EMA checkpoints only at registered looks. The `RETAIN_STEPS` declaration was not used by the save path. The prelock therefore inventories exact historical states and exits 2 with a durable table if any requested onset cell is unavailable.

No missing state will be substituted, reconstructed by training replay, or approximated from logs under the present authority. If the inventory is incomplete, strategy must amend the onset arm before any model is loaded. The endpoint, amplitude, component, attractor, and objective-task arms remain technically runnable, but they do not run piecemeal without that amendment.

## 8. Decision mapping

- H-B magnitude: a lower gamma outperforms the matched initialization comparator; prioritize margin-indexed radius control and a trainable gate temperature.
- H-C constitutive: disabling the constitutive innovation restores materially more capability than disabling the inherited residual flow; replace the constructor with gated additive innovation.
- H-A attractor: task-margin correlation collapses or question states converge while held-out CE/KL improves or remains flat; require an explicit task-preservation anchor in every successor.
- Composite outcomes remain composite. The autopsy is not forced to name one exclusive cause.

## 9. Immediate sequence

1. Complete the score-blind CPU prelock and bank its manifest and inventory receipt.
2. Return any inventory block and the ablation interpretation to strategy.
3. Fill the machine lock from ratified values and obtain Mark's signature.
4. Run both seeds in one A100-40GB session.
5. Validate row and aggregate receipts, prepare figures where they clarify the mechanism, publish the final handoff, and release all paid compute.
