# Handoff: T1-lite-R Replication Result and Paper Two Decision Point

**Date:** 2026-07-25  
**Run:** `stage5_paper2_t1_lite_r_20260725`  
**Registered attempt:** seed 1, attempt 2  
**Primary endpoint:** raw final-step weights  
**Registered verdict:** `registered_negative`  
**Operational status:** complete; A100 may remain shut down

## 0. Purpose

This handoff closes the registered T1-lite-R replication, reconciles it with
the seed-0 T1-lite result and EMA audit, and presents the next decisions for
strategy review. It separates the strict preregistered verdict from the
mechanistic result.

The strict result is negative because the raw seed-1 endpoint missed the
forced-chain preservation floor by four rows. The mechanistic result is
stronger: across two raw endpoints, the internal token pathway selected every
trained depth exactly and obeyed every causal override, while preserving the
answer operation within 3.32 to 3.71 percentage points of the matched
non-halting reference. The correct bounded reading is therefore exact causal
depth control at a small, reproducible preservation cost, not an unrestricted
T1 pass.

## 1. Why the replication ran

Seed 0 was registered with the continuous-EMA endpoint as primary. That
endpoint failed badly, while its raw final-step secondary selected all
`1,024/1,024` trained depths exactly, obeyed all `5,632/5,632` causal
interventions, and missed the preservation floor of `975/1,024` by eight
rows. This was the preregistered near-threshold condition requiring seed 1.

A read-only audit then showed that continuous EMA destroyed the learned
function inside the recurrent block despite small global parameter distance.
T1-lite-R therefore changed one registered factor only: raw final-step weights
became primary. Continuous EMA and a stage-reset EMA remained passive
descriptive shadows. Seed, endpoint policy, and preservation infrastructure
were locked before training in commit `ae2793ac`.

## 2. Experimental design

### 2.1 Substrate and trainable set

- Backbone: Qwen2.5-0.5B with the repaired recurrent surgery.
- Lineage: fresh-base full-block, not a continuation from seed 0.
- Seed: `1`.
- Trainable parameters: `180,559,617`.
  - Recurrent block: `178,948,608`.
  - Repaired split bridge: `1,608,321`.
  - Three control-token rows: `2,688`.
- Frozen pretrained weights, old embedding rows, prelude, and coda remained
  unchanged under end-to-end hashes.
- Added control tokens: continue, stop, and readout at new IDs `151936` through
  `151938`; they were masked from visible output.

### 2.2 Training recipe

The recipe was unchanged from the seed-0 lock:

- Optimizer: AdamW with the registered split-prelude parameter groups.
- Total training: `10,500` steps.
- Staged support curriculum with boundaries at steps `500`, `2,500`, `6,500`,
  and `8,500`, ending at step `10,500`.
- Trained depths: 1 through 8.
- Control-loss coefficient: `0.5`.
- Stop and continue class weights: equal.
- Mechanism rehearsal: 30 percent.
- Exact target: continue before the row's stated depth and stop at that depth.
- Liveness guardrail: abort only if control loss stayed flat for the complete
  stage and stop recall was exactly zero on trained depths.
- Tier-1 canary hard floor: `60/64`.

### 2.3 Endpoint policy

Three states were maintained:

1. Raw final-step weights, the only registered primary.
2. Continuous EMA at decay `0.999`, descriptive only.
3. Stage-reset EMA at decay `0.999`, reset from raw at each support expansion,
   descriptive only.

No endpoint was selected from intermediate accuracy. Raw final step was fixed
before training.

### 2.4 Registered gates

All four gates were required for a positive verdict:

| Gate | Locked requirement |
|---|---|
| 1. Forced-chain preservation | At least `975/1,024`, within 3 points of the full-block reference `1,005/1,024` |
| 2. Self-halted accuracy | Within 3 points of paired forced-depth accuracy |
| 3. Control selection | At least `115/128` at every trained depth and `922/1,024` pooled |
| 4. Causal override | Exact on all `4,608` forced-stop and `1,024` forced-continue executions |

Depths 9 through 14 and four training-free exit baselines were descriptive.

### 2.5 Preservation policy

Raw, continuous-EMA, and stage-reset-EMA trainable states were atomically
saved at steps `500`, `2,500`, `6,500`, `8,500`, and `10,500`. The run could
not be scored unless all fifteen files existed in Drive and local/Drive hashes
matched.

## 3. Integrity and completion

- Run status: `finished`.
- One-loop identity maximum logit difference: `0.0`, below `1e-3`.
- Frozen-base SHA-256 unchanged: yes.
- Old embedding-row SHA-256 unchanged: yes.
- All parameters finite: yes.
- Tier-1 canary by boundary: `60/64`, `60/64`, `61/64`, `61/64`.
- Liveness aborts: none.
- Stage manifest: complete.
- Required stage states: `15/15`.
- Missing stage states: none.
- Local/Drive hash mismatches: none.
- Raw final checkpoint SHA-256:
  `93d2e5f9a941bbe79a0b2fc3f9bf43d582bf054990c14b1a93ff67024140062d`.
- Continuous-EMA final checkpoint SHA-256:
  `5e5488d1a783176d84ff3f5f1fe2987a2c68ff5e8aeb2dd1d62676c5e6c6f18a`.
- Stage-reset-EMA final checkpoint SHA-256:
  `69d11bc7d0f4f9f74210bcf9f04ddcd4c317e85d1093f0ccf957a5dd32d1b0e5`.
- Final GitHub receipt commit: `5b5379bd`.

The original launch stopped before model loading because a Windows CRLF hash
was compared to Git's LF bytes in Colab. The canonical-newline repair changed
no experimental factor and consumed no attempt.

## 4. Registered raw-primary result

| Gate | Result | Verdict |
|---|---:|---:|
| Forced-chain preservation | `971/1,024 = 94.824%` | **Fail**, floor `975` |
| Self-halted accuracy | `971/1,024 = 94.824%` | Pass |
| Exact depth selection | `1,024/1,024 = 100%` | Pass at every depth |
| Continue decisions | `3,584/3,584 = 100%` | Pass |
| Stop decisions | `1,024/1,024 = 100%` | Pass |
| Causal overrides | `5,632/5,632 = 100%` | Pass |

The endpoint missed Gate 1 by four rows, or `0.391` percentage points below
the registered floor. It was `3.320` percentage points below the matched
non-halting reference of `1,005/1,024`. Because all four gates were jointly
required, the registered verdict is negative.

Forced and self-halted answer correctness were identical row for row because
the controller selected the required depth exactly on every gated row.

## 5. Seed-to-seed replication

### 5.1 Per-depth raw endpoints

| Depth | Seed 0 raw | Seed 1 raw | Seed-1 change |
|---:|---:|---:|---:|
| 1 | 128/128 | 127/128 | -1 |
| 2 | 126/128 | 126/128 | 0 |
| 3 | 127/128 | 124/128 | -3 |
| 4 | 128/128 | 125/128 | -3 |
| 5 | 122/128 | 126/128 | +4 |
| 6 | 119/128 | 118/128 | -1 |
| 7 | 113/128 | 114/128 | +1 |
| 8 | 104/128 | 111/128 | +7 |
| **Pooled** | **967/1,024** | **971/1,024** | **+4** |

Both raw seeds achieved `1,024/1,024` exact depth selection and exact causal
control. The answer errors remain concentrated toward the deeper trained tail,
but seed 1 improved depth 8.

On paired rows, seed 1 alone was correct on 52 rows and seed 0 alone was
correct on 48. A post-hoc paired two-sided sign test gives `p=0.764`. This is
not evidence that seed 1 is a better preservation model. It is evidence that
the near-threshold preservation cost and exact control behavior replicated in
kind.

### 5.2 Replicated bounded reading

- Seed 0 raw missed the floor by 8 rows and the reference by 3.711 points.
- Seed 1 raw missed the floor by 4 rows and the reference by 3.320 points.
- Both selected every trained depth exactly.
- Both obeyed every causal override exactly.

The actuator result is seed-replicated. The strict joint pass is not.

## 6. EMA shadow results

| Endpoint | Forced answer | Self-halted answer | Exact selection | Continue | Stop |
|---|---:|---:|---:|---:|---:|
| **Raw primary** | **971/1,024** | **971/1,024** | **1,024/1,024** | **3,584/3,584** | **1,024/1,024** |
| Continuous EMA | 216/1,024 | 89/1,024 | 128/1,024 | 0/3,584 | 1,024/1,024 |
| Stage-reset EMA | 1,003/1,024 | 741/1,024 | 731/1,024 | 3,584/3,584 | 731/1,024 |

Continuous EMA reproduced the seed-0 collapse. It always stopped at loop 1,
so only depth-1 selections were correct.

Stage-reset EMA produced a different and informative tradeoff. Its forced
answer accuracy nearly matched the non-halting reference, but its stop policy
was incomplete. It selected depths 1 through 4 and depth 8 exactly, while
depths 5, 6, and 7 reached only `37/128`, `0/128`, and `54/128` exact
selections. This separates operation preservation from controller
preservation:

- Continuous EMA damaged both the recurrent answer function and depth control.
- Stage-reset EMA largely preserved the answer function but lagged the stopping
  policy introduced across support transitions.
- Raw final weights preserved exact control with a small answer cost.

This is descriptive because neither EMA shadow was a registered endpoint.

## 7. Curriculum and dose readout

| Boundary | Trained support | Exact pilot selection | Stop recall | Tier-1 canary |
|---:|---:|---:|---:|---:|
| 500 | 1 | 32/32 | 32/32 | 60/64 |
| 2,500 | 1-2 | 64/64 | 64/64 | 60/64 |
| 6,500 | 1-4 | 128/128 | 128/128 | 61/64 |
| 8,500 | 1-8 | 168/256 | 200/256 | 61/64 |
| 10,500 | 1-8 gated set | 1,024/1,024 | 1,024/1,024 | final evaluation |

The first support-8 boundary was not consolidated. The final 2,000-step dose
converted partial depth control into exact depth control. This is consistent
with Paper One's dose result: support expansion initially disrupts the new
tail, and additional training consolidates it.

The saved raw and shadow states now permit the stage-lag analysis that was
impossible for seed 0. That analysis has not yet run.

## 8. Training-free baseline comparison

Thresholds were fit on the separate 512-row calibration set and evaluated on
the 1,024 gated rows.

| Exit rule | Exact depth | Mean selected loops |
|---|---:|---:|
| Learned control-token pathway | **1,024/1,024** | required depth |
| Fixed depth 1 | 128/1,024 | 1.000 |
| Answer-logit margin | 135/1,024 | 1.140 |
| Hidden-state update norm | 253/1,024 | 2.753 |
| Successive-output KL | 96/1,024 | 4.364 |

The internal token pathway is not merely matching an obvious confidence or
convergence heuristic on the trained family. This comparison remains
descriptive and does not establish natural difficulty inference.

## 9. Extrapolation beyond trained support

On 768 rows at depths 9 through 14:

- Forced-depth answers: `263/768 = 34.24%`.
- Self-halted answers: `10/768 = 1.30%`.
- Exact selected depth: `0/768`.
- Continue recall: `5,094/8,064 = 63.17%`.
- Stop recall at the eventual selected point: `768/768`.

The controller stops too early outside trained support. Exact causal control on
depths 1 through 8 is therefore a learned in-support policy, not an
algorithmically extrapolating halting rule.

## 10. Interpretation

### 10.1 Supported

1. The explicit internal token pathway can causally control recurrent depth on
   this substrate at every trained depth.
2. This control replicated across two independently trained raw endpoints.
3. Self-halting itself adds no answer loss when the correct depth is selected;
   forced and self-halted accuracy are identical for both raw seeds.
4. The learned controller substantially outperforms four training-free exit
   heuristics on exact in-support depth selection.
5. Continuous EMA is unsafe for this staged iterative mechanism under the
   tested recipe, across two seeds.
6. Resetting EMA at support boundaries preserves the answer operation much
   better, but does not fully preserve the evolving stop policy.
7. Support-8 control required additional consolidation dose after the first
   support-8 boundary.

### 10.2 Not supported

1. A preregistered all-four-gate T1 success.
2. Cost-free learned halting relative to the matched non-halting reference.
3. Depth selection beyond trained support.
4. Content-determined difficulty inference. The row depth is stated by the
   synthetic task structure.
5. Natural-task halting, reasoning improvement, or measured efficiency.
6. A general claim that EMA is harmful. The result is specific to this staged
   recurrent mechanism and recipe.
7. A claim that stage-reset EMA is a validated production endpoint.

## 11. Limitations

- The family is synthetic and has an exact known depth target.
- Only two raw training seeds are available.
- The same frozen evaluation family is used for both seeds.
- The strict threshold was preregistered; a four-row miss remains a failure
  even though it is near the boundary.
- The causal sweep was run on the raw primary only. Shadows are descriptive.
- Stage states are preserved, but boundary-by-boundary raw-versus-shadow
  evaluations have not yet been computed.
- No natural-surface retention battery was part of T1-lite-R.
- No wall-clock advantage was measured for self-halting.
- The experiment covers the full-block budget only and makes no adapter-budget
  T1 claim.
- The controller learns the task's stated-depth policy. It does not prove that
  the model can infer how much computation an unseen problem requires.

## 12. Plain-language summary

We taught the recurrent model to use hidden continue and stop tokens. On every
test problem whose required depth was within the training range, it chose the
right number of loops. When we deliberately forced the hidden decision to stop
or continue, the model executed exactly what the intervention commanded. This
happened in two separate training runs, so the control mechanism is real and
reproducible.

The tradeoff is that installing this perfect controller slightly reduced the
underlying answer accuracy. The two runs missed the preservation requirement
by eight and four questions out of 1,024. For that reason the formal experiment
did not pass. The most accurate plain statement is: we achieved exact causal
control of recurrent depth, but not for free.

The averaging method commonly used to smooth model weights was particularly
damaging here. Ordinary continuous averaging erased the controller. Resetting
the average whenever the curriculum added harder depths preserved the answer
skill but still lagged some stopping decisions. The unaveraged final weights
were the only endpoint that retained exact control.

The controller also did not generalize to depths beyond those used in
training. It is an effective learned actuator over trained depths, not yet a
general rule for deciding how long to think.

## 13. Questions for strategy review

1. **Verdict language:** Approve the dual statement: strict registered
   negative, bounded positive for replicated exact causal control at a measured
   preservation cost?
2. **T1 training closure:** Is the stated-depth T1 training line now closed
   after two raw seeds, with no additional seed, optimizer, EMA-decay, or
   threshold sweep?
3. **D0 authorization:** Does the replicated actuator justify advancing D0 to
   final preregistration, despite missing the preservation floor? D0 remains
   build-only until explicitly approved and locked.
4. **D0 endpoint:** If D0 opens, should the seed-1 raw endpoint be locked as the
   drafter because it is the registered replication primary, rather than
   choosing opportunistically among seed 0, seed 1, or a shadow?
5. **Stage-lag reconstruction:** Should the five saved boundary triplets be
   evaluated on an L4 to identify when continuous and stage-reset EMA diverge?
6. **Seed-0 audit extras:** Are the authorized layer-group swap and per-depth
   interpolation breakdown still worth running now that seed 1 reproduced the
   raw/continuous-EMA split?
7. **Paper framing:** Should T1 be presented as the positive causal-control
   contrast to Arm G, with preservation cost and failed extrapolation as the
   boundary, or should Paper Two retain a broader controllability-boundary
   framing until D0?
8. **Content-determined bridge:** Before natural traces, should the next halting
   family hide depth behind an iterate-until-predicate or sentinel condition,
   so the controller must infer required computation rather than read a stated
   target?

## 14. Proposed next steps for approval

### Step 1: Bank and close T1-lite-R, no GPU

- Mark both registered attempts final.
- Use the bounded interpretation above.
- Do not change either historical verdict.
- Update Paper Two's status and figures with raw seed 0, raw seed 1, continuous
  EMA, and stage-reset EMA as separately labeled endpoints.

### Step 2: One read-only stage-lag receipt, L4

- Evaluate raw, continuous EMA, and stage-reset EMA at all five saved stages.
- Use the same fixed pilot and gated-reader definitions.
- Report function versus support, not checkpoint selection.
- Do not alter any registered verdict or promote a shadow.

### Step 3: Complete already-authorized read-only localization, L4 or CPU

- Seed-0 recurrent layer-group swap.
- Per-depth raw-to-EMA interpolation breakdown.
- Stop after these named audits; no EMA tuning sweep.

### Step 4: Strategy decision on D0, no labeling or training yet

- Resolve teacher size, corpus hashes and licenses, trainable set, calibration
  constants, and endpoint SHA.
- Lock a complete D0 preregistration before any teacher forward labeling.
- If the preservation miss is judged disqualifying, keep D0 closed and state
  the reason rather than silently substituting another endpoint.

### Step 5: Continue COCONUT engineering independently, L4

- Run only the already-authorized RG-4 epsilon sweep and RG-11 precision-policy
  comparison.
- Keep RG-12 and all C-track training unauthorized until those contracts pass
  and strategy explicitly approves the next phase.

### Step 6: Convene the Paper Two framing decision

Use T1-lite-R, the stage-lag receipt, and the D0 authorization decision to
choose between:

1. a causal-control paper centered on successful in-support token control with
   measured preservation and extrapolation boundaries; or
2. a broader frozen-substrate controllability paper combining Arm G's failed
   branch interface with T1's successful but costly depth interface.

## 15. Do not do

- Do not call the registered T1 experiment a pass.
- Do not rerun until a desired threshold is crossed.
- Do not tune the preservation floor after observing the four-row miss.
- Do not select an EMA endpoint from descriptive results.
- Do not claim natural adaptive reasoning or efficiency.
- Do not authorize D0 labeling or training from this handoff alone.
- Do not reopen Arm G or authorize COCONUT RG-12 from this result.

## 16. Canonical artifacts

- `outputs/stage5/stage5_paper2_t1_lite_r_20260725/summary.json`
- `outputs/stage5/stage5_paper2_t1_lite_r_20260725/preregistration.json`
- `outputs/stage5/stage5_paper2_t1_lite_r_20260725/replication_basis.json`
- `outputs/stage5/stage5_paper2_t1_lite_r_20260725/train/training_summary.json`
- `outputs/stage5/stage5_paper2_t1_lite_r_20260725/train/stage_checkpoint_manifest.json`
- `outputs/stage5/stage5_paper2_t1_lite_r_20260725/eval/raw_primary/summary.json`
- `outputs/stage5/stage5_paper2_t1_lite_r_20260725/eval/continuous_ema_shadow/summary.json`
- `outputs/stage5/stage5_paper2_t1_lite_r_20260725/eval/stage_reset_ema_shadow/summary.json`
- `docs/PHASE_T1_LITE_R_PREREGISTRATION_20260725.md`
- `docs/PAPER2_T1_LITE_EMA_AUDIT_HANDOFF_20260725.md`
- `docs/PAPER2_T1_LITE_R_LAUNCH_HASH_CORRECTION_20260725.md`

Drive checkpoint root:
`/content/drive/MyDrive/recurrent-qwen-svgd-checkpoints/stage5_paper2_t1_lite_r_20260725/`.
