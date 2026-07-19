# Paper One Finishing Receipts

**Date:** 2026-07-19  
**Scope:** receipt resolution only; no new experiments and no manuscript prose rewrite  
**Canonical evidence rule:** landed JSON artifacts override provisional handoff
numbers.

## 0. Completion Status

This packet resolves all twelve requested receipt questions and records the
four requested ledger claims. Figure 4 has been regenerated.

One mechanical action remains outside this repository: apply these
paste-ready resolutions to the authoritative manuscript v2 file. That file is
not present in the repository, so this batch cannot truthfully certify that
the literal `[RESOLVE-*]` strings have been removed from that external file.

The provisional E-closure numbers in the request do not match the canonical
landed artifacts. The manuscript and ledger must use the artifact-backed
values in Section 4 below.

## 1. RESOLVE-GUARDRAIL

### 1.1 Battery composition and method

The keeper-lineage loop-1 regression battery comprised:

| Benchmark | Rows |
|---|---:|
| AI2 ARC Easy, all validation and test rows | 5,197 |
| AI2 ARC Challenge, all validation and test rows | 2,590 |
| Total paired rows per checkpoint | 7,787 |

Four promoted checkpoints were compared with the same base model:

1. `stage5_reentry_recovery_20260625_154210`
2. `stage5_depth_support_route_20260705_124320`
3. `stage5_support8_dose_arm_20260706_153028`
4. `stage5_n24_support12_rung_20260707_140139`

The preregistered noninferiority margin was `-0.03` accuracy. The primary
family comprised eight checkpoint-by-benchmark cyclic-label comparisons.
Exact paired sign/McNemar p-values were Bonferroni-corrected across those
eight comparisons. Content-question-only scores were secondary descriptive
diagnostics and were not part of the multiplicity-controlled family.

### 1.2 Primary cyclic-label results

| Checkpoint | Benchmark | Base | Recurrent | Delta rows | Raw p | Corrected p |
|---|---|---:|---:|---:|---:|---:|
| Re-entry recovery | ARC Easy | 3976/5197 | 3984/5197 | +8 | 0.512255 | 1.000000 |
| Re-entry recovery | ARC Challenge | 1522/2590 | 1526/2590 | +4 | 0.746534 | 1.000000 |
| Support 6 | ARC Easy | 3976/5197 | 3971/5197 | -5 | 0.560065 | 1.000000 |
| Support 6 | ARC Challenge | 1522/2590 | 1524/2590 | +2 | 0.831812 | 1.000000 |
| Support 8 | ARC Easy | 3976/5197 | 3968/5197 | -8 | 0.268187 | 1.000000 |
| Support 8 | ARC Challenge | 1522/2590 | 1523/2590 | +1 | 1.000000 | 1.000000 |
| N24 support 12 | ARC Easy | 3976/5197 | 3962/5197 | -14 | 0.038477 | 0.307818 |
| N24 support 12 | ARC Challenge | 1522/2590 | 1520/2590 | -2 | 0.838820 | 1.000000 |

All eight comparisons remained above the locked noninferiority margin. The
largest row-count decrease was the N24 checkpoint on ARC Easy: `-14` rows,
unadjusted `p=0.0384773083`, Bonferroni-corrected `p=0.3078184663`. It is not
a multiplicity-corrected significant regression and it does not breach the
three-point noninferiority margin.

### 1.3 Secondary content-reader scores

| Checkpoint | ARC Easy base -> recurrent | ARC Challenge base -> recurrent |
|---|---:|---:|
| Re-entry recovery | 3115 -> 3145 | 886 -> 894 |
| Support 6 | 3115 -> 3110 | 886 -> 886 |
| Support 8 | 3115 -> 3110 | 886 -> 887 |
| N24 support 12 | 3115 -> 3118 | 886 -> 891 |

These are descriptive because content-reader scoring was not the corrected
primary family.

### 1.4 Permutation zero-shot control

The N24 checkpoint was evaluated on a new permutation-table set with 128 rows
per depth for depths 1-12. It scored `9841/9984 = 98.57%` across active
labels. Its per-depth diagonal differed from the arbitrary-table reference by
at most `0.03125`, within the locked `0.05` tolerance at every depth. The
receipt status is `permutation_zero_shot_parity_pass`.

### 1.5 Natural and Tier-1 canary references

Natural-canary values are lineage-specific and must not be mixed:

| Lineage/use | Baseline | Candidate | Delta | Reading |
|---|---:|---:|---:|---|
| Full-block inverse cap-3 rehearsal | 227/256 | 212/256 | -5.86 pp | hard-stop breach |
| Full-block N24 inverse continuation | 227/256 | 171/256 | -21.88 pp | red |
| Arm E E4 own natural baseline | 60/256 | 49/256 | -4.30 pp | hard-stop breach |
| Bounded PEFT Tier-1 arithmetic | 60/64 | 61/64 at checks | +1 row | preserved |
| Arm E E4 Tier-1 arithmetic | 59/64 | 59/64 at step 100 | 0 rows | green |

The bounded PEFT checks used a `60/64` registered reference and a maximum
three-point drop. Arm E correctly measured its own natural baseline because
the full-block natural reference was not applicable to that lineage.

### 1.6 Ledger wording

- Claim ID: `general_capability_preservation`
- Status: `supported_bounded`
- Scope: `loop-1, evaluated battery, keeper-lineage promoted checkpoints`
- Safe claim: promoted checkpoints were noninferior to base at the
  preregistered three-point margin on this evaluated loop-1 ARC battery.
- Prohibited expansion: broad natural-capability preservation or superiority.

Primary receipt:
`outputs/stage5/stage5_lineage_regression_battery_current/summary.json`.

## 2. RESOLVE-EARLY

### 2.1 Era configuration

The June Stage 4 configuration used Qwen2.5-0.5B-Instruct with a
Prelude/Recurrent Block/Coda split of layers `0:6`, `6:18`, and `18:end`,
with `max_loops=4`.

Phase 1:

- LoRA rank `8`, alpha `16`, dropout `0`;
- LoRA on `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`,
  and `down_proj` in the twelve-layer recurrent block, 84 modules total;
- learning rate `1e-5`, weight decay `0`, max gradient norm `0.3`;
- Ponder initial halt probability `0.15`, KL weight `0.08`;
- maximum 500 configured steps.

Phase 2:

- four trajectories;
- particle mode `svgd`;
- latent sampling disabled in the recorded configuration;
- initialization noise `0.02`, epsilon `1.0`, repulsion scale `0.5`;
- median bandwidth, floor `1e-6`, max repulsion norm `1.0`;
- latent dimension `256`, latent scale initialization `0.01`;
- adapter initialization standard deviation `0.0001`;
- injection position `pre`;
- maximum 100 configured steps from the Phase 1 checkpoint.

The bridge gate was initialized to zero. Because the projected branch was
also exactly zero at that initialization, the bridge output was
identity-preserving but gradient-dead.

The era did not have the later preregistered acceptance framework. Its
recorded 128-row recovery result was base `72/128`, Phase 1 `70/128`, and
Phase 2 mean/vote `69/128`; Phase 2 best was one row below Phase 1 and three
below base. These are historical screening outcomes, not gate-qualified
evidence.

Configuration receipts:

- `outputs/stage4/stage4_opus_a100_20260620/phase1.yaml`
- `outputs/stage4/stage4_opus_a100_20260620/phase2.yaml`
- `outputs/stage4/stage4_opus_a100_20260620/recovery_summary.json`

### 2.2 Halting archaeology

Recorded aggregate halting moments exist, but no calibrated per-task
distribution supports a useful learned-depth claim.

| Era/readout | Expected loops | Halt entropy |
|---|---:|---:|
| Phase 1 validation | 2.6920 | 1.3444 |
| Phase 2 validation | 2.8763 | 1.2814 |
| Original Stage 4 aggregate | 2.8102 | 1.3024 |
| Later reconstructed early archives | about 3.05-3.07 | about 1.17-1.18 |

These means show a broad four-loop stopping distribution, not task-adaptive
depth selection. They are directly consistent with the later bounded selector
failure and must not be described as a successful Ponder policy.

### 2.3 Dead-bridge receipt

The read-only re-entry diagnostic found:

- `bridge_gate=0`;
- bridge bias maximum `0`;
- projection identity maximum difference `0`;
- bridge delta RMS `0`;
- gate, weight, and bias gradient magnitudes all `0`.

The diagnostic control made the bridge gate one and recovered nonzero
projection gradients: weight RMS `0.0009283`, bias RMS `0.0009042`. This
isolated the dead initialization rather than an optimizer failure.

Receipt:
`outputs/stage5/stage5_reentry_drift_20260625_011444/summary.json`.

### 2.4 Tail damper and Parcae

The original damper was an eval-only, covariance-calibrated
principal-component tail intervention. On 512 ARC-Challenge rows it swept
strengths `0`, `0.5`, and `1.0`; strength `1.0` reduced the loop-8 tail trace
ratio from `33.65` to `2.91` but did not produce a clean answer-selection
gain. The score changed from:

- strength 0: loop-1 `180`, loop-2 `167`, loop-3 `160`, oracle `229`,
  rescued `49`, harmed `70`;
- strength 1: loop-1 `180`, loop-2 `165`, loop-3 `159`, oracle `232`,
  rescued `52`, harmed `67`.

The damper was later retired from production. The active training forward path
records no damper path and strength zero.

Receipt:
`outputs/stage5/stage5_reentry_tail_damper_sweep_arc_challenge_train_offset0_20260626_233857/summary.json`.

[Parcae (Prairie et al., arXiv:2604.12946)](https://arxiv.org/abs/2604.12946)
constrains injection spectral norms through a discretized negative-diagonal
parameterization. That is not mathematically equivalent to the project's
post-hoc PC-tail attenuation. The correspondence survives only as a high-level
stability analogy. Appendix A.4 should use the fallback wording and must not
call the two mechanisms equivalent.

## 3. RESOLVE-REF

### 3.1 Verified identifiers and authors

- [Hierarchical Reasoning Model, arXiv:2506.21734](https://arxiv.org/abs/2506.21734):
  Guan Wang, Jin Li, Yuhao Sun, Xing Chen, Changling Liu, Yue Wu, Meng Lu,
  Sen Song, and Yasin Abbasi Yadkori.
- [Less is More: Recursive Reasoning with Tiny Networks,
  arXiv:2510.04871](https://arxiv.org/abs/2510.04871): Alexia
  Jolicoeur-Martineau.

The latter title should not be shortened in the bibliography to "Tiny
Recursive Models" unless that is explicitly introduced as an informal name.

### 3.2 Memories versus procedures

[Ruis et al., Procedural Knowledge in Pretraining Drives Reasoning in Large
Language Models, arXiv:2411.12580](https://arxiv.org/abs/2411.12580) supports
the narrower proposition that influential pretraining documents for reasoning
often contain reusable procedural knowledge and differ from fact retrieval.
It does not establish a strict memories-versus-procedures dichotomy. Keep the
discussion hedged: the results are consistent with recurrence implementing or
preserving procedural transformations rather than merely retrieving stored
answers.

### 3.3 McLeish et al. recipe and novelty fit

[McLeish et al., Teaching Pretrained Language Models to Think Deeper with
Retrofitted Recurrence, arXiv:2511.07384](https://arxiv.org/abs/2511.07384)
is verified.

Claim-fit findings:

1. Their own retrofit recipe uses continued pretraining/full-model
   optimization and reports Muon as preferable to AdamW for their recurrent
   models. Their architecture contains a full linear re-entry adapter after
   concatenating prelude and recurrent state, but this is an architectural
   adapter, not a parameter-efficient LoRA recipe.
2. The paper explicitly discusses Bae et al. (2024) as a prior pretrained
   recurrent retrofit that required low-rank adapters to recover base-model
   performance. Therefore the broad claim that depth-recurrent LoRA itself is
   novel is not supportable from this pass.
3. A defensible differentiator is narrower: this work uses one rank-16 LoRA
   parameterization shared across repeated executions of a surgically isolated
   Qwen recurrent block, together with a repaired split bridge,
   intermediate-state curriculum, and a matched depth-profile evaluation.
   Use "we study" or "to our knowledge in this exact configuration," not
   "the first," unless the cited Bae work and wider literature are separately
   audited.

## 4. Arm E Ledger Closures

The values below are canonical. They supersede the provisional numbers
`622/640`, `349/384`, relay `13.9%`, pointer `12.4%`, and the proposed E4
joint-pass numbers in the finishing request.

### 4.1 `adapter_persistence`

- Status: `supported`
- Active diagonal: `636/640 = 99.375%`
- Above-diagonal continue: `380/384 = 98.958%`
- Hold: `0/384`
- Full-block references: `625/640`, `357/384`, and `1/384`
- Receipt:
  `outputs/stage5/stage5_adapter_parity_e2_20260719/summary.json`

### 4.2 `adapter_zero_shot_transfer_minimal`

- Status: `supported_bounded`
- Relay: `249/1536 = 16.211%`
- Pointer: `264/1536 = 17.188%`
- Full-block descriptive references: relay `1321/1536 = 86.003%`,
  pointer `1213/1536 = 78.971%`
- Scope: no adapter-budget verbal training; the full-block reference did
  receive natural-surface development.
- Receipt:
  `outputs/stage5/stage5_adapter_parity_e3a_20260719/summary.json`

### 4.3 `adapter_retention_joint_pass`

- Status: `not_supported`
- Registered verdict: `wall_holds`
- Hard stop: step 100
- Inverse acquisition: `2/64`, required `46/64`
- Synthetic retention minimum: `0.09375`, required `0.93`
- Own natural baseline: `60/256 = 23.438%`
- Step-100 natural canary: `49/256 = 19.141%`, delta `-4.297` points
- Tier-1 arithmetic: `59/64`, green against the registered `60/64`
  reference and `0.9075` floor
- Joint pass at any checkpoint: false
- Scope: `one additional operation, single seed, this substrate`
- Receipt:
  `outputs/stage5/stage5_adapter_parity_e4_20260719/summary.json`

### 4.4 `adapter_budget_depth_profile`

- Arm A: `1506/1792 = 84.040%`
- Arm E: `1501/1792 = 83.761%`
- Paired two-sided p: `0.81275`
- Arm E depth frontier: `11.56`
- Frontier divided by trained support 8: `1.44x`
- Derivation: linear interpolation between Arm E depth 11
  (`111/128`) and depth 12 (`75/128`) at the `0.71` bar.
- Cross-check: Arm A ladder-official frontier `11.61`
- Registered profile verdict: tail-concentrated deficit, not
  budget-independent parity.
- Receipt:
  `outputs/stage5/stage5_adapter_budget_arm_e_20260718/summary.json`

## 5. Figure 4

The regenerated figure contains five curves:

- Arm A: `full block, 180.6M trainable`;
- Arm B: dense direct 0.5B;
- Arm C: dense scratchpad 0.5B;
- Arm D: dense direct 1.5B;
- Arm E: `R16 + bridge, 6.0M trainable`.

Depths 1-8 are shaded as trained support. The empirical Arm A/Arm E curve
crossover between depths 11 and 12 is marked at `d11.54`. This crossover is
distinct from Arm E's threshold frontier `11.56`.

Artifact:
`docs/figures/figure4_phase_a_depth_profile.svg`.

Regenerator:
`eval/build_paper_one_figure4.py`.

## 6. Marker Closure Map

| Marker family | Count | Closure receipt |
|---|---:|---|
| RESOLVE-GUARDRAIL | 3 | Section 1 and lineage regression/permutation JSON |
| RESOLVE-EARLY | 4 | Section 2 and Stage 4/re-entry/damper JSON |
| RESOLVE-REF | 5 | Section 3 and linked primary papers |
| Total | 12 | This packet |

The evidentiary work is complete. Literal marker deletion remains a manuscript
integration task because the authoritative v2 manuscript is external to this
checkout.

## 7. Publication-Safe Bottom Line

The bounded evidence supports:

1. loop-1 noninferiority on the evaluated ARC battery across promoted keeper
   checkpoints;
2. a documented early dead-bridge failure and its later repair;
3. strong adapter-budget persistence after intermediate supervision removal;
4. minimal zero-shot verbal transfer at the adapter budget;
5. a replicated acquisition-retention wall in the registered Arm E probe;
6. a `30.1x` trainable-budget reduction with near pooled parity but a measured
   far-tail capacity-depth interaction.

It does not support broad natural-capability preservation, successful learned
halting, mathematical equivalence between the damper and Parcae, general
novelty for depth-recurrent LoRA, or an Arm E retention joint pass.
