# Handoff: Phase-2 Experiments 0A and 0B - Canonical Geometry Before the Student Build

**Date:** 2026-08-04

**Program:** Paper Two, Phase 2, DC2/E1 preparation

**Status:** Experiments 0A and 0B complete on DEV-C; no frozen evaluation partition touched; no E1 training authorized by this handoff

**Decision requested:** canonicalizer carried into the student build, matched-pilot matrix, alpha-selection rule, and disposition of the weak per-example path monotonicity

## 0. Executive verdict

Experiments 0A and 0B clear the narrow engineering question they were built to answer. The cached Qwen2.5-14B teacher states contain compressible predictive structure beyond an unsupervised PCA baseline, all tested canonical geometries are numerically finite, affine endpoint paths improve average probe KL, and small disposable serial-flow modules reduce held-out canonical MSE in every method-by-alpha arm.

They do **not** select the final canonicalizer or whitening exponent, validate the complete student architecture, establish downstream acceptance, or open E1. Two findings require strategy review before the pilot design is locked:

1. Predictive RRR and Tucker predictive are substantially better than PCA, but Tucker's advantage over predictive RRR is small. The present receipts do not contain a preregistered inferential rule that converts that small edge into a final method choice.
2. Average affine paths improve probe KL, but only about 50-55% of individual paths improve monotonically. The disposable flow fits canonical MSE, yet some predictive arms improve KL while reducing top-1 teacher agreement. This makes the full module's control, acceptance, and preservation measurements load-bearing.

The clean next action is therefore **build-only**: implement the complete student module and pass its no-loss assertion battery. The matched DEV alpha pilots should wait until strategy locks the canonicalizer, the pilot matrix, and a non-arbitrary alpha-selection rule.

## 1. Place in the program

The governing implementation order is:

1. V1d capped-radius diagnostic.
2. Stage 0A teacher-state collection and metric repair.
3. Experiment 0A canonicalizer and whitening screening.
4. Experiment 0B interpolation and disposable flow screening.
5. Full student-module build and assertion battery.
6. Matched DEV-only alpha pilots on the built module.
7. Alpha selection from flow convergence, upper-model quality, and verified acceptance.
8. E1 lock, followed by the registered training phases.

Steps 1-4 are complete. This handoff concerns the transition from step 4 to step 5. It does not authorize steps 6-8.

## 2. Experimental integrity and lineage

### Experiment 0A

| Field | Receipt |
|---|---|
| Kind | `paper2_phase2_exp0a_canonicalizer_screening` |
| Status | `complete_development_only` |
| GitHub receipt commit | `3cc4b842` |
| Public summary | `outputs/stage5/stage5_paper2_phase2_exp0a_20260804/summary.json` |
| Public-summary SHA-256 | `ffb49b48fa20bbe686a16d9c0e7402c64438a21c4bba5a567034fa54fe07b4b7` |
| Stage 0A source-summary SHA-256 | `f6e83574ab3b42a4f9fb9c17bb88ff0ae831fc58f3d67aca093b25c4ce824680` |
| Stage 0A repair-summary SHA-256 | `7ee733f81751ec1db8c46fb9ea51149796b5b71f9b48a6d45574cb8f77676e6e` |
| Source-manifest SHA-256 | `43edbb74c5edf84dc5e6512dbe4beb1bbbf0f4df31b2c6830714ce2c8fc7ba93` |
| Backbone training | none |
| Optimizer steps | 0 |
| Frozen evaluation partitions touched | none |
| Elapsed time | 838.7 seconds |

The development split was document-disjoint: 166,708 calibration samples from 356 documents and 33,292 holdout samples from 76 documents, seed 20260804. Candidate artifacts were subsequently refit on all 200,000 DEV-C samples. Those all-sample refits are deployment artifacts, not held-out evidence.

### Experiment 0B

| Field | Receipt |
|---|---|
| Kind | `paper2_phase2_exp0b_flow_path_screening` |
| Status | `complete_development_only` |
| GitHub receipt commit | `69da6a7e` |
| Public summary | `outputs/stage5/stage5_paper2_phase2_exp0b_20260804/summary.json` |
| Public-summary SHA-256 | `082395712cc46d4bdd504cdadf721301f359c40f1fec887fc2c4f634788464f9` |
| Source Experiment 0A SHA-256 | `ffb49b48fa20bbe686a16d9c0e7402c64438a21c4bba5a567034fa54fe07b4b7` |
| Backbone training | none |
| Backbone parameters mutated | false |
| Disposable-flow optimizer steps | 5,400 total, 600 per arm |
| Frozen evaluation partitions touched | none |
| Elapsed time | 59.6 seconds |

All private canonicalizer, endpoint, target, and flow-pilot tensors were written to Drive with paths and SHA-256 hashes in the public receipts. Nothing required for reconstruction remained only in Colab runtime memory.

## 3. Experiment 0A design

### Question

Can a fixed low-dimensional canonical representation preserve predictive information from the teacher states while supplying a tractable geometry for the later recurrent flow?

### Inputs and geometry

- Teacher: cached Qwen2.5-14B boundary states from Stage 0A.
- Source population: 200,000 DEV-C boundary samples.
- Three selected teacher layers, initially mixed uniformly unless the method learns a mixture.
- Canonical shape: eight slots by 128 coordinates, with internal rank 256.
- Four future slots populated.
- Four trace/span slots masked because Stage 0A found no span boundaries.
- One frozen PCA orientation shared by every alpha arm.
- Effective eigenvalues computed once as `max(lambda_raw, 1e-4 * lambda_max, 1e-8)`.
- No second forward-pass epsilon.
- No renormalization of endpoints or persistent state.

### Methods

1. **PCA:** mandatory unsupervised linear baseline.
2. **Predictive RRR:** supervised reduced-rank projection toward the multi-target teacher representation.
3. **Tucker predictive:** predictive projection with learned layer-mixture weights and factorized structure.

The conditional nonlinear attention-pooling and deterministic-autoencoder arms were held unless the predictive linear model failed to improve on PCA by at least 1%. That trigger did not fire.

### Whitening grid

`alpha` in `{0, 0.5, 1.0}` under a common orientation:

- `alpha=0`: PCA rotation only, no variance equalization.
- `alpha=0.5`: partial whitening.
- `alpha=1.0`: full whitening under the regularized metric.

Because the orientation is shared, the grid isolates variance scaling rather than coordinate rotation. Alpha was screened, not selected.

### Metrics

- Conditional top-K future-token KL.
- Teacher top-1 agreement.
- Observed-token accuracy.
- Hidden-state cosine and normalized MSE.
- Canonical condition number.
- Per-coordinate gradient RMS dispersion.
- Probe parameter count and projection cost.

The sparse top-K KL is not full-vocabulary KL, and development probe fidelity is not downstream task quality.

## 4. Experiment 0A results

### Primary comparison at `alpha=0.5`

| Canonicalizer | Future KL, mean | Teacher top-1 agreement | Observed-token accuracy | Hidden cosine | Gradient CV |
|---|---:|---:|---:|---:|---:|
| PCA | 5.1492 | 7.69% | 7.46% | 0.5778 | **0.0215** |
| Predictive RRR | 4.7260 | 15.56% | 16.93% | **0.6276** | 0.6333 |
| Tucker predictive | **4.7225** | **16.24%** | **17.91%** | 0.6265 | 0.6295 |

The predictive methods approximately doubled teacher top-1 agreement and more than doubled observed-token accuracy relative to PCA. Their KL was about 8.2% lower. Tucker was directionally best on KL, teacher agreement, and observed-token accuracy, but its margin over predictive RRR was small.

### Alpha dependence

| Method | Alpha | Condition number | Future KL | Teacher agreement | Gradient CV | Gradient max/median |
|---|---:|---:|---:|---:|---:|---:|
| PCA | 0.0 | 6.49 | 5.1492 | 7.69% | 0.0528 | 1.10 |
| PCA | 0.5 | 2.55 | 5.1492 | 7.69% | 0.0215 | 1.04 |
| PCA | 1.0 | 1.00 | 5.1492 | 7.68% | 0.0231 | 1.01 |
| Predictive RRR | 0.0 | 10,000 | 4.7375 | 15.51% | 1.1970 | 17.10 |
| Predictive RRR | 0.5 | 100 | 4.7260 | 15.56% | 0.6333 | 2.87 |
| Predictive RRR | 1.0 | 1.00 | 4.7257 | 15.57% | 0.1064 | 1.12 |
| Tucker predictive | 0.0 | 10,000 | 4.7342 | 16.12% | 1.1897 | 17.91 |
| Tucker predictive | 0.5 | 100 | 4.7225 | 16.24% | 0.6295 | 2.93 |
| Tucker predictive | 1.0 | 1.00 | 4.7222 | 16.24% | 0.1059 | 1.12 |

Whitening barely changed the representational fidelity metrics, which is expected when the same invertible orientation and scale are carried consistently through the fixed probe. Its main measured effect was optimization geometry. Full whitening reduced predictive-method gradient dispersion by more than an order of magnitude relative to `alpha=0`, while partial whitening landed between the endpoints.

### Method-specific observation

Tucker learned layer weights of approximately `[0.584, 0.332, 0.084]`, rather than the uniform `[1/3, 1/3, 1/3]` mixture retained by PCA and predictive RRR. That asymmetry may be useful structure, but it also means Tucker differs from RRR by more than factorization alone. The present screen does not isolate which part produced the small gain.

### Screening disposition

- PCA: numerically valid mandatory baseline.
- Predictive RRR: numerically valid and predictively stronger than PCA.
- Tucker predictive: numerically valid and directionally strongest.
- Nonlinear attention pooling: not triggered.
- Deterministic autoencoder: not triggered.

Important terminology: `screening_survivor` in the receipt means that the metrics were finite. It is not a scientific pass threshold and should be read as **numerically valid arm**.

## 5. Experiment 0B design

### Question

Do affine paths between early and later canonical endpoints remain meaningful under the frozen probe, and can a small newly initialized serial flow learn to move along those paths without changing the backbone?

### Path audit

For every method-by-alpha arm, 0B evaluated an affine path between horizon-one and horizon-four canonical endpoints at `tau={0, 0.25, 0.5, 0.75, 1.0}`. It recorded:

- probe KL at each path point;
- fraction of individual paths with monotonically non-increasing KL;
- mean KL change from start to stop;
- hidden-path second difference;
- midpoint norm contraction.

The path was affine by construction. Near-zero second difference is therefore an implementation check, not independent evidence that a trained nonlinear module will follow the path.

### Disposable serial-flow pilot

Each of the nine method-by-alpha arms received a 600-step serial-flow pilot on newly initialized parameters only. The pilot used 6,657 training anchors and 1,666 validation anchors. It measured canonical validation MSE and frozen-probe behavior before and after the pilot.

This pilot is development scaffolding. It is not the complete student module, an E1 run, or the matched alpha-selection pilot.

## 6. Experiment 0B results

### Affine-path audit

| Method | Alpha | Mean KL start-to-stop | Monotonic-path fraction | Midpoint norm ratio |
|---|---:|---:|---:|---:|
| PCA | 0.0 | -0.5036 | 50.40% | 0.818 |
| PCA | 0.5 | -0.5037 | 50.44% | 0.813 |
| PCA | 1.0 | -0.5036 | 50.41% | 0.809 |
| Predictive RRR | 0.0 | -0.6942 | 54.93% | 0.745 |
| Predictive RRR | 0.5 | -0.7124 | 55.26% | 0.770 |
| Predictive RRR | 1.0 | -0.7132 | 55.11% | 0.796 |
| Tucker predictive | 0.0 | -0.7481 | 55.36% | 0.743 |
| Tucker predictive | 0.5 | -0.7664 | **55.38%** | 0.765 |
| Tucker predictive | 1.0 | **-0.7672** | 55.33% | 0.787 |

Every arm improved mean endpoint KL. Tucker produced the largest average improvement. The stronger predictive paths were nevertheless monotonic on only about 55% of individual samples. Midpoint norms contracted by about 21-26% in those arms, confirming that the unnormalized affine path does not remain on a constant-radius shell. That behavior is permitted by the governing geometry, but it must be tracked in the built module rather than normalized away.

### Disposable-flow trainability

| Method | Alpha | Validation MSE ratio, after/before | Probe-KL change | Top-1 agreement, before to after |
|---|---:|---:|---:|---:|
| PCA | 0.0 | 0.626 | -0.249 | 4.26% to 5.16% |
| PCA | 0.5 | 0.623 | -0.220 | 4.26% to 5.22% |
| PCA | 1.0 | 0.621 | -0.196 | 4.26% to 5.28% |
| Predictive RRR | 0.0 | 0.502 | **+0.357** | 5.76% to 4.92% |
| Predictive RRR | 0.5 | 0.531 | -0.168 | 5.82% to 5.28% |
| Predictive RRR | 1.0 | 0.594 | -0.229 | 5.82% to 5.58% |
| Tucker predictive | 0.0 | **0.496** | **+0.245** | 6.00% to 5.58% |
| Tucker predictive | 0.5 | 0.529 | -0.180 | 5.94% to 5.34% |
| Tucker predictive | 1.0 | 0.586 | -0.188 | 5.94% to 5.34% |

All pilots learned the canonical MSE objective. The predictive `alpha=0` arms fit canonical MSE best but **worsened** probe KL. Partial and full whitening reversed that KL failure, at the cost of a somewhat weaker MSE ratio. Even where KL improved, predictive-arm top-1 agreement declined slightly.

This is the most important 0B result. A canonical loss can improve while a decision metric worsens. Alpha and method therefore cannot be selected from canonical MSE alone, and the built-module pilots must include upper-model quality and verified acceptance as primary evidence.

## 7. Combined interpretation

### What the receipts support

1. **There is predictive teacher structure in a compact linear canonical space.** Predictive RRR and Tucker outperform PCA on every primary 0A predictive metric.
2. **Nonlinear canonicalizers are not currently justified.** The pre-stated linear-underfit trigger did not fire.
3. **The alpha grid changes optimization geometry more than static fidelity.** The gradient audit, not 0A token accuracy, supplies the useful alpha signal.
4. **A small flow can fit the canonical update.** Every arm reduced validation MSE substantially.
5. **The flow is not automatically decision-aligned.** The `alpha=0` predictive flows provide the cleanest counterexample: strong MSE convergence with worse probe KL.
6. **Per-position arbitration remains necessary.** Only about 55% of predictive paths were individually monotonic even though average KL improved.

### What remains unresolved

1. Tucker versus predictive RRR as the production canonicalizer.
2. Whether `alpha=0` should remain in the matched pilot despite its anisotropy and probe-KL regression.
3. Whether the weak per-example monotonicity requires a design change before the build or is properly delegated to the existing gate and masked-improvement loss.
4. Whether four populated slots are sufficient for v1, given the absent trace/span targets.
5. The exact lexicographic rule for selecting alpha without an arbitrary weighted composite.
6. The E1 thresholds and which parallel receipts must be complete before E1 lock.

## 8. Options for strategy review

### Canonicalizer choice

**Option A: Tucker primary, RRR fallback, PCA receipt-only baseline.**

This follows the direction of every predictive metric and minimizes the built-module matrix. The risk is over-reading a small Tucker-versus-RRR difference without paired uncertainty.

**Option B: Carry Tucker and RRR through matched module pilots.**

This gives the strongest empirical comparison but doubles the pilot matrix from three to six arms before seeds. It also mixes method selection and alpha selection in one experiment.

**Option C: Run a cheap paired DEV bootstrap or sign analysis first.**

Use the cached holdout outputs to determine whether Tucker's small edge is stable across rows and workloads. This remains post-hoc development analysis, not confirmation, but can justify reducing the module matrix before GPU training.

**Coding recommendation:** Option C if the cached row-level predictions make it cheap; otherwise Option A with predictive RRR retained as the explicit fallback. There is no evidence for nonlinear canonicalizers.

### Alpha pilot matrix

**Option A: Keep `{0, 0.5, 1.0}` exactly as planned.**

This preserves the full A35 ablation and measures whether `alpha=0`'s fast MSE fit survives the complete objective despite its probe regression.

**Option B: Retire `alpha=0` now and pilot `{0.5, 1.0}`.**

This saves one-third of pilot cost and is supported by extreme gradient anisotropy plus probe-KL worsening in both predictive methods. The cost is losing the clean no-equalization control at the stage where upper-model behavior is finally measured.

**Coding recommendation:** Keep all three. The no-equalization arm is scientifically useful, and the governing spec already anticipated that alpha cannot be selected before the complete module exists.

### Alpha-selection rule

The current prose names three criteria but not their ordering. An arbitrary weighted score would make the result hard to defend. Recommended lexicographic form:

1. Exclude any arm that violates identity, frozen-lineage, tube, or preservation assertions.
2. Among valid arms, rank by verified acceptance and upper-model quality on the matched DEV set.
3. Use flow convergence, gradient balance, and clipping burden as tie-breakers and stability diagnostics.
4. If no arm dominates within a pre-stated practical-equivalence band, retain `alpha=0.5` as the registered default rather than selecting on noise.

The equivalence bands, pilot budget, and seeds must be written before the first pilot step.

## 9. Questions for the strategy agent

1. Should Tucker be selected now, should Tucker and RRR both enter the module pilots, or should a paired cached-output analysis arbitrate first?
2. Is the approximately 55% per-example monotonic-path fraction adequate for build-only progression under the existing masked-improvement and gating design, or does it trigger a design amendment before module construction?
3. Should `alpha=0` remain in the matched full-module pilot as the no-equalization control despite its two probe-KL regressions?
4. What is the primary alpha-selection metric: verified acceptance, upper-model KL/quality, or a lexicographic combination? What practical-equivalence band prevents selecting on small DEV differences?
5. Do the four masked trace/span slots remain reserved and empty in v1, or should Stage 0A be extended before the build? The coding recommendation is to keep them masked unless a specific downstream loss requires them.
6. Should `screening_survivor` be renamed to `numerically_valid_arm` in the strategy ledger and future summaries to prevent accidental overstatement?
7. Are Stage 0B baseline mapping, V3 deployable-proxy probes, EVAL-D/E, the resource note, and the AngelSpec assessment parallel work or E1-lock blockers? The implementation spec explicitly blocks E1 on the resource note, V1d, the RMS cap, and 0A/0B, but the broader sequencing memo lists the others in parallel.
8. Does the top-1 decline under KL-improving predictive pilots require a dedicated pilot gate, or is it fully covered by verified acceptance and upper-model quality?

## 10. Recommended next sequence pending review

1. Bank this handoff and strategy's answers.
2. If authorized, run the cheap paired Tucker-versus-RRR DEV analysis from cached outputs.
3. Lock the canonicalizer carried into the build.
4. Implement the complete student module without attaching training losses.
5. Pass the full assertion battery:
   - loop cap `K <= 4`;
   - no persistent-state or target renormalization;
   - RMSNorm only on module inputs and innovations;
   - single stored effective-eigenvalue rule with no second epsilon;
   - shared PCA orientation across alpha arms;
   - nonzero bridge and flow output initialization;
   - position-zero bridge gate forced closed;
   - frozen teacher, canonicalizer, probe, and backbone hashes;
   - exact inactive-path identity and zero backbone mutation;
   - stage-C-ready control-read plumbing;
   - gradient-atlas and three-surface telemetry wiring.
6. Write and lock the matched-pilot protocol, including seeds, budget, stopping rules, equivalence bands, and lexicographic alpha selection.
7. Run matched DEV-only alpha pilots on the selected canonicalizer.
8. Return the pilot results to strategy before selecting alpha or locking E1.

The student build can be implemented without an A100. The next material GPU expense should be the matched full-module pilots, not another canonicalizer or disposable-flow sweep.

## 11. Do-not-claim boundaries

Do not claim that:

- Experiment 0A selected the final canonicalizer or alpha.
- All listed screening survivors passed an efficacy threshold.
- Sparse top-K conditional KL is full-vocabulary KL.
- Development-only probe fidelity is downstream model quality.
- Affine-path smoothness proves a trained recurrent module will follow the path.
- The disposable serial-flow pilot is an E1 result.
- Canonical MSE convergence establishes acceptance improvement.
- The all-200,000-sample refit has a held-out estimate.
- These results generalize to frozen evaluation partitions.
- The missing trace/span slots are known to be unnecessary.

## 12. Plain-language summary

The teacher's future behavior can be compressed into a small state better by supervised predictive projections than by ordinary PCA. A tiny recurrent flow can learn to move that state toward a later teacher state. That is the positive result.

The caution is equally important. The average direction is useful, but it is not useful for every example, and fitting the latent target does not always improve the model's token decision. The next architecture therefore still needs its gate, bounded writeback, direct-logit path, and acceptance checks. We have enough evidence to build that machine and test its inactive-path integrity. We do not yet have enough evidence to choose its whitening strength or start E1 training.
