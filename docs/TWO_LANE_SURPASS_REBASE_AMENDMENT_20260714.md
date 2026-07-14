# Two-Lane Surpass and Width Re-base Amendment

**Date:** July 14, 2026
**Status:** Amendment of record for the Phase A comparison, inverse-composition staircase, and Phase G-alpha queue

## 1. Decision

The program now proceeds in two coordinated lanes:

1. **Width lane:** formalize the synthetic Phase A comparison, establish deterministic validity on an inverse-rendered non-injective task, and then run the first attributable guided stochastic-width experiment.
2. **Curriculum-science lane:** determine why a non-native inverse transition did not install under the matched staircase, repair retention, measure position transfer, and rerun the canonical forward-table arm under a corrected curriculum.

The lanes answer different questions and must not be conflated. The inverse-rendered task can provide a valid multimodal coverage assay without claiming that canonical backward search has been solved. The canonical forward-table task remains the stronger scientific target for non-native operation installation.

F9, the multi-channel bridge precursor battery, remains diagnostic-only. Its architecture gate requires a priced staircase reading, and the staircase stopped without a finite arm-F dose-to-bar estimate. No F9 outcome authorizes an architecture change in the current queue.

## 2. Verified evidence and claim boundary

### 2.1 Phase A result

All four systems were evaluated on the same frozen depth-1-through-14 set, with 128 rows per depth and the same final-symbol answer space.

| Arm | Description | Correct | Accuracy | Depths 11-14 |
|---|---|---:|---:|---:|
| A | Recurrent 0.5B system | 1506/1792 | 84.04% | 272/512, 53.13% |
| B | Dense 0.5B direct SFT, step 4000 | 470/1792 | 26.23% | 60/512, 11.72% |
| C | Dense 0.5B serialized scratchpad SFT, step 4000 | 952/1792 | 53.13% | 56/512, 10.94% |
| D | Dense 1.5B direct SFT, step 4000 | 322/1792 | 17.97% | 58/512, 11.33% |

The recurrent system exceeds the strongest evaluated dense 0.5B control by 554 rows, or 30.92 percentage points. Its advantage is especially large beyond depth 10.

This supports the following claim:

> On this frozen synthetic composition family, the trained recurrent 0.5B system outperforms the evaluated dense 0.5B direct and serialized-scratchpad recipes, with the largest advantage beyond the scratchpad control's observed depth horizon.

It does **not** yet support any of these stronger claims:

- superiority on natural reasoning benchmarks;
- superiority at matched total training lineage, tokens, optimizer updates, or FLOPs;
- a causal claim that recurrence alone produced the difference;
- superiority to a tuned 1.5B system;
- a general frontier-model or cross-scale claim.

Arm A and arm C are matched on base-model scale and frozen evaluation rows, not on complete training history. The result is meaningful system-level evidence, but the architecture, supervision, and training histories differ.

### 2.2 Statistical status

The original preregistered primary gate was **A over B** at three or more consecutive depths using a one-sided Fisher exact test with `p < 0.05` at each depth. On the landed counts, A clears that gate at all 14 depths.

A also exceeds C at all 14 depths under the same conservative count-based Fisher calculation. That is the scientifically stronger control comparison, but it is an analysis extension rather than the original primary gate and must be labeled accordingly.

The current Fisher helper treats the two depth-wise count tables as independent samples. Because the systems were evaluated on identical row IDs, the final receipt must also report a paired row-level test, such as an exact McNemar or sign test, as a secondary analysis. It must not describe the Fisher test itself as paired.

The preregistered primary dense checkpoints are the step-4000 checkpoints. Step 2000 remains a useful efficiency and saturation secondary:

- B gained 6 net rows from step 2000 to 4000 (`p=0.771` paired sign test);
- C gained 22 net rows (`p=0.00319`);
- D lost 28 net rows (`p=0.161`).

Step 2000 must not replace C step 4000 as the primary comparator after seeing the results.

### 2.3 Staircase result

The inverse-table/forward-lookup arm C reached:

- cap 2: `62/64`;
- cap 3: `63/64`.

The cap-3 run failed the existing synthetic retention guardrail at `26/32 = 0.8125`, below the `0.93` floor. Cap 4 correctly did not run.

The forward-table/reverse-search arm F reached `3/64` at cap 2 after its first inverse transition reached `55/64`. The correct landed verdict is `experiment_stalled_at_matched_dose`; no finite F-to-C dose ratio was observed.

The historical recovery's held-out loop-2 result, `42/112`, is real and much higher than arm F's `3/64`. However, the two runs differ in initialization and curriculum. Their performance contrast does not identify a unique cause. Primitive consolidation and missing rehearsal are therefore **causal hypotheses to test**, not established explanations.

## 3. Revised dependency graph

```text
Current F9 battery -> diagnostic scoring only

Width lane:
Phase A formal receipt
  -> inverse-rendered non-injective frozen-set construction
  -> zero-shot deterministic validity on C cap 3
  -> optional bounded deterministic tune if authorized
  -> deterministic validity + retention gate
  -> G-alpha K=1 parity
  -> G-alpha coverage and iso-compute comparisons

Curriculum-science lane:
C cap-3 replay with rehearsal
  -> position-transfer micro-test
  -> corrected primitive-first arm F
  -> seed-swapped C/F publication comparison
```

Parallelize within a decision level, never across a gate. CPU-only Phase A receipt work and frozen-set construction may proceed while F9 or a retention run uses an L4. G-alpha training cannot launch before the re-based deterministic gate passes.

## 4. Lane 1: width program

### W0. Score F9 as diagnostics

Complete M1-M3 against their registered random-partition controls. Record the result in the mechanism section. Regardless of outcome, leave F9 banked because the staircase did not produce reading one.

### W1. Produce the Phase A surpass receipt

Create one durable machine-readable receipt and one figure with curves A, B, C, and D over depths 1-14.

Required receipt fields:

- exact checkpoint paths and SHA256 values, resolving any currently missing arm-A hash;
- frozen-evaluation row and row-ID hashes;
- per-depth numerators and denominators;
- pooled and depth-11-through-14 summaries;
- the original A-over-B Fisher gate;
- the A-over-C extension, explicitly labeled;
- paired row-level A-versus-B and A-versus-C tests;
- arm-C step-2000/step-4000 saturation comparison;
- context-growth, generated-token, and sequential-decode ledger by depth;
- claim-boundary text from section 2.1.

The paper's synthetic surpass section may draft only after this receipt is written.

### W2. Build the inverse-rendered non-injective validity assay

This is a re-base of the deterministic prerequisite, not a claim that canonical backward search is solved.

Construction:

- `N=24` arbitrary non-bijective mappings;
- inverse relation explicitly rendered in the prompt;
- depths 1-4;
- exact predecessor chains retained in the manifest;
- disjoint calibration and test splits;
- three exact-preimage strata per split:
  - unique: exactly 1;
  - small: 2-4;
  - large: 5 or more;
- 128 rows per stratum, balanced at 32 rows per depth within each stratum;
- 384 rows total per split and 96 rows per depth;
- manifest and row-ID hashes frozen before model evaluation.

The unique stratum is a competence control. Coverage claims are made on the small and large multimodal strata.

Validity means the emitted chain is verified against the rendered inverse relation and terminates in a member of the exact valid-preimage set. Exact-match to one arbitrarily selected chain is not the validity metric.

### W3. Zero-shot deterministic gate

Evaluate the exact C cap-3 checkpoint, SHA `83767ebf...9ac5`, on the frozen calibration split before any further training.

Planning gates, converted to integer gates by the fixed construction above:

- pooled validity: at least `288/384` (75%);
- every depth: at least `58/96` (60%);
- report every preimage stratum separately;
- no training occurs during this readout.

If this passes, proceed directly to the test split and the retention check. If it is materially above a preregistered chance/null baseline but below gate, one bounded deterministic tune may be designed after review. A zero-shot failure near the null returns the task design to review rather than automatically spending dose.

### W4. Optional bounded deterministic tune

This stage is conditional on the W3 review. Before launch, lock:

- one initialization checkpoint and SHA;
- injective/non-injective row balance;
- one forward-rehearsal fraction, informed by lane C1 rather than selected post hoc;
- exact dose in weighted active labels per loop;
- canary cadence and hard stops;
- the same W3 validity gates on a disjoint test split;
- synthetic retention of at least `30/32` at every guarded depth, the operational integer form of the 0.93 floor.

Only one bounded tune is authorized without another strategy review.

### W5. G-alpha

On a green deterministic validity and retention gate:

1. Freeze the keeper block and assert zero frozen gradients on every backward pass.
2. Train only the conditional prior, target-conditioned posterior, and injection scale.
3. Require K=1 parity before interpreting stochastic width.
4. Compare latent-K against entropy-matched answer-head K on identical frozen rows.
5. Compare K trajectories at depth T against one trajectory at depth K*T under the locked compute ledger.
6. Score exact distinct-valid coverage on small and large preimage strata.

The claim is guided stochastic branching on an explicitly rendered inverse relation. Canonical forward-table abduction remains a separate, stronger target.

G-beta, LPRM selection, per-trajectory learned halting, and SVGD remain closed unless latent-K improves coverage over both locked comparators.

## 5. Lane 2: curriculum science

### C1. Cap-3 retention replay

Continue from exact C cap-3 under one predeclared rehearsal mix, preferably 25% forward synthetic unless the pre-run power/dose audit requires another fixed value. Hold task rows, optimizer, dose, and evaluation constant.

Primary question: does rehearsal restore the synthetic guardrail to at least `30/32` at every guarded depth while preserving cap-3 task accuracy at or above the existing bar (`46/64`)?

This is the causal test of the rehearsal hypothesis. A pass authorizes cap 4 once under the same mix. A failure means rehearsal alone does not explain the retention loss.

### C2. Position-transfer micro-test

Train the canonical inverse primitive at loop position 1 only until it reaches `46/64` on held-out primitive rows. Freeze the checkpoint, then evaluate the same primitive without updates at positions 2, 3, and 4 by forcing the readout position.

Required controls:

- identical held-out mapping rows at each position;
- unchanged prompt surface and answer reader;
- fixed loop count with no learned routing;
- per-position random/chance baseline;
- a no-training identity receipt before the forced-position intervention.

Interpretation:

- strong zero-shot transfer: the failure is not a per-position primitive repurchase problem;
- monotonic transfer decay: position-specific installation cost is supported;
- no position-1 generalization: the primitive itself remains unconsolidated and arm F must not advance.

### C3. Corrected arm F

Only after C1 and C2 are read:

1. train the inverse primitive to the cap-1 bar with the locked rehearsal mix;
2. save and hash the consolidated primitive checkpoint;
3. open cap 2 under the existing equalized-mass accounting;
4. stop on task, retention, or canary failure;
5. continue to later caps only under the existing stage gates.

The current F result is retained as a valid result for its tested curriculum, but not as the publication estimate of direction cost.

### C4. Publication comparison

Run the corrected C and F designs with swapped seeds and identical dose accounting. A direction-effect statement requires agreement across both seed assignments and green retention. Report labels-to-bar ratios only when both arms actually reach bar; otherwise report a bounded stall without inventing a finite ratio.

## 6. Resource and review schedule

CPU, immediately:

- Phase A receipt and figure;
- inverse-rendered frozen-set generator and exact scorer;
- G-alpha spec amendment and tests;
- ledger and paper claim-boundary updates.

L4 slot 1 after F9:

- C1 retention replay.

L4 slot 2, parallel with C1 if available:

- W3 zero-shot inverse-rendered validity;
- then C2 position-transfer micro-test.

No A100 is required for the current two-lane gates. Reserve larger GPUs for an explicitly authorized scale control or later G-alpha scaling, not for the deterministic re-base.

At most two result streams should land in one review sitting. W3 and C1 form the first paired review because together they determine whether width can open cheaply and whether rehearsal repairs retention.

## 7. Ledger amendments

Record these findings:

- Phase A: recurrent A exceeds dense B and C on the frozen synthetic family; claim remains system-level and synthetic.
- Staircase: C composes through cap 3, but cap-3 retention fails; F stalls at cap 2 under the tested curriculum.
- Historical F-versus-recovery contrast: performance gap verified; causal attribution unresolved.
- F9: diagnostic-only and banked until a priced staircase reading exists.
- Width re-base: inverse-rendered non-injective validity becomes the G-alpha prerequisite; canonical forward-table search remains lane 2.

Record these falsified or corrected assumptions:

- the original locked Phase A primary did not include A over C;
- Fisher count tests are not paired merely because row IDs match;
- step 2000 cannot replace preregistered step 4000 as the primary dense comparator;
- the F-versus-recovery contrast does not by itself prove primitive simultaneity or missing rehearsal caused the stall.

## 8. Next decision points

1. **W3 green, C1 green:** open G-alpha preparation and cap 4 in parallel.
2. **W3 green, C1 red:** G-alpha may proceed from a keeper that independently passes its own retention gate; canonical curriculum science continues separately.
3. **W3 red but above null:** review one bounded deterministic tune after C1 fixes the rehearsal choice.
4. **W3 near null:** re-base design review; do not train stochastic heads.
5. **C2 shows position transfer:** redesign F around primitive consolidation but do not assume per-position repurchase.
6. **C2 shows position decay:** use the measured curve to set the corrected F curriculum and dose.

This amendment controls the queue until one of these decision points lands.
