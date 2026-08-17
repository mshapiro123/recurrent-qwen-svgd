# Paper Two Stage 2A Charter Draft: Teacher-Fingerprint Content Memory

**Date:** 2026-08-16  
**Status:** draft awaiting Mark's signature; implementation build only  
**Strategy authority:** Drive `1GhTafphPuxmfq8hDn0Y-BrjWJCmafZKu`, 11,079 bytes, SHA-256 `57c41caa6d9bfe0174f295f6b7a56634ad26718dbb868f3cac9a3d857405ba2c`  
**Lock rulings:** Drive `1UOfWqVpV4ByPRVeTbsabgaID-6LohmkF`, 8,972 bytes, SHA-256 `445e77a089148d58910acabf1e277600e966068068300df26e11e02b12c98c73`
**Machine companion:** `training/paper2_stage2a_preregistration.draft.json`  
**Training:** prohibited until all signature fields are bound and the three authorization flags flip together

## 0. Program position

Stage 2A begins from two separately ratified findings.

1. KP-1R returned `NO POSITIVE KNOWLEDGE-PRESENCE GATE`. This means the registered readout did not establish recoverable answer content. It must never be translated into "knowledge absent."
2. Teacher-fingerprint retrieval was strongly positive and replicated. Student layer-6 states retrieved matching teacher layer-12 items at 40.7% and 42.2% top 1 against 0.244% chance after a split-fit coordinate transport.

The resulting design premise is: **use the student's early state as an address for teacher-derived content, then deliver the retrieved content into the recurrent pathway downstream.** High item retrieval is not answer recovery. Stage 2A tests the missing causal step.

## 1. Primary question

Does a static teacher-derived content memory, addressed by fixed early-state student fingerprints and attached through an exactly inert downstream injection, improve task accuracy and answer-token margins on the frozen DEV panel relative to the paired no-memory system?

This phase tests content delivery before an expert bank exists. Expert access remains a secondary Stage 2A program and is not evidence that content was already latent in the student.

## 2. Hypotheses

### H1: fingerprint-keyed content is useful

T3a-C improves paired net rows and continuous answer-token margins over the no-memory baseline on both registered seeds.

### H2: the gain depends on retrieval specificity

The shuffled-values arm remains flat under the preregistered equivalence band. A gain shared by T3a-C and shuffled values is an injected-energy effect, not evidence for fingerprint-addressed content.

### H3: the fingerprint address beats surface lookup

T3a-C exceeds or more consistently improves upon the parameter-matched T3b literal 2/3-gram arm. T3b is a real memory baseline, not a null control.

### H4: the write remains selective and bounded

The registered chi audit, preservation metrics, and battery decomposition remain within their locked bounds. A task gain that violates those bounds is not a clean positive.

## 3. Data and seal contract

### 3.1 Evaluation

- The only scored partition in Stage 2A is the existing frozen 1,024-row DEV panel, SHA-256 `c0e15a890b598544059ac337cc475123f97c05e3c1626febcdee1c6d8fe02615`.
- CONFIRM and EVAL-E remain sealed.
- All task comparisons are paired row by row against the no-memory baseline under the same reader and inference graph.

### 3.2 Memory population

- Candidate contents come only from the 8,712 `verified_train` rows outside DEV. The broader 9,207-row outside-panel count includes 495 DEV rows that the lock ruling now prohibits from fit, keys, and contents.
- Teacher outputs must pass `V(x)`: the pinned 14B teacher must be correct under the registered reader and the pinned 32B verifier must return the same normalized final answer. No confidence threshold is used.
- Slot selection is deterministic after admission: ascending SHA-256 rank of `seed:battery:item_id:content_sha256`.
- The selected manifest, source manifest, `V(x)` rule, and all hashes are frozen before any DEV score is computed.
- The memory manifest must prove zero overlap with the DEV panel by item, document, and content hash.

### 3.3 Geometry leakage correction

The positive teacher-fingerprint diagnostic used a 614/410 split inside DEV. That fit established the scientific premise, but its PCA and Procrustes tensors are not eligible model artifacts.

For Stage 2A, every learned or fitted coordinate transform must be rebuilt only from non-DEV reference rows, frozen, hashed, and receipted before DEV scoring. DEV contributes neither memory content nor transform fitting. Reusing the diagnostic DEV transform is a stop-class fault.

## 4. Arms

### 4.1 T3a-C: fingerprint-keyed teacher content

- Slots: 8,192.
- Fixed keys: student layer-6 fingerprints, PCA-128 from the non-DEV fit.
- Initial values: teacher layer-12 PCA-128 representations kept in teacher coordinates. The Procrustes map is diagnostic only and is absent from the live path.
- Retrieval: top-8 maximum inner-product search over normalized fixed keys at temperature 0.07.
- Compatibility gate: learned from retrieval confidence telemetry.
- Injection: once per emitted token, immediately after `ScratchpadInitializer` returns `S0` and before flow step 1: `S0 <- S0 + g_L * w_L outer (W_L m)`. Here `W_L` maps 128 teacher-PCA dimensions to the 128-dimensional scratch state, `w_L` is an eight-slot nonnegative `2 * sigmoid` write vector, and `g_L` is exactly zero at attachment. Memory reaches hidden states only through scratchpad -> flow -> `AnchoredBridge` -> hidden; no new substrate writer exists.
- Seeds: 0 and 1.

### 4.2 T3b: literal n-gram memory

- Hashed causal 2/3-gram keys over prefix token IDs.
- Same effective value and injection budget as T3a-C.
- No substrate token or weight modification.
- Seeds: 0 and 1.

### 4.3 Shuffled-values control

- T3a-C architecture and fixed keys.
- Complete values permuted across slots under a recorded seed.
- Values remain frozen after permutation so the arm cannot relearn the key-content association it is intended to destroy. Its compatibility gate, injection map, slot weights, and injection gate still train.

### 4.4 Frozen-random-values control

- T3a-C graph with a recorded random-value initialization.
- Values remain frozen.
- Its compatibility gate, injection map, slot weights, and injection gate still train.
- Both honesty controls run on seed 0. A control exceeding +3 paired rows triggers seed 1 before interpretation.

## 5. Trainable and frozen surfaces

The only allowed trainable names are:

- `memory.values`
- `memory.compatibility_projection.*`
- `injection.projection`
- `injection.slot_logits`
- `injection.gate`

The substrate, existing sidecar, memory keys, PCA/transport tensors, and query transform are frozen. A startup allowlist assertion and end-of-run tensor digests enforce the boundary.

## 6. Identity and causal contracts

Before any optimizer is constructed:

1. Attach each arm with its injection gate exactly zero.
2. Run the paired inference path against the no-memory source checkpoint.
3. Require bit-exact task logits, requested/executed loop accounting, and unchanged substrate/sidecar hashes.
4. Open the gate under a synthetic nonzero memory value and require a nonzero downstream delta at the named injection tensor. This is a positive control, not a task score.

The zero-gate test establishes inert attachment. The positive control establishes a live write path. Both are required.

## 7. Training recipe

The registered run is 1,200 steps at batch 128 with AdamW, learning rate `5e-4`, weight decay `0.01`, betas `(0.9, 0.999)`, and 50 warmup steps. The learning rate cosine-decays over the final 120 steps. EMA decay is `0.999`; EMA is primary and raw is secondary. Checkpoints and looks occur every 200 steps. Training uses the amplitude lottery over `[0.02, 0.11]`; the registered read is `0.05` and endpoint-only `0.08` is telemetry. Every checkpoint stores the exact optimizer, EMA, data cursor, generator, and CPU/CUDA RNG states needed for bit-identical resume.

The objective is bound by `STRATEGY_T3A_OBJECTIVE_BINDING_20260817.md` (Drive `1-2iiv8aaTrBvUR2Zxs4V6BW1P8OLotb_`, 4,821 bytes, SHA-256 `78cbf2fb397cf2c6319636523a7feea44b1e21e8941ee32e898323e697f18a22`). It is `L = 0.5 * L_CE + 0.5 * L_KL`. `L_KL` is forward teacher-to-student KL at temperature 1.0 over the cached top-128 14B teacher lattice, renormalized on that lattice. `L_CE` targets the teacher token. Both losses apply only at KP-1R-repaired answer-bearing positions; prompt positions, formatting-only tokens, and position zero are excluded. Reduction is the mean over unmasked positions within each example, then the mean over the batch.

Every teacher-forced loss position runs the deployment graph at K=4, with top-k 8 fingerprint retrieval at temperature 0.07. A fingerprint training row's own memory slot is excluded before top-k. The literal n-gram arm has no row-owned slot, so this exclusion is vacuous for that control rather than silently omitted. A 512-row admitted non-DEV validation split monitors training. DEV is used only for the registered 200-step looks and EMA endpoint.

No alpha, rank, slot-count, or amplitude sweep is authorized by this charter.

## 8. Evaluation and decision rules

At every registered read, retain exact row predictions and report:

- correct counts and paired fix/regression tables against no memory;
- paired sign test;
- mean and row-minimum answer-token margins, pooled and by battery;
- retrieval top-k slot IDs, scores, weights, entropy, and compatibility gate;
- retrieval-hit statistics where a true content identity is defined;
- chi/selectivity audit at amplitude `0.05`;
- gate and write amplitudes;
- frozen-tensor hashes and loop accounting.

A positive T3a-C reading requires all of the following:

1. mean two-seed paired delta is at least +8 rows and each seed is strictly positive at its EMA endpoint;
2. the one-sided paired sign test passes at alpha 0.05;
3. continuous margins agree in direction;
4. the shuffled-values control lies within the inclusive -3 to +3 row equivalence band;
5. the standing confident-agreement collateral flip fraction chi is at most 0.05% at amplitude 0.05, and quality, lineage, and seal contracts pass.

T3b is compared descriptively and under its preregistered paired rule. The frozen-random arm is an honesty control. Neither may be selected post hoc as the headline arm.

## 9. Interpretation branches

### Branch A: fingerprint-specific positive

T3a-C passes, shuffled values remain flat, and controls remain healthy. Interpretation: teacher-derived content can be addressed by early student fingerprints and causally improve the recurrent system under this bounded memory interface.

### Branch B: generic memory positive

T3a-C and T3b improve similarly while shuffled values remain flat. Interpretation: external content helps, but the fingerprint address has not established an advantage over literal lookup.

### Branch C: injected-energy effect

T3a-C and shuffled values both improve. Interpretation: the gain is not attributable to correct key-content pairing. The retrieval-specificity claim fails.

### Branch D: safe null

All active memories remain within safety bounds but task effects are flat. Interpretation: the fingerprint map does not by itself make the teacher representation useful through this value/injection contract.

### Branch E: unsafe or unhealthy

Identity, lineage, quality, chi, or seal contracts fail. Stop with receipts. Do not interpret task effects until the instrument is repaired under an amendment.

## 10. Do-not-claim boundaries

- Do not say knowledge is absent.
- Do not treat item retrieval as answer recovery.
- Do not call teacher-derived memory internalized knowledge.
- Do not reuse the DEV-fitted diagnostic transform.
- Do not describe a shuffled-control gain as retrieval specificity.
- Do not spend CONFIRM or EVAL-E in Stage 2A.
- Do not construct an optimizer from this draft.

## 11. Signature blockers

The machine companion carries the authoritative blocker list. The substantive open decisions are:

1. resolve the slot-count contradiction: after reserving the required 512 admitted non-DEV validation rows, at most 5,332 rows remain before the required 32B concurrence filter, so 8,192 one-row-per-slot entries are impossible;
2. bind and hash a deterministic, battery-stratified 512-row admitted non-DEV validation split that has zero overlap with memory;
3. materialize and hash the non-DEV `V(x)` source and amended matched-slot manifests;
4. fit the non-DEV-only geometry, score held-out non-DEV retrieval, and bind its manifest and artifact hashes;
5. Mark signs the fully materialized executed lock.

The eventual Stage 2A main campaign selected from this screen, not T3a itself, must be designed to a CONFIRM-resolvable target of at least 2 absolute points (approximately 21 of 1,024 DEV rows) with power arithmetic in its own lock.

## 12. Execution order after signature

1. Freeze and hash the non-DEV source, admission, and selected memory manifests.
2. Fit and hash the non-DEV-only PCA/transport artifact.
3. Build fixed keys and initial values; produce data-separation receipts.
4. Run zero-gate identity, positive-control, allowlist, lineage, and seal assertions.
5. Train only the signed arm matrix.
6. Score DEV under the registered reads and write row-level receipts.
7. Apply the scripted branch; do not select an arm after inspection.
8. Prepare the standard strategy handoff. Keep CONFIRM and EVAL-E sealed.

---

**Draft authorization line:** Stage 2A loss-free implementation, data-contract code, tests, and lock assembly are authorized. Training is not authorized. Mark's signature on a fully bound executed lock is required before the first optimizer is constructed.
