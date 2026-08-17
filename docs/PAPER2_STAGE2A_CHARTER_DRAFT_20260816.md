# Paper Two Stage 2A Charter Draft: Teacher-Fingerprint Content Memory

**Date:** 2026-08-16  
**Status:** draft awaiting Mark's signature; implementation build only  
**Strategy authority:** Drive `1GhTafphPuxmfq8hDn0Y-BrjWJCmafZKu`, 11,079 bytes, SHA-256 `57c41caa6d9bfe0174f295f6b7a56634ad26718dbb868f3cac9a3d857405ba2c`  
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

- Candidate contents come only from the approximately 9,200 KP-1R reference-table rows outside DEV.
- Teacher outputs must pass the exact, locked firm-knowledge rule `V(x)` before selection.
- Selection of 8,192 slots is deterministic after admission: ascending SHA-256 rank of `seed:battery:item_id:content_sha256`.
- The selected manifest, source manifest, `V(x)` rule, and all hashes are frozen before any DEV score is computed.
- The memory manifest must prove zero overlap with the DEV panel by item, document, and content hash.

### 3.3 Geometry leakage correction

The positive teacher-fingerprint diagnostic used a 614/410 split inside DEV. That fit established the scientific premise, but its PCA and Procrustes tensors are not eligible model artifacts.

For Stage 2A, every learned or fitted coordinate transform must be rebuilt only from non-DEV reference rows, frozen, hashed, and receipted before DEV scoring. DEV contributes neither memory content nor transform fitting. Reusing the diagnostic DEV transform is a stop-class fault.

## 4. Arms

### 4.1 T3a-C: fingerprint-keyed teacher content

- Slots: 8,192.
- Fixed keys: student layer-6 fingerprints, PCA-128 from the non-DEV fit.
- Initial values: teacher layer-12 representations under the ratified transport convention.
- Retrieval: top-k maximum inner-product search over normalized fixed keys.
- Compatibility gate: learned from retrieval confidence telemetry.
- Injection: learned 128-to-hidden map and scalar zero gate at the exact registered downstream recurrent tensor.
- Seeds: 0 and 1.

### 4.2 T3b: literal n-gram memory

- Hashed causal 2/3-gram keys over prefix token IDs.
- Same effective value and injection budget as T3a-C.
- No substrate token or weight modification.
- Seeds: 0 and 1.

### 4.3 Shuffled-values control

- T3a-C architecture and fixed keys.
- Complete values permuted across slots under a recorded seed.
- Its training/freeze policy must be fixed before lock. The coding recommendation is to freeze values after permutation so the arm cannot relearn the key-content association it is intended to destroy.

### 4.4 Frozen-random-values control

- T3a-C graph with a recorded random-value initialization.
- Values remain frozen.
- Compatibility and injection behavior follow the lock-listed control policy.

## 5. Trainable and frozen surfaces

The only allowed trainable names are:

- `memory.values`
- `memory.compatibility_projection.*`
- `injection.projection`
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

The registered design calls for a bounded run near 1,000 steps, the standing optimizer and batch recipe, amplitude lottery over `[0.02, 0.11]`, EMA-primary selection, and a pinned `0.05` read. The lock must replace every approximate term with an exact value before signature:

- step count and checkpoint cadence;
- batch size and data sampling;
- optimizer, learning rate, weight decay, and parameter groups;
- EMA decay and initialization;
- resume state and random-stream contract;
- control-arm seed allocation.

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

1. both seeds are positive at their EMA endpoints under the locked net-row criterion;
2. paired support passes the locked sign-test criterion;
3. continuous margins agree in direction;
4. the shuffled-values control is flat under its locked equivalence band;
5. chi, quality, lineage, and seal contracts pass.

The exact numerical thresholds remain signature fields. They may not be inferred from DEV after training begins.

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

1. exact `V(x)` admission rule and receipt;
2. non-DEV PCA/transport fit population, direction, seed, and hash;
3. retrieval constants and control seeds;
4. shuffled-values training policy;
5. exact downstream injection tensor and timing;
6. P3.5 endpoint initialization ratification;
7. exact training and resume recipe;
8. exact positive, equivalence, sign-test, margin, and chi thresholds;
9. control-arm replication allocation;
10. whether the 2-point CONFIRM-resolvability target binds this screen or the selected successor campaign.

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

