# Phase DC1 Stage A Preregistration — Bridge-Only Interface Adaptation, Draft 1

Date: 2026-07-30. Strategy lane. Governing design: COMPOSITE_TRAINING_DESIGN_20260729.md (SHA `0ae848f560dda18abc89deb7716b53b24f40b49f5a7d44a6d5f2e514c9d5ed7b`), as amended by STRATEGY_TO_CODING_AGENT_DC1P_BANK_STAGE_A_PREP_20260730.md (Drive `12wp0ovgsW83FW5LQOPV7Tf8YBVRZMFfl`, SHA `8662a78d…46854`). Responds to: PAPER2_DC1_FOLLOWUPS_STRATEGY_HANDOFF_20260730.md (Drive `1FXzhzo2daFJZVnfTR-GqGVESW4oVYl2o`, SHA `c38f1e9a…f8520a`). This document locks Stage A. Training may not begin until this file is stored in the Drive research folder with recorded SHA-256 and the machine-readable lock is committed with `locked_before_training`. Every section carries a plain-language companion, marked *In plain terms*.

## 1. Question and scope

Stage A asks one question: **can a bounded, bridge-only training run make one forced scratchpad step safe?** Formally: after training only the horizontal bridge under forced k=1 with teacher cross-entropy at the appended-slot readout, is the trained model's forced-k=1 net utility on untouched EVAL-C at or above zero relative to its own k=0 baseline?

Stage A is an actuator qualification, parallel to T1-lite's role on the vertical axis. It does not test routing, dynamic k, reasoning quality, speculative-decoding speedup, persistent scratchpads, L above 1, or any teacher other than the cached 7B. A qualifying result authorizes only the drafting of the Stage B preregistration.

*In plain terms: we freeze the whole model except the one small connector that feeds the temporary thinking slot, train it briefly, and ask whether the forced thinking step stops breaking more answers than it fixes. Nothing else is being tested, and success only unlocks designing the next experiment.*

## 2. Lineage and data

- Initialization checkpoint: post-D0 EMA, SHA-256 `8245cabfe7639dcd442c19e03496623b9d59eec31b39e253e08fd6f78b1086cf`, asserted at launch and at every resume.
- Training data: DEV-C, JSONL SHA-256 `05bca2ee3ba71421296b2e31a0439746eb9c1b0e15e2cea4471be202ab6ac29d`, private manifest SHA-256 `1816d9e953280cfb335c23de80292b64e36270599c3b4d273474b25f2e476caf`, with its cached teacher pass (Qwen/Qwen2.5-7B-Instruct, revision `a09a35458c702b33eeacc393d103063234e8bc28`).
- Evaluation data: EVAL-C, frozen and unread. Its manifest SHA-256 and teacher-cache SHA-256 are transcribed verbatim from `outputs/stage5/.../eval_c/summary.json` into the JSON lock at commit time by the coding agent during reconciliation; the scoring pass asserts both at runtime. EVAL-C bytes are not read by any human or process before the single registered pass. EVAL-B is never read again, by anyone, for any purpose.
- Training seed: **0**. Any future replication uses seed 1. Bootstrap seed: **20260730**.

*In plain terms: we name the exact model we start from, the exact text we train on, and the exact untouched text we will test on, with fingerprints for each, so nothing can be quietly substituted.*

## 3. Architecture and trainable set

- Trainable set: the horizontal bridge parameters only. The complete parameter-name allowlist is enumerated in the JSON lock at commit; the launcher prints the trainable-name set and its hash at startup, and the optimizer group must cover exactly that set (hash-compared, T1-lite convention).
- Every non-bridge parameter is frozen via requires_grad False (never no_grad wrappers, per M6), with start-of-run and end-of-run weight-hash assertions proving zero drift.
- Bridge initialization: **identity at raw hidden-state scale**. This is an operational initialization, not a claim of optimality. Recorded here so it cannot later be read as an oversight (follow-ups question 1): the scale probe found 2x raw descriptively least harmful, and the initialization deliberately remains raw identity — a scalar gain is the single easiest direction for the optimizer to learn, so a genuinely better scale is reachable within early steps, whereas baking a DEV-C-derived descriptive optimum into the lock is post-hoc tuning of exactly the kind the boundaries prohibit.
- Conventions: advancing position ids (t+1); forced k=1 in training and registered evaluation; global L=1; hard-asserted k ≤ 3 invariant; recompute-only execution (no cache path in the training graph); transient eviction with the position-id, cache-length, and downstream-identity assertions of the markup addendum.
- k=0 bit-identity contract: the k=0 forward is bit-identical to the registered surgery before training and after the final step (fp32 exact-match assertion, both budgets' convention). The control-read plumbing may be present and stage-C-ready but is inactive and receives no gradient; its rows are in the frozen set.

*In plain terms: only the connector learns. Everything else is provably untouched, the model without the scratchpad stays exactly the model we registered, and the slot bookkeeping is asserted every step.*

## 4. Objective, optimization, and label-to-objective alignment

Objective: teacher cross-entropy at the appended-slot readout, forced k=1, over all DEV-C positions, against the cached 7B greedy targets.

Locked optimization constants (resource note of 2026-07-30, confirmed unmodified): A100-SXM4-80GB or equivalent; step ceiling 2,000 optimizer steps; microbatch 1, accumulation 1, effective batch 1 row; maximum sequence length 512; AdamW, learning rate 1e-4, weight decay 0.0; gradient clipping 0.5 global norm; full fp32 for model, feedback boundary, gradients, and optimizer state (RG-11 declared policy); passive checkpoints at steps 500, 1,000, 1,500, 2,000 under the standing stage-checkpoint manifest policy; **final-step raw weights primary**. No DEV-C-guided early stopping, no DEV-C-guided hyperparameter selection, no EMA primary.

**Label-to-objective alignment** (mandatory section, standing policy): the deployment question is forced-append safety — does the slot readout stop destroying correct predictions when it cannot improve them. The CE objective's optimum contains the fallback policy, because the fed state carries the k=0 prediction through the interface: on every position where extra computation cannot help, reproducing the k=0 prediction is available and loss-optimal, and on positions where it can help, improvement is loss-optimal. Every labeled position is therefore causally actionable under the objective. This is the alignment D0's disagreement label lacked, and no position's label asks the model to do something the substrate cannot do.

*In plain terms: the training signal is simply "predict what the big teacher predicts, from the thinking slot." The perfect solution includes "when thinking doesn't help, just repeat your original answer" — which the slot has the information to do — so the training goal and the safety goal are the same goal.*

## 5. The single registered EVAL-C pass

One scoring pass, run once, after the final training step. The pass reads EVAL-C and its teacher cache into an immutable scoring cache, written once and hashed; all arms are scored from that cache; the verdict script runs against it; no arm-specific rescoring occurs afterward (follow-ups question 6 confirmed — this is the registered meaning of read-once).

Arms scored in the one pass:

1. Registered k=0 baseline (full-sequence path).
2. Trained append, forced k=1 (final-step raw weights).
3. Untrained append, forced k=1 (initialization bridge — same-partition anchor).
4. In-place forced depth 2 — descriptive anchor.
5. In-place forced depth 3 — descriptive anchor.
6. Code/general strata and the matched-layer descriptive table for all of the above.

Metric: the helps/hurts/neutral transition decomposition against the cached 7B teacher, with net utility = helps − hurts. Tie policy: deterministic argmax, lowest token id, fixed fp32 logit space, tie cells flagged (standing policy). Execution-path anchor disagreements, if any, are disclosed with the DC0/DC1-P convention.

**Verdict statistics, frozen here** (follow-ups questions 4 and 5): the decision statistic is the normalized net utility u = (helps − hurts) / scored positions, for trained append versus the registered k=0 baseline. Row-cluster bootstrap: resample EVAL-C source rows with replacement, 10,000 replicates, seed 20260730, computing u per replicate; the 95 percent interval is the 2.5th and 97.5th percentiles. The confidence floor applies directly to the normalized statistic: CI lower bound ≥ −0.0025. The verdict computation is a committed script named in the JSON lock, runnable by anyone against the immutable cache.

*In plain terms: the untouched test text is read exactly once, all comparisons come from that one reading, and the pass/fail arithmetic is a script written and frozen now — before anyone has seen the test data — so the verdict cannot be argued with after the fact.*

## 6. Locked decision mapping

- **Qualifies:** trained-append point estimate u ≥ 0 and bootstrap 95 percent CI lower bound ≥ −0.0025. Consequence: the actuator is safe under forced application; the Stage B preregistration drafts. No other claim is licensed.
- **Partial domestication:** hurts ≤ 50 percent of same-partition untrained-append hurts, helps ≥ untrained-append helps, and u < 0. Consequence: the decision point weighs exactly one further bounded round (wider trainable set or longer budget, one amendment) against reversion; Mark decides.
- **No material improvement:** hurts reduction versus untrained append < 50 percent. Consequence: transient append retires on this substrate; the program returns to the in-place lane with the parity-ledger result in hand (no free wins there either — that decision point is convened fresh).
- In-place depth-2 and depth-3 arms are **descriptive anchors, never qualification gates** (both checkpoints' in-place behavior is net harmful per the parity ledger, so "gentler than in-place" carries no qualification weight).
- A blocked, aborted, or infrastructure-failed run writes receipts and does **not** consume the registered attempt; a completed run that reaches the EVAL-C pass consumes it regardless of outcome.

*In plain terms: three outcomes are defined in advance — it worked, it half-worked, it didn't work — each with its consequence already chosen. A crashed run doesn't count against us; a completed run counts whatever it says.*

## 7. Do-not-claim boundaries

All follow-ups handoff boundaries carry forward verbatim, including: append is not safe, useful, or accuracy-positive until the registered pass says so within its bands; raw scale is not optimal; the U-shaped trough has no established mechanism; D0 did not create the in-place harm asymmetry; parity results do not generalize beyond the tested checkpoints and the DEV-C diagnostic; DEV-C numbers are never registered headline evidence; no Stage B/C/D, policy training, persistent scratchpad, RG-12, GRAM, or width work. Additionally: a qualifying Stage A result licenses only "the trained bridge made one forced latent step non-destructive on this substrate" — not that the composite reasons, compresses, or accelerates anything.

## 8. Amendment policy

Amendment before the Drive lock is free and versioned in this document. After lock and before launch, amendments require a strategy-signed addendum with its own SHA. After launch, no amendment touches the objective, data, bands, or verdict script; only infrastructure repairs that provably do not change the computation are permitted, each with a receipt.

## Appendix A. Machine-readable lock (committed as `stage_a_prereg.json` alongside this document)

```json
{
  "phase": "DC1_STAGE_A",
  "version": "draft1_20260730",
  "locked_before_training": true,
  "init_checkpoint_sha256": "8245cabfe7639dcd442c19e03496623b9d59eec31b39e253e08fd6f78b1086cf",
  "train_partition": {
    "name": "DEV-C",
    "jsonl_sha256": "05bca2ee3ba71421296b2e31a0439746eb9c1b0e15e2cea4471be202ab6ac29d",
    "manifest_sha256": "1816d9e953280cfb335c23de80292b64e36270599c3b4d273474b25f2e476caf"
  },
  "eval_partition": {
    "name": "EVAL-C",
    "manifest_sha256": "TRANSCRIBE_FROM_EVAL_C_RECEIPT_AT_COMMIT",
    "teacher_cache_sha256": "TRANSCRIBE_FROM_EVAL_C_RECEIPT_AT_COMMIT",
    "read_once": true,
    "immutable_cache": true
  },
  "teacher": {"model": "Qwen/Qwen2.5-7B-Instruct", "revision": "a09a35458c702b33eeacc393d103063234e8bc28"},
  "trainable": {"set": "horizontal_bridge_only", "allowlist": "ENUMERATE_AT_COMMIT", "frozen_hash_assertions": true},
  "bridge_init": {"type": "identity", "scale": "raw_hidden_state", "status": "operational_not_optimal"},
  "mechanics": {"k_forced": 1, "k_cap": 3, "L": 1, "position_ids": "advancing", "execution": "recompute_only", "eviction": "transient_with_assertions", "k0_bit_identity": "before_and_after"},
  "optimization": {"steps": 2000, "microbatch": 1, "accumulation": 1, "seq_len": 512, "optimizer": "AdamW", "lr": 1e-4, "weight_decay": 0.0, "grad_clip": 0.5, "precision": "full_fp32", "seed": 0, "checkpoints_passive": [500, 1000, 1500, 2000], "primary": "final_step_raw", "early_stopping": false},
  "objective": "teacher_ce_at_slot_readout",
  "evaluation": {
    "arms": ["k0_registered", "trained_append_k1", "untrained_append_k1", "inplace_depth2_descriptive", "inplace_depth3_descriptive"],
    "strata": ["code", "general"],
    "tie_policy": "fp32_argmax_lowest_token_id",
    "verdict_statistic": "net_utility_per_scored_position",
    "bootstrap": {"cluster": "eval_c_source_row", "replicates": 10000, "seed": 20260730, "ci": 0.95},
    "verdict_script": "NAME_AT_COMMIT"
  },
  "bands": {
    "qualifies": {"u_point_min": 0.0, "u_ci_lower_min": -0.0025},
    "partial_domestication": {"hurts_vs_untrained_max_ratio": 0.5, "helps_vs_untrained_min_ratio": 1.0, "u_point_max_exclusive": 0.0},
    "no_material_improvement": {"hurts_reduction_below": 0.5}
  },
  "consequences": {"qualifies": "stage_b_prereg_drafts", "partial": "one_bounded_round_or_reversion_mark_decides", "none": "transient_append_retires"}
}
```

*The two TRANSCRIBE fields and the allowlist/script names are filled by the coding agent from existing receipts during reconciliation, before the Drive lock — they reference artifacts that already exist and are hash-verified at runtime, so no degree of freedom is introduced.*
