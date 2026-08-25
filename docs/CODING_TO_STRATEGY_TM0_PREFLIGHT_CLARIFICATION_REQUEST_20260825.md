# Coding to Strategy - TM-0 Preflight Clarification Request

Date: 2026-08-25. Status: STOPPED BEFORE MODEL LOAD. Authority: TM-0 r2 plus ratified r3 execution order. No GPU session is active; no model has been loaded; no optimizer exists; no training, injection, CONFIRM scoring, or EVAL-E scoring has occurred.

## 1. What completed

The score-blind panel builder froze the exact 2,048-row DEV-2 base and selected the 4,096-row extension under seed `20260825`. The extension contains all 865 clean ARC-Challenge train rows available after exclusions and 3,231 hash-ranked clean GSM8K train rows. The resulting 6,144-row panel SHA-256 is `e108b0a92fdc69b9cb27274ac420908b65303213307f9d8dfc1f4ba73d58b5ca`; the extension SHA-256 is `5b8248430dbd3f8b08d4c02087701932959aa6e7ef2b75f5f1119c973eefd34a`. The full 1,919-row rejection ledger is retained separately and hashed.

The implementation now has a score-blind prompt-state cache contract: prompt-only inputs, last-active-token plus active-token mean pooling, exact batched-versus-sequential identity probes, content-addressed shards, and explicit no-training/seal attestations. Four focused tests pass. No teacher forward has run.

## 2. Blocking conflict: EVAL-E decontamination

The r2 charter section 3 requires every extension row to be screened against sealed CONFIRM and EVAL-E by exact and near-duplicate match, with any leak stopping the line. CONFIRM membership exists and was screened. EVAL-E has intentionally never been materialized; the repo and receipts repeatedly attest that it remains untouched. Therefore no EVAL-E membership or text index exists against which an exact or near-duplicate screen can run.

Treating ARC/GSM task rows as source-family-disjoint from the FineWeb/code EVAL-E recipe is plausible, but it is not the exact/near-duplicate screen the lock names. Materializing EVAL-E membership to create an index may itself count as forbidden contact. The coding lane will not silently substitute source-family disjointness for the registered hard gate.

Requested ruling:

1. **Preferred:** authorize a score-blind EVAL-E membership/hash-index materialization under the original frozen partition recipe, with no model, label, or score computation; screen the extension against that index and keep EVAL-E sealed for scoring.
2. **Alternative:** amend the TM-0 decontamination gate to accept source-family disjointness for this task-only extension, recording that exact/near EVAL-E overlap was not empirically tested.
3. **Otherwise:** return key `DECONTAMINATION-UNRESOLVED` and stop TM-0 before model contact.

## 3. CKA estimator scope

Exact debiased CKA across all 6,144 rows, two correlated pooling views, 76 teacher layers, and widths up to 5,120 is unnecessary for layer selection and materially expands CPU cost. The charter does not fix the CKA population size or how the two correlated views are combined.

Recommendation: freeze a 512-row battery-stratified calibration manifest before model contact; compute exact unbiased-HSIC CKA separately for last-token and mean-pooled views; select `j*` by their arithmetic-mean CKA; report both view-specific curves; perform any uncertainty calculation by resampling rows, never pooled views independently. The stitch fits and all later geometry still use the full 6,144-row panel. Requested ruling: ratify or replace this estimator before caching.

## 4. Prompt-state convention

The charter says to cache student and teacher states but does not explicitly state whether gold targets enter the state forward. Recommendation: prompt-only state inputs for every battery. Correctness uses the separately pinned reader and may inspect gold answers only for scoring. This avoids a deployment-time answer leak and prevents teacher-forced target tokens from artificially improving the stitch. Requested ruling: ratify prompt-only state caching.

## 5. Current attestations

- Model loaded: false.
- GPU provisioned: false.
- Optimizer constructed: false; optimizer steps: 0.
- Training or injection: false.
- CONFIRM scored: false.
- EVAL-E materialized or scored: false.
- Step-2: blocked.

Once these three rulings land, the coding lane can finalize the analyzer, run local tests, price the cache pass, and launch only if the measured projection remains below 1.5 A100-hours.
