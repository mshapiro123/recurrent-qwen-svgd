# Coding Handoff to Strategy — W2′ Authority Clarifications Before Phase D Fits

Date: 2026-08-25. Responds to: `STRATEGY_BICAMERAL_W2P_CONDITIONAL_MIXER_CHARTER_20260825.md`, Drive `1jfIkThIq_ts5_oxS_Rck-sTiQ6El4bvd`, SHA-256 `f89b45ef100fa46536dd93a3ef936aa8c9cfa1fc624b401b4bfc0d2b50bc2aa4`.

## 1. Status

The target-independent W2′ implementation is built and its local contract tests pass. The D4 prompt-only cache is safe to run because it does not select or score a correction target. D1–D3 and Phase G remain blocked on two authority conflicts. No optimizer has been constructed, no sealed partition has been touched, and no desk outcome has been observed.

## 2. Target-symbol conflict

Charter §2 registers `c*_c = L0c teacher-forced correction delta`. The banked W1 extractor defines the families differently:

- `L0a = -∇ CE` at answer-bearing positions.
- `L0c = ∇ margin` at answer-bearing positions.
- `L0d = h_gold - h_free`, the teacher-forced correction-state delta.

The symbol and description therefore identify different banked tensors. Choosing silently would alter the registered secondary estimand and could change the `TARGET-FAMILY-SPLIT` branch.

**Recommendation:** bind the semantic definition and amend the secondary family to banked `L0d`. The primary `L0a` target is unaffected. Do not add `L0c` as a third scored family in this confirmatory desk gate; it can remain a separately named exploratory diagnostic only if strategy explicitly authorizes it before D1–D3.

**Requested ruling R-W2P-1:** select exactly one registered secondary target: banked `L0c` margin gradient or banked `L0d` teacher-forced state delta.

## 3. FS-2 leak-boundary conflict

Charter §2 registers FS-2 as `(m, d, trajectory statistics exactly as in the W3 desk fit)` while also binding a strict deployment-input boundary: no gold answer, teacher forward, or oracle routing at fit or score time. The banked W3 trajectory cache was produced from `prompt + gold target` forced sequences. It therefore fails the charter's own deployment-feature provenance contract.

Recomputing prompt-only trajectory features now would not be “exactly as in W3”: it would change the input sequence and potentially the source graph. Reusing the cache would leak gold-conditioned state. Either silent choice would violate the lock.

**Recommendation:** withdraw FS-2 from W2′ and gate this wave on FS-1 only. Record FS-2 as `BLOCKED_SOURCE_CONFLICT`, not as a failed model. If prompt-only trajectory features remain valuable, specify and build them prospectively as a new feature set in a later amendment before viewing W2′ desk outcomes.

**Requested ruling R-W2P-2:** either (a) withdraw FS-2 for this wave, recommended, or (b) provide an exact prospective prompt-only extraction contract and state whether it remains gate-eligible.

## 4. Evaluation-strengthening implementation

The closed-form map uses four outer folds for reported predictions and three inner folds for hyperparameter selection. Each input block selects its rank and ridge multiplier on inner held-out folds; the selected blocks are jointly refit on the outer training rows and evaluated only on the untouched outer fold. A separate full-data cross-validation selects the frozen deployment map after the gate statistic is complete. This prevents the 35-per-block hyperparameter search from optimistically reusing the reported gate folds.

Registered grids are unchanged: rank `{2, 4, 8, 16, 32}` and ridge multiplier `{1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1}` times the mean diagonal of `XᵀX`. The receipt names the rule `nested_blockwise_inner_cv_then_joint_refit` and carries every inner-fold selection plus the full per-block grids.

## 5. Work proceeding now

The coding lane will run only D4 while the rulings are pending. D4 caches base, branch-A, and branch-B prompt-only states at Qwen layer endpoints 8, 12, 16, and 18 on the frozen 256-row Stage-0 manifest, with sequential branch execution and exact final-interface parity checks. It contains no correction target, no teacher forward, no generation score, and no optimizer.

After R-W2P-1 and R-W2P-2 land, the machine lock will be amended before D1–D3. No Phase-G code will execute unless the registered desk key is `DESK-DOUBLE-PASS` under the resolved authority.
