# STRATEGY — Handoff Packet 2026-09-03: Everything Bound Today, in Build Order

**Date:** 2026-09-03 · **Status:** HANDOFF PACKET for the coding agent. Consolidates the bindings issued across today's records into one authority list so nothing has to be reconstructed from five documents. **This packet adds no new bindings**; where it and a source record differ, the source record wins (SEQ-1). Ratified by Mark: D-MC-1, EG-1, D-NB-1 (all 2026-09-03). Everything else here is a receipt-schema addition, a test addition, a named alternative, or an obligation on a later phase.

---

## 1. Document register (verify before use)

| # | document | bytes | SHA-256 | Drive id | authority |
|---|---|---|---|---|---|
| R0 | STRATEGY_ARCHITECTURE_RECONCILIATION_20260903.md | 20,695 | `0d81e9ab63d21720fecfbfcb629aaa5eeae6693eabbd9682b82adc7e3792ea8e` | `1_2OLc0i7weKolHP2PSdQ_7P8o65N44Q4` | already passed; vocabulary, R-1..R-8, re-cut queue |
| R1 | STRATEGY_MATH_CHECK_20260903.md | 16,587 | `509cac8c7f5f82a6a70d0bcc8494b02967d3f545e4e875e9bbfcdc2b93dedcff` | `1lTnggBLgw6gqt9c5t_7btG-Zjqq4VgKZ` | already passed; four checks |
| R1s | math_check_20260903.py | 7,586 | `9dbe3724345382d451fd03af7e57d9503a1fe4d626d0c4cba9a4acc80b08195b` | project `claude/math-check-20260903.py` | reference script |
| R2 | STRATEGY_MATH_CHECK_RATIFICATION_20260903.md | 2,868 | `9c5822daef5dbb0609bc3e46019cc4b1e332991c30e8a42c1b4432800a747ab1` | `1cRCClUvvkl2d6HlEKZ0l5KvInuNH5LBP` | **D-MC-1 ratified** |
| R3 | STRATEGY_FARSKIP_AUTHORITY_LRNBA_ADJUDICATION_20260903.md | 12,853 | `c1703fbc5d37c3280bbdf42cfad72aee9fa3ab2787c0261f175fd18cb234ce71` | `1TvEZdZwC8lgUd0PrJgdhxtbAGNzce7Vv` | receipt/obligation additions only |
| R4 | STRATEGY_ENGRAM_GATE_RATIFICATION_20260903.md | 4,398 | `36f0255c1cc0e61b2d9019ce86b3b1e7446b0a2c3445a42ce20334213deae780` | `1ppyeXo1Og4YFDGpmQZ-BVleB18QScEOn` | **EG-1 ratified**; catch #39; D-COND |
| R5 | STRATEGY_NANBEIGE_ADJUDICATION_20260903.md | 12,518 | `eb8ab8b1a4ebeffbc6c43db411dd5b2a0d5008c6bd15030e2f67096a293aad2d` | `1fqz_Pya82bdRzCLGgHtD8V2b9jR-t_B-` | KV-LIVE contrast registered; **§4/§6 superseded by R6** |
| R6 | STRATEGY_KV_LIVE_RATIFICATION_20260903.md | 4,329 | `489729907fe672f811015ff961f8731c6a9f775a5119347c819184d368e238b0` | `1YEHZNSVLlYFLKHRVZdC8qV-gOPjgNOGc` | **D-NB-1 ratified**; amends §5.3, S-3, A1-Q1 |
| R7 | STRATEGY_NANBEIGE_AMENDMENT_A1_20260903.md | 21,756 | `7afc73725bdb9a60bbf1e8317896a261ac34391ddca360eab413f9f8a3df56ad` | `1hII-SaFIiB-aY1piUvR4I6bAbsQEK404` | named alternatives; one optional switch value |
| R7s | nanbeige_check_20260903.py | 2,724 | `3d6bce43ae6b601545a197428faf70091aae3d562b0b7927bac7b8c4bda60f25` | project `claude/nanbeige-check-20260903.py` | reference script |

## 2. Precedence notes (read first — these are the places a fail-closed stop would otherwise fire)

1. **Engram gate.** R2 restates catch #37's form ("memory-space dot, /√d_m"); **R4 (EG-1) is the later ratified record and controls:** form A as built, `d_m = 64`, **plus trainable RMSNorm gains `γ_q, γ_k`** (vector class, LR `η_base`, no decay, init 1). Handoff §5.11 is amended to EG-1 in place. D-COND (R4) governs how defect-fixing bindings are read from here on.
2. **K/V policy.** R5 §4/§6 say "step 2 proceeds with the static cache"; **R6 (D-NB-1) supersedes: default is `live`.** R5 §3's KV-LIVE contrast stands with roles swapped (live = default, static = control). R7 §2.1 adds an *optional* fourth switch value; it does not alter the default or the S2 arms.
3. **Coda decodes.** Handoff §5.1's `step_logits` loop (per-visit decode) is **amended by R2 (D-MC-1)** to final + one sampled earlier visit. Any allocation figure derived from the old 1.95× multiplier is stale.
4. Nothing in R3 or R7 changes the build queue (R0 §re-cut: steps 2–7) or the first run.

## 3. Bindings by build step

**Step 2 — bicameral block, K/V, S-2 combine, legacy retirements, visit_schedule receipt (R0 + R6).**
- `kv_policy ∈ {live, static, midpoint}`, **default `live`**: K/V recomputed each visit from each hemisphere's own current state; `W_K, W_V` μ-only (S-3 unchanged); no cross-hemisphere K/V reads. The integrated static single-stream path is retained as the `static` arm's implementation. (R6)
- Tests: T15 equivalence retained for `static`; **new live-path equivalence test anchored on the K = 1 identity** (`live ≡ static` at K = 1). (R6)
- Receipts: `kv_policy`, `kv_cache_multiplier_at_serving` (= 2K / 2 / 1). (R6)
- Optional, decline if not one line: `kv_policy = first` (K/V computed at visit 1, reused thereafter — the exact Nanbeige "shared" axis). Not an S2 arm. (R7 §2.1)

**Step 1 backfill — engram (prelude block 1).**
- EG-1: add `γ_q, γ_k`; **re-run T2 on this form** (nonzero step-1 gradient on `W_Q`, `γ_q`, `γ_k`, tables, `W_V`, `γ_m`); receipt the gate form as `EG-1`. (R4)
- From R1/R2, still owed: T2 asserts nonzero `∂L/∂dU` and `∂L/∂dV`; T4 asserts `perm[k] = bitrev(gray(k))` and reports which variant passed; `2⁻ᵖ` convention stays, round-trip exactness claim withdrawn.
- Named only, no action: `EG-1-SQRT` (signed-sqrt logit compression) in the engram-sweep alternatives list. (R7 §2.4)

**Step 3 — carrier / rank-8 write / bridge_out / retention gauge.** No new bindings today.

**Step 4 — per-band callosum.** No new bindings. Named only: `VISIT-MIX` (per-visit `ρ_b`, `θ_b` tables) in the MEM-OP alternatives list. (R7 §2.2)

**Step 5 — sidecar (S-4′/S-5).**
- Receipt line **ROUTE-STAB**: per eligible step, (i) fraction of examples whose top-3 selection changes between consecutive visits; (ii) fraction whose selection differs train-mode vs eval-mode on the same input; (iii) the same two under the occupancy router's hysteresis when it exists. (R3 §1)
- Named only: `DELAY-1`, `MEM-SYN-STATIC` join the MEM-OP battery; `DEPTH-ANCHOR` in the K/V alternatives list. (R3, R7 §2.3)

**Step 6 — objective stack.**
- D-MC-1: per micro-batch, decode the final executed visit for `L_LM` and exactly one earlier visit `j ~ Uniform{0..K_exec−2}` for `L_stage`, drawn from O-9 stream `weft.lstage.sample`; `L = L_LM(final) + λ_stage·L_stage(j)`; at `K_exec = 1`, `L_stage = 0` for that example (recorded, not padded). STOCH-K composes unchanged. (R2)
- Receipts: `coda_decodes_per_step` (= 2), `lstage_sampled_visit`. (R2)
- **Allocation re-derived at 1.24× (K = 4; 1.32× at K = 2, 1.19× at K = 6) and reported before S2**; pre-registered de-scope order (rung B first) applies if over the allowance. (R2)
- Registered contrast, exploration allocation only: `LSTAGE-FULL`. (R2)

**Step 7 — certificates / A7 / production.**
- C-JAC / PF-1.4: `Λ̂_core` is **re-measured under `kv_policy = live`**; no certificate claim changes (attention was already the empirical factor). (R6)
- **REPLAY-SEL** (obligation for P4 / any RL phase): exact selection codes and gate decisions used to generate a token are recorded in the step receipt and replayed in the update. Schema field only for now. (R3 §1)

**S2 registry (no build action; for the registry file).** KV-LIVE contrast first in S2, now live (default) vs static (control), proxy rung, both seeds, branches as written in R5 §3. COMP-LOCALITY joins COMP-SKILL with its `C_cross/C_within ≈ 0.16` prior (R3 §2).

## 4. Owed back to strategy (in the order they unblock things)

1. T2 on EG-1 (with the `dU/dV` assertions) and T4's sequency variant report — closes the R1/R4 items.
2. Allocation re-derivation at 1.24× — needed before S2 is scheduled.
3. Step-2 receipt with `kv_policy = live`, the K = 1 identity test, S-2 combine, FRONT-WHT / H0-REENTRY / TwoLaneBirkhoffMixer retirements, `visit_schedule` line.
4. Build-status matrix update and figure r2 (WEFT naming, five paired, closed items) — strategy will issue diagram r3 (live K/V) alongside.
5. Still open from earlier: C1 under PF-3.1/S-1; C-S5/S6 dispositions; quarantine review 2026-09-04; P-A ETA.

## 5. Explicitly not required

No action on: `first` (unless one line), `EG-1-SQRT`, `VISIT-MIX`, `DEPTH-ANCHOR`, `DELAY-1`, `MEM-SYN-STATIC`, `LSTAGE-FULL`, the WEFT-2 seed item on exact multi-lane Birkhoff parameterization (R7 §2.2(d)). Do not implement any Nanbeige component; R7 is a reading, not a spec.

---

**Strategy:** everything above traces to a hash in §1; the three precedence notes are where the day's records could read as conflicting and are resolved in favor of the later ratified record each time. **Coding agent:** verify §1, apply §3 in step order, report §4. **Mark:** pass this packet with R2, R4, R6 (the three ratified records) and R7 if not already passed; nothing to decide.
