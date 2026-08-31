# WEFT-1 G-TOK semantics implementation audit

**Date:** 2026-08-31
**Status:** authority artifact verified; no behavioral change; no GPU spend; implementation remains fail-closed
**Scope:** reconcile `STRATEGY_GTOK_CONFIRMATION_SEMANTICS_20260831.md` with the accepted `f8176fec` scaffold before changing the governed execution path

## Authority and current posture

The new strategy ruling is present locally and is byte-exact against the delivery record:

- file: `docs/STRATEGY_GTOK_CONFIRMATION_SEMANTICS_20260831.md`
- bytes: `13,975`
- SHA-256: `2e42664d0062a119c9fadcb76bf227a91134914920116627f9244f650defe72d`
- Drive: `1FzxF7OxK6W1Fk1Os5CnyfHrwAhjEm2qV`

The accepted scaffold checkpoint remains `f8176fec92f2647fc483adc231063e21f13f57da`. The current branch contains later P-A durability work, but no G-TOK base or confirmation run has launched, no A100 compute has been consumed by this implementation review, and no `V` receipt exists.

The fresh P-A replay `pa-v4-c19766a1-r3` lost its Colab backend after 82 Dolma checkpoint artifacts. It is immutable and non-resumable. Its termination receipt is bound by SHA-256 `a3d0aee379ec4119d13a12ee4a5b7f5ff268935916ed8a7a1d8b95efbe87de7a`; no P-A or P-B gate minted.

## One upstream execution-order conflict

The new ruling says both that the A100 base screen is unlocked on hash verification and that G-TOK GPU work does not wait on P-A. The executable authority chain still requires, before either production G-TOK CLI can reach CUDA:

1. a factory-validated frozen P-B screen corpus;
2. D1-D6, C1-C3, and DECON evidence;
3. tokenizer artifacts fitted to that frozen T identity; and
4. CPU-precompute evidence bound to the same code closure and runtime.

Those inputs do not exist while P-A is incomplete, and the original run-axis order is P-A → P-B → tokenizer fit/precompute → P-C. This is an authority-versus-dependency conflict, not a reason to weaken a guard. Recommended posture: treat the new document as authorization to implement and, once all upstream artifacts exist, to spend A100 compute. Preserve the frozen-corpus, gate-bundle, DECON, tokenizer, and CPU-precompute checks exactly. No GPU launch should bypass them.

## Six literals still needed before behavior can change

The following are scientific/execution semantics, not replay-library choices. The code cannot choose among them locally without manufacturing authority.

| ID | Underspecified literal | Why the current scaffold cannot infer it | Literal options to ratify |
|---|---|---|---|
| S-1 | Exact runner-up `U` | The selector returns one asymmetric-band winner and an agreed raw-BPB order; it does not produce a complete post-band ranking. “Top-2 after the tie rule” therefore does not uniquely identify `U` when `W` is outside the raw top two. | **A:** `(W, best distinct arm in the agreed strict raw-BPB order)`, matching the prior clarification recommendation. **B:** recursively reapply the asymmetric traversal to the remaining arms and take that winner as `U`. Bind pair ordering as `(W,U)` in either case. |
| S-2 | Dropped-tail accounting | The accepted packing contract executes an actual-size terminal batch and requires terminal trained bytes to equal frozen realized T. SEM-1(a) instead drops it, but does not say whether `N_tokens` counts content tokens, BOS/EOS/boundary tokens, valid predictions, padded sequence slots, or another quantity. Dropping the tail also makes trained terminal bytes smaller than manifested T. | Bind the exact token counter; add source-token, trained-token, dropped-token, dropped-fraction, dropped-byte, and dropped-document fields; state whether terminal BPB’s `training_raw_bytes` is the trained prefix or manifested source total; then update the frozen-T invariant explicitly. |
| S-3 | Seed-sign pairing in the hybrid confirmation | The min-FLOP evidence reuses two byte-matched seeds while the max-FLOP arm uses new `gtok.confirm.{arm}.{seed}` streams. “Both seeds agree in sign” needs an exact pairing, and “identical byte stream” constrains data-order RNG while “new streams” appears to prohibit reuse. | Bind seed-index pairing (`base seed row i` versus `confirmation seed row i`), the exact root and derivation for every new initialization/run/module RNG, and whether the base data-order permutation is replayed or independently redrawn. If replayed, say that the data order is evidence, not a reused RNG stream. |
| S-4 | Integer `F*` and overshoot behavior | `F*` is the minimum of two-seed means, which can be half-integral. SEM-1(c) says floor by `f_step`; SEM-1(d) permits a realized value up to 1% above F, while the prose also says never overshoot. These do not define one receipt-valid integer target. | **A:** require the two per-arm seed totals to be equal and use that exact integer. **B:** represent `F*` as an exact rational and define the integer comparison without rounding. Separately state whether any realized `F_i > F*` is invalid even inside the ±1% band. |
| S-5 | Initial-versus-steady FLOP pricing | The complete counter prices the first AdamW step separately from steady steps. One scalar `f_step` from a calibration burst does not uniquely produce the largest whole-step prefix, and a mean-step floor can cross F once the initial-step delta is restored. | **A:** use the calibration burst’s ordered complete per-step ledger and solve the exact prefix `initial + (n-1)×steady ≤ F*`. **B:** define a named steady-step estimator plus an explicit first-step correction. In either case record next-step quantum and prove the chosen prefix does not cross the bound. |
| S-6 | Confirmation curve coordinates | Existing base receipts use `after_1b`, `after_2b`, and terminal realized raw-byte milestones. SEM-2(e) requires 0.25×/0.5×/1.0× of confirmation token count for fresh runs while saying the reused curve stands. Those coordinates are not generally the same. | **A:** add confirmation-token-fraction milestones and reconstruct equivalent points for the reused runs only if the base event ledger contains them. **B:** keep the registered raw-byte points for both arms. **C:** rerun neither arm but score only terminal confirmation BPB and explicitly drop the three-point confirmation-curve claim. |

## Code surface once the literals are bound

The minimum behavior-bearing surface is:

- `training/weft1_gtok_training_v2.py`: terminal-batch policy, plan/receipt accounting, schedule horizon, complete-FLOP projection check;
- `training/weft1_gtok_v2_contract.py`: winner/runner-up pair, reused-versus-fresh provenance, reversal statistic, seed-split and reversal escalation, freeze eligibility;
- `training/weft1_gtok_campaign_v2.py`: verified semantics authority identity, 10% analytic-reference stop, and removal of the old unresolved-semantics stop only after the new literals are encoded;
- `training/weft1_gtok_confirmation_v2.py`: arm-mean F*, min-arm reuse, max-arm fresh runs, exact prefix pricing, new seed registry, confirmation curves, ±1% validity, and stop receipts;
- `scripts/run_weft1_gtok_campaign_v2.py` and `scripts/run_weft1_gtok_full_campaign_v2.py`: authority verification without weakening P-B prerequisites;
- `training/weft1_gtok_code_closure_v2.py` and focused G-TOK tests: bind every changed behavior byte and exercise all stop branches.

At minimum, focused tests must prove: exact authority identity; no partial-tail execution; dropped-token receipt math; >10% projection divergence stop; deterministic `(W,U)` construction; arm-mean F*; no fresh work for the reused arm; exactly two fresh runs under registered confirmation seeds; tokenizer identity reuse; exact prefix/no-overshoot and ±1% validity; close confirmation leaves W unchanged; larger-U uses 3ŝc; smaller-U uses 2ŝc; seed split escalates; qualifying reversal escalates and never mints `V`; and all P-A/P-B prerequisites remain fail-closed.

## Requested strategy return

Please bind S-1 through S-6 as literals. Until then the correct implementation state is:

`AUTHORITY_VERIFIED__SEMANTICS_NOT_YET_EXECUTABLE__NO_GPU_SPEND`

Build-axis work and a fresh P-A replay may continue under standing authority. Production base or confirmation compute must not launch from an implementation that silently chooses among the options above.
