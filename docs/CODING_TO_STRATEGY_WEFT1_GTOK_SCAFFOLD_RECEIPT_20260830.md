# WEFT-1 G-TOK governed execution scaffold receipt

**Date:** 2026-08-30
**Status:** build axis complete and audited; P-A continues; authoritative A100 base screen remains deliberately blocked on two strategy semantics

## Authority verified

The release/license closure was fetched independently and matched its delivery record byte for byte:

- `docs/STRATEGY_RELEASE_POSTURE_AND_LICENSE_CLOSE_20260830.md`
- 5,746 bytes
- SHA-256 `d8c4f3bf8829bbe48e2464bf758ec3594ef730a0f952712099b45d183ca2ab3e`
- Drive `1ZaBmFnlHhAEiGGUGMQXq3Wo6V2ud5ds-`

This clears the human license gate for P-B. Mechanical D1-D6, C1-C3, DECON, tripwire, and seed-split gates remain fail-closed.

## Implemented surface

Implementation commit `83d72753b99b67efa989f40554a28bea1a91ea33` adds the governed G-TOK tokenizer, runtime, offline launcher, CPU precompute, base campaign, confirmation campaign, and code-closure surfaces. The proxy is the ratified ten-block `4 + 2 + 4` structural-OFF graph, not an eight-block graph.

The scaffold now binds and reauthenticates:

- the physical P-B/V4 corpus identity and exact seed-specific T order before consumption, plus the H identity before every held-out read;
- the tokenizer parent receipt, both isolated fit-worker receipts, the tokenizer artifact itself, and all tokenizer-local invariants;
- the exact Python/PyTorch/CUDA/cuDNN runtime, stable A100 device identity, hash-pinned wheelhouse, closed environment, and offline parent policy;
- the clean Git code closure and authenticated CPU-precompute producer;
- fixed `microbatch_sequences = 8`, 32 accumulation slices, flat A1 AdamW, and exact packed-stream digests;
- both initialization seeds against CPU-precomputed state before fresh or resumed full runs;
- complete profiler plus unsupported-operation FLOP ledgers and the registered output/evaluation surfaces.

Checkpoint-free resume uses a durable SQLite/lifecycle ledger. Orphan recovery conservatively charges the last durable reading plus one full heartbeat cadence. Completed relaunches reauthenticate and reuse immutable evidence instead of rerunning rows.

## Determinism and meter

Before calibration, the campaign runs one fresh replay pair for every distinct `(vocabulary size, terminal rows)` key in canonical order. It binds deterministic algorithms, cuBLAS `:4096:8`, TF32-off, cuDNN deterministic/no-autotune, and Flash-only SDPA. Exact model state, optimizer state, and evaluation output must match across replicas.

Projection, the 2x watchdog, pair receipts, and the cumulative 12 A100-hour meter all use outer lifecycle charges. Inner CUDA work time remains a diagnostic and must not exceed the outer charge. The second outer lifecycle crossing 2x terminates as `aborted_watchdog`; replay preamble and model construction are inside the timed boundary.

## Verification

- independent focused and adjacent matrix: `188 passed in 47.62s`;
- second independent audit: `158` adjacent P-A/P-B/P-C tests and `88` focused G-TOK tests passed;
- production modules and scripts compile;
- all four production CLI help surfaces return zero;
- `git diff --check` is clean;
- final P0/P1 boundary audit: clean.

The repository-wide raw suite remains red only at the governed Paper-2 evidence-ledger node: `1 failed, 3829 passed, 19 warnings`. No failure was added, removed, or renamed. The quarantine is rolled forward without editing its predecessor:

- `training/ablation_lm_engineering_quarantine_20260830_gtok.json`;
- review due 2026-09-04;
- repo-wide green claim remains prohibited;
- strict quarantine-aware gate: PASS.

## Live P-A state

The Pharma Initiatives Pro+ Colab is still materializing P-A from clean commit `2d9278c09187ebfcb10f2c8271c0ce45815d862b`. At the final receipt check, both workers were alive after 2h28m and the durable Dolma parse ledger was 707,221,225 bytes. `_INCOMPLETE` remained present and no parent replay receipt existed, so no P-A/P-B gate is claimed here.

No production G-TOK run, A100 base result, checkpoint, or sealed-evaluation read occurred in this build receipt.

## Strategy return required before A100 base spend

The authoritative campaign stops before offline/runtime/GPU work until strategy answers the three questions in `docs/CODING_TO_STRATEGY_WEFT1_GTOK_CONFIRMATION_CLARIFICATION_REQUEST_20260830.md`:

1. the physically realizable common-FLOP rule and tolerance;
2. the exact construction and ordering of the confirmation pair; and
3. the reversal rule when the asymmetric-band winner is not the raw-BPB winner.

Recommended defaults remain largest whole-step prefix below `F*` with a registered `5e-4` relative-slack cap, and selected `V` plus the best distinct raw-BPB alternative. These are recommendations, not execution authority.
