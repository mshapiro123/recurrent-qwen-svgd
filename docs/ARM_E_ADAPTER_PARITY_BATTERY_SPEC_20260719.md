# Arm E Adapter Parity Battery: E3a, E2, and E4

**Locked:** 2026-07-19
**Arm E checkpoint:** `bffa8c4277ce82ae9f662db3243a21a50a08c4c041820c9d7506d8f250e82839`
**Trainable budget:** rank-16 LoRA over all recurrent-block projections plus the repaired split bridge (`6,007,425` forward-active parameters)
**Frozen set:** all pretrained Qwen parameters

## Purpose

This battery gives the adapter-budget Arm E a measured counterpart for the
full-block model's verbal transfer, outcome-only persistence, and inverse-task
retention characterizations. It is a labeled Paper One extension, not a new
keeper lineage and not a rank sweep.

## Immutable Order

1. **E3a:** zero-shot relay and pointer transfer, evaluation only.
2. **E2:** 1,000 outcome-only continuation steps and the standard persistence
   readout.
3. **E4:** inverse-table retention, authorized only when E2 preserves the
   active-label diagonal at or above `0.93`.

E3b, matched natural-surface training, is not authorized. E4 branches from the
original Arm E checkpoint; E2 is an authorization test, not E4's initialization.

## E3a: Zero-Shot Verbal Transfer

- Frozen rows:
  `outputs/stage5/stage5_natural_surface_transfer_20260708_230229/data/relay_test_chain_mcq.jsonl`
  and `pointer_test_chain_mcq.jsonl`.
- Depths `1-12`, forced loops equal row depth.
- Same-reader, full-symbol scoring with `name:` symbols.
- Reporting bands: strong `>=0.70`, partial `[0.40,0.70)`, minimal `<0.40`.
- Full-block step-6000 profiles are descriptive references, not a parity gate.

## E2: Adapter-Budget Persistence

- Initialization: locked Arm E final checkpoint.
- Data: the exact training and held-out rows used by
  `stage5_chain_anneal_20260703_160250`, verified by SHA-256.
- Training: AdamW, `1e-5`, split-bridge prelude multiplier `10`, 1,000 steps.
- Supervision: target outcome only. The chain-label coefficient is zero from
  step one (`chain_anneal_hold_frac=1.0`).
- R16 LoRA and the bridge remain trainable; pretrained Qwen weights remain
  frozen and hash-checked.
- Strong: active diagonal `>=0.93` and continuation `>=0.85`.
- Partial: diagonal `>=0.93` and continuation in `[0.50,0.85)`.
- Failed: diagonal `<0.93`, or continuation `<0.50`.
- E4 authorization depends only on preserving the diagonal at `>=0.93`.

Full-block references remain `625/640` active labels, `357/384` continuation,
and `1/384` hold.

## E4: Adapter-Budget Retention

- Initialization: locked Arm E final checkpoint, never the E2 continuation.
- E2 receipt must identify the locked Arm E source and explicitly authorize E4.
- Task: explicit inverse-table branch, cap 3.
- Rehearsal: additive 25% forward rehearsal, preserving the original inverse
  dose; 334 optimizer steps at effective batch size 8.
- Optimization: AdamW, `1e-5`, R16 LoRA plus split bridge only.
- Online Tier-1 hard stop: Arm E baseline `60/64`, hard stop below `57/64`.
- Natural guardrail: establish Arm E's own pre-training relay/pointer baseline;
  every saved checkpoint must remain within 3 absolute points.
- Synthetic guardrail: every depth stratum at every saved checkpoint must
  remain at or above `0.93`.
- Acquisition: depth-3 inverse-table accuracy at least `46/64`.

Readings:

- **WALL HOLDS:** no checkpoint jointly passes acquisition and both retention
  guardrails.
- **WALL MOVES:** at least one checkpoint jointly passes, but the complete
  checkpoint trajectory or final point violates the registered retention rule.
- **WALL VANISHES:** the final point jointly passes and every registered
  retention checkpoint remains green.

Blocked outcomes return exit code 2 after writing and publishing their tables.

## Adapter Frontier Receipt

Arm E crosses the `0.71` threshold between depth 11 (`111/128`) and depth 12
(`75/128`). Linear interpolation gives:

```text
frontier = 11 + ((111/128 - 0.71) / (111/128 - 75/128)) = 11.559...
frontier / trained support 8 = 1.4449...
```

The ledger records this as frontier `11.56` and support ratio `1.44x`,
cross-validated against Arm A's ladder-official frontier `11.61`.

## Colab Targets

- `adapter_parity_e3a`
- `adapter_parity_e2`
- `adapter_parity_e4`

All three use `colab/STAGE5_ADAPTER_PARITY_BATTERY_CELL.py`. The E4 target
fails before model loading if the E2 authorization receipt is absent or red.
