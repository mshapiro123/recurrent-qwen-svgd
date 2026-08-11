# P3.3 Lock Implementation Audit

Date: 2026-08-11. Status: resolved by strategy errata e1 and e2 and the landed
token-retention preflight. P3.3 training is authorized. No P3.3 training step
has yet run and no CONFIRM or EVAL-E row has been scored.

## Governing record

- Protocol lock: `STRATEGY_P33_PROTOCOL_LOCK_20260811.md`, SHA-256
  `45e2221bc94cf6c13df38c7d0bcdbb4075256792dc5968cb33b1076336455c8d`.
- Ratification: `PROGRAM_RECORD_P33_PROTOCOL_LOCK_RATIFIED_20260811.md`,
  SHA-256
  `4f3a333dbdee1d9379fbca2711fefbadd5bda0c17717e90f4988cba2c6af68f2`.
- Ratification authorizes training only after A1-A5 pass and are receipted.

## Source-to-lock discrepancies

### 1. The fixed gate ceiling conflicts with migration equivalence

The lock requires both exact E1 checkpoint writeback reproduction (A1) and a
fixed `gamma = 0.02` ceiling for the whole pilot. The banked E1 confirmation
receipt reports mean full-system bridge gates of `0.0840322599` for seed 0 and
`0.0884889439` for seed 1. Clamping either checkpoint to `0.02` changes its
writeback and therefore cannot be bit-exact with the Phase 2 source.

Required ruling: either A1 is performed on the unclamped migrated model and the
clamp is then recorded as an explicit pre-training intervention, or continuity
takes precedence and the ceiling is amended. The two conditions cannot both
hold on the optimizer's starting graph.

### 2. `c = 0.15` is not a symbol in the implemented bridge equation

The canonical Phase 3 bridge in `models/paper2_dc2_student.py` computes

```text
delta_hat = delta / RMS(delta) * min(RMS(h0), p99)
writeback = position_gate * delta_hat
hidden = h0 + rho * (previous - h0) + writeback
```

The source binds `p99`, `rho`, and the gate. It does not multiply by the legacy
diagnostic radius constant `c = 0.15`. Adding that factor now would change the
banked model and fail A1. The recommended erratum is to retire `c` from the P3.3
forward-equation binding while preserving it in the V1d diagnostic receipt.

### 3. The locked audit slice cannot estimate gate precision

`prepare_training_rows` draws all 4,096 audit records from the strict positive
write-candidate population before selecting negatives. This correctly supports
the aim-capture denominator and positive-class gate recall. It contains no
negative labels, so gate precision against the tri-state labels is undefined.

Recommended repair: preserve the locked 4,096-row positive audit hash for
`pi_dir` and `pi_dep`, and add a separately seeded, separately hashed,
evaluation-only negative cohort for gate precision and collateral `chi`.
Training counts and the original audit hash remain unchanged.

### 4. The optimizer budget and look cadence are not numerically bound

The lock specifies two seeds, one A100 session per seed, and a 20-point curve,
but not optimizer, learning rate, batch size, warmup, update budget, or the
update interval between looks. The guardrail calibration requires exactly 20
looks, so these values cannot be selected after training begins.

Recommended inherited pilot schedule: AdamW, learning rate `3e-4`, batch size
`128`, 100-update linear warmup, 1,000 updates, and one registered look every
50 updates. Parameter exclusions for weight decay follow the Phase 2 pilot:
biases, normalization gains, and learned scalar parameters receive no decay.

## A5 resolution

The linear-decodability forecast was regenerated and banked on 2026-08-11.
Both seed lineages and all four loops completed over 43,204 strict concurrent
positions with document-disjoint splits and zero optimizer steps. The
ridge-extended loop-4 fits selected ridge `1e5` and produced holdout cosine
`0.0952` (95% CI `[0.0842, 0.1077]`) for seed 0 and `0.0874` (95% CI
`[0.0792, 0.0993]`) for seed 1. See
`docs/PAPER2_PHASE3_LINEAR_DECODABILITY_FORECAST_RECEIPT_20260811.md` and
`outputs/stage5/stage5_paper2_phase3_oracle_forecast_20260810/summary.json`.

Pre-run assertion A5 is satisfied. Erratum e1 resolved all four discrepancies:
migration identity precedes clamp activation, `c = 0.15` is an audit magnitude
rather than a forward factor, a disjoint negative audit supports precision and
`chi`, and the 1,000-update AdamW schedule with twenty looks is bound.

Erratum e2 then replaced the unavailable task-level guardrail estimator with an
exact 1,024-position token-retention panel. The no-update preflight passed:
step-zero retention is `1024/1024` in both seeds, Tier S controls the simulated
familywise false-stop probability at `0.00003`, and Tier W controls the null
warning probability at `0.00424`. Receipt:
`outputs/stage5/stage5_paper2_phase3_retention_preflight_20260811/summary.json`.

## Work that may proceed

- Treat A5 as complete and preserve its hashes in the P3.3 execution receipt.
- Mirror the governing documents and bind their hashes.
- Build tests, telemetry, resumable checkpointing, and an optimizer-free A1-A4
  preflight.
- Construct the optimizer only inside the registered P3.3 runner after it
  re-verifies the landed preflight hash and all A1-A5 assertions.
- Run exactly 1,000 updates per seed with exactly twenty looks.
- Continue to prohibit task-level capability scoring until the P3.4 inference
  graph is separately locked.
