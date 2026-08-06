# Phase-2 Option B Drafter-Power Exploration Protocol

Date: 2026-08-06. Status: **draft, not locked, training prohibited**.

Governing charter: `STRATEGY_TO_CODING_AGENT_OPTION_B_CHARTER_20260806.md`,
Drive `1LkyqpSY-uQkv4VX-V--xyj_UOcwLjnRs`, 6,454 bytes, reported SHA-256
prefix/suffix `fbe65a44...9ccb95`. Governing guardrail doctrine: Drive
`1R40TawfW-ZJcec4dkRnxGir9c3v4a5IP`.

## 1. Lock blockers

1. The helped/harmed localization receipt must land and strategy must accept or
   reject its proposed structural mask.
2. The data-population unit error in
   `PAPER2_PHASE2_OPTION_B_DATA_UNIT_AUDIT_20260806.md` must be amended by
   strategy. Stage 0A contains 50,000 anchors, not approximately 200,000.
3. The curve must be named correctly. With one fixed training population, its
   x-axis is cumulative anchor presentations, not unique-data scale.
4. The final training-population manifest, document partition, exclusion proof,
   and hashes must be filled in before `locked_before_training`.

No training launcher may exist while any blocker remains.

## 2. Fixed design after blocker resolution

- Four arms: full writeback and no-writeback control, seeds 0 and 1.
- Continue from the four banked A2 step-2,000 endpoints:
  - seed 0 full: `aef5d3fc07ca0319fd60a306f5b4711126137930d498fc838cfda262a0d6b2a6`;
  - seed 0 control: `34cae7a307495b86016f6dd9db4c85e8934aee95612450077864529d47bc1cc4`;
  - seed 1 full: `d8904f9bb2241faf42a7f486e829baed1b68720298b8416847799e7490a59373`;
  - seed 1 control: `704620b74711c8371ba94e3f4d52b1eaa3fdf2fb3dc34d238e1ff89d0c13ce03`.
- Fresh AdamW state, 200-step linear warmup.
- Alpha 0.5. A1 flow frozen. No joint fine-tuning.
- Batch size 128; 20,000 optimizer updates per arm.
- Learning rate `3e-4`, cosine decay to `3e-5` over the full budget.
- Existing weight-decay exclusions preserved exactly.
- Identical sampled-anchor schedule within each same-seed full/control pair.
- Resume-safe Drive checkpoint and fixed 8,031-anchor DEV evaluation every
  1,000 updates.
- Directional audit every 2,000 updates on the locked matched 51 by 128
  estimator.
- Endpoint quality metrics are evidence, not pass/fail, because this is an
  exploration run.

## 3. Measurements

At step zero and every 1,000 updates:

- mean expected accepted length for all four arms;
- gain over the matched zero-loop path;
- full-minus-control writeback increment and its share of total gain;
- baseline-correct retention, point estimate and Wilson 95 percent lower bound;
- quality-safe oracle headroom;
- bridge and draft gate means;
- cumulative anchor presentations and, once resolved, cumulative distinct
  anchors observed.

The result contains all 21 points, including step zero. Curves are reported by
seed and as a two-seed descriptive envelope. No arbitrary weighted aggregate is
used.

## 4. Pre-stated readings

1. **Exploration supports E1:** endpoint full-system relative EAL gain over
   zero-loop reaches at least 1 percent, or the second-half slope against the
   correctly named exposure axis is positive with a document-block bootstrap
   95 percent interval excluding zero.
2. **Writeback retained for E1:** the full-minus-control increment or its share
   of total gain grows over the curve in both seeds. A flat or shrinking share
   sends E1 forward drafter-only and banks writeback as real but bounded.
3. **Bounded at tested scale:** both arms plateau below 1 percent and the
   second-half slope interval includes zero. No automatic extension follows.
4. **Quality trajectory:** a full arm reaching 99.7 percent point retention is
   banked descriptively. Degradation is governed only by the tripwires below.

These readings do not authorize serving-throughput, unique-data-scaling, or
deployable-router claims.

## 5. Draft rule inventory

| Rule | Threshold | Estimator/reference | Cadence | Disposition | Named cliff |
|---|---|---|---|---|---|
| Non-finite loss | Any non-finite weighted loss | Current pre-update batch | Every attempt | Stop | Garbage training |
| Non-finite gradient | Any non-finite active gradient element | Current pre-update batch | Every attempt | Stop | Garbage training |
| Relative gradient explosion | Raw global norm above 10 times the prior-100 median for three consecutive attempts | Current excluded; 100-update warmup logs only | Every attempt | Stop | Garbage training escaping into budget |
| Source checkpoint identity | Exact four endpoint SHAs and metadata | Banked A2 receipts | Startup | Stop | Corrupted lineage |
| Training-population identity | Exact manifest, document partition, and exclusion hashes | Locked Option B population | Startup | Stop | Invalidated data comparison |
| Frozen parameter identity | Exact digest before and after | Reconstructed endpoint | Startup and completion | Stop | Corrupted frozen lineage |
| Evaluation contact | Training loader touches zero fixed 8,031-row evaluation anchors and zero frozen confirmatory partitions | Explicit access ledger | Startup and completion | Stop | Invalidated science |
| Matched pair schedule | Exact sampled-anchor hash sequence within each seed | Full versus control | Every update and resume | Stop | Invalid paired comparison |
| Control path identity | No-writeback hidden path bit exact | Fixed DEV probe | Every 1,000 | Stop | Invalid control |
| Resume durability | Checkpoint, optimizer, RNG, sampler, and receipt hashes all persisted | Drive readback | Every 1,000 | Stop before further updates | Unrecoverable lineage |
| Wilson quality floor | Wilson 95 percent lower bound at least 0.99 | Fixed DEV baseline-correct decisions | Every 1,000 | Stop | Irreversible quality damage escaping into exploration |
| Init-relative retention | More than 0.003 below run-specific step zero twice consecutively | Same arm step zero | Every 1,000 | Stop | Irreversible quality drift |
| Directional gross miss | Primary below 0.40 or any auxiliary above 0.35 | Matched 51 by 128 independent-gradient estimator | Every 2,000 | Stop | Objective inversion |
| Directional repeated marginal miss | Primary 0.40-0.50 or auxiliary 0.25-0.35 twice consecutively | Prior matched audit | Every 2,000 | Stop | Sustained objective displacement |
| Directional target | Primaries at least 0.50 and auxiliaries at most 0.25 | Same estimator | Every 2,000 | Log/pass | Intended objective balance |
| Endpoint evidence | 1 percent gain, exposure slope, writeback share, quality trajectory | Full curve | Endpoint | Evidence only | None |

Every stop-authority rule names an irrecoverable failure mode. Numeric endpoint
readings do not acquire stop authority.

## 6. Localization integration

The CPU localization evaluates only structural candidates observable without a
teacher or hindsight: stratum, position bucket, and their intersections. A
single candidate may enter the full arms only if it clears the pre-stated
two-seed document-bootstrap rule and strategy names it in the locked protocol.
Diagnostic quantiles may explain the outcome but cannot become router features
or masks in this run.

## 7. Do not claim

- The curve is serving throughput.
- A fixed-population dose curve is a unique-data scaling law.
- Post-hoc localization is confirmation evidence.
- Oracle headroom is achievable by a deployable selector.
- A plateau at this scale proves architectural impossibility.
