# Phase-2 Option B Staged Dose-Then-Data Exploration Protocol

Date: 2026-08-06. Status: **draft, not locked, training prohibited**.

Governing documents:

- Option B charter, Drive `1LkyqpSY-uQkv4VX-V--xyj_UOcwLjnRs`;
- data-resolution amendment, Drive `1BkxgDfdLzDAKTiresWTbomY8LOzsx89I`;
- Guardrail Doctrine, Drive `1R40TawfW-ZJcec4dkRnxGir9c3v4a5IP`.

## 1. Remaining lock sequence

The CPU localization landed at commit `17151603`. No structural candidate
cleared the pre-stated two-seed rule. The protocol therefore recommends no
mask, pending strategy ratification. The existing-data hashes are now banked:

- training manifest: `03ce3e1877f4e79f0952ab7054b16c0fb823fe9c9de03ee7c9088d8aa271201a`;
- document partition: `7b4fcdfad21b940ea8a5d51d4310d3a9b4ac851d27df2542004a9182f8398e81`;
- evaluation exclusion proof: `c751de988b7c83fd1bfed4a409174d99ed79b02657a06f156672df73537b7f5f`;
- fixed old-train diagnostic subset: `0f5d114c3dcf6c856956ba9a618f7957f0c3d18c317415c3a1eb23420cd609c5`.

Remaining lock sequence:

1. Strategy ratifies the no-mask reading.
2. Approve the teacher-pass resource note and lock the generation procedure,
   pinned model revisions, minimum/target anchor counts, quarantine rules, and
   hash-only amendment procedure.
3. Resolve whether 14B layer states are collected for every new anchor or only
   the 14B-threshold subset. The current resource estimate conservatively
   assumes every anchor; the implementation may not guess.
4. Strategy locks this protocol and its machine-readable JSON before either
   segment training or the teacher pass begins.

The expanded data cannot have a manifest hash before it is generated. The lock
therefore binds its generation procedure and admission tests. A later hash-only
amendment records the landed manifest and partition hashes and is required
before the splice. It may not change the recipe, target/floor, source rules, or
analysis.

No Option B training or teacher-pass launcher may exist before the strategy
lock. This repository currently provides only the authorized CPU localization
launcher.

## 2. Scientific design

The run has four arms: full writeback and no-writeback control for seeds zero
and one. They continue from the four banked A2 step-2,000 endpoints. Alpha is
0.5, the A1 flow remains frozen, and AdamW starts with fresh optimizer state.

The 20,000 updates are one staged intervention:

- **Segment 1, dose:** train on the existing 41,969 anchors while the fresh-data
  cache is generated. This measures additional exposure and overfitting onset.
- **Segment 2, fresh data:** at a durable 1,000-step checkpoint boundary, add
  the new anchors and continue without resetting model, optimizer, rule state,
  or counters. Step 4,000 is the target splice; the recorded boundary governs.

The target expansion is 140,000 new training anchors from new documents. The
minimum admissible expansion is 100,000. With the target, the post-splice pool
contains about 181,969 training anchors. The fixed 8,031-anchor evaluation slice
and all confirmatory partitions remain quarantined.

The splice is a causal data intervention, not a general unique-data scaling
law. The primary contrast is EAL change per 1,000 updates over the 2,000 updates
immediately before versus after the splice, with document-block bootstrap
intervals. The constant learning rate and continued optimizer state make fresh
data the only intended discontinuity.

## 3. Teacher/cache expansion contract

The pass reuses Stage 0A's units and pinned revisions:

- the general source is FineWeb-Edu `CC-MAIN-2025-26` at revision
  `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`; the code source is
  `bigcode/the-stack-smol` at revision
  `4a6938ce94446f324c6629e7de00ac591710044b`, with a new selection seed
  `20260806`, a 50/50 stratum mix, and exclusion of every previously used
  document id;
- anchors are the training unit; each anchor has horizons one through four;
- union top-K is 128 with per-model tail mass and a stable one-percent full-logit
  audit subset;
- 7B is the broad teacher, 14B is queried under the standing thresholds and
  supplies canonicalizer states at layers 16, 32, and 44, and 32B is used only
  under the existing cascade rule;
- student, 7B, 14B, and 32B revisions are copied verbatim into the machine lock;
- every anchor comes from a new document excluded from the existing evaluation
  documents and every frozen confirmatory partition;
- the manifest names anchor count and horizon-sample count separately;
- resume shards, audit hashes, document partitions, and model-cache hashes are
  durable before ephemeral storage is released.

One coverage detail remains open for strategy lock. The amendment says 14B is
queried on thresholds while also requiring 14B canonicalizer-input states. A
threshold-only state cache is cheaper but is not an all-anchor future-flow
dataset; all-anchor state collection is reusable but changes the resource bill.
The locked protocol must choose one explicitly.

The fixed diagnostic subsets are document-stratified and immutable: one from
the old training population and one from the new population. Once the new cache
lands, all stored pre-splice checkpoints are evaluated on both subsets so the
train-eval trajectory is aligned across the intervention.

## 4. Optimizer and cadence

- Batch size: 128 anchors.
- Budget: 20,000 updates per arm.
- Learning rate: 200-step linear warmup to `3e-4`; constant `3e-4` through step
  18,000; linear cooldown to `3e-5` over steps 18,001 through 20,000.
- Existing weight-decay exclusions are preserved exactly.
- Full/control pairs use identical sampled-anchor schedules within seed.
- Checkpoint, fixed evaluation, and train-subset diagnostics run at step zero
  and every 1,000 updates.
- The locked 51 by 128 directional audit runs every 2,000 updates.
- Checkpoints persist model, optimizer, RNG, sampler, population version,
  cumulative presentations, and distinct-anchor ledger.

## 5. Measurements

At all 21 checkpoints, report by arm and seed:

- expected accepted length and relative gain over the matched zero-loop path;
- full-minus-control writeback increment and share of total gain;
- baseline-correct retention and Wilson 95 percent lower bound;
- quality-safe oracle headroom;
- bridge and draft gate means;
- fixed evaluation EAL and loss;
- fixed old-train-subset EAL and loss;
- fixed new-train-subset EAL and loss once the cache exists, including
  retrospective reads of pre-splice checkpoints;
- cumulative anchor presentations, distinct anchors observed, population
  version, and actual splice step.

Report the two seeds separately and as a descriptive envelope. Do not collapse
them into an arbitrary weighted score.

## 6. Pre-stated readings

1. **Data starvation confirmed:** the fresh-data segment slope exceeds the dose
   segment slope with separated document-bootstrap intervals. E1 is justified
   and its data budget is sized from the observed intervention response.
2. **Exposure suffices:** both slopes are positive and their intervals overlap.
   Unique data is not yet binding; the expanded cache remains banked.
3. **Bounded at tested scale:** both slope intervals include zero and the
   train-eval gap is small. Only this joint reading can return the bounded
   program decision. A dose-only plateau cannot.
4. **Overall E1 support:** the endpoint full-system relative EAL gain reaches
   at least one percent, or the second-half exposure slope is positive with a
   document-bootstrap 95 percent interval excluding zero.
5. **Writeback at scale:** a growing full-minus-control increment or share in
   both seeds retains the bridge for E1. Flat or shrinking share sends E1
   forward drafter-only.
6. **Overfit:** a widening old-train versus evaluation gap during Segment 1 is
   banked as the exposure ceiling on 41,969 anchors. It is not a stop.

## 7. Rule inventory

| Rule | Threshold | Cadence | Disposition | Named cliff |
|---|---|---|---|---|
| Non-finite loss | Any non-finite weighted loss | Every attempt | Stop | Garbage training |
| Non-finite gradient | Any non-finite active gradient element | Every attempt | Stop | Garbage training |
| Relative gradient explosion | Raw global norm above 10 times the prior-100 median for three consecutive attempts | Every attempt | Stop | Escaping numerical instability |
| Endpoint lineage | Exact four source SHAs and metadata | Startup | Stop | Corrupted lineage |
| Frozen lineage | Exact digest before and after | Startup/completion | Stop | Frozen-parameter mutation |
| Existing population | Locked manifest, document partition, exclusions, and fixed-subset hashes | Startup | Stop | Invalid dose segment |
| Expanded population | Hash-only amendment, minimum 100,000 new anchors, document quarantine, audit completeness | Before splice | Refuse splice | Invalid data intervention |
| Evaluation contact | Zero training access to fixed evaluation or confirmatory partitions | Startup/completion | Stop | Invalidated science |
| Pair schedule | Exact sampled-anchor sequence within each seed | Every update/resume | Stop | Invalid paired comparison |
| Splice identity | Only population version changes at recorded checkpoint boundary | Splice | Stop | Confounded intervention |
| Control identity | No-writeback hidden path bit exact | Every 1,000 | Stop | Invalid control |
| Resume durability | Model, optimizer, RNG, sampler, population, counters, and receipt hashes persist | Every 1,000 | Stop before updates | Unrecoverable lineage |
| Wilson quality floor | Lower bound at least 0.99 | Every 1,000 | Stop | Irreversible quality damage |
| Init-relative retention | More than 0.003 below step zero twice consecutively | Every 1,000 | Stop | Sustained quality drift |
| Directional gross miss | Primary below 0.40 or any auxiliary above 0.35 | Every 2,000 | Stop | Objective inversion |
| Directional repeated marginal miss | Primary 0.40-0.50 or auxiliary 0.25-0.35 twice consecutively | Every 2,000 | Stop | Sustained objective displacement |
| Directional target | Primaries at least 0.50 and auxiliaries at most 0.25 | Every 2,000 | Log | Intended objective balance |
| Endpoint readings | Slopes, one-percent gain, writeback share, train-eval gaps | Endpoint | Evidence only | None |

Only tripwires with named cliffs stop training. Exposure plateau, overfit onset,
and endpoint thresholds remain measurements.

## 8. Localization integration

The CPU localization tests only structural candidates observable without a
teacher or hindsight: stratum, position bucket, and intersections. A single
candidate can enter the full arms only if it clears the two-seed
document-bootstrap rule and strategy names it in the lock. Diagnostic quantiles
can explain the outcome but cannot become router features or masks.

The landed result found `3,270/8,031` rows helped in both seeds,
`2,904/8,031` harmed in both, and `1,857/8,031` with opposite signs. Only
`61/8,031` rows lost quality in both seeds. No structural candidate qualified.
The nearest apparent pocket, token position zero, contained only 24 rows and
failed both the 200-row minimum and the replicated interval rule. Coarser groups
with adequate support generally showed a benefit from writeback, not a
maskable harm pocket. The recommended locked arm is therefore unmasked.

Receipt:
`outputs/stage5/stage5_paper2_phase2_a2_localization_20260806/summary.json`
(SHA-256 `848afffcd0c1eaffed61cb1524870246a522689e436d32a0cfa560fbdb1ae222`).

## 9. Do not claim

- This is serving throughput.
- A single splice identifies a general unique-data scaling law.
- A fixed-population dose curve measures unique-data scaling.
- Post-hoc localization is confirmation evidence.
- Oracle headroom is achievable by a deployable selector.
- A plateau without flat fresh-data response and a small train-eval gap proves
  the substrate is bounded.
