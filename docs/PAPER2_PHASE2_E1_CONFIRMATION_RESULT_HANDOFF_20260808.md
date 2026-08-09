# Handoff: Phase 2 E1 Read-Once Confirmation Result

Date: 2026-08-08. Audience: strategy and research agents. Status: complete,
published, and ready for strategy review. No further E1 scoring is permitted.

## 0. Executive verdict

The frozen, read-once E1 confirmation pass completed on the preregistered
EVAL-D population. The full writeback system beat its matched drafter-only
control with paired document-bootstrap intervals excluding zero in both seeds.
The causal mechanism effect therefore replicated under the registered primary
test.

The full systems narrowly missed the separately ratified 99.5 percent point
retention requirement in both seeds, although both Wilson lower bounds remained
well above the 99.0 percent requirement. The registered joint verdict is:

`EFFECT_REPLICATES_QUALITY_BOUNDARY_FAILS`

This is not a null result and it is not a full confirmation pass. It is a split
result: the incremental writeback effect replicated, while the endpoint lies
outside the preregistered preservation region.

The read-once lease is `spent_complete`. EVAL-E remains untouched. The A100 was
released after receipt verification.

## 1. Question and rationale

Option B established a small full-over-control EAL advantage on DEV after
20,000 updates:

| Seed | DEV relative gain | DEV document-bootstrap 95% CI |
|---:|---:|---:|
| 0 | 0.351% | [0.240%, 0.477%] |
| 1 | 0.496% | [0.370%, 0.655%] |

E1 asked whether that writeback increment would replicate on a frozen,
document-disjoint population without any additional optimization. It also
tested whether the promoted endpoints satisfied the confirmation-scale quality
margin ratified before the pass.

The design deliberately separated two claims:

1. Does writeback causally improve expected accepted length relative to a
   matched no-writeback control?
2. Does that improvement remain inside the ratified quality-preservation
   region?

The scripted verdict required both answers to be positive for a full pass.

## 2. Locked design

### Systems

Four final Option B checkpoints were frozen and hash-asserted:

| Seed | Arm | Checkpoint SHA-256 |
|---:|---|---|
| 0 | full system with writeback | `c1f5a6f217342ad721267a08d16c1bca75c8308d03f471db9d28ff3f319c777f` |
| 0 | drafter-only control | `8c9a7f6573bd268d67592b271a1b10a37c1f882681dc20efdbd8e9a5232bd681` |
| 1 | full system with writeback | `ccebda5c0b4bb1832194f690075b0be9ac1a96c557e63978ebb97a8632d278f7` |
| 1 | drafter-only control | `b26ca18e76fc60a622d6056b2957d31ee37e0c6c26dde88a3250b9bbd54a2424` |

No optimizer was constructed, no parameters were trainable, no optimizer step
occurred, and the endpoint parameter digests were unchanged after evaluation.

### Population and procedure

- Partition: frozen EVAL-D; EVAL-E untouched.
- Anchors: 8,000, balanced as 4,000 general and 4,000 code.
- Horizons per anchor: 1 through 4.
- Same cache and row order for all four arms.
- Primary statistic: paired full-minus-control difference in mean expected
  accepted length, separately by seed.
- Inference: 10,000 paired percentile cluster-bootstrap replicates over
  represented documents, seed `20260808`.
- Primary pass: 95 percent interval lower bound strictly above zero in both
  seeds.
- Quality pass: point retention at least 99.5 percent and Wilson 95 percent
  lower bound at least 99.0 percent in both full-system seeds.
- Former 1 percent EAL target: descriptive aspiration only, not a gate.
- Alpha: fixed at 0.5 as an unselected design prior.

## 3. Primary result

| Seed | Full EAL | Control EAL | Absolute gain | Relative gain | Paired document-bootstrap 95% CI | Primary |
|---:|---:|---:|---:|---:|---:|:---:|
| 0 | 1.953958 | 1.944340 | 0.009617 | 0.495% | [0.401%, 0.597%] | pass |
| 1 | 1.956061 | 1.944927 | 0.011135 | 0.572% | [0.459%, 0.752%] | pass |

Both independent required seed instances passed. The point effects are slightly
larger than their corresponding DEV estimates, and all four DEV and EVAL-D
intervals exclude zero. The one-percent exploratory target remained unmet in
both E1 seeds.

The correct primary reading is that the full writeback path contributes a
small, reproducible increment beyond the matched trained drafter path on this
frozen population. The intervals provide inference over the represented
documents within each fixed training seed; two seeds do not estimate a seed
population.

## 4. Quality result

| Seed | Baseline-correct decisions | Retained | Lost | Point retention | Wilson 95% lower | Point-gate miss | Quality |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 0 | 22,642 | 22,520 | 122 | 99.461% | 99.357% | 0.039 percentage points | fail |
| 1 | 22,642 | 22,511 | 131 | 99.421% | 99.314% | 0.079 percentage points | fail |

The Wilson requirement passed in both seeds. The point requirement failed in
both. The misses are narrow relative to the 99.5 percent boundary, but they are
directionally replicated and cannot be rounded into a pass after the fact.

The matched controls retained 100 percent of baseline-correct decisions. The
quality cost therefore belongs to the tested writeback system, not to the
shared drafter-only training path.

The EVAL-D quality losses, 0.539 and 0.579 percentage points, are somewhat
larger than the preregistered DEV expectation of approximately 0.35 to 0.45
points. The confirmation result supports a measured Pareto tradeoff, not a
quality-neutrality claim.

## 5. Secondary localization

### By stratum

| Seed | Stratum | Full EAL | Control EAL | Relative full gain |
|---:|---|---:|---:|---:|
| 0 | code | 2.280568 | 2.270818 | 0.429% |
| 0 | general | 1.627347 | 1.617863 | 0.586% |
| 1 | code | 2.282636 | 2.272942 | 0.427% |
| 1 | general | 1.629487 | 1.616913 | 0.778% |

The effect is not carried by only one workload stratum. All four stratum-level
point estimates are positive. General text is larger in both seeds, but this
secondary was descriptive and was not separately powered as a gate.

### By position bucket

| Position bucket | Anchors | Seed 0 full-minus-control EAL | Seed 1 full-minus-control EAL |
|---|---:|---:|---:|
| 0 | 22 | 0.118057 | 0.231353 |
| 1-3 | 55 | 0.001268 | 0.002929 |
| 4-31 | 524 | 0.002246 | 0.000109 |
| 32-127 | 1,658 | 0.005804 | 0.008758 |
| 128+ | 5,741 | 0.011056 | 0.012062 |

The large position-zero values are based on only 22 anchors and should not be
promoted as a stable effect. The broadest evidence is the positive increment in
the 128-plus bucket, which contains most anchors and is similar across seeds.

### Mechanism use and selector headroom

| Seed | Mean bridge gate | Mean draft gate | Probe KL versus EAL correlation | Probe top-1 versus EAL correlation | Quality-safe oracle headroom |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.0840 | 0.9823 | 0.198 | 0.027 | 2.84% relative |
| 1 | 0.0885 | 0.9815 | 0.248 | 0.005 | 2.99% relative |

The bridge is active and its use level closely matches the final Option B
operating point. The current probe signals remain weak selectors. A
quality-safe oracle could in principle support materially more gain, but the
observed effect captures only roughly 17 to 19 percent of that oracle ceiling.
This is evidence of unrealized arbitration headroom, not evidence that a
deployable router already exists.

Among gate-open rows showing accepted-length improvement, final quality
worsened on approximately 1.35 percent in seed 0 and 1.48 percent in seed 1.
That overlap makes selective suppression of harmful writes a plausible repair
target, but EVAL-D must not be used to fit that selector.

## 6. Registered verdict

The primary effect passed in both seeds. The quality endpoint failed in both.
The locked table therefore maps the result to:

`EFFECT_REPLICATES_QUALITY_BOUNDARY_FAILS`

Permitted claim:

> On the frozen EVAL-D population, the trained writeback system produced a
> small positive EAL increment over a matched no-writeback control in both
> tested seeds, while narrowly missing the preregistered point-retention floor
> in both seeds.

Prohibited claim:

> E1 fully passed, or the mechanism improved acceptance without a quality cost.

## 7. Scientific interpretation

E1 strengthens the causal mechanism claim. The comparison is paired, the
control shares the drafter training path, the only functional distinction is
writeback, and the effect replicated on a frozen document-disjoint population.
The primary effect is therefore no longer only a DEV observation.

At the same time, the result locates the current boundary sharply. The bridge
can improve expected acceptance, but the learned arbitration does not yet keep
that gain inside the chosen preservation region. Because the control remains
bit-exact and the quality miss repeats in both seeds, this should be treated as
a real property of the present system rather than random noise or a scoring
artifact.

The result favors a quality-allocation problem over a basic state-construction
problem. The bridge state is useful, the writeback path moves EAL in the desired
direction, and oracle headroom remains large. What is missing is reliable
per-position discrimination between helpful and harmful writes.

## 8. Execution and receipt integrity

### Read-once discipline

- Lock commit: `ebe4ea4b27d0633ca5b5dcaabc1cb3cdacc4ca37`.
- Final launcher commit: `5ae94582ddab8d95bb97cba96afc3c7b9baca3de`.
- Result commit: `7f5743b4577c1f3b22efe39290bb37dce6a1ac4b`.
- Summary SHA-256:
  `3853b34b78593287317fbd0d95ae056045593acdead873994202faaa9f749e58`.
- Lease status: `spent_complete`.
- `read_once_scoring_spent=true`.
- `optimizer_constructed=false`; `optimizer_steps=0`.
- `training_started=false`; `eval_e_touched=false`.

Two infrastructure faults occurred before score exposure:

1. The launcher invoked the runner by file path, leaving the repository root
   off `sys.path`; this failed on `import training` before staging or scoring.
   Fixed in `60569fab` by module execution, with a regression test.
2. The runner omitted the durable `private/a1/` path component. It failed while
   staging A1 inputs, before the lease. Fixed in `5ae94582` by resolving both A1
   paths and SHA-256 values from the banked A1 completion receipt.

The successful run claimed the lease once, completed all four arms and the
10,000-replicate bootstrap, and wrote a complete summary to Drive. Anonymous
CLI Git transport then failed during publication only. The byte-identical Drive
summary was downloaded, hash-verified, and published from the authenticated
local checkout without rerunning evaluation. Drive carries a publication
recovery receipt recording that sequence.

### Document-count terminology erratum

The locked registration records `actual_document_count=132`. The cache freeze
also records 132 frozen source documents. The selected 8,000-anchor population,
however, represents 131 unique documents, and the completed bootstrap correctly
resampled those 131 observed document clusters.

The difference is definitional: one document in the frozen source universe
contributed no selected anchor under the locked selection rule. It does not
change membership, scores, or bootstrap arithmetic. However, the machine field
was mislabeled and the runner did not assert source-universe count separately
from represented-bootstrap count. This should be recorded as a post-lock
terminology erratum and corrected in future registrations. The result should be
reported as 8,000 anchors from 131 represented documents, drawn from a frozen
132-document source universe.

## 9. Limitations and do-not-claim boundaries

- Two seeds are required replications, not a sample supporting inference over
  training seeds.
- Document-bootstrap intervals describe the represented EVAL-D document
  population under this fixed protocol.
- The effect is approximately 0.5 percent, not at least 1 percent.
- Alpha 0.5 was a design prior, not selected or shown optimal.
- EAL is a simulation metric, not serving throughput or wall-clock speed.
- The current endpoint is not quality-neutral and did not meet the registered
  preservation region.
- Oracle headroom is an upper descriptive construction, not a demonstrated
  deployable policy.
- No claim extends to EVAL-E, other model families, other teachers, or other
  decoding protocols.
- No secondary analysis rescues the failed joint quality requirement.

## 10. Questions for strategy review

1. Ratify the split reading: primary causal effect confirmed; joint E1 endpoint
   not confirmed because the quality boundary failed.
2. Decide whether Paper Two should bank this as the final measured Pareto
   boundary or authorize an E2 quality-repair program.
3. If E2 opens, confirm that EVAL-D is analysis-only and cannot be used for
   threshold fitting, model selection, or iterative repair.
4. Decide whether E2 should focus narrowly on selective write suppression or
   include broader refiner or alpha changes. The current evidence favors the
   narrow arbitration question first.
5. Decide the role of EVAL-E. Recommended: preserve it for one newly locked
   confirmation after a DEV-only repair recipe is selected, rather than spend
   it on diagnostics.
6. Ratify the 132-source versus 131-represented document-count erratum as
   non-outcome-changing and require separate fields and assertions going
   forward.
7. Decide whether a roughly 0.5 percent EAL gain against a roughly 0.54 to 0.58
   point retention cost is scientifically sufficient for the main positive
   result, or should be framed primarily as a controllability boundary.

## 11. Recommended next sequence

1. Bank E1 exactly as the scripted split verdict; do not rerun it.
2. Add the source-versus-represented document-count erratum and separate these
   fields in future registration schemas.
3. Update the Paper Two status and claim ledger only after strategy ratifies
   the interpretation language.
4. If E2 is authorized, begin with a read-only DEV localization of the harmful
   write subset and candidate pre-write signals. Do not fit on EVAL-D.
5. Draft and lock a bounded quality-repair protocol. Its central test should be
   whether the EAL increment can be retained while recovering the 99.5 percent
   point-retention floor in both seeds.
6. Keep EVAL-E untouched until a single repair recipe and confirmation rule are
   locked.
7. Do not spend GPU on a broad alpha matrix, more unchanged Option B dose, or a
   selector sweep before the quality-repair decision is made.

## 12. Plain-language summary

The added recurrent writeback mechanism worked on the final unseen test. In
both runs it improved the model's expected accepted output length relative to
an otherwise matched system that could not write the recurrent state back. The
gain was small, about one-half of one percent, but statistically positive in
both runs and present in both general text and code.

The mechanism also changed a small number of answers that the base system had
gotten right. It retained about 99.42 to 99.46 percent of those decisions,
slightly below the 99.5 percent requirement chosen before the test. That means
the scientific effect replicated, but the current system did not pass the full
quality-preservation standard.

The next decision is not whether the bridge carries useful information; E1 now
answers that positively. The decision is whether to stop with an honest,
measured speed-quality tradeoff or open one tightly scoped repair phase aimed at
learning when the bridge should remain silent. Any repair must be developed on
DEV and confirmed once on still-untouched EVAL-E, not tuned on the now-spent
EVAL-D results.

## 13. Canonical artifacts

- E1 result:
  `outputs/stage5/stage5_paper2_phase2_e1_confirmation_20260808/summary.json`
- Locked preregistration:
  `docs/PAPER2_PHASE2_E1_CONFIRMATION_PREREGISTRATION_LOCKED_20260808.md`
- Machine registration:
  `training/paper2_phase2_e1_confirmation_preregistration.json`
- EVAL-D freeze receipt:
  `outputs/stage5/stage5_paper2_phase2_e1_eval_d_20260808/receipts/e1_eval_d_freeze_summary.json`
- Option B result handoff:
  `docs/PAPER2_PHASE2_OPTION_B_FINAL_RESULT_HANDOFF_20260808.md`
- Durable Drive run:
  `/content/drive/MyDrive/recurrent-qwen-svgd-artifacts/stage5/stage5_paper2_phase2_e1_confirmation_20260808/`
