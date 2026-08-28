# Recirculation Phase-A Completion - Strategy Handoff

Date: 2026-08-27. Coding-agent result relay under the paper-native
recirculation contract.

**Status:** `PHASE-A-COMPLETE / STRATEGY-KEY-UNRESOLVED`. The complete
score-only sweep finished with 111 registered measurements. Static
recirculation produced a real but narrow language-modeling improvement: the
best configuration reduced perplexity by 2.638%. It did not produce a
three-cell qualifying coarse region, and both selected configurations caused
large losses on the 461-row generative battery. No optimizer was constructed,
Phase B remains unauthorized, CONFIRM and EVAL-E remain sealed, and all Colab
sessions are closed.

## 1. Executive finding

The experiment separates three claims that should not be collapsed:

1. **The evaluator and mechanism are live.** Alpha-zero identity remained
   bit-exact, and increasing alpha produced a smooth local response around
   destination layer 4 and source layer 16.
2. **Static recirculation can improve teacher-forced next-token prediction.** The
   best registered cell moved perplexity from 13.9152 to 13.5480, a 2.638%
   reduction.
3. **The measured static intervention is not task-safe.** The two battery arms
   scored 96/461 and 108/461 against the 160/461 paper-native baseline and the
   151/461 neutral lower edge. The loss was concentrated in GSM8K.

Plain-language reading: the model contains a narrow route through which
recirculated state can improve average token prediction, but applying that
route uniformly at every active position destabilizes long-form task
generation. This is evidence for an effect, not evidence for a usable
inference architecture.

| Registered read | Result | Mechanical comparison |
|---|---:|---|
| Coarse sweep | 96 cells | complete |
| Refinement sweep | 13 cells | complete |
| Battery sweep | two cells, 461 rows each | complete |
| Best coarse cell | +1.141% PPL reduction | clears 1% alone |
| Largest coarse component | one cell | misses required size of three |
| Best refined cell | +2.638% PPL reduction | clears 1% |
| Battery rank 1 | 96/461 | 55 rows below neutral edge |
| Battery rank 2 | 108/461 | 43 rows below neutral edge |
| Strategy key | unresolved | registered mixed-pattern relay required |

## 2. Governing authorities and lineage

Authorities:

- `STRATEGY_RECIRCULATION_PROBE_HANDOFF_20260823.md`
  - Drive `1vBn5JpoGl2cz7WyqGJobJlPkpHmEad3I`
  - 20,116 bytes
  - SHA-256 `c0e358bf21cc0d330871ac5e58264d03ac4e22c0b078096ca0b93576e6b64db1`
- `STRATEGY_RECIRCULATION_SEMANTICS_RULING_20260827.md`
  - Drive `1uAjmvAinYaNezVXq1IGp_TPZvHicjBwl`
  - 15,353 bytes
  - SHA-256 `794548b168d3423ea2aaebec08317384e47e3635e6a4d18ba73137e8b162a553`
- `STRATEGY_RECIRCULATION_COMPARATOR_RULING_20260827.md`, citable v2
  - Drive `1zuedpHN5LBJq2RJfTB3fH3J0srwcrBrR`
  - 12,733 bytes
  - SHA-256 `e3d60feed134a46ca0ee968b8886cb7784c2aed05d4d805e8aa3b5d94407dbfd`
- `STRATEGY_RECIRCULATION_PHASE_A_AUTHORIZATION_20260827.md`
  - Drive `1bPp69nGPIB-RPchXcSILfcM2Ptrugj9t`
  - 9,078 bytes
  - SHA-256 `14342d76432c37e7c1f144fdd760525e993138b65d9c780b9fd87bd11342f794`

Executed lock:

- `training/paper2_recirculation_phase_a_lock.json`
- 5,579 bytes
- SHA-256 `087804f0ce7ddb9ed7ac0371e970b208a552bb296dd4ebd521059f0d3540524a`

Code lineage on `codex/bicameral-stage0`:

- Phase-A lock and runner: `f31f4aa3f1170d119f4970c162ab00a5c0614a3d`
- module entrypoint repair: `a58dd636e5510459cb65c4ae6d245db71d155dfb`
- LF-canonical summary transport: `94b4209cb5146988b50c0368e5ad109dd75b7ff9`
- corpus receipt correction: `04ac6299c5b0ce70bfd7819d192915e3f5e45b37`
- preserved false-stop publication: `c9b24a8e43e2c2a1426fa187ef0e6c3d5cfcad67`
- resume-checkpoint repair: `84344ef22016b9ea7fe5efe74d2e8f021da3b907`
- complete published result: `5c8a2aa9479190828bce7660cf33b70c189e13a8`

## 3. Exact runtime and evaluator identity

- GPU: NVIDIA A100-SXM4-40GB, 40,441 MiB
- PyTorch: 2.11.0+cu128
- CUDA: 12.8
- Transformers: 5.14.1
- datasets: 5.0.0
- dtype: BF16
- attention backend: SDPA
- evaluator: `paper_native_serial_first_iteration_readout_v1`
- weights frozen: true

The registered serial schedule, first-iteration readout, post-block tap, and
K/V ownership remained unchanged. The final identity recheck produced zero
logit differences and zero committed-cache differences at alpha zero.

## 4. Coarse sweep

The coarse grid evaluated 32 registered source/destination pairs at alpha
values 0.05, 0.10, and 0.16. Only one coarse cell cleared the 1% materiality
floor:

| Destination | Source | Alpha | PPL reduction |
|---:|---:|---:|---:|
| 4 | 16 | 0.16 | +1.141% |
| 4 | 16 | 0.10 | +0.932% |
| 2 | 6 | 0.16 | +0.795% |
| 2 | 6 | 0.10 | +0.639% |
| 6 | 18 | 0.16 | +0.540% |
| 4 | 16 | 0.05 | +0.506% |

The pre-registered connected-region selector returned three pairs:

- `(d=4, s=16)`: +1.141%;
- `(d=6, s=16)`: +0.371%;
- `(d=6, s=18)`: +0.540%.

Their mean was +0.684% and their minimum was +0.371%. The registered
component calculation therefore found no qualifying region at alpha 0.05 or
0.10 and only a one-cell component at alpha 0.16. The required component size
was three.

The response was spatially structured rather than uniformly positive. At
alpha 0.16, late source-22 routes were strongly harmful: `(d=10,s=22)` was
-4.045%, `(d=12,s=22)` was -2.881%, and `(d=14,s=22)` was -2.269%.

## 5. Registered refinements

At `(d=4,s=16)` under convex, norm-matched mixing, the alpha curve was smooth:

| Alpha | PPL reduction |
|---:|---:|
| 0.02 | +0.226% |
| 0.04 | +0.416% |
| 0.07 | +0.750% |
| 0.10 | +0.932% |
| 0.13 | +1.092% |
| 0.16 | +1.141% |
| 0.20 | +1.232% |
| 0.25 | +1.073% |

The mechanism refinements increased the next-token effect:

| Configuration | PPL reduction |
|---|---:|
| `(d=4,s=16,a=0.20)`, additive, norm-matched | **+2.638%** |
| `(d=4,s=16,a=0.20)`, convex, identity normalization | +2.229% |
| `(d=6,s=18,a=0.16)`, additive, norm-matched | +1.801% |
| `(d=6,s=16,a=0.16)`, additive, norm-matched | +1.751% |
| `(d=4,s=16,a=0.20)`, convex, 10-token ramp | +1.030% |

The best cell reduced mean NLL from 2.632980 to 2.606242 and perplexity from
13.915174 to 13.548039. The smooth alpha response and the gains under multiple
mixing/normalization variants argue against a dead path or a single numerical
artifact. They do not supply the missing spatial component or task safety.

## 6. Generative battery

The two unique configurations with the lowest mean NLL were selected before
generative scoring, as registered.

### 6.1 Rank 1: additive, norm-matched

Configuration: `(d=4,s=16,a=0.20)`.

- total: **96/461**, baseline 160, delta -64;
- fixes: 21; regressions: 85;
- GSM8K: 44/369 versus 107 baseline, 18 fixes and 81 regressions;
- MBPP: 30/67 versus 33 baseline, one fix and four regressions;
- Tier-1: 22/25 versus 20 baseline, two fixes and zero regressions.

Accuracy fell from 34.71% to 20.82%, a decline of 13.88 percentage points.

### 6.2 Rank 2: convex, identity normalization

Configuration: `(d=4,s=16,a=0.20)`.

- total: **108/461**, baseline 160, delta -52;
- fixes: 23; regressions: 75;
- GSM8K: 59/369 versus 107 baseline, 20 fixes and 68 regressions;
- MBPP: 27/67 versus 33 baseline, one fix and seven regressions;
- Tier-1: 22/25 versus 20 baseline, two fixes and zero regressions.

Accuracy fell from 34.71% to 23.43%, a decline of 11.28 percentage points.

Both row-level receipts reconcile exactly to the public summaries. Across all
rows in both arms, the position gate mean was 1.0 and the realized writeback
ratio mean was 0.20. The intervention was therefore uniformly active at the
registered amplitude. The data establish neither that gating would repair the
harm nor that it would fail; they identify the selectivity problem Phase B
would have to solve.

## 7. Registered decision map

Coding did not resolve a strategy key. The observed pattern is explicitly
mixed under the original map:

- `AFFORDANCE-PRESENT` does not apply because there is no qualifying
  three-cell coarse region and neither battery is neutral or better.
- `AFFORDANCE-PRESENT-BATTERY-ADDITIVE` does not apply because neither battery
  approaches 180/461.
- `PERPLEXITY-ONLY-WITH-BATTERY-HARM` requires a qualifying perplexity region;
  the battery-harm clause is met but the region clause is not.
- `ABSENT` does not apply because several registered refinement cells exceed
  the 1% perplexity floor.

The charter names isolated, non-contiguous winners and additive-only effects
as stop-and-relay patterns. Strategy must therefore adjudicate rather than
having coding stretch one of the four labels.

## 8. Blind predictions: mechanical comparison

Formal scoring remains strategy's responsibility. The direct comparisons are:

- **P-A1** `[0.5%, 2.5%]`: observed 2.638%, outside the upper edge by 0.138
  percentage points.
- **P-A2** at least one cell clears 1%: yes.
- **P-A3** no battery below neutral: no; both were below 151.
- **P-A4** best source-destination span in the upper half and wider than 16->8:
  yes; the best pair was 16->4, span 12.

## 9. Cost, resumability, and execution archaeology

The completed-work ledger reports:

- actual Phase-0-plus-A scientific cost: **3.631429 A100-hours**;
- expected cost at completion: 3.761908 A100-hours;
- actual/expected multiplier: 0.965316;
- cost ceiling: 8.0 A100-hours;
- completed measurements: 111.

The independent Colab assignment history reports six A100 allocations totaling
**4.422350 assigned hours**. The 0.790922-hour difference covers installs,
tests, discarded work after two backend reclamations, one pre-science false
resume stop, and a deliberately discarded partial battery pass used to rotate
before another expected backend timeout. This is the honest physical-spend
number and remains below the registered ceiling.

Execution incidents were preserved rather than hidden:

1. Before science, module invocation, CRLF-sensitive summary hashing, and one
   transcribed corpus hash were repaired under tests.
2. Two Colab backends were reclaimed after about 71 minutes. Checkpoints at
   measurements 24, 48, and 72 prevented loss of the completed prefix.
3. A resume from measurement 72 exposed a bookkeeping defect: a historical
   24-cell overrun checkpoint was incorrectly compared against cumulative
   72-cell cost. It stopped before new science. Commit `84344ef2` limits
   checkpoint rechecks to newly crossed checkpoints while keeping the global
   8-hour ceiling active. The superseded receipt remains in commit `c9b24a8e`.
4. Later checkpoints at 96, 109, and 110 allowed proactive VM rotation around
   the observed backend lifetime.
5. After the final runner printed
   `paper2_recirculation_phase_a_complete=true`, the local CLI timed out waiting
   for Jupyter's final reply. Direct inspection confirmed status complete,
   publication complete, archive hash exact, and all 111 measurements present.

The targeted local suite after publication passed: **17 passed**.

## 10. Durable artifacts

Published branch commit:

- `5c8a2aa9479190828bce7660cf33b70c189e13a8`

Downloaded completion archive:

- local title `recirculation-phase-a-artifacts.tar.gz`
- 620,397 bytes
- SHA-256 `273997d71be5d6f7fae9f799ac328f5a1db083cbf22504c1a26bd0fa7ccd8b3f`

Canonical archive receipts:

- Phase-A result summary: 6,649 bytes, SHA-256
  `5831b2b9c70e604a525e07edce22027f2738366a8281c527d1ef5a676f1b67f2`
- coarse cells JSON: 108,383 bytes, SHA-256
  `0f7b58a0e50a68777888fbc58c49b7cc8d9ce3753b160d6a2bf5e2be5d9c0bbe`
- refinement cells JSON: 14,834 bytes, SHA-256
  `1289b96bb06547cebec7315246d22be928f92b8df587f11ced5eeb9379c6b36c`
- status: 953 bytes, SHA-256
  `1c2a198b446ee5f7f66ea6dafde48145fa0e435ea3895a8d847d0084af61ade0`
- runner status: 1,204 bytes, SHA-256
  `1135f0aff2bb46c1f11a61de1414e7cb3588ef6c0d38be2c8198b1c729b2f31f`
- heatmap PNG: 147,767 bytes, SHA-256
  `2d2e2bda4efa99ae9d5d6190dd74db07c73db174b4913454551c274cdf53a5a2`
- heatmap SVG: 113,609 bytes, SHA-256
  `e4968293a43220643348ea6b890df5442c83d2ddd3481f66d278204fd9cbff84`
- battery rank-1 rows: 1,044,007 bytes, SHA-256
  `9a06aa04ff67837c65009eb94e31224bb3e6cc25926317c6246eb2e60afa3b3d`
- battery rank-2 rows: 1,258,532 bytes, SHA-256
  `cd3ffbb7aec104200efbfb482c0da600985f572db74c111be585f8e66a670ebe`

The final checkpoint archive recorded by the status receipt is 618,589 bytes,
SHA-256 `05ae3219aceb3a5f5da844abf5af9d108fb3c40be0f2391a0fb0ce110ae3e66c`.

## 11. Safety and seal state

- optimizer constructed: false
- optimizer steps: 0
- weights frozen: true
- Phase B training authorized: false
- CONFIRM scored: false
- EVAL-E scored: false
- strategy key resolved by coding: false
- active Colab sessions after closeout: 0

## 12. Do-not-claim boundaries

Do not claim any of the following from this wave:

- that Qwen has a broad or task-usable recirculation affordance;
- that the isolated best cell satisfies the registered region criterion;
- that perplexity improvement predicts task improvement;
- that static recirculation is harmless on reasoning tasks;
- that GSM8K harm has a proven causal mechanism;
- that an adaptive gate would necessarily repair the observed harm;
- that Phase B, CONFIRM, or EVAL-E has run;
- that any of the four strategy keys has already been assigned.

Supported claims are narrower: the paper-native mechanism is live and
bit-exact at alpha zero; static recirculation creates a narrow, smooth, and
material next-token improvement around `(d=4,s=16)`; the same always-on
intervention substantially harms the registered generative battery, especially
GSM8K.

## 13. Requested strategy adjudication

Coding recommends the following response:

```text
Bank Recirculation Phase A as complete under the paper-native evaluator. Record
the result as a registered mixed pattern: no qualifying three-cell coarse
region, multiple tuned cells above the 1% perplexity floor, and both selected
generative batteries below the 151/461 neutral lower edge. Do not run more
static score-only tuning and do not open CONFIRM or EVAL-E. Resolve the Phase-A
key explicitly, score P-A1 through P-A4, and decide whether the narrow NLL
signal plus task harm is sufficient to justify a separately locked Phase-B
adaptive-gating test. If Phase B is authorized, its burden is selective
delivery: preserve alpha-zero identity, keep task retention armed, and show
that token-conditioned writes retain the next-token gain without reproducing
the GSM8K regression channel.
```

Until an explicit strategy ruling and user ratification land, Phase B remains
blocked and no paid session will be opened.
