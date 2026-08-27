# Recirculation Phase-0 Anchor Stop — Strategy Handoff

Date: 2026-08-27. Coding-agent hard-stop report and ruling request.

**Status:** `HARD-STOP / AWAITING STRATEGY`. The paper-native evaluator passed
its bit-exact identity gate but scored 160/461 at alpha = 0, versus the
registered 162/461 result from the earlier Stage 2B-S K1 execution graph. The
run stopped before Qwen timing, the Gemma anchor, any Phase-A grid cell, any
optimizer, or any sealed evaluation. The A100 is released.

## 1. Executive finding

This is not a two-row model regression. It is a comparator-identity defect in
the lock.

The 162 score was produced by `Stage2BScheduleGraph` under the prior batched
full-sequence execution schedule. The amended recirculation authority requires
the paper-native token-sequential dependency graph and makes that schedule part
of evaluator identity. Under that required evaluator, alpha = 0 is bit-exact
against the intact model for both scored logits and future-visible committed
K/V, but greedy generation scores 160/461.

The correct sets expose broad boundary churn:

| Read | Correct | Tier-1 | GSM8K | MBPP |
|---|---:|---:|---:|---:|
| Prior Stage 2B-S K1 graph | 162 | 20 | 108 | 34 |
| Paper-native serial alpha = 0 | 160 | 20 | 107 | 33 |

The sets intersect on 147 rows. Fifteen prior-correct rows become wrong and
thirteen prior-wrong rows become correct: 28 changed row outcomes, net -2.
That pattern is consistent with BF16 schedule-sensitive greedy boundary churn,
not with two isolated scoring errors.

Strategy must decide which estimator owns the baseline. Coding recommends
ratifying 160 as the paper-native same-estimator baseline, retaining 162 as a
descriptive prior result only, and preserving the registered **relative** bars:
+20 rows for additivity and -9 rows for the neutral lower edge. That yields
paper-native absolute thresholds of 180 and 151. No threshold should move
without a written amendment.

## 2. Authorities and code provenance

Governing documents:

- `STRATEGY_RECIRCULATION_PROBE_HANDOFF_20260823.md`
  - Drive `1vBn5JpoGl2cz7WyqGJobJlPkpHmEad3I`
  - 20,116 bytes
  - SHA-256 `c0e358bf21cc0d330871ac5e58264d03ac4e22c0b078096ca0b93576e6b64db1`
- `STRATEGY_RECIRCULATION_SEMANTICS_RULING_20260827.md`
  - Drive `1uAjmvAinYaNezVXq1IGp_TPZvHicjBwl`
  - 15,353 bytes
  - SHA-256 `794548b168d3423ea2aaebec08317384e47e3635e6a4d18ba73137e8b162a553`
- Executed lock: `training/paper2_recirculation_phase0_lock.json`
  - 2,409 bytes
  - SHA-256 `fc6da70c54ec46fe82d9f291a37c959c9a90c97c6308a3ae20133b8d41106c30`

Code lineage:

- evaluator implementation: `20e79322cf0aa08ada71d6e775664af50caaa53d`
- Colab transport: `1accbf4564b7b65980042cd2b2f1d14e0728209f`
- line-ending-portable panel lock: `dc43f01a9210e1befd536dedc18cdbf2cb2f1815`
- branch: `codex/bicameral-stage0`

The portability commit accepts only the two proven transports of the exact
JSONL bytes: CRLF and canonical LF. It records both the transport hash and the
canonical hash, rejects lone carriage returns, and does not change rows,
scores, evaluator semantics, or the 160/162 gate.

## 3. What ran

Pinned runtime:

- GPU: NVIDIA A100-SXM4-40GB, 40,441 MiB
- PyTorch: 2.11.0+cu128
- CUDA: 12.8
- dtype: BF16
- attention backend: SDPA
- Transformers: 5.14.1

Completed stages:

1. authority and panel preflight;
2. public corpus construction;
3. Qwen model load;
4. graph receipt generation;
5. Qwen alpha = 0 identity gate;
6. all 461 generative anchor rows.

The identity gate passed exactly:

- scored-logit maximum absolute difference: 0.0;
- committed-cache maximum absolute difference: 0.0;
- mismatched cache tensors: none.

The generative anchor then ran for 1,705.1757 seconds and produced:

- rows: 461;
- correct: 160;
- expected by the unamended lock: 162;
- row file: 1,748,126 bytes;
- row SHA-256:
  `79fb3b1a28780b24b2a8db0a99f701f8ab86af707086de778f975696e072e41e`.

The registered hard stop fired immediately.

## 4. Why 160 and 162 are different estimators

The original handoff called the old K1 graph “base-identical.” That is true at
the architecture/weights level, but it is not sufficient under this program's
evaluator-identity doctrine.

The prior 162 receipt came from the Stage 2B-S wrapper running the intact K1
graph under its full-sequence generation path. The semantics ruling later made
the token-sequential paper graph the registered recirculation evaluator:

- positions execute serially because later shallow computation consumes deep
  K/V constructed from earlier positions;
- scored logits come only from the first stack;
- the recirculated stack constructs future-visible K/V;
- batching is permitted only across independent sequences;
- schedule, batching, precision, reader, cache ownership, and serving graph are
  evaluator identity.

At alpha = 0, the paper-native graph is an exact intact-model evaluator under
its own schedule. The identity receipt proves that. It does not prove that its
BF16 greedy trajectory must reproduce a score obtained under a different
batch/execution schedule. Requiring 162 while simultaneously requiring the
paper-native serial graph therefore imports an outcome from one estimator as a
hard gate on another.

The row churn is the decisive evidence. If two scoring rows alone were broken,
the symmetric difference would be two. Instead, 28 greedy outcomes move while
the total shifts by only two.

## 5. Exact changed rows

Prior correct, paper-native wrong (15):

```text
gsm8k-evaluation-1085  gsm8k-evaluation-1107  gsm8k-evaluation-1112
gsm8k-evaluation-1160  gsm8k-evaluation-1176  gsm8k-evaluation-1186
gsm8k-evaluation-1317  gsm8k-evaluation-190   gsm8k-evaluation-212
gsm8k-evaluation-335   gsm8k-evaluation-495   gsm8k-evaluation-836
gsm8k-evaluation-853   mbpp-240                mbpp-246
```

Prior wrong, paper-native correct (13):

```text
gsm8k-evaluation-10    gsm8k-evaluation-103   gsm8k-evaluation-1042
gsm8k-evaluation-1229  gsm8k-evaluation-210   gsm8k-evaluation-511
gsm8k-evaluation-674   gsm8k-evaluation-736   gsm8k-evaluation-742
gsm8k-evaluation-901   gsm8k-evaluation-924   gsm8k-evaluation-941
mbpp-138
```

The machine-readable companion receipt carries the same sets.

## 6. Safety, seal, and cost state

- optimizer constructed: false
- optimizer steps: 0
- Phase B authorized: false
- Phase-A timing cell run: false
- Phase-A heatmap cells run: 0
- Gemma loaded or scored: false
- CONFIRM scored: false
- EVAL-E scored: false
- paid Colab sessions after closeout: 0

No recirculation effect has been measured. This stop is not evidence for or
against `AFFORDANCE-PRESENT`; it is exclusively a baseline-estimator ruling.

## 7. Durable receipts

Failure archive:

- Drive `1QjN3pr8fak1RtEkqyaC5_B88TmWPAN8X`
- title `PAPER2_RECIRCULATION_PHASE0_FAILURE_ARCHIVE_20260827.tar.gz`
- 598,627 bytes
- local/source SHA-256
  `927add34c0bcafc063c6732ebb177e107797d36180cbdc53a7b3fdcf977a14d2`

Companion receipt:

- `PAPER2_RECIRCULATION_PHASE0_ANCHOR_STOP_RECEIPT_20260827.json`

The archive includes runtime, graph, identity, anchor, row-level, corpus, log,
and failed-status receipts. The Drive connector reports the same 598,627-byte
size as the source archive.

## 8. Mechanical transport finding

Before scientific scoring, Linux Git materialized the frozen panel with LF
bytes (`c0e15a...2615`) while the lock carried the Windows CRLF hash
(`2e7e1d...70642`). The files were semantically and canonically identical:
1,024 rows, exactly 1,024 CRLF substitutions, no content difference after
normalization. The exact locked CRLF file was restored for this run.

Commit `dc43f01a...` now validates canonical LF content while receipting the
actual transport hash. This removes a portability-only failure class without
loosening row identity.

## 9. Requested rulings

### R-A — baseline estimator

**Recommended:** ratify the observed 160/461 row receipt as the paper-native
alpha = 0 baseline. Keep 162/461 in the record as the Stage 2B-S K1 result, not
as an absolute score requirement on the new evaluator.

Alternative: require the paper-native evaluator to reproduce 162. Coding does
not recommend this because doing so would require changing the registered
schedule or searching implementation details against the desired score.

### R-B — interpretation bars

**Recommended:** preserve the registered deltas, not the old absolute totals:

- additive: paper-native baseline +20 = at least 180/461;
- neutral lower edge: paper-native baseline -9 = at least 151/461.

If strategy intended 182 and 153 as estimator-independent absolute bars, state
that explicitly instead. Coding will not infer it.

### R-C — receipt preservation and resume

**Recommended:** preserve the failed v1 receipt unchanged. After amendment,
create a v2 adjudication receipt that points to the already-complete 461-row
file and authority SHA, then resume at Qwen timing. Do not spend another
28 minutes regenerating identical rows unless strategy requires an independent
replication.

### R-D — Phase-A gate

Confirm that Phase A remains blocked until the resumed Phase 0 completes Qwen
timing, the Gemma directional anchor, and the cost projection, and those results
are relayed. Coding assumes yes.

## 10. Recommended binding response

The minimal executable ruling is:

```text
Ratify R-A through R-D. The paper-native token-sequential alpha=0 evaluator
owns the recirculation comparator at 160/461. Preserve relative bars (+20 and
-9), yielding 180 and 151. Preserve the failed v1 receipt; issue a v2
adjudication receipt over the same row SHA and resume after the anchor without
regeneration. Phase A remains blocked on the rest of Phase 0 and its relay.
```

Until that or a different explicit ruling lands, no compute will be relaunched
and no scientific constant will change.
