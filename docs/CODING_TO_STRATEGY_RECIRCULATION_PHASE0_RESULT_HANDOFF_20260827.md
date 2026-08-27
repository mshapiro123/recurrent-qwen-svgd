# Recirculation Phase-0 Completion - Strategy Handoff

Date: 2026-08-27. Coding-agent result relay under the paper-native recirculation
contract.

**Status:** `PHASE0-PASS / PHASE-A-BLOCKED-PENDING-RELAY`. The corrected
paper-native evaluator passed both bit-exact identity gates, reproduced the
registered directional Gemma anchor with an 8.268% perplexity reduction, and
projects the complete Phase-0-plus-A plan at 3.762 A100-hours against the
8-hour ceiling. No Phase-A grid cell ran. No optimizer was constructed.
CONFIRM and EVAL-E remain sealed. The paid Colab session is closed.

## 1. Executive finding

Phase 0 validates the evaluator and clears the pre-registered economic gate.

| Gate/read | Result | Registered reading |
|---|---:|---|
| Qwen alpha=0 logits | max abs diff 0.0 | bit-exact pass |
| Qwen alpha=0 committed K/V | max abs diff 0.0 | bit-exact pass |
| Gemma alpha=0 logits and K/V | max abs diff 0.0 | bit-exact pass |
| Paper-native battery anchor | 160/461 | passed by the ratified same-estimator adjudication |
| Qwen timing cell `(16,8,0.10)` | PPL 13.9152 -> 13.8640 | +0.368%, timing pilot only |
| Gemma directional cell `(11,4,0.15)` | PPL 20.9834 -> 19.2485 | +8.268%, directional gate pass |
| Full Phase 0 + A projection | 3.7619 A100-hours | below 8-hour ceiling |

The Qwen timing cell is mildly positive but below the registered 1% materiality
floor. It is not an affordance verdict and was not selected from the registered
grid. Its purpose was timing. The Gemma result is the required directional sign
check, not a claim of exact magnitude replication against the paper.

The next scientific question is therefore still open and now properly posed:
does a contiguous Qwen layer region clear the 1% perplexity floor without
damaging the matched 461-row battery? Only Phase A answers that question.

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
- `STRATEGY_RECIRCULATION_COMPARATOR_RULING_20260827.md`
  - Drive `1gOT00J5xOeUA4jBmUr4BQMeIrgQWXOgk`
  - 9,536 bytes
  - SHA-256 `f4127faf06b026885613329bfee28d324f970c43e0aae9e61739eff20b8f9785`

Executed lock:

- `training/paper2_recirculation_phase0_lock.json`
- 3,499 bytes
- SHA-256 `7360ccc2e0c379b51f7d5f8705f0e36afe18fc4fa8eb13ed0134b5d51f9ed208`
- lock status `LOCKED_PHASE0_PHASE_A_BLOCKED_PENDING_RELAY`

Code lineage on `codex/bicameral-stage0`:

- paper-native evaluator: `20e79322cf0aa08ada71d6e775664af50caaa53d`
- CLI-native artifact transport: `1accbf4564b7b65980042cd2b2f1d14e0728209f`
- line-ending-portable panel identity: `dc43f01a9210e1befd536dedc18cdbf2cb2f1815`
- comparator adjudication and resumability: `785d52d4a37e7a6b5567313ad3f49afd8664603d`
- publication-path repair: `3aabe0329d9a7eb1c1bf4a3577126d5c22b9743f`

## 3. Exact runtime

- GPU: NVIDIA A100-SXM4-40GB, 40,441 MiB
- PyTorch: 2.11.0+cu128
- CUDA: 12.8
- Transformers: 5.14.1
- datasets: 5.0.0
- dtype: BF16
- attention backend: SDPA

The evaluator remained the serial paper-native reference. No optimized
execution schedule was introduced, so no schedule-equivalence gate was needed.

## 4. Comparator adjudication and preservation

The original v1 battery receipt remains unchanged:

- 461 rows;
- 160 correct versus the superseded cross-estimator expectation of 162;
- 302 bytes;
- SHA-256 `254789f708c88c41d042774130e5d2cbf60fd5887da0b72f05fdf12e670903c0`;
- row file 1,748,126 bytes, SHA-256
  `79fb3b1a28780b24b2a8db0a99f701f8ab86af707086de778f975696e072e41e`.

The v2 adjudication receipt points to those exact rows and records:

- paper-native comparator: 160/461;
- prior Stage2BS K1 comparator: 162/461 under its own evaluator;
- additive bar: 180/461 (`160 + 20`);
- neutral lower edge: 151/461 (`160 - 9`);
- generation replayed: false;
- status: `passed_by_strategy_adjudication`.

No 28-minute regeneration was performed. The failed v1 receipt is preserved as
archaeology, exactly as R-C required.

## 5. Phase-0 measurements

### 5.1 Identity and graph contract

Qwen and Gemma both produced:

- scored-logit maximum absolute difference: 0.0;
- committed-cache maximum absolute difference: 0.0;
- mismatched committed-cache tensors: none;
- overall bit-exact identity: true.

The graph receipt is 329,255 bytes, SHA-256
`7954d2ad8ed56a5aa0d499632edff079ffbc6bda437b30d8d5414cee8322bc31`.
It freezes the first-iteration readout, post-block tap convention, serial token
schedule, and scored/provisional/committed/discarded K/V ownership required by
the semantics ruling.

### 5.2 Qwen timing pilot

Registered cell: source 16, destination 8, alpha 0.10, 32 windows of 1,024
tokens (32,736 predicted tokens).

| Read | Intact | Recirculated |
|---|---:|---:|
| elapsed seconds | 31.5917 | 63.5160 |
| mean NLL | 2.632980 | 2.629292 |
| perplexity | 13.915174 | 13.863955 |

Measured perplexity reduction: **0.3681%**. The recirculated cell took about
2.01 times the intact evaluation time. This is a useful low-cost positive sign,
but it is below the registered 1% materiality floor and is not a selected-grid
result.

### 5.3 Gemma directional anchor

Registered cell: source 11, destination 4, alpha 0.15, 128 windows of 1,024
tokens (130,944 predicted tokens).

| Read | Intact | Recirculated |
|---|---:|---:|
| elapsed seconds | 216.9743 | 454.3670 |
| mean NLL | 3.043730 | 2.957431 |
| perplexity | 20.983361 | 19.248450 |

Measured perplexity reduction: **8.2680%**. Direction is positive, so the
registered implementation sanity gate passes. This validates the evaluator
against the paper's central qualitative result before a Qwen conclusion is
drawn.

### 5.4 Complete cost projection

The cost receipt prices the actual serial evaluator that passed the gates:

- elapsed Phase 0: 2,497.65 seconds;
- 96 coarse cells across 32 source/destination pairs: 6,173.76 seconds;
- 13 refinement perplexity cells: 949.56 seconds;
- two battery cells: 3,921.90 seconds;
- projected total: 13,542.87 seconds = **3.7619 A100-hours**;
- registered ceiling: 8.0 A100-hours;
- headroom: 4.2381 A100-hours;
- gate: `within_ceiling=true`.

No pruning or optimized schedule is required on cost grounds.

## 6. Free churn rider

The non-gating desk rider prices the 28 row outcomes that differed between the
prior Stage2BS K1 graph and the paper-native alpha=0 graph:

- 25 GSM8K, three MBPP, zero Tier-1;
- both flipped and stable groups have a median generated length of 256 tokens;
- GSM8K mean generated length is similar (231.4 flipped versus 235.7 stable);
- flipped-row minimum answer-token margins are lower, including within GSM8K,
  but the statistic is strongly zero-inflated.

Bounded reading: the churn is task-concentrated and consistent with greedy
boundary sensitivity; generation length does not explain it. This is
descriptive, not causal, and gates nothing.

Receipt: `PAPER2_RECIRCULATION_PHASE0_CHURN_DESK_RIDER_20260827.json`, 4,138
bytes, SHA-256
`cc6bcef4d5fcfe241a496257aece6a6a660f47d700dabd0b67c1153b69fa11b3`.

## 7. Post-run publication defect

The scientific child process ended with
`status=phase0_pass_awaiting_relay`. The wrapper then copied the public
receipts and attempted a normal `git add` inside the repository's broadly
ignored `outputs/` tree. Git correctly rejected the ignored path, so the
top-level wrapper status is `failed` even though all Phase-0 gates and receipts
had already completed.

This was a packaging-only defect after measurement. Coding:

1. downloaded and hashed the durable archive before releasing the VM;
2. reconstructed the public summary from the exact downloaded receipts;
3. changed publication to the repository's existing lightweight-artifact
   whitelist plus scoped `git add -f`;
4. added a regression test proving JSON receipts are staged while checkpoints
   remain excluded;
5. ran the recirculation test selection: **11 passed, 171 deselected**.

No model computation was repeated and no scientific receipt was altered.

## 8. Durable artifacts

Public summary:

- path `outputs/stage5/stage5_paper2_recirculation_20260827/phase0/summary.json`
- 7,041 bytes
- SHA-256 `e64009c58c3223c94aede72ffa71edaaa4cfe88b0fa630b7502deb8a0918f459`

Downloaded completion archive:

- title `PAPER2_RECIRCULATION_PHASE0_COMPLETION_ARCHIVE_20260827.tar.gz`
- 602,548 bytes
- SHA-256 `268ea5089e3143553a1b2ae149dc6c2eec6f3656be5cea3cc2133bec167ab3f3`

The archive contains runtime, graph, identity, v1/v2 anchor, row-level,
corpus, Qwen timing, Gemma anchor, cost, log, and top-level status receipts.
Drive publication IDs are recorded in the companion publication receipt.

## 9. Safety and seal state

- optimizer constructed: false
- optimizer steps: 0
- Phase-A grid cells: 0
- Phase B training authorized: false
- CONFIRM scored: false
- EVAL-E scored: false
- active Colab sessions after closeout: 0

Phase 0 measured evaluator integrity, one timing cell, and the registered Gemma
sanity anchor. It did not measure a Qwen heatmap or resolve any Phase-A key.

## 10. Do-not-claim boundaries

Do not claim any of the following from this wave:

- that Qwen has or lacks a usable recirculation affordance;
- that the Qwen timing cell clears the 1% materiality floor;
- that the Gemma magnitude exactly replicates the paper;
- that 160 and 162 are interchangeable model-quality estimates;
- that the churn rider establishes a causal source of row flips;
- that any Phase-A or Phase-B result exists.

The supported claims are narrower: the paper-native evaluator is bit-exact at
alpha=0, it reproduces the published Gemma effect direction, and the complete
registered score-only Qwen probe fits the cost ceiling.

## 11. Requested strategy ruling

Coding recommends the following binding response:

```text
Bank Recirculation Phase 0 as PASS under the paper-native evaluator. The Gemma
directional anchor passed at +8.268% perplexity reduction, and the complete
Phase-0-plus-A projection is 3.7619 A100-hours within the 8-hour ceiling.
Authorize Phase A exactly under the existing amended charter and comparator
bars (160 baseline, 180 additive, 151 neutral lower edge), with no Phase B
training. Preserve all Phase-0 receipts and the post-run publication-defect
record. Resolve Phase-A keys only after the complete score-only handoff.
```

Until that or a different explicit ruling lands, Phase A remains blocked and
no paid session will be opened.
