# Paper Two Bicameral W0 Result Handoff

Date: 2026-08-24  
Status: W0 complete; all W0 hard gates passed; no training performed  
Implementation commit at execution: `fd7651bf4a31f1d5f65e0a36715bec3c95fa8b53`  
Governing adjudication: Drive `1sDoD6HXM63saL2KrWfL9AmD06S0gMOCH`, 15,854 bytes, SHA-256 `056337f3cef59a190f5511087fa7704eec44b5d837d937b21d1b605e99624d7f`

## 1. Plain-language summary

The Bicameral evaluator now behaves exactly like the locked reference when its gates are closed. The earlier failure was not a weakness of the architecture: concatenating the two branches into a doubled BF16 batch changed GPU arithmetic. W0 removed that optimization and made the two branch calls sequential. On the pinned A100 runtime, the repaired graph produced exactly the same 9,723,904 logits as the base graph, with zero differing values.

The intended operating gates also work mechanically. With `g_A = g_B = s_A = s_B = 1`, the two branches are substantially different under both seed-specific initializers rather than collapsing onto the same state. The cold-start gradient contract passes: the gates and mean combiner receive live finite gradients, while the disagreement combiner and bank gains remain exactly zero-gradient at initialization as designed.

The sequential repair is slower than the invalid batch-concatenated evaluator, but caching remains negligible relative to full evaluation. The revised projection is 0.00735 A100-hours for both seeds across the 256 training rows, 461-row slice, and 2,048-row panel. W0 therefore clears the engineering block for W1 and Step-1. It does not clear the registered residual-correlation escalation or authorize Step-2.

## 2. Bound design and scope

W0 implemented only the ratified repair package:

1. The shared frozen middle is called sequentially for branch A and branch B. Batch concatenation is prohibited.
2. Evaluator provenance declares `sequential_shared_middle_v1`.
3. The operating point is fixed by design at `g_A = g_B = s_A = s_B = 1.0`; no gate search occurred.
4. T1 exact identity and T2 cold-start gradients were rerun on the real Qwen2.5-0.5B substrate.
5. The preserved arm-6 population was frozen at 256 rows with native reader lengths.
6. One forward-only pinned A100-40GB preflight measured divergence and cost.

No optimizer was constructed. No optimizer step ran. No sealed partition was opened. Step-1 and Step-2 training remained disabled in the machine lock.

## 3. Implementation changes

- `models/bicameral.py`: removed branch batch-concatenation; declared the sequential schedule; added a receipt-bound strategy operating-point helper.
- `analysis/build_paper2_bicameral_step1_manifest.py`: builds the exact 256-row manifest from the preserved arm-6 identities and the original P3.1 source rows using the registered MCQ and generation readers.
- `training/paper2_bicameral_w0_lock.json`: machine-readable W0 scope, authority, gates, runtime, and prohibitions.
- `colab/run_paper2_bicameral_step1_cost_probe.py`: byte-lock validation, real-substrate T1/T2, both-seed divergence, native-manifest timing, and hard-gate aggregation.
- Tests assert sequential call count, schedule provenance, fixed gates, and W0 lock boundaries.

The artifact byte contract is protected with repository LF attributes. A first successful scientific preflight was superseded because its Git archive converted LF to CRLF and therefore changed artifact hashes in transit. The canonical rerun added an explicit summary-versus-file hash assertion and is the sole result of record.

## 4. Population manifest

The 256 rows are the exact preserved arm-6 population in both correction artifacts. They originate from the frozen DEV-2 source population, not the later 1,024-row P3.4 panel.

| Property | Result |
|---|---:|
| Rows | 256 |
| Seed populations identical | Yes |
| Native input length, minimum | 32 tokens |
| Native input length, mean | 102.770 tokens |
| Native input length, maximum | 194 tokens |
| ARC-Challenge | 32 |
| ARC-Easy | 1 |
| GSM8K | 216 |
| MBPP | 5 |
| MMLU | 1 |
| Tier-1 | 1 |

Manifest SHA-256: `06b2ab04bde4eb0a66bfb2db21600ef31637940d6d7c84d916e358bade4c7bea`.

The divergence probe is a deterministic nine-row sample: the first two manifest rows per battery where available. It spans all six batteries and is identified by SHA-256 `fad30fb92e9d212f0afe85a9c0707a90bc8f0498aa1c62886a424c69daa24441`.

## 5. Hard-gate results

### T1: exact identity

| Metric | Result |
|---|---:|
| Compared logits | 9,723,904 |
| Nonzero differences | 0 |
| Maximum absolute difference | 0.0 |
| Exact equality | PASS |

This repairs the prior batch-concat failure of 8,969,555 differing logits and maximum absolute difference 0.375 without weakening the gate.

### T2: cold-start gradient contract

Live, finite, nonzero gradients were observed for callosum gates A/B, bank gates A/B, and combiner `mu`. Gradients were exactly zero for combiner `delta` and bank gains A/B. T2 passed in full.

### Operating-point divergence

| Seed | Branch correlation | Cosine similarity | RMS difference | L2 difference | Max absolute difference |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.744590 | 0.744574 | 3.002717 | 3,280.352 | 686.0 |
| 1 | 0.744580 | 0.744564 | 3.002848 | 3,280.495 | 686.0 |

The branches are correlated because they share the frozen substrate, but they are not mechanically collapsed. The near-identical seed readings show that the differentiation is stable to the two registered initializers.

## 6. Cost result

Pinned runtime: NVIDIA A100-SXM4-40GB, PyTorch 2.11.0+cu128, CUDA 12.8, BF16/SDPA.

| Read | Result |
|---|---:|
| Batch 8, length 256 median | 38.264 ms/batch |
| Batch 8, length 256 | 4.783 ms/row |
| Exact 256-row manifest, one seed | 1.253 s |
| Exact 256-row manifest, two seeds | 0.000696 A100-hours |
| Projected two-seed cache, 256+461+2,048 rows | 0.007347 A100-hours |
| Closed-form 256-row combiner fit | 1.121 ms |
| Peak allocated memory | 1.111 GiB |

Sequential execution is approximately 40.5% slower per row than the invalid 3.403 ms/row batch-concat receipt. That cost is scientifically required and operationally immaterial relative to the approximately 4.9 A100-hour evaluator envelope in the strategy adjudication.

## 7. Interpretation

W0 resolves the implementation uncertainty cleanly:

- The exact-identity failure was schedule-induced BF16 evaluator drift, not evidence against the architecture.
- The fixed conditioning gates create two measurably distinct branch states on real prompts in both seeds.
- The zero-initialized architecture has the intended gradient topology.
- The sequential evaluator is cheap enough that no throughput-driven approximation is justified.

W0 does not establish that either branch contains useful causal information, that the closed-form combiner improves task performance, or that the conditional map is learnable. Those are the questions for X-1, Step-1, and D-M5.

## 8. Limitations and standing boundaries

1. Divergence was measured on a deterministic nine-row mechanical probe, not as a task-effect estimate.
2. Gate values were design-bound and not optimized. The result establishes non-collapse at the registered point only.
3. The exact manifest timing measures frozen-state caching. Full task evaluation remains the dominant cost and is still a projection until W1/Step-1 dry runs.
4. The `RHO_ESCALATE_AT_RANK_CAP` result remains banked. W0 does not restore the invalid mean-direction power model.
5. Step-2 remains blocked pending X-1 and D-M5. No result here relaxes that condition.

## 9. Decision requested from strategy

Bank W0 as `PASS` and proceed in the registered order:

1. W1: cost the amended eleven-rung X-1 ladder under the sequential evaluator; run only if the dry run remains within the eight A100-hour cap.
2. W2: run Step-1 with the 256-row manifest and existing additions only after W1 adjudication.
3. W3: execute X-6 and D-M5 as CPU desk work in parallel.
4. Keep Step-2 blocked until the consolidated adjudication.

Any W1 seed disagreement, mixed ladder result, runtime mismatch, cost-cap breach, or evaluator schedule change returns to strategy rather than being resolved locally.

## 10. Receipt ledger

| Artifact | SHA-256 |
|---|---|
| W0 lock | `ae8ca2c34d41f09cafe2fab942fd9a89f9a94d2354e2805c19bc19ad167cac8e` |
| Step-1 manifest | `06b2ab04bde4eb0a66bfb2db21600ef31637940d6d7c84d916e358bade4c7bea` |
| Manifest summary | `7a8f92ffdcbf2d145bb09899d1fab0c0e440278435a2b01bed24ffb5821c85b5` |
| Probe batch | `fad30fb92e9d212f0afe85a9c0707a90bc8f0498aa1c62886a424c69daa24441` |
| Canonical A100 W0 receipt | `39fdadf4a6de8f5c1d65cc79e0f530f790e628b8659c979314ea44f29cfe5e95` |

Targeted validation: 14 tests passed. The paid Colab session was terminated after receipt download, and `colab sessions` reported no active sessions.
