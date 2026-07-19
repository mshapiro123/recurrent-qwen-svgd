# Wall-Clock Latency Receipt Specification

**Status:** pre-registered descriptive measurement; no pass/fail gate.  
**Scope:** single hardware configuration, batch size 1, registered evaluation paths.

## Systems and locked lineage

| Arm | Registered path | Checkpoint SHA256 | Decode cap |
|---|---|---|---:|
| A | Full-block recurrent, forced loops equal row depth | `dc00f7b694ce32427eb13b0b85d365bc15e0c0317130bd22d4bbc3568544f71b` | one symbol |
| E | R16 recurrent plus split bridge, forced loops equal row depth | `bffa8c4277ce82ae9f662db3243a21a50a08c4c041820c9d7506d8f250e82839` | one symbol |
| C | Dense 0.5B serialized-orbit scratchpad | `f2e7d600e057cb742b28d2f053615520e5257a16c39a3057ad34d89d4301c801` | 96 tokens |
| B | Dense direct 0.5B | `bb4fbaa628c11bc40f9d21f8e8f08c42b064463cd6cf357f196dadef27d0fa74` | 32 tokens |
| D | Dense direct 1.5B | `1e2999731352ac8f36d6cbd03359f4e68e6f93e8c5f7c9c35e04cdfc72b118d2` | 32 tokens |

The frozen input is the 1,792-row Phase A family at depths 1-14. Measurements are
interleaved by row index across depth to reduce order and thermal confounding.

## Timing method

All GPU phases use `time.perf_counter_ns()` around synchronized CUDA work. Model
loading and the excluded warmup observation are not timed. Tokenization is recorded
separately. Total row latency is tokenization plus actual model-path latency.

For dense arms, a manual cached greedy loop is checked token-for-token against the
registered Transformers `generate` call before measurement. The first full-prompt
forward is prefill; subsequent cached forwards are decode.

For recurrent arms, total model latency is the actual registered forced-depth call.
The prefill/decode split is explicitly a subtraction decomposition: a separate
synchronized one-loop call on the same encoded prompt estimates prefill, and the
nonnegative difference from the forced-depth call is decode-side recurrent work.
This is not presented as an internal kernel-profiler decomposition.

## Stability and interruption handling

Depths 4, 8, and 12 use the first 32 frozen rows and three repeats. Raw observations
and status are mirrored to Drive every 128 completed measurements. Resume keys are
`phase|repeat|row_id`, so an interrupted runtime continues without duplicating rows.
The hardware signature must match across all arms in the final receipt.

## Reporting

The receipt reports per-arm and per-depth median and IQR for total, prefill, decode,
model-total, tokenization, and generated-token count. The paper table uses depths
1, 2, 4, 8, 11, and 14 plus median generated tokens. Accuracy is logged only as a
reader-path diagnostic. This result describes interactive latency and makes no
batched-throughput claim.
