# P3.5 Lock Draft: Stabilized Landing and Probe-Reader A/B

Date: 2026-08-15. Status: ratified and approved for training.

## 1. Authority and scope

The governing strategy response is `STRATEGY_P35_CHARTER_RESPONSE_20260815.md`, Drive `1ZzWO3MzkFW5Ph0wAuF5r-ZpdYCzAEi6F`, 13,242 bytes, SHA-256 `3bf476f1db8ebe451d798c941aeef3110129b8d946515125488b7871e0cf7c82`. The machine companion is `training/paper2_phase3_p35_preregistration.draft.json`.

Execution authority: `docs/STRATEGY_P35_EXECUTION_HANDOFF_20260815.md`, Drive `15ylK6jk2vXDTX0TRk5XJ949D6xzIAA-X`, 4,876 bytes, SHA-256 `314b51a34a0d451dd8045b69eb794ebff4ecd3efb3483dcebcdc78bae3628efe`. Mark ratified the assembled lock and authorized Arm S seeds 0 and 1 plus Arm R seed 0 on 2026-08-15.

The registered P3.4 verdict remains `REPLICATED_POSITIVE_BELOW_TRIGGER_B`. CONFIRM and EVAL-E remain sealed. This lock tests two questions only: whether a deliberate landing stabilizes the existing mechanism, and whether a detached four-probe reader improves the gate/control path relative to the landed mean reader.

## 2. Initialization decision

P3.5 uses a 400-step continuation from the exact scored P3.4 step-4,000 checkpoints. A fresh 4,000-step replay is rejected because the source checkpoints retain model, optimizer, RNG, batch schedule, objective-weight, and runtime-controller state. The continuation is the direct causal test of landing and costs one tenth of the original training dose.

| Seed | Source SHA-256 |
|---:|---|
| 0 | `381955ec5b78d0a00883c29e9f940feac8cfc8665f7a3a4446c79734532f4ed7` |
| 1 | `97ad532a5bffd72b2563799047b517e531e00115793bf4808f060148dfffc1ec` |

Arm S runs both seeds with the existing mean reader. Arm R runs seed 0 and replaces only the scratch reader in the gate/control path with four learned attention probes over the eight final flow-state cells. Cells are detached. The reader attaches as exact mean pooling by initializing all probes uniformly and the output as averaged identity blocks. This prevents an initialization distribution shift from masquerading as a reader result.

## 3. Landing contract

- Steps 4,001 through 4,400.
- AdamW state restored by parameter name; fresh reader parameters begin with empty Adam moments.
- Batch-generator and global RNG states restored from each source endpoint. Arm S seed 0 and Arm R therefore receive the same continuation schedule.
- Cosine learning-rate decay from `3e-4` to exactly zero.
- Original batch size, depth lottery, data, losses, and scalar objective weights retained.
- Runtime annealing controller frozen; both seeds train at rung 0 and gate ceiling `0.02`.
- Objective-share weights frozen. Shares and counterfactual controller decisions remain logged but cannot shape the landing.
- Four looks at steps 4,100, 4,200, 4,300, and 4,400.

EMA decay is `0.995`, initialized from the source endpoint and updated after each optimizer step. EMA is primary. Raw-after-decay is secondary. Raw and EMA checkpoints are saved at every look.

## 4. Evaluation and measurement

The unchanged 1,024-row DEV panel is scored under the registered fresh-scratch, four-loop, greedy task graph. Every primary look uses ceiling `0.02` regardless of training controller history. A score-only bundle evaluates raw and EMA at both `0.02` and `0.08` after the landing completes.

Every row receipt carries its exact item ID, correctness transition, per-token answer margin, and row-minimum answer-token margin. Margins are summarized pooled and by battery. Adjacent EMA looks report changed-row counts and fix-set Jaccard. The continuous margins remain telemetry, not a replacement for task accuracy.

The inherited task-collapse guardrail remains armed. Non-finite loss or gradient, frozen-lineage mutation, sealed-data contact, and a source-anchor identity below 100% are immediate hard failures. Loss-share boundaries are observed during landing but do not change weights or rung because doing so would invalidate the stabilization test.

## 5. Estimator repair

Before P3.5 training, the oracle direction cache is rebuilt on all 4,096 positive rows with the exact BF16 serving matmul used by audit. No mismatched rows may be dropped. Cache construction and audit must report 4,096/4,096 source-token identity. Until that receipt lands, registered `pi_dir` is suspended and the runner remains disabled.

This prerequisite landed on 2026-08-15. The registered 4,096 audit IDs were selected without loss from the 43,204-row source cache and preserved in audit order. Exact source-token identity was `4,096/4,096`. The repaired cache is
`/content/drive/MyDrive/recurrent-qwen-svgd-artifacts/stage5/stage5_paper2_phase3_p35_prerequisites_20260815/private/serving_oracle/agreement_oracle_directions_v2.pt`, SHA-256 `294358a7dacc746b733e9f08296494c6f461443a92c093f8019a1dda56422294`. Its summary SHA-256 is `ab584b6ba008b0ade9247bee099f9bee4cce02ed1c58de94be68a8bb5c4197e6`.

## 6. Persistence probe

The no-training probe compares fresh scratch against controlled cross-token carry on a deterministic, hash-ranked DEV sample of up to 128 GSM8K and 128 MBPP rows. It never reuses a frozen-source direction: direction telemetry is recomputed from the current source token at every generated step. The probe is mandatory before any persistent training, but it does not block these nonpersistent landing arms once the serving-reader cache is repaired.

The probe landed on 195 available rows: 128 GSM8K and 67 MBPP. Fresh scratch scored `76/195`; controlled carry scored `75/195`, with two fixes and three regressions. Carry changed later generated tokens on `35/195` rows, including `29/128` GSM8K and `6/67` MBPP rows, but produced no pooled accuracy gain. The bounded reading is `no_free_persistence_gain_on_this_seed0_dev_probe`. Persistent training remains gated; the nonpersistent Arm S/Arm R landing comparison is unaffected.

## 7. Registered readings

Arm S succeeds if its EMA endpoint reaches the predeclared late-window benchmark, at least +8 rows for seed 0 and +9 for seed 1, while adjacent churn falls and continuous margins hold or rise. Raw checkpoints remain a declared secondary.

Arm R is compared with Arm S seed 0 on gate precision at matched recall, repaired `pi_dep` enrichment, row-minimum margin, and net correct rows. It is promoted only if task net is noninferior and gate behavior improves. No result permits post-hoc choice among raw/EMA or ceilings.

## 8. Approval boundary

The code, tests, and no-training prerequisites are complete. The exact serving-cache path and SHA, its 100% identity receipt, persistence receipt, and implementation commit `87bac2a4364284dccb08976aa9b048521cde1469` are inserted into the machine lock. Strategy and Mark ratified the assembled lock in the execution authority above. The four authorization fields are set, the unresolved list is empty, and training may start only after the registered preflight receipt passes for all three arms.
