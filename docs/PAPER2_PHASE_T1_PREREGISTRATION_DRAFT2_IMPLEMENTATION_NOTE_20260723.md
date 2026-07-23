# Phase T1 P0 Implementation Note - Draft 3 Amendment

**Status:** P0 is authorized. Registered T1 remains unlocked and no registered
T1 training launcher exists.

Draft 3 descopes registered T1 to one full-block T1-lite lineage. P0 remains
authorized on the R16 adapter lineage because Mark elected to run it, but its
role is now limited to loss feasibility and selection of candidate constants.
It is not matched-lineage evidence for T1-lite.

## Implemented now

- The ten-cell P0 grid is machine-readable: lambda in `0.5, 1, 2` crossed
  with stop-to-continue ratios `1, 3.5, 7`, plus the lambda-zero reference.
- Every cell uses seed `9999`, 1,500 steps, and readouts at steps 500, 1,000,
  and 1,500.
- The dedicated pilot evaluation set contains 256 rows, 32 per depth for
  depths 1-8, generated from a separate seed and marked as excluded from all
  registered sets.
- The training stream is exactly 70% control-bearing rows and 30% unchanged
  mechanism-rehearsal rows, balanced by depth.
- A private `<|recur_readout|>` prompt position is separated from answer
  supervision by a masked delimiter. Loop-specific continue/stop logits are
  read at that position, while per-loop mechanism and answer labels remain at
  their original output positions.
- Only the three added token rows are trainable. Because Qwen ties input
  embeddings and the LM head, the resized tensor is factored exactly into a
  frozen pretrained matrix and one shared three-row parameter used by both
  input and output modules. This preserves the tie and parameter count without
  allocating AdamW state for the full vocabulary. The pretrained-row hash is
  verified unchanged at the end of each cell.
- Compact checkpoints store LoRA, bridge, and only the three control rows.
  Checkpoints are copied to Drive at each 500-step readout. JSON and log
  receipts land in GitHub after each completed cell.
- Both class recalls must reach 0.60 at step 1,500, then answer-accuracy drop
  versus the lambda-zero reference is minimized over all nine non-reference
  cells, with ties toward lambda 1 and then ratio 3.5. If no cell qualifies,
  the run exits blocked and does not extend the grid.

## Receipt reconciliation completed before lock

The mechanism-installation recipe is copied from
`outputs/stage5/stage5_adapter_budget_arm_e_20260718/preregistration.json` and
`docs/ARM_E_ADAPTER_BUDGET_PUBLICATION_HANDOFF_20260718.md`: AdamW, batch size
1, gradient accumulation 1, weight decay 0, gradient cap 0.5, base bfloat16,
trainable adapters float32, stage learning rates `2e-5` then `1e-5`, and bridge
Prelude multiplier 1 then 10.

The T1-lite Gate 1 floor is integer-correct. The full-block reference is
1005/1024; subtracting 0.03 gives 974.28 rows, hence a minimum of 975.
The former adapter gate is no longer part of registered T1. Canonical receipts
and checkpoint SHAs are encoded in
`training/internal_think_token_t1_spec.py`.

## Still required before registered T1

1. Run P0 and apply its selection rule.
2. Insert the selected lambda, realized label counts, and normalized class
   weights into Draft 3 and `preregistration.json`, explicitly recording that
   calibration transferred from the adapter P0 to full-block T1-lite.
3. Commit the final Draft 3 and machine-readable preregistration with status
   `locked_before_training`.
4. Only then add and run registered T1 training launchers.
