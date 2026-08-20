# Strategy Request - Stage 2B-A Autopsy Onset Amendment

Date: 2026-08-20. Status: awaiting two narrow rulings. The build and score-blind prelock are complete. No model was loaded, no optimizer was constructed, no optimizer step ran, and CONFIRM and EVAL-E remain sealed.

## 1. Why this request exists

The registered autopsy requested model-state reads at steps 0, 20, 60, 100, 200, 300, 500, 700, and 1,000 for both stopped Stage 2B-D seeds. The prelock found the deterministic step-0 construction and the signed step-1,000 EMA endpoint for each seed, but none of the 14 intermediate checkpoints.

This is a retention fact, not a search failure. The source runner atomically overwrote `resume.pt` every 20 steps and emitted named EMA snapshots only at registered looks. The declared `RETAIN_STEPS` list was not connected to the save path. Replaying training would create new trajectories rather than recover the historical ones and is not authorized.

## 2. Receipts

- Prelock receipt: `docs/receipts/paper2_stage2b_autopsy_prelock_20260820.json`, 11,939 bytes, SHA-256 `4c01812ecd545c076ae540b3e3bdbce5e314933157ea5e6a8aa7d0ef1c95d732`.
- Drive receipt path: `/content/drive/MyDrive/recurrent-qwen-svgd-artifacts/stage5/stage5_paper2_stage2b_autopsy_20260820/receipts/prelock.json`.
- Frozen 256-row DEV-2 sample: SHA-256 `d48d5d7c0ad9d6b9a6267e50fb0d0c753aa5ff2a7bc883ab00508c6bd8d84bc4`.
- Sample-selection receipt: SHA-256 `19a9df5ced8ce6f52ac87c32c3251eb67f63eb450f3fd9c01601ae09236463b1`.
- Seed-0 step-1,000 EMA: SHA-256 `50cbf437adda668812dbe53a015792d3dc8ebc02cb785fba594c512b64bf2f58`.
- Seed-1 step-1,000 EMA: SHA-256 `830bbfa11dca4d3b9ed56db96a7c40c887f56fb4a5227555edc1bd447b6662bc`.
- Missing cells: steps 20, 60, 100, 200, 300, 500, and 700 for both seeds.

## 3. Requested ruling A - onset arm

Recommended amendment: replace the unavailable nine-point model-state onset curve with a descriptive two-source reconstruction:

1. Compare the exact deterministic initialization and exact step-1,000 EMA endpoint on the frozen DEV-2 panel.
2. Report the already-banked per-step or per-segment training-log trajectory for objective losses, controller state, and registered training telemetry over steps 0 through 1,000.
3. Label the log trajectory as training-process telemetry, not a checkpointed causal or task-score trajectory.
4. Do not interpolate missing model-state scores, replay training, or imply that the endpoint comparison locates onset timing.
5. Run the four executable score-only arms together after the amended lock is signed: amplitude response, component attribution, attractor diagnostics, and objective-task divergence.

Alternatives are to cancel the full autopsy, or to authorize a separately preregistered replay study. Replay is not recommended because it answers a replication question rather than recovering the stopped runs and would add unauthorized optimizer contact.

## 4. Requested ruling B - inherited-flow ablation

Please confirm or replace this implementation reading:

`inherited_flow_off` disables the inherited `SharedResidualFlow` state update while preserving final bridge execution.

This is the nontrivial component attribution. Disabling final bridge execution is already represented by the zero-write plumbing control and cannot distinguish inherited-flow damage from absence of all recurrent writeback.

## 5. Proposed binding language

If accepted, strategy may bind the following text:

> The missing intermediate Stage 2B-D checkpoints are banked as a retention limitation. The onset model-state arm is amended to an exact initialization-versus-step-1,000 endpoint comparison plus descriptive banked training-process telemetry; no missing score is reconstructed or replayed. The remaining four score-only arms run together. `inherited_flow_off` disables the inherited `SharedResidualFlow` update while preserving final bridge execution. All other panels, metrics, sealed-partition exclusions, hashes, and decision mappings remain unchanged.

## 6. Next action after ratification

The coding agent will update the machine lock with the strategy record ID and SHA, preserve the superseded onset specification as archaeology, obtain Mark's signature, run one signed A100-40GB score-only pass across both seeds, validate all receipts, publish figures and the standard handoff, and release compute.
