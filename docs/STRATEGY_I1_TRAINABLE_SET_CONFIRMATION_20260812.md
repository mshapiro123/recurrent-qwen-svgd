# Strategy — i1 Trainable-Set Confirmation: Freeze the Complete Selector, Train the Output Projection, Two Riders

Date: 2026-08-12. Responds to the i1 build report (commit `83522aa6`, 27 tests passing). Mark approved this ruling in session.

## The confirmation

**The implemented interpretation is confirmed: freeze the complete functional selector — gate head, control.*, query, key, value — and train only `bridge.output_projection.weight` (114,688 parameters).** The memo's phrase "direction pathway + control state" was architecturally self-contradictory in this implementation, since control and Q/K/V feed the gate. The narrow set is the only reading under which "gate frozen" is literally true: the gate's behavior stays bit-identical to the measured P3.3 gate, every π_dir movement is attributable purely to direction learning, and the shared-representation drift that would silently change gate behavior without gate training cannot occur. This is the comparability clause and the rebalance-not-redesign clause enforced together. The capacity question the narrow set defers — whether the aim deficit lives upstream in what the cross-attention reads — belongs to the P3.5 capacity lever, priced by the observatory's A_r measurement, not to the single preregistered iteration.

## Two riders, binding on the run

- **R1 — the freeze is a verified invariant, not an intention.** Gate statistics at every audit (recall, precision, open-set membership on the audit slices) must be bit-identical to the P3.3 endpoint values. The selector is fully frozen, so this assertion is free, and any deviation is a stop-class implementation fault, not a result.
- **R2 — the aim-loss convergence curve is a required deliverable, reported with the re-read.** Its purpose is to disambiguate a middle-band outcome: aim loss still descending at budget end keeps undertraining live and makes duration the next lever, while an early plateau at mediocre capture localizes the constraint upstream and makes capacity the next lever. The re-read memo will classify the outcome using this curve explicitly.

## Also confirmed

The framing note is accepted as written: π_dep exceeding π_dir is described as **measured enrichment** by the gate on its selected rows, not as semantic routing — consistent with the standing claim discipline. The mirrored ruling, ledger reconciliation to canonical BF16, pinned checkpoint hashes, AdamW recipe, 1,000 updates, 20 looks, calibrated shares (aim ≥ 70 percent, preservation ≤ 25 percent), endpoint receipts, and the resumable separate runner are all banked as reported.

## Effect

Mark the preregistration locked and launch both seeds. Re-read thresholds unchanged against the canonical baseline of 14.901 percent: at or above 25 percent proceeds to the P3.4 charter, the middle band brings the full evidence including R2's classification to Mark's decision, and below 5 percent writes the boundary memo. Plain language: the aiming half of the machine gets the whole training budget while the selector half — which already learned its job — is locked in glass and checked for fingerprints at every audit, so whatever the number says at the end, we will know exactly which part of the machine said it.