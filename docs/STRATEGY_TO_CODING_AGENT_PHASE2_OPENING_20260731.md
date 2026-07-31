# Strategy Handoff to Coding Agent — Phase 2 Opens: Theory First, Then the Windows

Date: 2026-07-31. Responds to: DC1 Stage A Result Handoff (Drive `1kaMAA7o9AJ3KR1wq53YhvL2RwowOBz3U`, SHA `5544b83c…fb8a7d0`). Program decision (Mark, 2026-07-31): the program does not retire — Stage A's verdict and consequence stand untouched in the ledger, and by explicit program-level override two new lines open under new locks. Governing documents, both mandatory reading before any phase-2 work:

1. **PAPER_TWO_PHASE2_PROGRAM_DECISION_AND_DESIGN_20260731.md, draft 3** (Drive `1LGSwyw9rYqwO0oLzgE1GiL5TLp27PM9b`, SHA-256 `47916cdc88652c3de39c75b634c3f6b2fcc0fd73bf75cef75d3fb8880c97c9de`, 29,166 bytes) — the program structure: the override record, the exploration/confirmation methodology, DC2 (bounded-correction sidecar) and D1 (utility-labeled in-place controller), the gradient atlas, budgets, and decision rules.
2. **THEORETICAL_FOUNDATIONS_AUDIT_20260731.md** (Drive `1xEVW1T4Kul01hLP8OD0uxqKWseUv3dEA`, SHA-256 `a43858623ab88c0710469cdd6b12f3d9ab10c882aea7436291a3ffd4b52e8bdc`, 16,088 bytes) — the mathematical grounding. This is not background reading; it is binding design context. Its amendments A1–A4 bind every phase-2 build, its verification computations V1–V5 are scheduled work, and its central reduction — every phase-2 mechanism succeeds or fails on the separability of improvable from fragile positions in accessible features — is the interpretive frame for every result you produce. When an implementation choice arises that the design documents do not settle, resolve it in the direction the audit's bounds favor and note the choice in the receipt.

## 1. Stage A record updates

Execute the result handoff's section 9 as written: Stage A `complete`, verdict `none`, consequence `transient_append_retires` in the status, experiment log, and claim ledger; the final checkpoint archival only; EVAL-C `read_once_scoring_spent=true`; handoff and figure attached to the artifact map. Add one line to the claim ledger: the retirement consequence is overridden at program level by Mark's 2026-07-31 decision recorded in the phase-2 design — the Stage A claim scope itself is unchanged.

## 2. Pre-window verification jobs (authorized now — forward-only, dev material only, no training infrastructure yet)

These land before either window opens. All on DEV-C or its successors; nothing touches any frozen slice.

1. **V1 — the expressivity check (the one that can reshape the design).** From dev forced passes over the Stage A trained bridge (archival checkpoint, read-only): the logit-margin distribution of oracle-help positions (positions where the slot prediction is correct and k=0 is wrong). Separately, measure the local Lipschitz gain of the upper stack Lip(F_{>L}) by JVP probes on dev states. Overlay: what fraction of oracle-help margins fall within the reachable bound Lip(F_{>L})·γM/(1−ρ) for c ∈ {0.01, 0.02, 0.05}, ρ = 0.8? Report the reachable fraction per c. Pre-stated readings are in the audit section 2: helps concentrated at small margins validates the bounded-correction premise; helps at large margins pre-names the remedy (larger c, or E4's upper-layer LoRA) before any window spend.
2. **V2 — block iteration gain.** JVP gain distributions ‖J_F v‖/‖v‖ of the pretrained recurrent block at iterates 1 through 4 on dev states, both checkpoints if cheap. This quantifies the non-contraction account of the in-place harm ledger and supplies the manuscript's mechanism sentence for the asymmetry.
3. **Oracle receipts and hurt-overlap (from the phase-2 design section 2).** Formalize the composite oracle numbers (trained and untrained arms) and the cross-mechanism hurt-position overlap between trained-append and in-place arms, from the Stage A immutable cache — post-processing, labeled exploratory, no EVAL-C rescore.
4. **Data preparation.** Generate EVAL-D and EVAL-E (0.2M tokens each, standing partition discipline, disjoint by hash from all prior documents) and the frozen own-base feature caches for the sidecar's feature loss, amortized into one pass. Both slices frozen and untouched after generation.

## 3. Window structure (opens after V1/V2 land; summary — the design doc governs)

**DC2 window:** stages E1–E4 per design section 4.3, starting from the minimum-viable sidecar recipe at K = 1, existing surgery cut, every component addition justified by a named telemetry observation. The gradient atlas (design 4.4, including audit A3 conflict cosines and A4 collapse metrics) runs from step zero of every run. The E1 probe battery is V3 of the audit and carries the separability ceiling for both lines. Budget approximately 40 A100-hours; report-backs at stage boundaries, each a continue/stop decision point for Mark.

**D1 window:** label construction in both variants (with and without the 14B-referee exclusion) plus the label audit and pilot grid, per design section 5 and the 9.3 resolution. Budget approximately 20 A100-hours.

**Resource notes:** one per window at its opening (steps, batch, sequence length, LR, wall-clock under full fp32), so the eventual confirmatory preregistrations record real numbers. Full fp32 remains the declared precision policy for every composite gradient (RG-11; audit section 5 gives the mathematical reason: the loop's small-correction signals live below bf16's mantissa).

## 4. Boundaries

Exploration-window numbers are never citable as evidence and never enter receipts as findings. EVAL-B and EVAL-C stay spent; EVAL-D and EVAL-E are untouched after generation until their registered passes. No confirmatory training without a Drive-locked preregistration (which will follow the audit-then-register order). All Stage A do-not-claim items remain in force. V1's outcome feeds design, not program go/no-go — a poor V1 reading routes to the pre-named remedies, and a poor V3 separability ceiling is itself a bankable finding about where arbitration information lives. If anything in the audit appears to conflict with a design-document instruction, stop and report — the two documents are meant to agree, and a divergence is a bug in the paperwork, not a choice left to implementation.

## 5. Plain-language summary

Phase 2 starts with mathematics instead of GPUs. Two cheap checks come first: whether the corrections the new design is allowed to make are large enough to reach the errors worth fixing (V1), and a measurement that explains, once and for all, why rethinking-in-place was destructive (V2). Then two exploration periods open — one for the new anchored-scratchpad architecture, one for the think-token controller — each with a real budget, dense instrumentation, and regular check-ins, and each ending in a single locked, one-shot test on fresh data. The theory document is part of the toolkit: it says what should work, why the old designs failed, and what numbers to watch so that the next surprise, if there is one, is a surprise about the world and not about our own arithmetic.
