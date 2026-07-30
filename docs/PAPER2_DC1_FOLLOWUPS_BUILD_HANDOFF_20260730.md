# Paper Two DC1 Read-Only Follow-ups Build Handoff

**Date:** 2026-07-30  
**Status:** built and locally tested; GPU receipts pending  
**Training authorized:** no

## Binding correction

The existing DC1 result handoff previously inferred that severe loop harm was
absent before D0. That inference compared 53,389 pre-D0 loop-1 rejections with a
full roughly 200,000-position post-D0 population. The sentence is now
quarantined in the source handoff. The archived floor payload was produced
before all-position rows were added, so it cannot repair the comparison by CPU
post-processing.

The parity runner therefore takes the authorized fallback. It evaluates the
pre-D0 T1-lite-R seed-1 raw checkpoint and post-D0 EMA checkpoint at forced
depths 1, 2, and 3 on the identical selected 113 DEV-C rows and cached 7B
targets. The public receipt reports full-population and depth-1 accepted versus
rejected ledgers, with ungated pre-D0 depth-2 net utility prominent. Private
token grids remain in Drive. EVAL-C and EVAL-B are untouched.

## Scale-response probe

The second target reuses the banked DC1 scale grid, adds 1.5 and 2 times raw
hidden-state RMS, and records per-position final-slot cosine to both the fed
state and the registered full-sequence k0 state. Raw and 10x additionally
record layer-resolved residual-stream cosine. The receipt identifies the
accuracy trough and nearest measured cosine crossover but remains descriptive
and non-gating. It cannot authorize mechanism language or alter Stage A.

## Stage A preparation

The strategy-locked decision bands are encoded and unit-tested in
`training/paper2_dc1_followups.py`. The separate resource note proposes 2,000
steps, batch 1, no accumulation, AdamW at `1e-4`, full fp32, and 2 to 4 hours on
an A100-SXM4-80GB. No training launcher is present because the governing Stage A
preregistration has not yet arrived with a Drive SHA.

## Canonical build paths

| Item | Path |
|---|---|
| Parity evaluator | `eval/eval_paper2_dc1_parity_ledger.py` |
| Scale evaluator | `eval/eval_paper2_dc1_scale_response.py` |
| Scoring contracts | `training/paper2_dc1_followups.py` |
| Parity Colab target | `paper2_dc1_parity_ledger` |
| Scale Colab target | `paper2_dc1_scale_response` |
| Resource note | `docs/PAPER2_DC1_STAGE_A_RESOURCE_NOTE_20260730.md` |

## Required next actions

1. Run the parity target on an L4.
2. Run the scale-response target on an L4, concurrently if a second instance is
   available or consecutively otherwise.
3. Review and bank both descriptive receipts.
4. Reconcile the final Stage A preregistration against the resource note.
5. Only after its Drive lock, build and launch Stage A on an A100.
