# Phase-2 V1b Batch-Baseline Amendment

Date: 2026-08-01

## Trigger

The first V1b execution stopped after completing 14 of 128 row groups. A strict
causal-prefix assertion found an argmax change before the perturbed position.
The perturbation forward used batch size 8, while its reference prediction came
from a batch-size-1 forward. That comparison cannot distinguish backward causal
influence from numerical changes caused by different GPU kernel shapes at nearly
tied logits. No V1b scientific result is interpreted from that failed execution.

## Corrected contract

V1b v2 runs an unmodified neutral forward at the same batch size and batch index
as each perturbation. Causal-prefix, target-flip, and collateral effects are all
computed against that paired neutral prediction. Differences between the paired
neutral and the registered batch-size-1 prediction are retained as a separate
numerical-sensitivity diagnostic.

The causal assertion remains strict: if a perturbed prediction changes any prior
position relative to its batch-matched neutral, the run still aborts. Such a v2
failure would indicate an actual masking or graph issue requiring diagnosis.

## Artifact lineage

The failed v1 row artifacts remain untouched under `private/v1b`. V2 writes to
`private/v1b_neutral_v2`; the first 14 rows are recomputed because their earlier
collateral baseline is not methodologically equivalent. The run remains DEV-only,
read-only with respect to model parameters, and performs no optimization.
