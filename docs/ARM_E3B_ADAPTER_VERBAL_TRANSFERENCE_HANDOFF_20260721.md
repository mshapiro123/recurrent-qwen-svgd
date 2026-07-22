# Arm E3b: Adapter-Budget Verbal Transference

**Date:** 2026-07-21  
**Run:** `stage5_adapter_verbal_transference_e3b_20260720`  
**Canonical receipt:** `outputs/stage5/stage5_adapter_verbal_transference_e3b_20260720/summary.json`  
**Status:** completed with the fresh-control arm truncated by its preregistered guardrail

## Executive Result

The installed rank-16 recurrent mechanism accelerated adaptation to controlled
verbal relay and pointer tasks relative to fresh rank-16 surgery at every
matched checkpoint through step 3,000. At the last matched checkpoint, the
installed arm scored `1,852/3,072 = 60.29%`, versus `1,282/3,072 = 41.73%` for
the fresh control, a difference of `+18.55` percentage points. The paired
row-level comparison contained 763 installed-only wins and 193 control-only
wins (`p = 9.74e-81`, exact two-sided paired sign/McNemar test).

The effect was not confined to the shallow output contract. By step 1,000 both
arms were essentially saturated at depth 1, while the installed arm's later
advantage was concentrated at depths 3-11 and was largest at depths 6-8. The
same advantage appeared on pointer, which was held out from verbal training.
This is bounded evidence that symbolic recurrent history improved the rate and
depth profile of learning related verbal transition tasks.

The registered 6,000-step endpoint comparison is unavailable. The fresh arm
hard-stopped at step 3,000 after its Tier-1 canary moved from `60/64` to
`58/64`, just beyond the preregistered three-point boundary. The two changed
rows give an exact paired `p = 0.50`; therefore the stop was mechanically
correct under the registered rule but is not evidence of a statistically
resolved capability regression. The installed arm completed all 6,000 steps.
The paper should report the last matched result and the truncation, not call the
registered asymptotic endpoint positive.

## Question And Design

E3b asked three questions:

1. Does either training budget transfer zero-shot from the symbolic operation
   to controlled verbal surfaces?
2. Does installing the symbolic recurrent mechanism before verbal training
   improve verbal learning at a matched adapter budget and dose?
3. Does verbal training erase the installed synthetic mechanism or impair the
   Tier-1 arithmetic canary?

The test used two adapter-budget arms differing only in initialization history.

| Arm | Initialization | Trainable set | Role |
|---|---|---:|---|
| T | Arm E final checkpoint with the symbolic recurrent mechanism installed | R16 LoRA plus repaired split bridge, 6,007,425 forward-active parameters | Transfer arm |
| S | Fresh Qwen2.5-0.5B surgery with a fresh R16 adapter | Same R16 LoRA plus repaired split bridge, 6,007,425 forward-active parameters | No-history control |

Arm T's source checkpoint SHA-256 was
`bffa8c4277ce82ae9f662db3243a21a50a08c4c041820c9d7506d8f250e82839`.
Arm S passed exact one-loop identity with maximum absolute difference `0.0`.
Both arms retained the same pretrained-base SHA-256,
`960f8bf265ba2850c9cdd60a388a00f8f366464babe0507521f010cb7f34971f`,
through training. The frozen base assertion therefore passed for both arms.

### Training protocol

- Optimizer: AdamW.
- Learning rate: `1e-5`; seed: `0`; effective batch size: `1`.
- Planned duration: 6,000 steps, checkpoints every 1,000 steps.
- Training mix: 2,048 verbal relay rows plus 2,048 synthetic rehearsal rows.
- Pointer was excluded from training and served as related zero-shot transfer.
- Per-loop labels were active, with forced loop count equal to row depth and
  maximum training depth 8.
- The pretrained backbone was frozen. No halting, latent, or re-entry-adapter
  parameters were trained.
- Tier-1 arithmetic was a per-arm hard stop against each arm's own step-zero
  baseline. Synthetic regression was measured on Arm T rather than hard-stopped
  because forgetting was an outcome of interest.

### Evaluation protocol

- Frozen relay: 1,536 rows, depths 1-12.
- Frozen pointer: 1,536 rows, depths 1-12.
- Same-reader scoring, forced loops equal to row depth.
- Synthetic regression: 2,048 rows, depths 1-8, Arm T at every checkpoint.
- Tier-1 arithmetic canary: 64 rows per arm at every checkpoint.
- The interrupted run was completed by an evaluation-only salvage. All
  training checkpoints, data hashes, checkpoint hashes, and matched row IDs
  were preserved; no additional training occurred during salvage.

## Results

### P1: zero-shot verbal transfer

| Budget | Relay | Pointer | Reading |
|---|---:|---:|---|
| Full recurrent block, pre-verbal keeper | 267/1,536 (17.38%) | 311/1,536 (20.25%) | Minimal |
| R16 plus bridge, installed Arm T | 249/1,536 (16.21%) | 264/1,536 (17.19%) | Minimal |

Neither budget produced substantial zero-shot verbal competence. The recurrent
operation transferred only after matched verbal fine-tuning.

### P2: matched-dose verbal learning

| Step | Arm T | Arm S | T-S | Exact paired p |
|---:|---:|---:|---:|---:|
| 0 | 513/3,072 (16.70%) | 86/3,072 (2.80%) | +13.90 pp | 3.67e-82 |
| 1,000 | 1,012/3,072 (32.94%) | 571/3,072 (18.59%) | +14.36 pp | 3.76e-56 |
| 2,000 | 1,753/3,072 (57.06%) | 697/3,072 (22.69%) | +34.38 pp | 1.26e-210 |
| 3,000 | 1,852/3,072 (60.29%) | 1,282/3,072 (41.73%) | +18.55 pp | 9.74e-81 |

At step 3,000 the result held independently on each family:

| Family | Arm T | Arm S | T-S | Exact paired p |
|---|---:|---:|---:|---:|
| Relay, trained | 935/1,536 (60.87%) | 665/1,536 (43.29%) | +17.58 pp | 1.84e-36 |
| Pointer, held out | 917/1,536 (59.70%) | 617/1,536 (40.17%) | +19.53 pp | 3.42e-46 |

The held-out pointer effect is important: the benefit was not restricted to
memorizing the trained relay surface.

### Depth localization at the last matched checkpoint

Relay and pointer are pooled below, with 256 rows per depth.

| Depth | Arm T | Arm S | T-S | Exact paired p |
|---:|---:|---:|---:|---:|
| 1 | 252/256 (98.44%) | 255/256 (99.61%) | -1.17 pp | 0.375 |
| 2 | 245/256 (95.70%) | 240/256 (93.75%) | +1.95 pp | 0.424 |
| 3 | 233/256 (91.02%) | 208/256 (81.25%) | +9.77 pp | 4.70e-4 |
| 4 | 225/256 (87.89%) | 182/256 (71.09%) | +16.80 pp | 2.43e-6 |
| 5 | 200/256 (78.12%) | 139/256 (54.30%) | +23.83 pp | 9.93e-9 |
| 6 | 193/256 (75.39%) | 84/256 (32.81%) | +42.58 pp | 2.17e-21 |
| 7 | 149/256 (58.20%) | 44/256 (17.19%) | +41.02 pp | 4.44e-20 |
| 8 | 119/256 (46.48%) | 31/256 (12.11%) | +34.38 pp | 5.50e-17 |
| 9 | 96/256 (37.50%) | 31/256 (12.11%) | +25.39 pp | 6.81e-12 |
| 10 | 71/256 (27.73%) | 18/256 (7.03%) | +20.70 pp | 1.08e-9 |
| 11 | 43/256 (16.80%) | 22/256 (8.59%) | +8.20 pp | 0.00751 |
| 12 | 26/256 (10.16%) | 28/256 (10.94%) | -0.78 pp | 0.875 |

The arms converged at depths 1-2, after which Arm T separated sharply. The
largest gains occurred at depths 6-8, and the advantage disappeared only at
depth 12, where both arms were near their tail floor. That profile is more
consistent with transfer of iterative computation than with an answer-format
advantage alone. It does not prove that every learned internal state or
algorithm transferred unchanged.

### Arm T after the matched comparison ended

| Step | Relay | Pointer | Pooled |
|---:|---:|---:|---:|
| 0 | 16.21% | 17.19% | 16.70% |
| 1,000 | 32.81% | 33.07% | 32.94% |
| 2,000 | 56.77% | 57.36% | 57.06% |
| 3,000 | 60.87% | 59.70% | 60.29% |
| 4,000 | 72.59% | 70.31% | 71.45% |
| 5,000 | 66.54% | 68.75% | 67.64% |
| 6,000 | 71.29% | 58.07% | 64.68% |

Arm T first crossed the pooled `0.71` reporting threshold at step 4,000, but
did not sustain it. Its verbal curve was nonmonotonic, and pointer deteriorated
most strongly at step 6,000. The full-block trained references were 86.0% on
relay and 79.0% on pointer. Arm T therefore remained below the full-block
regime: by 14.71 and 20.90 points at the registered 6,000-step checkpoint, or
by 13.39 and 8.66 points at the post-hoc best pooled checkpoint. The 4,000-step
checkpoint must not replace the registered endpoint in the primary claim.

### P3: regression and guardrails

Arm T retained the synthetic mechanism under the registered 50% synthetic
rehearsal mix. Its minimum depth-stratum accuracy at steps 0 through 6,000 was:

| Step | Minimum synthetic stratum |
|---:|---:|
| 0 | 97.27% |
| 1,000 | 94.53% |
| 2,000 | 98.83% |
| 3,000 | 96.09% |
| 4,000 | 96.09% |
| 5,000 | 98.05% |
| 6,000 | 98.83% |

Every checkpoint remained above the registered 0.93 retained floor. Arm T's
Tier-1 arithmetic canary stayed at 59-60/64 throughout. This is evidence of
retention **under rehearsal**, not evidence that verbal training causes no
forgetting without rehearsal.

Arm S's canary changed `60/64 -> 58/64` at step 3,000. That is a `-3.125`
point change against a `-3.0` point boundary, exceeding it by only `0.125`
points while one row corresponds to `1.5625` points. The paired row table was
two baseline-only, zero observed-only, and 62 ties (`p = 0.50`). The registered
stop remains valid, but the manuscript should call it a near-boundary discrete
guardrail event, not demonstrated regression. Future small-sample hard stops
will use the subsequently adopted uncertainty-aware policy; that policy does
not retroactively alter this run.

## Findings And Interpretation

1. **Zero-shot transfer remained minimal.** Installing the synthetic operation
   did not by itself yield useful verbal performance.
2. **Installed history accelerated matched-dose learning.** Through the last
   common checkpoint, Arm T beat fresh surgery by 18.55 points with decisive
   paired support.
3. **The effect was depth-dependent.** Near-equality at depths 1-2 and large
   gains at depths 3-11 argue against reducing the result to answer formatting.
4. **The effect transferred across controlled surfaces.** Pointer was never in
   the verbal training mix, yet its matched-dose gain was at least as large as
   relay's.
5. **The adapter mechanism survived verbal training when rehearsed.** All
   synthetic strata stayed above 0.945, and Tier-1 stayed stable.
6. **The adapter did not match the full-block verbal endpoint.** Its curve was
   nonmonotonic and remained below the published full-block references.
7. **Asymptotic T-versus-S superiority was not tested.** The fresh arm's
   preregistered stop makes step 3,000 the final admissible paired dose.

The strongest defensible reading is therefore: at a frozen-base R16-plus-bridge
budget, prior installation of the symbolic recurrent mechanism substantially
improved the speed and depth profile of learning related controlled verbal
tasks, including a held-out surface, while synthetic rehearsal preserved the
installed operation. This is evidence for transfer-assisted adaptation, not
general natural-language reasoning, budget parity, or a completed endpoint
comparison.

## Manuscript-Ready Language

> With the pretrained base frozen, initialization from the installed
> R16-plus-bridge mechanism accelerated adaptation to controlled verbal
> transition tasks relative to fresh R16 surgery. At the last matched
> checkpoint (3,000 steps), the installed arm scored 1,852/3,072 versus
> 1,282/3,072 (+18.6 percentage points; exact paired p < 9.8e-81). Gains were
> concentrated at depths 3-11 and were also present on the pointer family,
> which was held out from verbal training. The fresh arm then triggered its
> preregistered Tier-1 stop after a near-boundary 60/64-to-58/64 change, so the
> planned 6,000-step asymptotic comparison is unavailable. In the installed
> arm, all rehearsed synthetic depth strata remained at or above 94.5% and the
> Tier-1 canary remained stable.

The limitations sentence should state: one seed; controlled generated verbal
tasks; pointer held out from training but drawn from a related generator
family; synthetic retention measured with 50% rehearsal; fresh-control
truncation before the planned endpoint.

## Do-Not-Claim Boundaries

- Do not call the registered 6,000-step transference endpoint positive.
- Do not claim asymptotic superiority over fresh adapter training.
- Do not use the post-hoc step-4,000 Arm T peak as the primary endpoint.
- Do not claim general natural-language or natural-reasoning transfer.
- Do not claim no forgetting; the retention result included rehearsal.
- Do not claim the adapter matched the full-block verbal regime.
- Do not describe the Arm S stop as statistically established regression.

## Questions For Strategy Review

1. Is the bounded claim, "accelerated matched-dose adaptation through step
   3,000," sufficient for Paper One? **Recommendation: yes.** It answers the
   transfer question without spending more GPU or weakening the endpoint
   boundary.
2. Should Arm S be extended to 6,000 under the new small-sample policy?
   **Recommendation: no for Paper One.** Such a run would be a post-hoc
   extension under a different stop rule and should be labeled accordingly if
   ever pursued.
3. How should the nonmonotonic Arm T curve be interpreted? The safe account is
   that transfer was real but adapter-budget verbal consolidation was unstable;
   the present experiment does not distinguish surface overtraining,
   optimization variance, or depth-specific interference.
4. Does the large step-zero difference confound the transfer result? It likely
   contains same-reader/output-contract competence from prior training, but the
   depth-1 gap disappears by step 1,000 while deeper and held-out-pointer gains
   persist. The later depth-localized comparison is the stronger evidence.

## Paper-One Closure Recommendation

No additional E3b GPU experiment is required. Add the bounded result and
checkpoint curve to Section 9, record the guardrail truncation and limitations,
and close `adapter_verbal_transference` as `supported_bounded`. Preserve the
full receipt and this handoff as the source of truth for any reviewer response.
