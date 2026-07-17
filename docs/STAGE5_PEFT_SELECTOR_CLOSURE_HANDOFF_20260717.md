# Handoff: Parameter-Efficient Recurrence and the Bounded Depth-Selector Verdict

**Date:** July 17, 2026  
**Repository:** `mshapiro123/recurrent-qwen-svgd`  
**Decision:** Bank the R16 parameter-efficient mechanism result; close the
bounded Ponder depth-selector line; return to manuscript consolidation and the
separately gated stochastic-width program.

## 0. Executive disposition

Two overdue questions are now answered on the corrected recurrent architecture.

First, a parameter-efficient recurrent-block adaptation can install the tested
iterative transition without changing the pretrained base weights. Rank-16
LoRA over the recurrent block, together with the repaired trainable bridge,
marked 7,613,953 parameters for optimization and used 6,007,425 parameters in
the split-mode forward path. It passed the registered depth-1-through-4
synthetic gate at 4,000 cumulative steps and finished at `64/64`, `64/64`,
`60/64`, and `53/64`. The base-weight hash was unchanged and the Tier-1
arithmetic canary remained green.

Second, the existing bounded PonderNet controller did not learn useful depth
selection. On a frozen mechanism that produced the correct answer at oracle
depth on `759/768 = 98.83%` of held-out rows, supervised stated-depth reading
selected the correct depth only `70/768 = 9.11%` of the time. Outcome-only
Ponder training collapsed to selecting depth 12 on all 768 rows. This is a
controller failure on a working mechanism, not evidence that recurrent depth
itself is ineffective.

No further selector GPU work is authorized by these results. Optimizer, KL,
learning-rate, and seed sweeps would refine a failed controller family without
a new causal design. The next action is strategy review and manuscript
integration.

## 1. Questions and answers

### 1.1 Can corrected recurrence be installed parameter-efficiently?

**Bounded yes.** The R16 arm passed the registered task gate with an unchanged
pretrained-base hash and a green capability canary.

Relative to the registered full-block reference, R16 was tied at depth 1, one
row better at depth 2, three rows worse at depth 3, and six rows worse at depth
4. Those depth-3/4 differences were not significant at this sample size
(`p=0.183` and `p=0.090`, one-sided Fisher tests). The accurate conclusion is:
no significant per-depth difference was detected, and the comparison is
underpowered to claim parity. The supported result is installation of a
working recurrent transition on this synthetic family.

### 1.2 Does the existing Ponder controller allocate depth reliably?

**No.** Three controller reads now point in the same direction:

| Controller test | Forced accuracy | Learned accuracy | Selection behavior |
|---|---:|---:|---|
| R16 attached Ponder | `241/256 = 94.14%` | `191/256 = 74.61%` | loop 2 on all rows |
| N24 S1 supervised depth | `759/768 = 98.83%` | answer `73/768 = 9.51%` | sparse modes at 1, 2, 4, 6 |
| N24 S2 outcome-only | `759/768 = 98.83%` | answer `173/768 = 22.53%` | loop 12 on all rows |

The collapse direction is substrate/objective dependent, but the common
failure is global-depth selection rather than row-conditional allocation.

### 1.3 Is this a gradient-path failure?

**No.** Both selector arms had strongly live gradients at startup. In S2 the
gradient norm decayed continuously as the halting distribution approached
probability one at the final depth. The final zero gradient was boundary
saturation. The runner was corrected to distinguish:

- zero gradient at step 1: wiring/liveness failure and hard abort;
- zero gradient after a verified live start: recorded saturation and normal
  blocked-result finalization.

## 2. Parameter-efficient experiment

### 2.1 Registered design

- Corrected split-bridge recurrent Qwen2.5-0.5B architecture.
- R16 LoRA over 84 recurrent modules, alpha 32.
- AdamW.
- Bridge Prelude learning-rate multiplier 10.
- Depths 1-2 for the first 2,000 steps, then depths 1-4 for 4,000 steps.
- Evaluations every 1,000 cumulative steps.
- Gate: at least `46/64` correct at every depth 1-4.
- Tier-1 capability canary with hard stop at a 3 percentage-point decline.
- Pretrained-base hash required to remain unchanged.

### 2.2 Parameter accounting

| Component | Optimizer-marked | Forward-active |
|---|---:|---:|
| Recurrent-block LoRA | 4,399,104 | 4,399,104 |
| Repaired split bridge | 3,214,849 | 1,608,321 |
| Total | 7,613,953 | 6,007,425 |

The R16 bridge instantiated the legacy concatenation tensors
`bridge.proj.weight` (`[896, 1792]`) and `bridge.proj.bias` (`[896]`), totaling
1,606,528 parameters. Split-mode forward uses `bridge.prelude_proj` and
`bridge.state_proj` instead, so the legacy tensors are optimizer-marked but
bypassed. The bridge remains a meaningful part of the active trainable budget;
the result must not be described as LoRA-only.

### 2.3 Dose curve

| Cumulative step | Depth 1 | Depth 2 | Depth 3 | Depth 4 | Gate |
|---:|---:|---:|---:|---:|---|
| 1,000 | 64 | 12 | 16 | 7 | fail |
| 2,000 | 64 | 51 | 16 | 4 | fail |
| 3,000 | 63 | 51 | 20 | 7 | fail |
| 4,000 | 63 | 60 | 60 | 52 | pass |
| 5,000 | 64 | 63 | 61 | 52 | pass |
| 6,000 | 64 | 64 | 60 | 53 | pass |

The earliest crossing is step 4,000. The later dose preserved the gate but did
not erase the deeper-depth deficit relative to full-block training.

### 2.4 Preservation receipts

- Pretrained-base SHA remained unchanged.
- Tier-1 baseline: `60/64 = 93.75%`.
- Tier-1 final: `61/64 = 95.31%`.
- One-loop identity gate: maximum absolute difference `0.0`.
- Final checkpoint backup SHA:
  `2d564440618e9d09b76111b35b77da0c74fc7a22a32d82ddffbdd7136c2e1f2b`.

### 2.5 Claim boundary

Supported:

> On the corrected N16 synthetic recurrence family, R16 recurrent-block LoRA
> plus the repaired bridge installed a gate-passing depth-1-through-4
> transition while preserving the pretrained base-weight hash and tested
> capability canary.

Not supported:

- full-block parity;
- broad natural-reasoning recovery;
- general parameter-efficient architectural conversion;
- Muon equivalence;
- learned adaptive halting.

## 3. Bounded selector experiment

### 3.1 Immutable substrate

- N24 step-6,000 keeper.
- Source checkpoint SHA:
  `898a259db2ab344ece4545e2910b051840e8408dbe4927f799e7cdb3cdd8c7dc`.
- Train rows: 3,072, balanced at 256 per depth 1-12.
- Held-out rows: 768, balanced at 64 per depth 1-12.
- Same-reader full-symbol scoring.
- Frozen loop states, per-loop target NLL, and forced predictions cached before
  controller training.
- Only halt projection, loop embedding, and loop bias were trainable.
- Target-loop oracle controls were frozen.
- Each arm used 2,000 AdamW steps, batch size 8, learning rate `1e-3`.

The forced-depth diagonal scored `759/768 = 98.83%`, isolating control from
mechanism capability.

### 3.2 S1: supervised stated-depth reading

S1 was an intentionally easy control: the required depth was stated in the
prompt. Passing would establish that the existing controller could read and
route an explicit depth request. It was not a test of inferred difficulty.

Results:

- Correct depth: `70/768 = 9.11%`.
- Correct selected answer: `73/768 = 9.51%`.
- Mean selected depth: `2.620`.
- Selected-depth histogram:
  - depth 1: 92;
  - depth 2: 527;
  - depth 4: 14;
  - depth 6: 135;
  - every other depth: 0.
- Only true depth 2 cleared the `46/64` selection bar.

This is a decisive S1 block. The existing pooled-state halt head did not
reliably expose the explicit depth field under the bounded training recipe.

### 3.3 S2: outcome-only Ponder discovery

S2 optimized weighted final-answer NLL plus `0.02` KL to a truncated geometric
prior with exact mean 6.

Results:

- Correct selected depth: `64/768 = 8.33%`, exactly the one-of-twelve rate
  produced by always choosing depth 12.
- Correct selected answer: `173/768 = 22.53%`.
- Mean selected depth: `12.0`.
- Mean expected depth: `11.99993`.
- Spearman selected versus true depth: `0.0`.
- Final selected-depth histogram: 768 at depth 12, zero elsewhere.
- First-window mean loss: `13.2627`.
- Last-window mean loss: `13.0731`.
- Loss ratio: `0.9857`, failing the registered decrease gate.
- KL stabilized near `1.64621`, but at the collapsed boundary.

S2 is a registered collapse, not a partial result.

The two arms are also logically connected. With a frozen executor that emits
the correct intermediate at each loop, outcome loss is minimized at the true
depth, so the optimal halting distribution is a point mass at that depth.
Outcome-only S2 therefore requires the same depth information that supervised
S1 directly asks the controller to recover. Because S1 failed even when depth
was stated in the prompt, S2's collapse is evidence of a starved information
path, not a reason to sweep optimizer, KL, learning rate, or seed on the same
head.

### 3.4 Frozen-contract receipt and limitation

Within each process, the source checkpoint SHA and frozen-parameter hash were
unchanged from start to finish, and no frozen parameter received a training
gradient.

The resumed S2 process reported a different aggregate frozen hash than the S1
process. The hash includes frozen auxiliary parameters that may be initialized
when absent from the source checkpoint, including controls excluded from the
active selector path. Both arms used the identical source checkpoint SHA and
the same Drive-backed frozen feature caches, so the controller verdict remains
valid. Cross-process equality of that aggregate hash is not claimed. A future
runner should seed or exclude absent, unused auxiliary parameters before using
that hash as a cross-process lineage identifier.

## 4. Integrated interpretation

The experiments separate three layers that had previously been conflated:

1. **Recurrent mechanism capacity:** positive under forced or supervised depth.
2. **Economical installation:** positive on the bounded synthetic family with
   R16 LoRA plus bridge training.
3. **Adaptive depth control:** negative for the existing bounded Ponder head
   and objectives.

The mechanism does not need to be abandoned because the selector failed.
Conversely, strong forced-depth performance cannot be described as learned
adaptive computation. Current models should use fixed, externally specified,
or oracle depth in mechanism claims.

The S1 failure is particularly informative. Because even explicit depth
reading failed, the next selector attempt cannot be justified as a small
optimizer or KL adjustment. It would need a changed information path, such as
direct prompt-token access, a dedicated depth-query representation, or a
separate calibrated policy. That is a new architecture study and is outside
the bounded closure.

## 5. Recommended queue

### Immediate, no GPU

1. Bank the R16 result in the claim ledger and manuscript.
2. Record the selector as a closed bounded negative.
3. Keep all adaptive-halting language out of positive claims.
4. Use forced depth for deterministic mechanism figures and comparisons.
5. Preserve both selector checkpoints and the frozen caches as audit assets.

### Next GPU program

Do not run another Ponder sweep. Resume the already gated program:

- finish manuscript v2 integration for the deterministic paper;
- execute Phase G-alpha only under its frozen-keeper, K=1 parity, coverage,
  entropy-matched output-sampling, and iso-compute width-versus-depth gates;
- keep SVGD as a G-beta ablation only after a G-alpha coverage win.

### Optional later selector redesign

Only reopen depth selection with a new pre-registration that changes the
controller information path and includes:

- explicit-depth S1 as a mandatory startup gate;
- fixed-depth and confidence-policy controls;
- row-conditional depth-distribution diagnostics;
- held-out natural reasoning after synthetic control passes;
- matched compute and calibration metrics.

## 6. Questions for strategy review

1. Should the R16 result enter the main paper as evidence that the repaired
   architecture can be installed economically, or remain an appendix result
   because it does not reach full-block parity at depths 3-4?
2. Should the paper report 7.61M total trainable parameters prominently, with
   the explicit 4.40M LoRA plus 3.21M bridge decomposition?
3. Is the bounded selector negative sufficient to remove adaptive halting from
   the current program, or should a future paper reserve a separate controller
   redesign track?
4. Should Phase G-alpha now take the next GPU slot, or should manuscript v2
   consolidation finish before any new training?
5. Does the strategy agent want the cross-process frozen-hash limitation
   repaired prospectively before Phase G, even though Phase G has separate
   checkpoint-SHA and frozen-block assertions?

## 7. Durable artifacts

Primary summaries:

- `outputs/stage5/stage5_peft_ponder_closure_20260717_182113/summary.json`
- `outputs/stage5/stage5_depth_selector_bounded_20260717_204109/summary.json`

Selector detail:

- `outputs/stage5/stage5_depth_selector_bounded_20260717_204109/S1/gate.json`
- `outputs/stage5/stage5_depth_selector_bounded_20260717_204109/S1/eval_summary.json`
- `outputs/stage5/stage5_depth_selector_bounded_20260717_204109/S2/gate.json`
- `outputs/stage5/stage5_depth_selector_bounded_20260717_204109/S2/eval_summary.json`

Drive checkpoints:

- R16 final:
  `/content/drive/MyDrive/recurrent-qwen-svgd-checkpoints/stage5_peft_ponder_closure_20260717_182113/R16/unfrozen_recurrent_step_4000.pt`
- S1 selector:
  `/content/drive/MyDrive/recurrent-qwen-svgd-checkpoints/stage5_depth_selector_bounded_n24_step6000/S1_supervised_depth_reading/S1_supervised_depth_reading_step_2000.pt`
- S2 selector:
  `/content/drive/MyDrive/recurrent-qwen-svgd-checkpoints/stage5_depth_selector_bounded_n24_step6000/S2_ponder_outcome/S2_ponder_outcome_step_2000.pt`

Preregistration:

- `docs/stage5_depth_selector_bounded_preregistration.md`

This handoff:

- `docs/STAGE5_PEFT_SELECTOR_CLOSURE_HANDOFF_20260717.md`
