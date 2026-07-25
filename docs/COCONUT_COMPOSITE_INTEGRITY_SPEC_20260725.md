# COCONUT Composite Integrity Specification

**Date:** 2026-07-25  
**Substrate:** fresh Qwen2.5-0.5B recurrent surgery  
**Mode:** engineering preflight, no training

## Purpose

Install a COCONUT-style horizontal latent-feedback path around the existing
vertical recurrent-depth model without allowing a silently detached path to
look trainable. The final post-norm hidden state immediately before each
`<|recur_readout|>` placeholder is fed through an identity-initialized
`I + delta` bridge and reconstructed into that placeholder's input-embedding
slot. The reference path recomputes the full prefix and preserves one autograd
graph. A sliced-cache path is an L=1 optimization candidate only.

## Frozen design decisions

- H=0 delegates to the registered recurrent wrapper.
- Horizontal feedback uses the final post-norm hidden state at the preceding
  position.
- Slot replacement reconstructs the embedding tensor and performs no in-place
  mutation.
- The horizontal bridge is exact identity at step zero. It stays frozen for a
  future RG-12 pilot and may train only in a separately authorized C1 run.
- The first integration pilot, if authorized later, uses vertical depth L=1.
- RG-12 and all training are outside this preflight.

## Required checks

1. **RG-1:** H=0 maximum logit difference below 1e-3 at L=1 and L=2, for
   full-block and zero-initialized R16 surgery.
2. **RG-2:** identity bridge and raw feedback logits are exactly equal in fp32.
3. **RG-3:** final loss has finite, nonzero gradients to the first fed state
   and prompt input activations.
4. **RG-4:** a random feedback-direction derivative agrees with a centered
   finite difference.
5. **RG-5:** sliced-cache and recompute logits and probe gradients agree in
   fp32. Cache is rejected for L>1 and with checkpointing.
6. **RG-6:** adapter base weights receive no gradients while LoRA and earlier
   feedback activations remain live.
7. **RG-7:** independent forward and backward counters equal `(H + 1) * L`,
   the H feedback-producing cells equal `H * L`, and every cell gradient is
   nonzero.
8. **RG-8:** original placeholder-slot input activations receive zero gradient,
   and instantiated optimizer and EMA parameter-name sets exactly match the
   intended set.
9. **RG-9:** one full backward passes anomaly detection.
10. **RG-10:** checkpointed and plain recompute logits and probe gradients
    agree.
11. **RG-11:** bf16 versus fp32 feedback-gradient cosine is at least 0.99 and
    every feedback boundary is finite.

Any failed check is a red engineering result written to the receipt. It does
not authorize repair training. RG-12 remains unrun until a separate
null-calibrated corruption design and training pilot are authorized.
