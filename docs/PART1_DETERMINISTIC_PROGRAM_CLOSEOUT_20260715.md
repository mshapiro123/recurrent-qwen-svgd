# Part 1 Closeout: Deterministic Recurrent-Qwen Program

**Date:** July 15, 2026
**Decision:** Deterministic mechanism program closed; paper consolidation and width-substrate screening opened
**Scope:** `Qwen/Qwen2.5-0.5B-Instruct`, the repaired Prelude/Recurrent-Block/Coda wrapper, and the frozen synthetic and natural-surface evaluations used through Stage 5

## Executive Finding

Part 1 established that a pretrained dense Qwen model can be surgically converted into an identity-preserving recurrent-depth model, trained to perform genuine iterative state updates, and made to exceed the registered dense controls on the frozen synthetic family. It also established a reproducible boundary: on this 0.5B substrate and the tested full-block continuation recipes, a new non-native inverse operation can be learned in isolation, but it cannot be installed while preserving the already consolidated mechanism. That is a **retention boundary**, not a learning boundary.

The deterministic program is therefore complete. The canonical forward-table inverse branch, explicit inverse-table branch, inverse-rendered W3/W4 branch, and F9 multi-channel bridge intervention are closed. Their negative results are retained as first-class boundary evidence. The clean deterministic keepers are frozen assets. The program now returns to its original question: whether learned stochastic guidance over multiple recurrent trajectories produces useful coverage beyond output sampling and additional deterministic depth.

## What Part 1 Established

### Architecture and forensic repair

The final recurrent architecture divides Qwen into a Prelude, repeated Recurrent Block, and Coda. The one-loop path preserves the base architecture's output exactly when recurrent additions are inactive. The program found and repaired a silent loop-closure defect: the recurrent path did not correctly re-inject the Prelude representation on later loops. Subsequent diagnostics verified bridge liveness, re-entry state flow, forced-loop execution, and intermediate-label connectivity.

This forensic arc matters. Early stochastic and deterministic negatives that predated the repair do not test the final recurrent machine. In particular, they do not test GRAM-style guided width.

### Installed iterative mechanism

Exact per-loop supervision produced a real state-update chain. The chain survived removal of intermediate-label supervision during outcome-only annealing, demonstrating persistence rather than temporary teacher forcing. Probe and extrapolation batteries localized the mechanism, distinguished reader artifacts from state failures, and showed that learned recurrence can extend beyond the directly supervised horizon when the transition task and curriculum align.

### Scaling and surpass evidence

The N24 sequence established a four-point support law and a strong frozen-evaluation frontier. The registered recurrent arm scored `1,506/1,792` on the depth-1-through-14 frozen rows. Registered dense controls scored:

| Arm | Recipe | Correct / 1,792 |
|---|---|---:|
| A | Recurrent, same-reader | `1,506` |
| B | Dense 0.5B direct | `470` |
| C | Dense 0.5B serialized scratchpad | `952` |
| D | Dense 1.5B direct | `322` |

Arm A cleared the original A-over-B gate at every depth and also cleared the labeled A-over-C extension. The paired significance receipts are approximately `p ~ 1e-264` and `p ~ 1e-120`, respectively. These are synthetic-family, recipe-bounded results, not general benchmark or FLOP-matched claims.

The dense arms are finished for the registered comparison. B was flat from step 2,000 to 4,000 (`p=0.771`), C added 22 rows (`p=0.00319`) but remained sharply tail-limited, and D declined by 28 rows (`p=0.161`). Extending C to 8,000-10,000 steps is optional and non-gating.

### Transfer and reader findings

Natural-surface transfer showed that a synthetic recurrent transition can move onto verbalized surfaces, but also exposed a tail inversion and a training-history dependency. The same-reader identity work showed that several apparent final-answer failures were surface/reader mismatches rather than failures of the intermediate mechanism. Retrieval organization was not a static architectural property: it depended on consolidated training history.

## Closed Deterministic Branches

### Canonical forward-table inverse branch

Three curriculum designs stalled at matched dose. The model did not acquire robust canonical backward inference while retaining the installed forward mechanism. No further continuation, optimizer sweep, or particle experiment is authorized on this branch.

### Explicit inverse-table branch

The isolated task reached `63/64`, proving the operation is learnable. Joint task-retention gates never passed. A four-checkpoint Pareto sweep found no admissible point. This separates acquisition from retention: the model can learn either objective, but the tested full-block continuations cannot hold both.

### Inverse-rendered W3/W4 branch

W3 produced a near-gate zero-shot result on an explicit inverse relation. The one authorized W4 continuation began from a source already below the synthetic floor (`0.8125` versus `0.93`) and worsened all relevant objectives:

| Measure | Before W4 | After W4 |
|---|---:|---:|
| Calibration total | `288/384` | `208/384` |
| Synthetic retention minimum | `0.8125` source audit | `0.125` |
| Natural canary | `227/256` reference | `171/256` |

The branch is closed. The launch incident produced the standing floor rule below.

### F9 multi-channel bridge

F9 required at least two of three positive precursor measurements plus a priced staircase reading. M1 was smeared on both tested checkpoints. M2 was locally positive on N24 but failed the preregistered backward-checkpoint replication: zero stable retrieval heads versus 37 on N24, and aggregate concentration below the matched-random p95. M3 could therefore produce at most one of three votes. The activation gate is mathematically unsatisfiable, so F9 is closed without running M3 or searching alternative bases.

Retained F9 assets are the attention-capture instrumentation and the training-history-specificity finding.

## Unified Boundary Result

The three inverse-task branches are one finding measured from three directions:

> On the tested 0.5B recurrent substrate, native forward operations install and generalize with little additional cost, while non-native inverse operations can be acquired alone but full-block continuation cannot retain them together with the previously consolidated mechanism.

The claim is about this substrate, task family, and training regime. It does not establish that inverse reasoning is impossible, that larger models share the boundary, or that adapter-only specialization cannot preserve both objectives.

## Standing Rules Earned by the Evidence

### Launch-time floor assertion

No continuation may start from a checkpoint below any registered guardrail floor. Every launcher must:

1. resolve the exact checkpoint and SHA;
2. resolve every source guardrail metric from a durable receipt;
3. print value, floor, and pass/fail at startup;
4. fail before model loading or training when any floor fails.

This is implemented in `training/continuation_policy.py` and is tested independently of any specific runner.

### Frozen-asset regime

The clean keeper is a frozen asset. No full-block continuation may produce a keeper successor. Capability-bearing adaptation is limited to detachable heads or adapters.

Full-block training remains legal only in a predeclared `disposable_measurement` branch. Such a run must set both `checkpoint_promotable=false` and `successor_source_allowed=false`. Its checkpoint cannot become lineage, regardless of its score.

## Final Deterministic Measurement

The loop-position transfer micro-test is the final deterministic curriculum measurement. It is not a repair attempt and cannot reopen a closed inverse branch.

- One rendering throughout: forward for `p` days, then once backward.
- Train `p=0,1`, placing the inverse operation at loop positions 1 and 2.
- Mix 30% pure-forward rehearsal.
- Require at least `0.71` on 64 held-out rows at each trained position.
- Measure zero-shot `p=2,3`, placing inverse at positions 3 and 4, on 128 rows each.
- Read `>=0.55` at both as substantially position-invariant, `<=0.15` at both as per-position installation, and intermediate results as partial transfer.
- Run as a disposable full-block branch with effective batch at least 8 and a 1,000-step guardrail hard stop.

Nothing downstream depends on this result. It sharpens the paper's boundary explanation.

## Width-Substrate Screen

The branching-relations screen replaces inverse abduction as the deterministic prerequisite for Phase G-alpha. It uses a native forward operation with intrinsic multimodality.

- Every source has exactly two successors.
- N20 verbal rows run on the natural step-2,000 keeper.
- N24 symbolic rows run on the N24 step-6,000 keeper.
- Depths 1-4 have 128 frozen rows each.
- Rows store the exact reachable set and one sampled valid chain.
- Reachable-set bins are balanced where mathematically feasible at each depth: depth 1 can only have size 2; larger bins enter as the branching horizon permits.
- Same-reader final-symbol argmax is valid when it belongs to the exact reachable set.
- Gate: pooled validity `>=0.70` and every depth `>=0.55` on either keeper, with no block unfreeze.

If one keeper passes, the remaining powered coverage margin is locked and G-alpha may launch. If both miss, the exact profiles return to strategy review. One detachable attention-LoRA touch-up, rank at most 16 with 30% forward rehearsal, is authorized only after a near-miss determination. Because no numeric near-miss band was preregistered, the shared screen does not auto-launch that adaptation.

## Phase G-alpha Contract

On a green branching screen, Phase G-alpha keeps the deterministic block frozen and trains only:

```text
phase_g_prior_head.*
phase_g_posterior_head.*
phase_g_injection_scale
```

The posterior conditions on the stored sampled valid chain during training; inference samples independently from the learned prior. Stochasticity enters only at the high-level re-entry state. `K=1` parity, exact frozen-gradient checks, per-trajectory RNG manifests, and target-leakage assertions are mandatory.

The primary test is paired exact oracle coverage at K against:

1. deterministic answer-head sampling at matched K and entropy;
2. one deterministic trajectory at matched `K*T` transition compute.

G-beta, learned selection, per-trajectory halting, and SVGD remain closed unless guided latent width beats both comparators.

## Paper-One Consolidation Checklist

- [x] Close the three inverse branches as one retention boundary.
- [x] Close F9 by its unsatisfiable activation gate.
- [x] Record the two standing lineage rules in code and methods.
- [x] Re-scope early stochastic negatives as pre-repair, unguided evidence only.
- [ ] Fold final Phase A paired receipts and all-depth profile into the Results module.
- [ ] Add the four-point scaling/seed figure and same-reader identity result.
- [ ] Add natural transfer, tail inversion, and training-history specificity.
- [ ] Add the inverse-task Pareto and W3/W4 boundary figures.
- [ ] Resolve remaining historical `PENDING` markers against the closeout ledger.
- [ ] Run the literature pass before hardening any novelty sentence.

Paper one does not wait for the loop-position measurement, branching screen, or G-alpha.

## Claim Ledger

| Claim | Status | Boundary |
|---|---|---|
| Exact one-loop identity after model surgery | Supported | Qwen2.5-0.5B wrapper and tested split |
| Trainable and persistent recurrent state-update chain | Supported | Synthetic transition families with exact intermediate labels |
| Recurrent arm surpasses registered dense controls | Supported | Frozen synthetic depth family and registered recipes only |
| Recurrence generally surpasses base Qwen on natural reasoning | Not supported | Requires external benchmark evidence |
| Full-block continuation preserves arbitrary new operations | Falsified in tested regime | 0.5B inverse-task branches and tested recipes |
| Early particle/SVGD negatives test GRAM-style width | Not supported | Runs lacked repaired loop closure and target-conditioned guidance |
| Guided latent width improves exact solution coverage | Open | Phase G-alpha |

## Immediate Sequence

1. Continue the paper consolidation sprint on CPU.
2. Run the shared L4 session: loop-position micro-test and both branching screens.
3. If a keeper passes the branching gate, lock powered margins and launch G-alpha.
4. If both screens narrowly miss, review and explicitly authorize the one adapter attempt.
5. If both screens clearly miss, reopen the substrate/scale decision; do not weaken the gate.

The deterministic program is closed. The next model-training claim belongs to width.
