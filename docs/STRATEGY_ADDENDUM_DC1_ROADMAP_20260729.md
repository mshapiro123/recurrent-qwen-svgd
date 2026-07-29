# Strategy Addendum to the DC1 Handoff — Roadmap, Locked Constants, Teacher Policy

Date: 2026-07-29. Amends: STRATEGY_TO_CODING_AGENT_DC0_BANK_DC1_AUTH_20260729.md (Drive `1Yu6EUsbFb9Z4tA2n_2F6moeM-EGyxNVK`, SHA `4e9fc18b895c15b739739aef1d08ad80cbf2c0ae0f91bd02a921564824463071`). Nothing in DC1's scope, preconditions, preflight, pre-commitments, or boundaries changes. This addendum attaches the destination so stage A is built with the whole road in view.

## 1. Governing forward design

COMPOSITE_TRAINING_DESIGN_20260729.md (Drive, SHA recorded at upload) is now the governing design for the composite lane, superseding concept-note section 6's program shape for this lane. DC1 is its stage A, verbatim. Read it before making any implementation choice that outlives stage A — in particular the control-read plumbing (decision logits read at position t and at each slot from the control rows, masked per the T0 contract) should be built stage-C-ready even though stage A never reads it.

## 2. Constants locked by Mark, 2026-07-29

1. **k cap = 3.** The append chain never exceeds three latent slots per position, at any stage. Build the cap as an asserted invariant now.
2. **No joint fine-tune in stage C's first pass.** The controller trains against a frozen interface. Any joint pass is a separately preregistered later decision.
3. **Hybrid retained; vertical routing retired.** L is a global configuration setting (1 through stage C, 2 at stage D), never a per-position decision. The stage D iso-compute shape test — (k = 2, L = 1) versus (k = 1, L = 2), both 72 layer-applications — is the vertical loop's registered trial.

## 3. Teacher policy (Mark, 2026-07-29): ladder, not mixture

The 7B remains the sole training teacher through stages A–C — DC1 is unchanged, and no 14B pass is commissioned for EVAL-C, exactly as registered. The 14B's standing roles: descriptive crossover column in each stage's evaluation (cache-covered where possible, commissioned per stage at markup otherwise); optional label-cleaning referee at stage C (7B-rejects/14B-endorses positions excluded from the continue class — decided at that preregistration); and full distillation target in a post-stage-C phase, where the signal also switches from greedy exact-match to distribution KL. No mixing of exact-match signals from both teachers at any stage — the measured ~16.5 percent disagreement set would receive contradictory supervision.

## 4. Figure filed

`composite_architecture_20260729.svg` (Drive, SHA recorded at upload) is the architecture figure, marked by Mark as a paper keeper. Please land it in the repo figure set (`outputs`/`docs` per house convention) alongside the DC0 figure. The HTML rendering copy is `composite_architecture_20260729.html`, same content.

## 5. Unchanged

All DC1 boundaries bind as issued: preflight before preregistration, RG-4/RG-11 before the training loop, EVAL-B never read again, EVAL-C untouched until the single registered pass, no policy training, no persistent scratchpad, no L above 1 in stage A, rung 0 and the pre-D0 floor decomposition still owed.
