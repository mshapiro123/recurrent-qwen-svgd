# Models Package Code-Review Response

Date: 2026-08-13

## Bottom line

The review is substantially correct. It identifies maintainability and public-release gaps, not evidence that the current experimental results are invalid. No behavior-changing `models/` edit should enter the locked P3.4 campaign lineage while that campaign is running.

## Scientific-risk triage

### Address immediately after P3.4 lands

1. **Constrain the rotary-embedding compatibility fallback.** `_rotary_embeddings()` currently catches every `TypeError` and returns `None`. Replace the broad fallback with explicit handling for supported Transformers signatures, emit a diagnostic for the intended compatibility path, and add a pinned-version assertion that position embeddings are present when the loaded Qwen implementation exposes them. This is the only review item that could hide a model-execution discrepancy rather than merely complicate maintenance.
2. **Remove Phase 2/Phase 3 forward duplication.** The two bodies are currently equivalent except for the Phase 3 bridge call's `control_state` argument. Factor the bridge call behind a protected hook and add equivalence tests at initialization and after checkpoint migration.
3. **Move `extract_horizontal_control_logits` out of `training`.** Put the pure readout helper in a shared model-side utility and retain a compatibility import if old scripts need it. Add an import-smoke test that exercises the control-token path without importing the training package first.

These changes must be behavior-preserving and should land on a post-campaign branch with the existing identity, migration, and accounting tests plus targeted equivalence tests.

### Complete before public repository release

1. Expand the root README with a module map, current-versus-historical status, an internal-codename glossary, and one minimal Phase 3 forward example.
2. Add provenance comments for calibrated defaults and fixed seeds, especially `rms_cap=0.550893`, the SVGD seed stride, halting heuristic coefficients, and the control-state feature width.
3. Add a license after the owner selects its terms. Add `CITATION.cff` when the paper metadata or preprint identifier is stable.
4. Add formatter/lint/type-check configuration and a package typing marker if the package is intended to be consumed as a typed library.
5. Correct the formatting issue in `halting.py` and document deliberate fp32 islands and compatibility fields that currently look accidental.

### Defer to a separately reviewed refactor

1. Split `RecurrentQwenForCausalLM.forward()` into subsystem configuration objects and validators.
2. Rename the trajectory-diversity `rho` without breaking old configs or receipts; use a compatibility alias and a deprecation window.
3. Replace output-by-mutable-argument APIs and bound or document module-level caches.

These are worthwhile, but their interaction surface is too large for publication cleanup or an active experimental lineage.

## Interpretation of the review

The strongest positive finding is that scientific contracts are executable: identity-at-initialization, trainable-set coverage, cache and loop accounting, inference privilege boundaries, and frozen-lineage checks are enforced in code. The strongest negative finding is architectural accumulation: the wrapper has become both model and experiment registry. The correct response is staged cleanup with equivalence tests, not an in-place rewrite.

## Campaign boundary

P3.4 remains pinned to commit `431fdfed30db8439e1ed9d60180e39ac28926ed8`. This review does not alter its code, thresholds, estimators, data, or interpretation rules. The post-campaign cleanup begins only after all three P3.4 runs and their receipts are durable.
