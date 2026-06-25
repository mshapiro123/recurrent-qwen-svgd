# Depth Router Transfer - stage5_heldout_router_validation_20260625_230408

- Discovery: `outputs/stage5/stage5_forced_depth_arc_challenge_loop123_20260625_194738/summary.json`
- Held-out: `outputs/stage5/stage5_heldout_router_validation_20260625_230408/summary.json`
- Score target / aggregate: `cyclic_label_aggregated` / `permutation_mean`
- Selector: `threshold:loop1<0.25->loop3`
- Gate status: `router_transfer_failed`
- Recommended next: `scale_probe_or_depth_signal_rethink`

## Discovery

- loop1: `148/256`
- base: `154/256`
- selected: `156/256` (delta vs loop1 `8`, delta vs base `2`)
- any-depth oracle: `164/256` (gain vs loop1 `16`)
- oracle gap capture: `0.5`

## Held-Out

### arc_easy

- loop1: `103/128`
- base: `103/128`
- selected: `101/128` (delta vs loop1 `-2`, delta vs base `-2`, W/L `0/2`)
- any-depth oracle: `105/128` (gain vs loop1 `2`)
- oracle gap capture: `-1.0`
- deeper unique over loop1: `2`
- loop1 harmed by deeper loops: `6`
- hit patterns: `{'111': 97, '110': 5, '010': 2, '000': 23, '100': 1}`

### arc_challenge

- loop1: `23/43`
- base: `23/43`
- selected: `21/43` (delta vs loop1 `-2`, delta vs base `-2`, W/L `2/4`)
- any-depth oracle: `25/43` (gain vs loop1 `2`)
- oracle gap capture: `-1.0`
- deeper unique over loop1: `2`
- loop1 harmed by deeper loops: `4`
- hit patterns: `{'000': 18, '111': 19, '001': 2, '110': 1, '100': 3}`

### open_hard_arc_challenge

- loop1: `69/128`
- base: `75/128`
- selected: `72/128` (delta vs loop1 `3`, delta vs base `-3`, W/L `4/1`)
- any-depth oracle: `76/128` (gain vs loop1 `7`)
- oracle gap capture: `0.42857142857142855`
- deeper unique over loop1: `7`
- loop1 harmed by deeper loops: `1`
- hit patterns: `{'111': 68, '001': 1, '000': 52, '010': 2, '110': 1, '011': 4}`

## Transfer Summary

- positive delta vs loop1 benchmarks: `1`
- negative delta vs loop1 benchmarks: `2`
- positive delta vs base benchmarks: `0`
- mean delta vs loop1: `-0.3333333333333333`
- mean delta vs base: `-2.3333333333333335`
- mean oracle gap capture: `-0.5238095238095238`
