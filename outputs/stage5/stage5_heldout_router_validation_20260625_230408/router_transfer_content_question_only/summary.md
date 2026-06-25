# Depth Router Transfer - stage5_heldout_router_validation_20260625_230408

- Discovery: `outputs/stage5/stage5_forced_depth_arc_challenge_loop123_20260625_194738/summary.json`
- Held-out: `outputs/stage5/stage5_heldout_router_validation_20260625_230408/summary.json`
- Score target / aggregate: `content_question_only` / `mean`
- Selector: `threshold:base<0.5->loop3`
- Gate status: `router_transfer_failed`
- Recommended next: `scale_probe_or_depth_signal_rethink`

## Discovery

- loop1: `89/256`
- base: `87/256`
- selected: `96/256` (delta vs loop1 `7`, delta vs base `9`)
- any-depth oracle: `117/256` (gain vs loop1 `28`)
- oracle gap capture: `0.25`

## Held-Out

### arc_easy

- loop1: `76/128`
- base: `71/128`
- selected: `73/128` (delta vs loop1 `-3`, delta vs base `2`, W/L `8/11`)
- any-depth oracle: `86/128` (gain vs loop1 `10`)
- oracle gap capture: `-0.3`
- deeper unique over loop1: `10`
- loop1 harmed by deeper loops: `23`
- hit patterns: `{'111': 53, '000': 42, '110': 12, '011': 8, '100': 10, '010': 2, '101': 1}`

### arc_challenge

- loop1: `11/43`
- base: `11/43`
- selected: `8/43` (delta vs loop1 `-3`, delta vs base `-3`, W/L `2/5`)
- any-depth oracle: `15/43` (gain vs loop1 `4`)
- oracle gap capture: `-0.75`
- deeper unique over loop1: `4`
- loop1 harmed by deeper loops: `7`
- hit patterns: `{'000': 28, '100': 5, '111': 4, '001': 2, '110': 2, '011': 2}`

### open_hard_arc_challenge

- loop1: `40/128`
- base: `39/128`
- selected: `38/128` (delta vs loop1 `-2`, delta vs base `-1`, W/L `7/9`)
- any-depth oracle: `52/128` (gain vs loop1 `12`)
- oracle gap capture: `-0.16666666666666666`
- deeper unique over loop1: `12`
- loop1 harmed by deeper loops: `21`
- hit patterns: `{'000': 76, '111': 19, '100': 9, '011': 8, '101': 3, '110': 9, '001': 3, '010': 1}`

## Transfer Summary

- positive delta vs loop1 benchmarks: `0`
- negative delta vs loop1 benchmarks: `3`
- positive delta vs base benchmarks: `1`
- mean delta vs loop1: `-2.6666666666666665`
- mean delta vs base: `-0.6666666666666666`
- mean oracle gap capture: `-0.4055555555555555`
