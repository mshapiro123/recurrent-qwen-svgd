# ARC-AGI Evaluation - base

- Tasks path: `/content/recurrent-qwen-svgd/data/arc_agi/ARC-AGI/data/evaluation`
- Grid format: `compact`
- Include original test pairs: `True`
- Include leave-one-out training pairs: `False`
- Geometry TTA: `none`
- Program parse mode: `fallback`
- Selection strategy: `heuristic`
- Symbolic candidate format: `grid`
- Examples with targets: `5`
- First-candidate exact: `0` / `5` = `0.0`
- Selected-candidate exact: `0` / `5` = `0.0`
- Best-of-K exact: `0` / `5` = `0.0`
- Selector-generated selected: `0`
- Selector-generated selected exact: `0`
- Selected exact beyond generated best-of-K: `0`
- Tasks solved best-of-K: `0` / `5` = `0.0`
- Valid candidate rate: `0.0`

This is exact-grid scoring on ARC-AGI-format tasks, not ARC-Challenge multiple choice.

## Candidate Sources
- `model`: count `5`, valid `0`, exact `0`, selected `5`, selected_exact `0`

## Task Families
- `arc`: selected `0` / `5`, best `0` / `5`, tasks `0` / `5`, valid_rate `0.0`

## Difficulty Buckets
- `hard`: selected `0` / `3`, best `0` / `3`, valid_rate `0.0`, score_range `230`-`926`
- `medium`: selected `0` / `2`, best `0` / `2`, valid_rate `0.0`, score_range `80`-`138`

## Parse Methods
- `none`: count `5`, exact `0`, selected `5`, selected_exact `0`

## Program Verifier
- Candidates with executable programs: `0`
- Candidates fitting all demonstrations: `0`
- Program-fit exact candidates: `0`
- Program-fit selected candidates: `0`
- Program-fit selected exact: `0`
