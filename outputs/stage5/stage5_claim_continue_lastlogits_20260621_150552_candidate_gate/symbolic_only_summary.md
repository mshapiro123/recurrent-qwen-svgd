# ARC-AGI Evaluation - base

- Tasks path: `/content/recurrent-qwen-svgd/data/arc_agi/ARC-AGI/data/evaluation`
- Grid format: `compact`
- Include original test pairs: `True`
- Include leave-one-out training pairs: `False`
- Geometry TTA: `none`
- Program parse mode: `fallback`
- Selection strategy: `heuristic`
- Symbolic candidate format: `grid`
- Examples with targets: `20`
- First-candidate exact: `0` / `20` = `0.0`
- Selected-candidate exact: `0` / `20` = `0.0`
- Best-of-K exact: `0` / `20` = `0.0`
- Selector-generated selected: `0`
- Selector-generated selected exact: `0`
- Selected exact beyond generated best-of-K: `0`
- Tasks solved best-of-K: `0` / `20` = `0.0`
- Valid candidate rate: `0.0`

This is exact-grid scoring on ARC-AGI-format tasks, not ARC-Challenge multiple choice.

## Task Families
- `arc`: selected `0` / `20`, best `0` / `20`, tasks `0` / `20`, valid_rate `0.0`

## Difficulty Buckets
- `hard`: selected `0` / `14`, best `0` / `14`, valid_rate `0.0`, score_range `197`-`968`
- `medium`: selected `0` / `6`, best `0` / `6`, valid_rate `0.0`, score_range `80`-`142`

## Program Verifier
- Candidates with executable programs: `0`
- Candidates fitting all demonstrations: `0`
- Program-fit exact candidates: `0`
- Program-fit selected candidates: `0`
- Program-fit selected exact: `0`
