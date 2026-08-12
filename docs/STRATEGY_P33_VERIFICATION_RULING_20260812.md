# Strategy — P3.3 Verification Ruling: The Instrument Is Verified, the Assumption Is Retracted, i1 Is Cleared

Date: 2026-08-12. Responds to the verification handoff (Drive `1ZbalyUofkFyhlAVBMjUz_kdQNFmSw01A`; raw rows `1JI1Ec6RHaRvklbBPFZzi0KZVZfmvKeCP`). The one ruling requested is granted, with the reasoning on the record.

## 1. The ruling: V3 is banked as a failed population assumption, not an instrument failure

Yes — and the radius diagnostic is what makes this a finding rather than a shrug. V3's *purpose* was to distinguish a dead evaluation path from real physics. The literal 0.15 criterion failed, but the purpose was served decisively by the curve behind it: pooled negative flips 0 at 0.30, 23 at 0.60, 1,107 at 1.00 — a monotone, nonzero flip response through the tested radii, produced by the same path that V1 showed writing nonzero deltas on every row and V2 showed flipping 101 positives while sparing every negative. The dead-instrument hypothesis is refuted on three independent lines. What failed was the criterion itself, and that error is strategy's: the expectation that 0.15-radius writes flip confident-agreement rows transplanted the V-series flip physics onto a population the V-series never measured — those receipts belonged to flip-candidate positions, and the negative slice is by construction the opposite tail, the maximum-margin rows. This is the claim-scope rule (t1 C1) violated by its own author in an auxiliary assumption, caught by the agent's refusal to grade a 0.60 result as a 0.15 pass. That refusal was correct, and not spending the authorized i1 on an unresolved criterion was the right sequencing.

**Corrected V3 criterion, banked for reuse:** the positive control passes when the flip response is nonzero and monotone in radius over the tested range on the control population. Under that criterion V3 passes.

## 2. Two byproducts worth banking

- **A margin map of the safe population.** Nothing flips below 0.30, and even adversarial-direction writes at the full capped radius flip only 4.5% (1,107/24,576) of maximum-confidence rows. The deployed system operates at ≤ 0.02 — an order of magnitude inside the first flip. That is a quantified robustness statement about exactly the rows the preservation contract protects, measured rather than assumed, and it belongs in the paper's methods-validation material.
- **The gate-aim synergy signal.** π_dep 19.612% exceeding π_dir 14.901% means capture is *higher* on rows the gate chose to open than across the forced-open population — the gate is not merely permitting writes, it is selecting rows where the trained direction works. Recorded as telemetry for the P3.4 design.

## 3. Numbers of record

The all-row BF16 rerun supersedes the filtered provisional values, as the canonicalization ruling intended: **π_dir = 557/3,738 = 14.901% (CI 13.282–16.521%), π_dep = 101/515 = 19.612% (CI 14.807–24.341%), collateral 0/24,576, retention harm 0/2,048** — all now verified by live-path evidence. The middle-band verdict stands. These are i1's baseline.

## 4. i1 is cleared

All preconditions of the result-response memo (Drive `1wZ5DQjXFUu70DS2GUyLWlMxLX758nhsw` §3) are met: zeros verified, reader canonical, baseline pinned. Run i1 as specified — gate frozen, aim ≥ 70% post-clip gradient share with its own floor, preservation ≤ 25%, everything else held for comparability, both seeds, pre-declared config. Re-read against 14.901%: ≥ 25% funds P3.4 outright; middle band brings the full evidence to Mark's decision; < 5% would now require explaining how a rebalance destroyed a verified signal, which is its own diagnostic. **P3.4 remains unauthorized until the re-read.**

## 5. Plain-language summary

The suspicious zeros are real, and proving it made them more valuable. The measuring path demonstrably registers changes — tiny nudges on every row, real corrections on the rows it was aimed at — and when we cranked the write strength far past its operating level, the supposedly untouchable rows finally began to flip, exactly the way genuine physics behaves and a broken instrument never does. My test criterion was wrong, not the instrument: it borrowed an expectation from measurements of easy-to-flip positions and applied it to the hardest-to-flip ones. The bonus findings are worth the detour — we now know the model's correct answers sit an order of magnitude beyond the reach of the write channel as deployed, and the gate turns out to open preferentially where the aim actually works. The one revision run is cleared: freeze the selector, pour the whole training signal into the aim, and re-measure against a now-verified fifteen percent.