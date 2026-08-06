# Handoff: Option B Localization and Pre-Lock Resolution

Date: 2026-08-06. Status: CPU localization banked; teacher generation and
training remain prohibited.

## 1. Executive reading

The localization job did not find a coarse, deployable structural pocket that
should be masked in the Option B continuation. This is not because every row
benefited. Effects were heterogeneous: `3,270/8,031 = 40.72%` of anchors were
helped in both seeds, `2,904/8,031 = 36.16%` were harmed in both, and
`1,857/8,031 = 23.12%` changed sign across seeds. The result is that the harm
cannot be separated reliably by the preregistered teacher-free features.

The correct pre-lock recommendation is therefore to run the full writeback arms
unmasked. A post-hoc diagnostic router is not authorized.

## 2. Design

The job was CPU-only post-processing of the banked A2 DEV row tensors. It used
all `8,031` fixed evaluation anchors and both trained seeds. It performed no
model inference, optimization, or frozen-partition contact.

Candidate masks were restricted before the analysis to observable structural
features: source stratum, token-position bucket, and their intersections. A
candidate needed at least 200 rows and a positive masking effect with the
required two-seed document-bootstrap support. Diagnostic gate and probe
quantiles were retained for explanation only and could not become mask or
router inputs.

## 3. Results

| Cross-seed class | Anchors | Share |
|---|---:|---:|
| Helped in both seeds | 3,270 | 40.72% |
| Harmed in both seeds | 2,904 | 36.16% |
| Opposite sign | 1,857 | 23.12% |
| Quality loss in both seeds | 61 | 0.76% |

Mean full-minus-control accepted-length increments remained positive in both
seeds: `+0.002378` for seed 0 and `+0.004982` for seed 1.

No structural candidate qualified. The closest apparent harm pocket was token
position zero, but it contained only 24 rows, far below the 200-row minimum,
and seed 1's bootstrap interval crossed zero. It cannot support a mask.

Several adequately sized groups showed replicated evidence in the opposite
direction. For example, code positions 32-127 and general positions 128 and
above had negative masking deltas in both seeds, meaning removing writeback
would reduce accepted length in those groups.

The diagnostic surfaces did not reveal a clean monotonic router. High probe
top-1 values coincided with larger average benefit in both seeds, but tied
quantiles left empty middle bins and the analysis was post hoc. Treating this
as a routing rule would violate the registered feature boundary.

## 4. Interpretation

The A2 writeback effect is real but row-heterogeneous. Coarse metadata does not
separate the helpful and harmful cases with replicated precision. This weakens
the case for a cheap static mask, not the case for continued exposure or fresh
data. Option B should test whether the net increment and its quality tradeoff
improve with dose and then with new anchors.

The `23.12%` cross-seed sign-flip rate is also important. A large fraction of
row-level effects is not stable enough to be treated as an intrinsic row class
under the current two-seed substrate. Any future router study would need a new
locked design, deployable pre-decision features, and held-out validation.

## 5. Banked integrity receipts

- Localization summary SHA-256:
  `848afffcd0c1eaffed61cb1524870246a522689e436d32a0cfa560fbdb1ae222`.
- Markdown receipt SHA-256:
  `903062e46bb17d81825899aa995616f5fd121f43c3ec965eaed66cd1700f73d2`.
- Existing training manifest:
  `03ce3e1877f4e79f0952ab7054b16c0fb823fe9c9de03ee7c9088d8aa271201a`.
- Document partition:
  `7b4fcdfad21b940ea8a5d51d4310d3a9b4ac851d27df2542004a9182f8398e81`.
- Evaluation exclusion proof:
  `c751de988b7c83fd1bfed4a409174d99ed79b02657a06f156672df73537b7f5f`.
- Fixed old-train diagnostic subset:
  `0f5d114c3dcf6c856956ba9a618f7957f0c3d18c317415c3a1eb23420cd609c5`.

## 6. Decisions requested from strategy

1. Ratify `structural_mask = null`. Recommendation: yes.
2. Choose 14B state coverage for the new cache: every admitted anchor or only
   the threshold subset. Recommendation: collect states for every admitted
   anchor if the cache is intended to support later flow training; otherwise
   the resulting state population is selected by the current teacher cascade
   and cannot be called an all-anchor future-flow dataset.
3. Approve the teacher-pass resource note and lock the amended human and
   machine protocols.

## 7. Next execution after lock

1. Build and test the resumable teacher/cache pass. Do not launch it before the
   strategy lock.
2. Generate at least 100,000 and target 140,000 fresh anchors from new,
   quarantined documents. Bank the new manifest, partition, audit, and model
   cache hashes.
3. Apply the hash-only amendment before the splice.
4. Run the four staged Option B arms, paired full and no-writeback controls for
   seeds zero and one, with the recorded dose-to-data splice.
5. Read the pre/post-splice slopes, endpoint gain, retention, train-eval gaps,
   and writeback increment without converting this single intervention into a
   general scaling-law claim.

## 8. Boundaries

This result does not establish that row-level harms are absent, that a learned
router is impossible, or that Option B will pass. It establishes only that no
allowed coarse structural mask cleared the pre-stated two-seed rule on this DEV
population.
