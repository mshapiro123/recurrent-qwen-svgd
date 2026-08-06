# Phase-2 A2 Helped/Harmed Localization Receipt

CPU-only post-processing of banked DEV row tensors. No model inference or training.

## Population units

- Stage 0A anchors: 50,000
- Stage 0A horizon samples: 200,000
- A2 training anchors: 41,969
- A2 evaluation anchors: 8,031

The existing Stage 0A lattice does not contain approximately 190,000 anchors. Reaching that anchor count requires a new teacher/cache pass.

## Cross-seed consistency

- Helped in both seeds: 3,270
- Harmed in both seeds: 2,904
- Opposite sign across seeds: 1,857
- Quality loss in both seeds: 61

## Structural mask decision

No structural group cleared the pre-stated two-seed mask rule.

## Pre-lock population hashes

- Existing training manifest: `03ce3e1877f4e79f0952ab7054b16c0fb823fe9c9de03ee7c9088d8aa271201a`
- Document partition: `7b4fcdfad21b940ea8a5d51d4310d3a9b4ac851d27df2542004a9182f8398e81`
- Evaluation exclusion proof: `c751de988b7c83fd1bfed4a409174d99ed79b02657a06f156672df73537b7f5f`
- Fixed old-train diagnostic subset: `0f5d114c3dcf6c856956ba9a618f7957f0c3d18c317415c3a1eb23420cd609c5`

This is a post-hoc DEV localization result. Any mask must be locked before the Option B curve and cannot be described as confirmatory evidence.
