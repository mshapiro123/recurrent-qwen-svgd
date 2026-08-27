# Worktree cleanup receipt — 2026-08-26

**Repository:** `mshapiro123/recurrent-qwen-svgd`

**Branch:** `codex/bicameral-stage0`

**Authority:** Mark's 2026-08-26 instruction to review and clean the repository

**Scope:** Git/worktree hygiene only; no training, checkpoint use, sealed-data read, or experiment execution

## Outcome

The cleanup was conservative and evidence-preserving. It did not run `git clean`, reset history, inspect sealed contents, or remove a unique research artifact.

- Before cleanup: 270 untracked files, 15,628,087,274 bytes.
- Hash-proven duplicate material removed: 6,218,010,976 bytes.
- Root TM-0 scratch/transport material archived outside the worktree: 24 files, 610,091,492 bytes.
- Canonical W1, W2-prime and TM-0 evidence remains local under its original dated paths.
- The three exact dated artifact roots are now narrowly ignored; there is no global `artifacts/` ignore.

## Deleted duplicate set

| Removed path | Bytes | Retained authority or reconstruction source |
|---|---:|---|
| `artifacts/tm0_20260825/remote/tm0_teacher_14b_parts/` | 6,040,332,232 | Retained `tm0_teacher_14b_bundle.tar`, SHA-256 `5a373ccf410758c1a892b5a173dca56addb0baf37f6d8baaf41066d7ba7b92d3`; every part matched its manifest before deletion. |
| `artifacts/bicameral_w1_20260824/extracted/` | 53,621,328 | Reconstructible from the retained four W1 bundles, which match their published hashes. |
| `artifacts/bicameral_w1_20260824/restore_bundle/` | 40,815 | Reconstructed receipt copies; the source Phase-B bundle remains. |
| `artifacts/bicameral_w2p_20260825/d4/paper2_bicameral_w2p_d4_bundle.tar.gz` | 19,864,515 | Drive `1i6_ho4VfDxv4tK2iu1KJwG3zMb2x8132`, SHA-256 `3664b8d8371c321fb2654047e5c90843d6e8a410033c2ad1279f165b124698bd`. |
| `artifacts/bicameral_w2p_20260825/paper2_bicameral_w2p_phase_d_bundle.tar.gz` | 1,207,253 | Drive `1kEt7j7sUMD_Mz-WNhU8X121EWZ_8aZfc`, SHA-256 `b088a19059c00683a29a7b266aa0348f92b375738529c1013dea1bb9660490f2`. |
| `artifacts/tm0_20260825/results/paper2_tm0_result_bundle.tar.gz` | 837,867 | Drive `1dwNWY7BmXEGFck7mnuodQgv-iyO7geDW`, SHA-256 `b55c4cbb8049cfa049d702049776bedf8a3314a625c41a01dd75b6d088364be0`. |
| `artifacts/bicameral_w1_20260824/remote/generation_recovery_bundle.part-00` through `part-08` | 102,106,966 | Concatenated byte stream matched retained `generation_recovery_bundle.tar.gz`, SHA-256 `86278f8ac33d37289020ce461123293b4f807943c7d691d2ef1052bb4f155aba`. |

## Archived root transport set

The 24 root `.tmp_*` files and the byte-duplicate `.tm0_r4_download.md` were moved, not deleted, to:

```text
.work/cleanup-archives/bicameral-stage0-tm0-transport-20260826/
```

The archive carries `SHA256SUMS` with 24 verified entries. Manifest SHA-256:

```text
2b0fa60ea31af370d4091257b2130e73d68164744600024e177a580ced6add05
```

`.tm0_r4_download.md` was byte-identical to tracked `docs/STRATEGY_TM0_R4_JET_AMENDMENT_20260825.md`, SHA-256 `aa354b8bd6735d2780ff7afb25925e9cb08cc325898495f6dd22146dd880080a`.

## Evidence deliberately retained

- `artifacts/tm0_20260825/sealed/` remains in place and was not opened or moved.
- The TM-0 student, 7B and 14B cache bundles remain in place and reverified against the master handoff hashes: `6cf58941…9ab`, `fbba8216…752`, and `5a373ccf…92d3`.
- TM-0 preflight, results, score recovery, smoke and GPU receipts remain in place.
- W1's four published bundles and full local recovery bundle remain in place.
- The two W2-prime superseded-incomplete receipts remain as provenance.

## Commit-range and test review

The 18 pre-cleanup local commits from remote `51c67d65` through `bd1affe2` form one merge-free linear chain. `git fsck --no-dangling`, range whitespace checks and secret/binary/path review passed. No artifact, sealed, checkpoint, archive, tensor, image, credential or oversized blob was committed. The chain must not be squashed or rebased because internal and external receipts cite its commit hashes.

During cleanup, `docs/STRATEGY_RECIRCULATION_PROBE_HANDOFF_20260823.md` arrived as a new provenance file. Its raw 20,116 bytes matched Drive `1vBn5JpoGl2cz7WyqGJobJlPkpHmEad3I` exactly and were committed separately as `4304b72a`; this cleanup neither implements nor executes that probe.

The raw repository suite reproduced the governed baseline exactly:

```text
3 failed, 3300 passed, 19 warnings in 88.13s
```

The three failures are the same machine-authorized legacy quarantine nodes; no failure was added, removed or renamed. The quarantine review remains due 2026-09-02.

The strict quarantine-aware gate then accepted that exact result:

```text
ablation engineering gate PASS: all tests ran and exactly 3 quarantined legacy nodes failed
full repository suite remains RED: 3 failed, 3300 passed, 19 warnings in 65.97s
```
