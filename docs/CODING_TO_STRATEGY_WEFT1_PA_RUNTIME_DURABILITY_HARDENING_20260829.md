# WEFT-1 P-A runtime and durability hardening receipt

**Date:** 2026-08-29

**Branch:** `codex/bicameral-stage0`

**Build base:** `32c0491cfdfad60fc55ac8c0f0e817d9ea135880`

**Implementation commit:** `8e1eba7a1984b3089ff81155e65b3272e0f85162`

**Disposition:** the exact-runtime builder and the production P-A durability,
isolation, provenance and replay boundaries are implemented and locally
verified. This is a build-axis engineering receipt. It is not an authoritative
P-A materialization receipt and mints no P-B, P-C, D1-D6 or G-TOK gate.

## 1. Authority and predecessor

The governing Amendment A2 identity is unchanged:

- Drive ID: `1f2iue1yYW2gpdamsqVlaQnijfllRCaVn`
- bytes: `8,615`
- SHA-256: `f7a2655b30f6c699035ec4ffdccee8c03068eeab8da94894be8e5818f955ce02`

This receipt rolls forward from
`docs/CODING_TO_STRATEGY_WEFT1_CORPUS_GTOK_A2_IMPLEMENTATION_20260829.md`,
9,764 bytes, SHA-256
`9236100e735e413f9422d6f1f53afe93ab24797a34dd8f879fe906d475e59a1e`.
Neither predecessor is edited or re-keyed.

## 2. P1 closure matrix

| Finding | Implemented closure |
| --- | --- |
| A path labeled durable could still be ephemeral | Production requires a pre-registered canonical marker on an observed `fuse.drive*` mount. Mount row, source, device and distinct local backing are bound and re-observed before each worker and final mint. |
| Child receipts were outside the final durable rehash | Each child receipt is closed, canonically reopened, rehashed and included as a receipt-role row in that child's final durable inventory. |
| The authoritative parent receipt was optional or outside durable storage | `full-pa` requires a fresh parent receipt beside and outside the two child trees on the registered durable backing. Publication uses a fresh partial, file fsync, atomic replace, close/reopen/hash, and records whether Drive supports directory fsync. |
| Workers could inherit hostile Python paths or import mutable repository code | Production launches `python -I -B` with a minimal reserved environment, `PYTHONHASHSEED=0`, no user site, and only the exact private code snapshot inserted after the network guard. The snapshot tree is exact and hash-bound. |
| Package names and wheel hashes did not attest installed bytes | The runtime records every installed distribution and every RECORD-owned regular file, rejects unexpected distributions/files/links, and recomputes the entire inventory before each child and final mint. |

The independent pre-run audit also closed four builder failures before Colab
compilation: CPython 3.11.9's exact 119-byte unowned `site-packages/README.txt`
is verified, removed and directory-fsynced; GNU `ldd` SONAME rows are parsed
and resolved rather than substring-matched; target startup and `readelf`
RUNPATH checks run with `LD_LIBRARY_PATH` absent; and live executable,
`libpython`, `_sqlite3` and `libsqlite3` bytes are bound into every repeated
runtime attestation.

The installer chain is explicit rather than overstated. CPython's pinned
ensurepip `pip==24.0` is the trusted installer. Installation is isolated and
offline from a one-wheel-per-lock-entry wheelhouse; a fresh `pip --report`
must exactly cover every locked name/version and selected wheel SHA-256. The
complete installed tree is then independently hashed and repeatedly checked.
Coherent malicious file-and-RECORD rewriting by the trusted installer before
the first receipt is outside this threat model.

The P-A import surface is also torch-free. Stable seed derivation moved to
`training/weft1_seed.py`; the model RNG retains a compatibility wrapper, so
the seed law and O-9 stream identities do not change.

## 3. Critical artifact identities

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| `scripts/build_weft1_pa_runtime.py` | 73,675 | `f61d05ea8f8c39b5a1e69821f5a14da7689635eb523b298f6dee83f24cb0f9fd` |
| `scripts/run_weft1_pa_runtime_builder_colab.sh` | 1,707 | `dcae22fca64dfbb2fba510e7675cf982d71dc0644a1ad73e6904c680b24c69e1` |
| `scripts/run_weft1_corpus_pa.py` | 54,377 | `579735f90001f97b9cc9ba6a617a94c0be842f3a4b2f12cbbd25306d3deabd3d` |
| `training/weft1_corpus_pa.py` | 72,260 | `cd05192eb754e796977c6de1a17bef5c3f0c227655684bdda4f1ca2a7c09559f` |
| `training/weft1_corpus_replay_a2.py` | 157,108 | `34cce5f0767aa4006ab6edd9432572ff9b10422bb6e5e9347facd421b41e7ee5` |
| `training/weft1_seed.py` | 2,282 | `18f2e239ec31078fd5627e66e9831488299426e94d03ad9a2ed8f1ba80a4a3dd` |
| `models/ablation_lm/rng.py` | 9,567 | `f2a495a591b6c85d604353519523c19054aee3ff6ba79f2d4a4e1223711f80e5` |
| `docs/WEFT1_PA_RUNTIME_COLAB.md` | 5,835 | `7e92f0fdffd2dbf7bb80cf1dcaf999a50bf6c3004a686cc4289e144199dc442b` |

The governed A2 inputs remain byte-identical:

- requirements lock: 65,717 bytes, SHA-256
  `bccb8e5b58b5e8fa9eee367fe9c26f59053fff5b7fadf81f23f96b83d1531860`;
- A2 bindings: 17,400 bytes, SHA-256
  `ee10e69a3ccd55f7960949f4c318daa4db1197c779f5e88fb67cec82ab7f263b`.

## 4. Runtime source pins

| Source | Bytes | Pinned identity |
| --- | ---: | --- |
| CPython `Python-3.11.9.tar.xz` | 20,175,816 | SHA-256 `9b1e896523fc510691126c864406d9360a3d1e986acbda59cda57b5abda45b87` |
| SQLite `sqlite-amalgamation-3450100.zip` | 2,730,697 | SHA-256 `5592243caf28b2cdef41e6ab58d25d653dfc53deded8450eb66072c929f030c4`; SHA3-256 `e311198775d5d5b2889d5fabe1d9a490567a14e605591d6a9e4c833804a8b4cb` |
| extracted SQLite `sqlite3.c` | governed by archive | SHA3-256 `0474604df9e1b69a5544295dd046aad954749279780d557da80f44b958100295` |

The builder records the exact resolved OS package versions and hashes of its
compiler/linker tools. Those Colab OS packages are observed build provenance,
not claimed as timeless content pins.

## 5. Verification

- Focused runtime/durability suite: `83 passed in 11.86s`.
- Wider corpus/G-TOK regression suite: `275 passed in 19.85s` under the
  recorded local Python 3.11 interpreter.
- `python -m compileall -q training scripts tests`: PASS.
- builder `--dry-run`: emitted `PLAN_ONLY_NO_EXECUTION`; no network or build.
- `git diff --check`: PASS.
- raw repository suite on the implementation tree:
  `1 failed, 3562 passed, 19 warnings in 114.29s`.
- exact-node engineering gate after the implementation commit: PASS; the
  underlying repository remains red at
  `1 failed, 3562 passed, 19 warnings in 109.50s`.
- quarantine schema test: `6 passed`.

The sole failure remains
`tests/test_paper2_claim_evidence_ledger.py::test_paper2_claim_evidence_paths_exist`.
No new, removed or renamed failure was accepted locally. No repository-wide
green claim is made.

The quarantine rolls forward without modifying its predecessor:

- `training/ablation_lm_engineering_quarantine_20260829_pa.json`
- 2,985 bytes
- SHA-256 `f5e6f6cba70b1c4d8f285fb67d8d731f6431bdbf275c13535c02ab2f0a4876b2`
- review due: 2026-09-04.

## 6. Operational state and authority boundary

The in-app browser is connected to the Pharma Initiatives Colab Pro+ account.
The declared compiler prerequisites are installed on the live CPU VM. No GPU
was requested because runtime compilation is CPU work.

This engineering receipt does not claim a successful exact runtime build. The
first real Linux compilation, GNU `ldd`/`readelf`, `/proc/self/maps`, and
pip-report checks follow after this code is synchronized to Colab. The current
Drive mount attempt fails before DriveFS starts because Colab's ephemeral
credential endpoint returns 404 and reports credential propagation failure.
Accordingly, no Drive durability marker or P-A materialization receipt exists.

No corpus source bytes were consumed. No calibration burst or tripwire meter
started. No tokenizer was fit or frozen; no optimizer was constructed; no
training, checkpoint, sealed-data access, P-B or P-C action occurred. The
programme-level license review remains the sole human action before P-B.

## 7. Next live receipt

The next live evidence is the exact runtime build receipt from the Pharma
Initiatives Colab VM. If it passes, its interpreter, linkage inventory,
installed-file inventory, selected-wheel hashes, trusted-installer chain and
canonical receipt identity will be reported. Corpus materialization remains
fail-closed until Drive mounts and the durable storage registration succeeds.
