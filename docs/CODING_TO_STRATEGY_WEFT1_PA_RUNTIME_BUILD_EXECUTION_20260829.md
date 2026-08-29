# WEFT-1 P-A exact-runtime build execution receipt

**Date:** 2026-08-29

**Branch:** `codex/bicameral-stage0`

**Execution commit:** `5e3abb465fe9dd29f247c404f207b9bfb88651c1`

**Colab surface:** Pharma Initiatives Pro+ · `WEFT1_CORPUS_GTOK_A2_PA_20260828`

**Disposition:** the pinned P-A runtime was built successfully on the live
Pharma Initiatives Colab VM. The second fresh-path build minted an
authoritative runtime-build `PASS`, and an independent exact attestation
matched its environment, executable, linkage and installed-file inventory.
This authority is narrow: it covers the pinned runtime installation only. It
is not a corpus-materialization, D1-D6, P-B, P-C or G-TOK execution receipt.

## 1. Authority boundary and outcome

The authorized action was installation and verification of the exact pinned
P-A environment. That action completed. The following actions did not occur:

- no corpus source or materialization bytes were consumed;
- no calibration burst ran and the cumulative tripwire meter did not start;
- no GPU was requested;
- no tokenizer was fit, selected or frozen;
- no training run or checkpoint was created or consumed;
- no sealed data was accessed; and
- no P-B or P-C action occurred.

The runtime-build receipt's `authoritative: true` therefore means
authoritative for the exact-runtime build contract. It does not promote the
result into run-axis authority for any later phase.

At execution time the Colab checkout was clean at
`5e3abb465fe9dd29f247c404f207b9bfb88651c1`, with zero commits ahead of and
zero commits behind origin.

## 2. Fail-closed first attempt and corrective commit

The first build used fresh work, prefix and receipt paths and failed closed at
repository runtime attestation. `setuptools==84` contains vendored nested
`.dist-info/RECORD` payloads. The initial inventory logic misclassified those
nested vendored RECORDs as additional primary distribution RECORDs and refused
the repository attestation.

No authoritative `PASS` receipt was minted from that attempt. Its failed work,
prefix and receipt-path state was preserved rather than overwritten or
relabelled as successful.

Commit `5e3abb465fe9dd29f247c404f207b9bfb88651c1` corrected the classifier: only a
distribution's top-level primary `.dist-info/RECORD` is accepted as its
primary RECORD, while vendored nested payloads remain part of the installed
tree and are still byte-hashed. A regression test covers this exact
`setuptools` layout. The correction narrows record classification; it does not
exclude the vendored bytes from the inventory.

## 3. Successful fresh-path build

The corrected build used a second, independent set of fresh paths:

| Role | Exact Colab path |
| --- | --- |
| Repository | `/content/weft1` |
| Work root | `/content/weft1-pa-build-run2` |
| Installation prefix | `/content/weft1-pa-runtime-run2` |
| Runtime-build receipt | `/content/weft1-pa-runtime-receipt-run2.json` |

The build completed with an authoritative `PASS` under the exact pinned
runtime contract.

### 3.1 Receipt identity

| Field | Exact value |
| --- | --- |
| Physical bytes | `2,277,385` |
| Physical SHA-256 | `108421db0ec2008f9cf99a858f1f80e17f096b0693114c7b845037bc7b49ab77` |
| Canonical receipt identity | `68ad983164e0e26f40ebcf606cee7d77b69a3bfcb3bbe2a2aa1037766189c142` |

The physical SHA-256 identifies the exact LF-terminated receipt file. The
canonical receipt identity is the receipt schema's internal semantic identity;
the two hashes have different domains and are not expected to be equal.

### 3.2 Pinned build inputs

| Input | Bytes | Bound identity |
| --- | ---: | --- |
| CPython 3.11.9 source archive | `20,175,816` | SHA-256 `9b1e896523fc510691126c864406d9360a3d1e986acbda59cda57b5abda45b87` |
| SQLite 3.45.1 amalgamation archive | `2,730,697` | SHA-256 `5592243caf28b2cdef41e6ab58d25d653dfc53deded8450eb66072c929f030c4` |
| SQLite archive | — | SHA3-256 `e311198775d5d5b2889d5fabe1d9a490567a14e605591d6a9e4c833804a8b4cb` |
| Extracted `sqlite3.c` | — | SHA3-256 `0474604df9e1b69a5544295dd046aad954749279780d557da80f44b958100295` |
| Hash-locked requirements | `65,717` | SHA-256 `bccb8e5b58b5e8fa9eee367fe9c26f59053fff5b7fadf81f23f96b83d1531860` |
| Runtime bindings | `17,400` | SHA-256 `ee10e69a3ccd55f7960949f4c318daa4db1197c779f5e88fb67cec82ab7f263b` |

Every selected wheel and resolved package hash is carried in the authoritative
JSON receipt. The wheelhouse and complete installed-file identities below bind
that detailed payload without duplicating 26 per-wheel records here.

### 3.3 Exact runtime observations

| Component | Observed pinned value |
| --- | --- |
| Python | `3.11.9` |
| SQLite | `3.45.1` |
| SQLite source ID | `2024-01-30 16:01:20 e876e51a0ed5c5b3126f52e532044363a014bc594cfefa87ffb5b82257cc467a` |
| Unicode database | `14.0.0` |
| `zstandard` package | `0.25.0` |
| `libzstd` | `1.5.7` |

### 3.4 Bound identities

| Evidence domain | SHA-256 identity |
| --- | --- |
| Execution environment | `ac8b747e23892088f979573a0302a10529a57f4b4f846ad5e08467d698cdd543` |
| Python executable | `a74055ace1460e7e15ca8160dfad5e7f0dbccdbcdef930eb0c3d124947879756` |
| Installed-distribution inventory | `159e9cbb2c5f074bcb8aef3b73c9507a807ebc1b624d4d8a711a3e77cdf00dc5` |
| Runtime linkage | `6beff012caabfd61690bb4fccc276093a9d0f659e380636b3a6afed64a07a188` |
| Trusted-installer chain | `4a96a388bd9a75a2160e3076dc972c59d6d8b9c41efb3c7adca2c086d970cfb1` |
| Selected wheelhouse | `402f84d5524a6bea92e1ef8719cb7ff7c8d99f23801bded00b6f6e037a525c31` |
| Runtime builder | `f61d05ea8f8c39b5a1e69821f5a14da7689635eb523b298f6dee83f24cb0f9fd` |
| Runtime contract | `252bd365995d58e574285b4727eb8e08339e40abc858b0ecc80d3b62ebf02676` |

The installed inventory covers **27 distributions, 5,409 files and
339,701,149 bytes**. The selected wheelhouse contains **26 wheels**. The
remaining distribution is the explicitly governed bootstrap installer rather
than an unregistered wheel.

## 4. Independent post-receipt verification

After the build receipt was closed, the installed interpreter was attested
again through the repository's exact runtime contract. The independent
observation matched the receipt byte-for-byte for all four repeated boundaries:

1. execution-environment payload and identity;
2. Python executable identity;
3. runtime linkage identity; and
4. complete installed-distribution file inventory and identity.

This second observation was post-receipt evidence, not a restatement of the
builder's in-process values. No mismatch was averaged, tolerated or waived.

## 5. Durable-storage blocker

Google Drive mounting was attempted three times and failed before DriveFS
started. Colab reported unsuccessful credential propagation. The metadata
guest-auth `user-id` endpoint returned HTTP 404 through both credential
brokers:

| Broker environment variable | Observed address | Result |
| --- | --- | --- |
| `TBE_EPHEM_CREDS_ADDR` | `172.28.0.1:8009` | guest-auth `user-id` HTTP 404 |
| `TBE_CREDS_ADDR` | `172.28.0.1:8008` | guest-auth `user-id` HTTP 404 |

No DriveFS mount, DriveFS log or durable-storage marker was created. The
successful runtime prefix and its PASS receipt therefore remain on ephemeral
Colab storage. The browser VM was intentionally left live to preserve that
state while the mount problem is resolved.

P-A materialization remains blocked. It may begin only after genuine durable
storage is present, the runtime receipt has been copied there, and the copied
file has been rehashed to the physical identity recorded in Section 3.1. A
local re-read on the ephemeral VM is not represented as a durable copy.

## 6. Repository verification and gate closure

The raw local repository suite at execution commit `5e3abb465fe9dd29f247c404f207b9bfb88651c1`
reported:

```text
1 failed, 3563 passed, 19 warnings in 109.35s
```

The one failing node is the governed Paper-2 evidence-ledger test, which
reports two missing evidence paths. It is not attributed to the runtime build,
and no repository-wide green claim is made.

The immutable quarantine was rolled forward without modifying its predecessor:

| Field | Exact value |
| --- | --- |
| Artifact | `training/ablation_lm_engineering_quarantine_20260829_pa_runtime.json` |
| Bytes | `2,988` |
| SHA-256 | `956f785f18a452d9b3cc9b9ffce66c2e2b84628879d7001a37ff7dfbb7255444` |
| Superseded artifact | `training/ablation_lm_engineering_quarantine_20260829_pa.json` |
| Superseded SHA-256 | `f5e6f6cba70b1c4d8f285fb67d8d731f6431bdbf275c13535c02ab2f0a4876b2` |
| Review due | `2026-09-04` |

The strict engineering gate then passed with exactly one quarantined failure
and `3563 passed, 19 warnings in 111.33s`. Its six gate/schema tests separately
reported `6 passed in 0.17s`. The initial raw full-suite observation was
`1 failed, 3563 passed, 19 warnings in 109.35s`; the two durations are retained
separately because they came from distinct executions. Repository-wide green
status remains explicitly prohibited.

## 7. Current state and next admissible action

The exact pinned runtime installation is complete and independently verified.
The corpus pipeline has not started. The next admissible operational step is
to restore a genuine durable-storage mount, copy the runtime-build receipt to
that backing, and verify the copied receipt as exactly `2,277,385` bytes with
SHA-256
`108421db0ec2008f9cf99a858f1f80e17f096b0693114c7b845037bc7b49ab77`.

Only then may the already-authorized P-A materialization proceed. P-B remains
closed pending the programme-level license review, and P-C remains closed
pending its own prerequisites. Nothing in this receipt changes either gate.
