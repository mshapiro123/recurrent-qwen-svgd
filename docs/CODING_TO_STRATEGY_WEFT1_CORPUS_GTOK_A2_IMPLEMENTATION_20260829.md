# WEFT-1 corpus/G-TOK Amendment A2 implementation receipt

**Date:** 2026-08-29

**Branch:** `codex/bicameral-stage0`

**Implementation commit:** `d9a790f520397aa848eee87172339a6558a553d8`

**Disposition:** A2 is implemented through the production P-A boundary. P-B remains closed pending Mark's programme-level license review. P-C remains closed pending a completed P-A replay and D1-D6 evidence. No gate is minted by this commit.

## 1. Authority and execution surface

The governing A2 artifact was fetched from Drive and verified before implementation:

- Drive ID: `1f2iue1yYW2gpdamsqVlaQnijfllRCaVn`
- bytes: `8,615`
- SHA-256: `f7a2655b30f6c699035ec4ffdccee8c03068eeab8da94894be8e5818f955ce02`

The A2 authority is appended to the existing v1/v2 receipt chain under the v3 domain; no banked receipt is re-keyed. W-1's two-axis interpretation is preserved: the build implemented the pipeline without consuming a vocabulary, training compute, sealed data, or a checkpoint. The authorized P-A run surface is now concrete but has not been executed in this receipt.

## 2. Implemented production boundary

The implementation supplies five composed boundaries:

1. **Authoritative enumeration.** Only an internally instantiated and version-attested `huggingface_hub==1.24.0` `HfApi` can mint an authoritative six-family Hugging Face enumeration. Injected tree callables are fixture-only. Xet-only identities fail closed because a Xet Merkle hash is not treated as a raw-content SHA-256.
2. **Pinned Wikipedia observation.** The factory reads the revision-pinned `urls/v1_7.txt`, requires 171,893 bytes and SHA-256 `9fa8c2f0eb57149ff7914b35ca2ffb8da221c02786d712bcba5f6c39d294b49e`, and selects exactly `wiki-0000.json.gz` and `wiki-0001.json.gz`. It streams and hashes the selected assets before minting the listing.
3. **Verified local cache and parser spine.** Download plans are factory-derived canonical subsequences of the complete enumeration. Finalization writes a canonical source manifest and independently rehashes every local asset. Parser evidence binds the cache asset identity and SHA, exact record ordinal, raw and canonical row sizes/hashes, full JSON or Arrow schema, every disposition, and the retained-text spool.
4. **Disk-bounded materialization.** The offline worker rejects injected production streams, reparses the exact verified cache, applies the general-only FastText filter, global exact uniqueness, Dolma-first exact/near cross-source deduplication, deterministic first-fit selection, real LSH/Jaccard cluster exclusion, T/H allocation, zstd JSONL shards, D3-D6 checks, per-seed consumer-order receipts, and the T-only tokenizer input contract.
5. **Parent-observed D1/D2 replay.** The fixed production parent runs the one registered child worker twice under `/usr/bin/unshare --net`. It binds the A2 authority, settled bindings, dependency lock, route ledger, enumeration/download/source manifests, FastText model, Python executable, worker arguments, and every consumed implementation module. The parent independently rehashes every child artifact and recomputes the D2 decision ledger, rates, dropped/top-up bytes, selection ledger and recall audit. The generic replay API can report `CHECK_PASS` but cannot set the production-profile bit or mint authoritative `PASS`.

The production worker calls exact runtime attestation before loading a receipt, cache asset, or FastText model. It has no downloader and cannot mint P-B, D1, D2 or G-TOK gates. The `full-pa` CLI is the only composed execution entrypoint for the prepared cache; source preparation is a separate online stage and materialization is network-isolated.

## 3. Correctness properties encoded

- The proxy remains `4 prelude + 2 core + 4 coda = 10` executing dense blocks. Structural OFF disables WEFT machinery and never removes core blocks.
- T is exactly targeted at 4,000,000,000 retained UTF-8 bytes and H at 80,000,000 bytes, by independent per-stratum document-aligned floor with the registered 0.5% tolerance.
- Dolma is fully selected and its dedup index sealed before any FineWeb query. Exact and near rates use FineWeb query decisions as their denominator. FineWeb top-up continues in ranked-stream order and re-runs both dedup stages.
- D6 excludes both exact document overlap and every registered near-duplicate connected component across T/H. Each training seed has a distinct deterministic T order; every vocabulary arm for that seed consumes the same ordered multiset and the same H identity.
- Tokenizer fitting accepts only the receipt-bound T shards in canonical manifest order. H cannot enter through the production iterator.
- FastText hashes the exact newline-terminated backend bytes and loads a privately snapshotted, post-copy-verified model, closing the path reopen/TOCTOU gap.
- Governed JSON rejects duplicate keys, non-finite values, BOM/NUL/CR/tab/trailing whitespace and noncanonical serialization where identity requires it. Governed paths reject lexical symlink and Windows reparse ancestors before resolve, stat or open.
- The checked-in A2 bindings ledger is production-loadable only at SHA-256 `ee10e69a3ccd55f7960949f4c318daa4db1197c779f5e88fb67cec82ab7f263b`. Alternate binding paths require an explicit nonproduction-fixture flag.
- The dependency lock is 65,717 bytes with SHA-256 `bccb8e5b58b5e8fa9eee367fe9c26f59053fff5b7fadf81f23f96b83d1531860`; a cross-platform Colab regeneration using the registered cutoff and `--upgrade` reproduced it byte-for-byte.

## 4. Colab substrate observations

The in-app browser is connected to the Pharma Initiatives Colab Pro+ account as `mshapiro@pharmainitiatives.com`. The connected P-A runtime is CPU; no GPU was requested.

Observed base runtime:

- Python `3.13.15`, SQLite `3.37.2`, Unicode database `15.1.0`
- Linux `6.6.122+`, 12.67 GB RAM and 225.83 GB disk
- `uv 0.12.5`, executable SHA-256 `b65f23a420c4acc96427efb30e5ed9bc0f7e25d2d712000f6ede77c1a0de5f46`
- `/usr/bin/unshare`, SHA-256 `72a34e6ba98a59f1da0c7b4d8c9722b746b5ade54e4d7e8de8e519c2993858ad`

The browser-run network probe connected normally outside the namespace and failed with `Errno 101` inside `unshare --net`, establishing the registered OS isolation mechanism. The browser also independently fetched and verified the Wikipedia locator parent and its exact two selected URLs.

The pinned Python 3.11.9 environment and packages have not yet been installed because software installation requires the user's action-time confirmation in the in-app browser workflow. Consequently no authoritative environment receipt, corpus download, materialization, D1/D2 PASS, tokenizer fit/freeze, training run, checkpoint, or sealed-data access is claimed here.

## 5. A2-R7 implementation bindings

The following literal implementation choices are delegated under A2-R7 and are replay-tested rather than inferred from library reputation:

- canonical JSON is sorted, compact UTF-8 with one terminal LF; JSONL identities use explicit length framing and registered domain strings;
- MinHash uses the in-repository NumPy `2.4.6` PCG64 coefficient construction, 128 uint64 components with wraparound, 16 bands by 8 rows, SHA-256-derived byte-shingle values, and exact Jaccard rejection at 0.8;
- the streaming dedup ledger is a resume-validated SQLite state machine with a byte-stable schema/catalog, phase ledger and foreign-key checks;
- shard identity hashes uncompressed canonical JSONL bytes; zstd settings are fixed independently from corpus identity;
- per-module/pipeline RNG names are derived from the A2 authority root; no stage consumes another stage's stream;
- authoritative replay uses the fixed Linux `unshare` binary and also injects a Python socket guard as defense in depth;
- complete parser observations record full Arrow IPC schema bytes rather than presenting a projected card schema as the observed schema;
- source-cache selection and online transport are resumable, content-verified and receipt-bound; completed verified assets are reused rather than overwritten.

## 6. Verification and open gates

The commit-producing verification run reported:

- focused WEFT-1 corpus/G-TOK suite: `179 passed`;
- compilation: `python -m compileall -q training scripts tests` passed;
- full repository suite: `1 failed, 3529 passed, 19 warnings in 119.47s`;
- exact-node engineering gate: PASS, because the only failure is the unchanged governed Paper-2 evidence-ledger node;
- `git diff --check`: passed.

The generic local fixture replay no longer skips when the pinned Colab runtime is unavailable. It uses its fixture-only backend locally; the production I/O hook is tested directly with an injected typed runtime receipt, while exact production runtime attestation remains a Colab execution requirement. No repo-wide green claim is made. Ruff remains optional and was unavailable in the local interpreter.

The live engineering quarantine was rolled forward to `training/ablation_lm_engineering_quarantine_20260829.json` because its exact pass count changed with this test addition. Its one expected failure, authority restrictions, 2026-09-04 review date and no-green-claim rule are unchanged.

Open programme actions are unchanged:

1. Mark performs the programme-level license review before P-B freeze.
2. P-A source preparation and the two authoritative offline materializations run in Pharma Initiatives Colab after explicit installation confirmation.
3. D1-D6 receipts are reviewed; only then may P-B freeze occur.
4. G-TOK calibration/training remains bounded by the 12 A100-hour mechanical tripwire and does not begin from this implementation receipt.
5. The existing engineering quarantine review date remains 2026-09-04; this work does not silently retire the governed Paper-2 evidence-ledger failure.
