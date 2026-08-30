# WEFT-1 P-A parser recovery receipt

**Date:** 2026-08-30

**Code repair commit:** `df5e19a72cbd6fd0ca2bd13cb1e0ce0be590182e`

**Disposition:** implementation and tests complete; the failed run remains immutable and no corpus gate was minted

## 1. Outcome

The stopped P-A replay was not a Colab eviction. Its durable log terminates on a strict SQLite identity collision while parsing `wikipedia_wikibooks`. A hash-only replay of the exact two records proves that upstream page ID `12` occurs under two different `metadata.provenance` values with different retained bytes. The parser had incorrectly treated the upstream page ID as globally unique across the combined route.

The repair preserves the collision guard and narrows only the identity assumption: Wikipedia/Wikibooks canonical IDs now include the exact upstream provenance namespace. The same native ID under different provenance values is distinct; the same native ID under the same provenance with different content still stops with hash-only evidence.

A separate pre-replay audit found that the selected StackEdu prefix contains two pinned physical schemas. Production now resolves an exact parser per asset and binds both variants into one composite parser identity. This issue had not caused the stopped run, but it would have stopped the fresh replay later if left unresolved.

## 2. Failed-run evidence

The immutable failed run is `pa-v4-2d9278c0-r7`, launched from commit `2d9278c09187ebfcb10f2c8271c0ce45815d862b` at `2026-08-30T06:25:30.786157+00:00`.

- Durable log: `/content/drive/MyDrive/WEFT1/a3-pa/pa-v4-2d9278c0-r7.log`
- Durable output: `/content/drive/MyDrive/WEFT1/a3-pa/pa-v4-2d9278c0-r7/`
- State: `_INCOMPLETE` present; no parent replay receipt
- Terminal exception: `CorpusMaterializationError: stable source record ID repeats with different bytes`

The hash-only diagnostic receipt is:

- Path: `/content/drive/MyDrive/WEFT1/a3-pa/diagnostics/wikipedia-native-id-collision-receipt-v1.json`
- Physical SHA-256: `ed216f1d9009aae2dd85692d8882613bb6544aeb82143286bf35285010684ae0`
- Classification: `PROVENANCE_NAMESPACE_COLLISION`

The two colliding observations are:

| observation | asset | row | native ID | provenance | retained bytes | retained-text SHA-256 |
| --- | --- | ---: | --- | --- | ---: | --- |
| first | `wiki-0001.json.gz` | 2,755,173 | `12` | `en_simple_wiki_v0-0001.json.gz:2755174` | 1,125 | `9c0a2a6afc7638142600d99b71e460d1fe7ad3318707c7f701ff328cc3ca99ff` |
| repeated | `wiki-0000.json.gz` | 3 | `12` | `en_simple_wiki_v0-0000.json.gz:4` | 18,795 | `dec81d60412af097e8f6468bfabc5ceac8cd063886c53538766b7c99ee103d55` |

No raw source text is present in the diagnostic receipt.

## 3. StackEdu schema evidence

The selected StackEdu cache contains 114 assets and 9,579,222,114 compressed bytes:

- 101 normalized Dolma `shard_*` assets, 8,535,089,600 compressed bytes;
- 13 direct Python `part-*` assets, 1,044,132,514 compressed bytes.

The direct-Python binding is anchored to `data/stack_edu-Python/part-000000054.jsonl.zst`. A live sample of 256 rows from every direct-Python asset, 3,328 rows total, observed one top-level key set with SHA-256 `3ab2bd0f989b98b34ab6ee2727c0450896737c88f708df4838e5975bf767778c`. Exact non-text kinds were:

- `blob_id`, `language`, `license_type`, `path`, `repo_name`, `src_encoding`: string;
- `detected_licenses`: array;
- `download_success`: boolean;
- `int_score`, `length_bytes`: integer;
- `score`: float.

The direct binding uses `blob_id` as the native ID and top-level `int_score`. The normalized binding continues to use `id` and `metadata.int_score`. Unmatched paths and a caller-supplied wrong variant fail closed. The production composite parser identity is `30236658d243ef29c06fbac12fdb999db036661fdd602740dc09a8f9665346f7`.

## 4. Defensive collision instrumentation

Production materialization now distinguishes:

- exact repeat: accepted, first occurrence remains canonical;
- StackEdu score-only variance: accepted only when retained bytes are identical and both scores are at least 3; first occurrence and first score remain canonical;
- content, retained-byte-count, or unauthorized-score divergence: stop with hashes, scores, and ordinals, never raw text.

Source parse receipts carry the count and deterministic framed digest of accepted StackEdu score-only variance. The optional standalone hash-only audit is additive; production's exact-byte comparison remains authoritative.

| artifact | SHA-256 |
| --- | --- |
| `training/weft1_stackedu_collision_audit_v1.py` | `37f58844cf5c39b7f3c4696dee417e25ec3280cbb4665ec3ee704513d3e41c76` |
| `scripts/run_weft1_stackedu_collision_audit_v1.py` | `64a2329c857a64b69120e720f862e99c9556e034334abf18fa318abe80e70a6d` |
| `tests/test_weft1_stackedu_collision_audit_v1.py` | `5560a2339d214593f2abb179027af80ba8a8c22377a8d8ccbf32c177e4cd90a0` |

## 5. Verification

- Primary focused source/materializer tests: `64 passed`.
- All corpus tests: `315 passed`.
- Standalone StackEdu audit tests: `8 passed`.
- Raw repository suite: `1 failed, 3848 passed, 19 warnings in 154.94s`.
- Strict quarantine-aware rerun: PASS; underlying suite `1 failed, 3848 passed, 19 warnings in 126.21s`.

The sole failure is unchanged:

`tests/test_paper2_claim_evidence_ledger.py::test_paper2_claim_evidence_paths_exist`

The quarantine was rolled forward without editing its predecessor:

- Artifact: `training/ablation_lm_engineering_quarantine_20260830_pa_recovery.json`
- SHA-256: `0bbe529556a73dc07072b4419791f3db72e854575cca8fda1a205d483dafc2eb`
- Review due: 2026-09-04
- Repository-wide green claim: prohibited

## 6. Run posture

No failed-run file was overwritten, no P-B freeze or vocabulary was minted, no sealed data was touched, no model checkpoint was created, and no GPU training was started. Resume requires a fresh pushed commit, a rebuilt exact CPython 3.11.9 runtime, a fresh local work root, and a fresh durable output root. The existing verified cache, route receipts, durable-storage marker, and fastText model remain eligible inputs and will be rehashed by the parent replay.
