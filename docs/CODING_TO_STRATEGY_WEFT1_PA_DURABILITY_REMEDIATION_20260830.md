# WEFT-1 P-A durability remediation receipt

Date: 2026-08-30

Status: **LOCAL REMEDIATION PASS; PHYSICAL BACKEND-LOSS ACCEPTANCE PENDING**

Core implementation commit: `de90992e0f2fd43e7d494f41a0a6edcd2f12dc01`

Authority: build, tests, and bounded P-A infrastructure only. No corpus gate, tokenizer freeze, sealed-data contact, optimizer construction, checkpoint training, or target training is minted here.

## 1. Outcome first

The failed `pa-v4-23d001b7-r2` run is not resumable and remains immutable. The replacement code no longer leaves the production source-parse ledger as one long-lived open DriveFS file. It publishes closed, immutable, receipt-bound chunks only after each parse event has completed its disposition handling and any retained-record insertion. A successful run reconstructs the exact legacy ledger bytes and removes all staging. An interrupted run remains `_INCOMPLETE`, cannot mint a gate, and cannot be used as a resume source.

Local and repository-wide verification is complete and green under the standing exact-node quarantine. A stronger physical acceptance remains deliberately pending: publish the exact probe stage in the Pharma Initiatives Colab, record its SHA outside the VM, replace the whole backend without calling `drive.flush_and_unmount`, rebuild the identical pinned runtime, remount Drive, and verify every surviving object against the stage projection. A deliberate flush/remount is a useful transport canary but is not evidence for surprise-backend-loss durability.

No P-A replay may relaunch until that stronger probe passes.

## 2. Incident classification

The preceding runtime disappeared without a Python traceback, exit receipt, or terminal replay receipt. On reconnection, the VM boot identity, repository checkout, exact runtime, local work root, process, and authoritative SQLite parse state were absent. The Drive copy retained only a stale prefix of the open Wikipedia ledger even though the prior observer had seen materially later events.

The evidence supports **infrastructure/backend loss with an unsafely long open-file durability window**. It does not distinguish eviction, reset, or another provider-side termination mechanism, so this receipt does not claim a more specific cause.

The durable r2 evidence is nonauthoritative:

- `dolma_web.jsonl`: 867,733,449 bytes, through event 2,788,971;
- `wikipedia_wikibooks.jsonl`: 950,904,914 bytes, through event 2,950,857 in asset 0;
- the old observer had seen later Wikipedia events, proving observer progress cannot promote unsynced bytes to durable progress;
- StackEdu had not begun durably;
- `_INCOMPLETE` remains;
- the parent replay receipt and terminal child receipt are absent.

A fresh sibling termination receipt must inventory and rehash those facts from Colab before r3. It must state `authoritative=false`, `gate_minted=false`, `reusable_output=false`, and `fresh_replay_required=true`; it must not modify r2.

## 3. Remediation

`_DurableSourceParseLedgerV3` now implements:

1. a 65,536-event checkpoint cadence plus every asset boundary;
2. `write()` followed by an explicit `commit_event()` only after the event is fully handled;
3. fresh local chunk staging, file flush and fsync, close, copy to a fresh Drive `.partial`, file fsync, atomic replace, directory fsync where supported, and close/reopen rehash;
4. canonical hash-chained receipts binding source family, chunk identity, event range, asset and source-record ordinals, payload bytes, and the preceding receipt;
5. explicit `PARSE_PROGRESS_ONLY_NO_RESUME` and `resume_authorized=false` fields;
6. maximal-closed-prefix validation that preserves earlier valid evidence while separately reporting a partial, orphan, or unexpected tail;
7. fail-closed `finish()` behavior for every non-clean tail;
8. strict regular-file and symlink/reparse checks with descriptor-based reads;
9. exact reconstruction of the pre-change final JSONL bytes followed by checkpoint cleanup.

The implementation deliberately narrows its claim. Closed chunks reduce process-local and open-file loss. Whole-backend survival is established only by the physical no-flush replacement test; the chunks never become resumable parser state.

## 4. Tests and identities

### Source identities after final local hardening

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `training/weft1_corpus_materialize_a2.py` | 191,235 | `bf06acf926978e71e4b7017d5d2465ea9e628d78af89e0f0ac0b7d6764f0d70c` |
| `tests/test_weft1_corpus_materialize_a2.py` | 62,737 | `7e9700cd2927b7fd13336621570e2a0b17fe18770fe2085814e7b4ee022c4f46` |
| `scripts/probe_weft1_source_parse_drivefs_v1.py` | 52,135 | `3e8aec112da5aaff24c117cd6b03c771c0347c5e44b6d2d51f5f75b3d1b53ef7` |
| `tests/test_probe_weft1_source_parse_drivefs_v1.py` | 7,380 | `97b9492c764b9e671de61f21b7aebe85cd765f3aa78650b119a634f6cebed541` |

### Verification results

- focused durability, recovery, production-path, and two-phase probe matrix: **17 passed**;
- complete modified materializer plus probe tests: **28 passed**;
- 25-file corpus/P-A adjacency matrix: **386 passed**;
- raw repository suite: **1 failed, 3,865 passed, 19 warnings in 183.25 s**;
- the one failure is exactly `tests/test_paper2_claim_evidence_ledger.py::test_paper2_claim_evidence_paths_exist`;
- successor strict quarantine gate: **PASS**, with the underlying suite still explicitly red at **1 failed, 3,865 passed, 19 warnings in 133.04 s**;
- `py_compile`, `git diff --check`, LF staging checks, canonical JSON checks, duplicate-run rejection, and local two-phase probe simulation: PASS;
- Ruff was unavailable and no Ruff claim is made.

The immutable quarantine successor is `training/ablation_lm_engineering_quarantine_20260830_pa_durability.json`, SHA-256 `80a53a3ad6e264b401d2e02cd1bfd4bed486edbc94d1fdae0cc74ecbbb4e33e3`. It supersedes v8 without editing it, retains the 2026-09-04 review date, and does not authorize training.

## 5. Physical acceptance and relaunch order

The exact next sequence is:

1. commit and push this receipt/quarantine integration and confirm a clean `0 0` branch;
2. rebuild the exact runtime in the Pharma Initiatives Pro+ Colab from that commit;
3. run probe `publish` to fresh Drive and local roots and capture the returned stage SHA outside the VM;
4. force a whole-backend replacement without invoking any Drive flush helper;
5. on the fresh backend, clone the same commit, rebuild the same exact runtime, mount Drive, and run probe `verify --barrier-kind unflushed-backend-replacement`;
6. require a changed kernel mount ID, identical runtime and code identities, two physical reopens of the stage manifest, and two complete rehash passes over every durable object;
7. publish the fresh sibling r2 termination receipt without touching r2;
8. rebuild or retain the verified exact runtime, rehash every reusable parent cache/route/enumeration/download/source-manifest/marker/FastText input, and launch a fresh absent r3 parent and local work root;
9. monitor closed checkpoint receipts, not open ledger size, through the first complete event after Wikipedia event 2,951,021 and then through two complete records from the direct StackEdu Python binding at source-local asset ordinal 4;
10. only after those checkpoints pass may normal P-A monitoring resume. No checkpoint produced here can mint P-B, G-TOK, or any training gate.

## 6. Standing caveat

The physical probe can prove that this exact closed-object protocol survived the deliberate no-flush backend replacement performed for acceptance. It cannot prove that every future provider failure mode will preserve the newest chunk. Any later replacement still validates only the maximal physically surviving closed prefix, keeps the run incomplete, and requires a fresh replay.
