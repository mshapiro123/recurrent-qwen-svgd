# WEFT-1 corpus/G-TOK Amendment A1 integration receipt

**Date:** 2026-08-28<br>
**Branch:** `codex/bicameral-stage0`<br>
**Implementation commit:** `d7ec98569d54c2acefe38225d5739b1307ef2098`<br>
**Disposition:** A1 is verified and its settled contract is integrated. Corpus materialization, tokenizer fitting/freezing, and proxy training remain fail-closed because A1 does not yet identify reproducible literal implementations for several run-consuming operations.

## 1. Authority verification

The raw Drive object `STRATEGY_CORPUS_GTOK_AMENDMENT_A1_20260828.md` was fetched and verified before implementation:

- Drive ID: `1mLBc0WenGJcxvlfTQoisew0hQN9WY8E7`
- bytes: `12,997`
- SHA-256: `e996f89fee81871a6432d90fabbaa0dc470b8f7643bc65756966e27883af3267`

The governing base handoff remains:

- Drive ID: `11TyC0GF3RlloZmB5J43h1_4WY10FzBvF`
- bytes: `16,431`
- SHA-256: `2aecb64711a2bf2776c8d1940350bc5d42b335f60eb774ac1e941f470b9cf74c`

A1 is appended through `GTOK_EXECUTION_AUTHORITY_CHAIN_V2`. The original `GTOK_EXECUTION_AUTHORITY_CHAIN` and `execution_authority_bound_sha256()` implementation are unchanged, so no banked v1 receipt is re-keyed. New A1-era objects require an explicit `_v2` schema.

## 2. Accepted corrections now encoded

### A1-R1: ten executing proxy blocks

The G-TOK proxy is `4 prelude + 2 core + 4 coda = 10` dense blocks. Structural OFF disables WEFT-specific machinery; it does not delete the two core blocks. A direct forward-hook test observes exactly `(4, 2, 4)` block calls, one core visit and two core-block passes, with recurrence, scratch, engram and long-term memory absent.

The exact unique-parameter counts are:

| vocabulary | unique parameters |
|---:|---:|
| 16,384 | 37,891,840 |
| 24,576 | 42,086,144 |
| 32,768 | 46,280,448 |
| 49,152 | 54,669,056 |

The projected screen cost is recorded as 6.5 A100-hours against the unchanged cumulative 12 A100-hour tripwire.

### A1-R2: source routes

The delegated route resolution is encoded as an ordered seven-row ledger. Every row binds repository, config, split, 40-hex revision, selector, selection rule, card URL, card SHA-256, enumerated asset count, available bytes and the required byte margin.

| family | exact repository/config | revision |
|---|---|---|
| Dolma web | `allenai/dolma3_pool` / `default` | `6462556697df1a8f5c953727e9c686629ad98b68` |
| Wikipedia/Wikibooks | `allenai/dolma` / `v1_7` | `7f48140530a023e9ea4c5cfb141160922727d4d3` |
| StackEdu | `allenai/dolma3_mix-6T` / `default` | `689a3ea2d8217e64d73a5058913fa43ad15e81aa` |
| FineMath-3+ | `HuggingFaceTB/finemath` / `finemath-3plus` | `e92b25a616738fe95dc186b64dfb19f9c8525594` |
| arXiv | `allenai/dolma3_mix-6T` / `default` | `689a3ea2d8217e64d73a5058913fa43ad15e81aa` |
| olmOCR | `allenai/dolma3_mix-6T` / `default` | `689a3ea2d8217e64d73a5058913fa43ad15e81aa` |
| FineWeb-Edu | `HuggingFaceFW/fineweb-edu` / `default` | `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9` |

Mechanical card checks report ODC-By for all seven bound routes and sufficient available bytes. This is not the programme-level P-B license approval; Mark's human review remains required.

Two discrepancies are retained in the ledger:

1. A1 says the preflight receipt contained draft route bindings; it did not. These are newly resolved A1-R2 bindings.
2. The pinned `dolma3_pool` card names olmOCR, but its pinned repository tree contains zero matching assets. The executable olmOCR route therefore uses the pinned `dolma3_mix-6T` tree. The mix is upsampled, so selection first deduplicates stable document identities and does not inherit mix multiplicity.

### A1-R3 through A1-R5

The contract now records:

- `T_target = 4,000,000,000` bytes and independent `H_target = 80,000,000` bytes;
- document-aligned floor separately by stratum and stream, no overshoot, at most 0.5% shortfall, plus rejected-boundary-document evidence;
- zero document and near-duplicate-cluster overlap between T and H;
- AdamW at LR `3e-4`, betas `(0.9, 0.95)`, epsilon `1e-8`, weight decay `0.1`, global clip `1.0`, batch `256 x 2048`, cosine to 10%, 1% warmup, bf16 compute and FP32 masters/loss reduction, all in one parameter group;
- the A1 regex, byte-level BPE headline, single-digit numeric splitting, minimum frequency 2, no lowercasing/NFKC/dropout and 256 byte atoms;
- exact SHA-1 followed by 128-component MinHash, `16 x 8` LSH, byte-level 13-grams and Jaccard threshold 0.8, with Dolma canonical over a matched FineWeb-Edu duplicate;
- separate pipeline and per-arm/per-seed RNG namespaces.

## 3. Hash-addressed implementation

| object | SHA-256 |
|---|---|
| `training/weft1_gtok_contract.py` | `c86d84f3c6926e15837e1e107f2750eea117404bfb2415f4874af1caa38de890` |
| `training/weft1_gtok_a1_contract.py` | `56bf20f94b7c5472ef13fb3eeacf546299135473b3dbd4c81f1a23b67c3105e4` |
| source-route JSON bytes | `1cf99ea33b72013f4bf07101aad8c9b5124879afe3de9f28991e6427ea861a6c` |
| source-route v2 receipt | `8455b63f8b0dde7f5a5bdb599bec7563ce2b8c9159a26b09f6302e6e326bb663` |
| A1 contract snapshot v2 | `1027ad932de92f21fc15695f5dc2d591295f03c7e2929b58f6e13d1dda5e3d03` |
| tokenizer headline receipt v2 | `90b9f85645472e9acd50dac64ea436c6f8d6fc7b3f35623ff7fecbd9b47cfd70` |
| dedup headline receipt v2 | `e7a560844f1312c1e6683fd69845dac6b82e883bb62d8fec79944af2ad9e26de` |
| optimizer headline receipt v2 | `10f2399658cf2a2a71dbeb5063c2ed93fa01117f46e373724e49581564d7d525` |

The new module deliberately contains no downloader, corpus/shard writer, tokenizer fit, optimizer constructor, gate minter or training launcher. It is a validated contract boundary, not a partial production implementation that could accidentally run.

## 4. Literal execution defects reported under A1-R5

A1 says execution unlocks after authority verification, and also requires exact replay, fail-closed gates and echo-in-receipt bindings. Those requirements cannot simultaneously be met until the following literal choices are supplied. They are therefore reported rather than locally guessed:

1. **Floor/termination conflict:** independent whole-document floors can make realized T smaller than 4B, while the inherited protocol requires a 4B terminal point and 1B/2B/4B BPB milestones. Final-batch, termination and absent-4B receipt semantics are unspecified.
2. **Language ID:** exact package, model artifact/hash, version, threshold and equality/tie behavior are absent.
3. **Match normalization:** the literal NFC/whitespace algorithm, whitespace set, newline and trim behavior, and Unicode version are absent.
4. **Production MinHash:** hash/permutation family, root seed, byte framing, short-document behavior, candidate ordering and recall audit are absent.
5. **Shard format:** serializer, compression parameters, record framing, timestamp policy and manifest self-hash exclusions are absent.
6. **Tokenizer serialization:** `tokenizers` version/lock, `Split` and `ByteLevel` flags, decoder/post-processor, initial alphabet, ordered special-token strings/IDs/roles and `AddedToken` flags are absent. The installed `tokenizers==0.22.2` `BpeTrainer` exposes no seed argument, so `gtok.bpe` cannot be implemented as written without a deterministic alternative definition.
7. **Seed values:** a campaign/corpus root seed and the two numeric training seeds are absent. Per-arm run-seed naming also conflicts with the requirement for one shared frozen corpus/tokenizer unless roles are clarified.
8. **Ranking/top-up:** tie behavior is absent where a route lacks an explicit scalar quality score, as is whether FineWeb-Edu top-ups must be re-deduplicated.
9. **Gate evidence:** D1/D2's independent-process proof and authoritative production evidence schemas for D3-D6 are not yet literal.
10. **Training/measurement:** packing, final batch, scheduler step count/rounding, undertrained-row threshold and runtime/FLOP/memory/throughput/latency schemas are absent.
11. **Tripwire enforcement:** no pre-dispatch reservation or in-flight cumulative-cost cancellation mechanism is specified.

This blocks run-consuming P-A/P-B/P-C actions, not ordinary build-axis work. No corpus bytes were downloaded or materialized; no tokenizer was fit or frozen; no optimizer or checkpoint was created; no sealed data was contacted; no gate was minted.

## 5. Quarantine review and verification

The dated engineering quarantine was reviewed as requested. Of the previous three exact expected failures, two now pass and one remains:

- remaining: `tests/test_paper2_claim_evidence_ledger.py::test_paper2_claim_evidence_paths_exist`
- resolved target-marker node: commit `20e79322`
- resolved Python/Markdown mirror node: commit `f31f4aa3`
- next review due: **2026-09-04**, or before the next repository-wide engineering receipt, whichever comes first
- new quarantine bytes SHA-256: `042e5c56eb487940b8434d6d21401fd9f75d08820d0468445d81962ed1d5f7b9`

The legacy Stage 2A final-receipts manifest was not edited. It is now explicitly LF-pinned, and its working SHA-256 remains the registered `39e22bfefd7b5ef7cb2cc7af31be78d912ba7fe0380e2a7c4fb0df383eef1dca` with an identical Git blob.

Verification results:

- A1/G-TOK/quarantine focused suite: `58 passed in 1.82s`
- full repository suite: `1 failed, 3363 passed, 19 warnings in 91.96s`
- strict engineering gate: PASS because the sole failure exactly matches the live quarantine; its rerun still reports the repository suite as red (`1 failed, 3363 passed, 19 warnings in 85.68s`)
- Python compilation: PASS
- Ruff: not run because Ruff is not installed in the active interpreter (`No module named ruff`)

No repository-wide green claim is made. `.runlogs/` was preserved unchanged at 189 files and 14,932,623 bytes.

## 6. Required next authority

One compact literalization amendment can unlock the run. It must settle the eleven items in section 4 and state whether materialization may begin before Mark's P-B license approval or only after it. Once that authority is hash-verified, the implementation can add the production pipeline and D1-D6 receipts without changing the contract, route ledger, v1 receipt hashes or proxy topology recorded here.
