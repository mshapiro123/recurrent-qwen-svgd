# Coding to Strategy: WEFT-1 Corpus / G-TOK Execution Preflight

**Date:** 2026-08-28

**Branch:** `codex/bicameral-stage0`

**Starting commit:** `31a1a86db764ada084bd02fad69bea371aaa4e2b`

**Status:** SCOPED PREFLIGHT TESTS GREEN / MATERIALIZATION, FREEZE, AND SCREEN BLOCKED

**Sealed data:** operator attestation and command-scope evidence say untouched and unscored

## 0. Executive result

The execution handoff is authentic and internally clear about its authority boundary, but
it is not yet executable. Two statements conflict with the ratified build contract, and
several fields that determine the corpus, tokenizer, and training trajectory are still
placeholders or prose rather than literal bindings.

The safe build-axis work is implemented. The repository now has an execution-authority
chain that does not rewrite already-banked design hashes, typed source/shard/split/stream
schemas, reference-only byte-shingle and MinHash fixtures, and D1-D6 diagnostics that are
structurally unable to claim gate PASS. Any attempt to mint an authoritative D1-D6 receipt
still calls the fail-closed G-TOK authority guard.

No corpus payload or dataset asset was downloaded, no corpus was selected or materialized,
no tokenizer was fit or frozen, no optimizer was constructed, no model was trained, no
checkpoint was written, and no sealed battery was read. These are operator attestations
bounded to the commands and tools used for this preflight; the Drive handoff itself was
downloaded as raw bytes for verification.

## 1. Authority verification

The governing artifact was downloaded as raw bytes from Drive and independently verified
before implementation:

| Field | Verified value |
|---|---|
| Document | `STRATEGY_TO_CODING_AGENT_CORPUS_GTOK_20260828.md` |
| Drive file | `11TyC0GF3RlloZmB5J43h1_4WY10FzBvF` |
| Parent folder | `1aSbU2i8JZ37g5bJpyweLuvFaV0y92Qjr` |
| Bytes | `16,431` |
| SHA-256 | `2aecb64711a2bf2776c8d1940350bc5d42b335f60eb774ac1e941f470b9cf74c` |
| Verification | byte count and full digest match |

The execution hash chain added to code follows the handoff's governing order:

1. build handoff `498f34b5…eb02`;
2. ratification `c5df7429…6d3a`;
3. G-TOK rulings `167fc17d…f0d2`;
4. English-scope ruling `19399342…02d3`;
5. engram/tokenizer amendment `0221545d…65b5`;
6. curriculum-data r2 `14f0ba5d…ea22`;
7. Qwen adjudication r2 `6c2568d5…a1f`;
8. curriculum decisions `61fc7727…8b6d`; and
9. this execution handoff `2aecb647…f74c`.

The older `GTOK_AUTHORITY_CHAIN` is deliberately unchanged. Existing design-receipt hashes
therefore remain stable; future run artifacts need an execution-envelope v2 rather than a
retroactive mutation of v1 receipts.

## 2. Stop-line findings

### P0-A — `4/2/4` currently means both eight and ten blocks

The execution handoff says that the `4/2/4` proxy with its core structurally OFF executes
eight dense blocks. The build contract and current PyTorch graph execute the two middle
blocks once as ordinary dense blocks when recurrence is OFF: four prelude + two core + four
coda = ten.

Both readings are explicit. Choosing either locally would alter the G-TOK training graph,
parameter count, FLOPs, and the validity of the paired screen. Strategy must bind one of:

- `4/0/4`, eight executed blocks; or
- `4/2/4`, ten executed blocks with the middle pair run once and recurrence disabled.

### P0-B — the two-pin source topology cannot supply the named strata

The handoff treats `allenai/dolma3` plus FineWeb-Edu as the two upstream repositories for
all six source families. The live `allenai/dolma3` endpoint resolves to
`allenai/dolma3_pool`. Its official dataset card describes the pool as Common Crawl plus
olmOCR; Wikipedia/Wikibooks, StackEdu, FineMath, and arXiv are separate upstream dataset
families ([official Dolma 3 dataset card](https://huggingface.co/datasets/allenai/dolma3_pool/blob/main/README.md)).
The governing files still contain literal `revision="<commit-sha>"` placeholders.

This needs a route table with one row per source family:

`source family -> requested repo -> resolved repo -> config -> split -> full revision -> asset enumeration rule`.

The observed live revisions are useful only as discovery evidence and were not adopted as
pins:

| Endpoint observed 2026-08-28 | Resolved revision |
|---|---|
| `allenai/dolma3` -> `allenai/dolma3_pool` | `6462556697df1a8f5c953727e9c686629ad98b68` |
| `allenai/dolma3_mix-6T` | `689a3ea2d8217e64d73a5058913fa43ad15e81aa` |
| `HuggingFaceFW/fineweb-edu` | `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9` |

### P0-C — the 4B / 2% holdout arithmetic has no exact integer solution as written

The handoff names an exact 4,000,000,000-byte training stream and a disjoint 2% holdout but
does not bind the 2% denominator. If 2% means `heldout / (training + heldout)`, exact integer
bytes require:

```text
49 * heldout_bytes = 4,000,000,000
```

`4,000,000,000 mod 49 = 3`, so no integer-byte heldout satisfies both statements exactly.
The amendment must bind the denominator, whole-document rule, rounding direction, permitted
tolerance, whether 4B is pre- or post-dedup, and whether tokenizer fitting uses the 4B screen
stream or the full non-heldout corpus.

### P0-D — production algorithms and run recipe are not literal yet

The following remain unbound and can change data membership, token IDs, or optimization:

- whitespace-normalization algorithm and version;
- language-ID package/model/version, threshold, and tie behavior;
- MinHash family, component count, seed, LSH bands/rows, byte order, short-document rule,
  candidate ordering, and recall target;
- shard format, framing, compression, sizes, naming, timestamp policy, and raw-file versus
  semantic-manifest hash scope;
- tokenizer library/version, exact regex engine/text/flags/Unicode version, normalizer,
  decoder, BPE parameters/tie-breaking/seed, and exact ordered special-token strings/IDs;
- numerical AdamW learning rate, betas, epsilon, weight decay, schedule, warmup units,
  precision, batch/accumulation, clipping, and the exact two seed identities; and
- undertrained-row threshold plus fertility, coverage, memory, throughput, latency, FLOP,
  and final selection receipt schemas.

The current Paper-2 hermetic-screen dedup code is not a drop-in implementation: it uses
NFKC plus case-folding and Unicode-character shingles. This handoff requires byte 13-grams
and prohibits that normalization path.

## 3. Implemented build-axis surface

### Execution authority without historical hash drift

`training/weft1_gtok_contract.py` now records the complete later authority chain separately
from the banked design chain. It exposes a domain-separated execution hash helper, expands
the fail-closed blocker to name the currently identified binding classes, and keeps all old
receipt classes on their old hashes.

### Choice-independent corpus schemas

`training/weft1_corpus_contract.py` adds:

- content-addressed draft source assets with exact revisions, sizes, hashes, and strict
  canonical locator syntax (future materialization still requires an authority-bound host
  allowlist and route table);
- document identity bound to the typed source asset and retained bytes;
- separately labeled normalization diagnostics so transformed bytes cannot masquerade as
  retained corpus bytes;
- reference-only byte 13-grams, exact set Jaccard, deterministic signature/LSH fixtures,
  SHA-1 candidate keys guarded by length, SHA-256, and byte equality, and enforced
  FineWeb-drop/Dolma-keep decisions;
- typed shard diagnostics with strict lowercase ASCII relative paths, logical-stream and
  file hashes, byte/record counts, serializer identity, and optional compression identity;
- typed split manifests that bind document IDs to cluster IDs and reject both document and
  cluster leakage between train and heldout; and
- typed training and heldout stream diagnostics with document-set, order, framed-payload,
  byte-count, stratum, codec, and seed identities.

The reference MinHash family and uint64 framing are quarantined fixtures, not production
choices. Their API names, schema names, docstrings, output status, and tests all say so.

### D1-D6 semantics

| Diagnostic | Implemented invariant | Why it cannot green production |
|---|---|---|
| D1 | two distinct run IDs; equal typed inputs and typed shard outputs | returns `DraftDiagnosticReceipt(authoritative=False)` |
| D2 | two distinct dedup runs; equal input/output ledgers; exact equality and exact Jaccard checks | reference algorithm remains unbound |
| D3 | reports observed bytes, target centers, and exact rational deviations; reconciles general-source accounting | tolerance semantics recorded as `UNBOUND` |
| D4 | counts invocations and rejections and requires both to be zero outside general | language-ID implementation remains unbound |
| D5 | joins a typed fixture manifest to real accented/NFD, CJK, Greek, indentation, RTL, tab, and punctuation bytes; exercises a callback; rejects empty/incomplete suites | codec is a draft spec, not the production tokenizer |
| D6 | complete 4-arm x 2-seed matrix; one split manifest; same multiset; paired order within seed; different order across seeds; fixed typed heldout; cluster disjointness | byte/holdout rules and seed identities recorded as `UNBOUND` |

`mint_authoritative_gate_receipt()` always delegates to the existing fail-closed execution
guard. There is no normal code path in the current diff that can emit an authoritative
D1-D6 PASS.

## 4. Repository hygiene

The pre-existing `.runlogs/` tree contains 189 local execution files totaling 14,932,623
bytes. Operator attestation and the before/after path inventory say it was preserved in
place and narrowly added to `.gitignore`; no log or research artifact was deleted. Exact LF
attributes were added for the four modified/new G-TOK Python files because this program has
already experienced byte-transport failures from implicit line-ending conversion.

## 5. Verification

Focused contract verification after the final implementation revision:

```text
81 passed in 1.86s
```

Command scope:

```text
python -m pytest \
  tests/test_weft1_corpus_contract.py \
  tests/test_weft1_gtok_contract.py \
  tests/test_ablation_lm_contract.py -q
```

The wider WEFT/ablation surface also passed without invoking the repository-wide
quarantine gate:

```text
17 test files
307 passed, 18 warnings in 21.79s
```

The warnings are PyTorch `torch.jit.script` deprecation warnings; there were no test
failures, skips, deselections, xfails, or errors in this scope.

Python byte-compilation and staged-diff `git diff --cached --check` also passed, including
the three new files. Ruff, Black, and mypy are not installed in this environment, so no
lint, formatter, or type-check claim is made.

The repository-wide quarantine-aware gate was not used for this receipt. Its own contract
says the three legacy failures must be reviewed before the next repository-wide engineering
receipt or on 2026-09-02, whichever comes first. This receipt therefore makes only the
scoped test claims above and does not claim a green full repository suite.

## 6. Exact amendment needed to start execution

One strategy amendment can unblock implementation without reopening the rest of the
programme. It should bind:

1. eight versus ten executed proxy blocks;
2. the complete source-family route table, approved source-host allowlist, exact revisions,
   configs/splits, asset enumeration, quality selectors, and deterministic tie-breaks;
3. heldout denominator, raw-byte/framing denominator, whole-document rounding, 4B join,
   tokenizer-fit corpus scope, and every per-stratum/per-source tolerance and pass predicate;
4. literal normalization, language-ID, dedup, shard/manifest, tokenizer, and AdamW tables;
5. exact model, data-order, tokenizer, MinHash, dropout, and permutation RNG algorithms,
   seed identities, derivation rules, and stream-reuse rules;
6. D1/D2 independent-process and rebuild semantics, including environment/code/dependency
   identities and the evidence that two runs are genuinely distinct;
7. authoritative D1-D6 receipt schemas, manifest self-hash/exclusion rules, failure branches,
   stop conditions, and the exact join from corpus identity through every run receipt;
8. undertrained-row, fertility, coverage, memory, throughput, latency, FLOP, runtime,
   precision, hardware, and final selection measurement/reporting contracts; and
9. an execution-envelope v2 whose hash covers those bindings and wraps the existing v1
   design receipts without changing their hashes.

Mark's license review remains a prerequisite to P-B freeze. Its approval and the
decontamination result must be hash-bound inside the execution envelope or joined through
separately hash-bound receipts before P-B/P-C. A decontamination hit and an arm-order seed
split remain strategy stop-lines exactly as the handoff states.

Until that amendment lands, the authorized state is:

```text
P-A code/schema build       ALLOWED AND IMPLEMENTED
P-A corpus materialization  BLOCKED
P-B tokenizer freeze        BLOCKED
P-C eight-run screen        BLOCKED
sealed batteries            UNTOUCHED
```
