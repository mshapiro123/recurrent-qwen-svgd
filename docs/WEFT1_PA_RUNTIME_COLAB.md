# WEFT-1 P-A exact runtime on Colab

This builder creates the exact Linux runtime required by
`RuntimeExpectationV3`: CPython 3.11.9, Unicode 14.0.0, SQLite 3.45.1,
`zstandard` 0.25.0 with libzstd 1.5.7, and every distribution in the checked-in
hash-locked requirements file.

The intended execution host is the Pharma Initiatives Colab account. The build
itself is CPU compilation and does not benefit from reserving a GPU. A GPU can
be attached later for an authorized G-TOK run after the runtime receipt passes.

## Run in the in-app Colab browser

Open a fresh x86-64 Linux Colab VM, put the repository in `/content`, change to
the repository root, and run this cell:

```bash
!bash scripts/run_weft1_pa_runtime_builder_colab.sh
```

The wrapper installs the declared compiler prerequisites and then executes the
governed Python builder. Its default outputs are:

- build workspace: `/content/weft1-pa-build`
- exact runtime prefix: `/content/weft1-pa-runtime`
- canonical receipt: `/content/weft1-pa-runtime-receipt.json`

All three destinations must be absent at startup. A retry must use new explicit
paths rather than overwriting evidence from an earlier attempt:

```bash
!bash scripts/run_weft1_pa_runtime_builder_colab.sh \
  --work-root /content/weft1-pa-build-run2 \
  --prefix /content/weft1-pa-runtime-run2 \
  --receipt /content/weft1-pa-runtime-receipt-run2.json
```

The successful interpreter is `/content/weft1-pa-runtime/bin/python3.11`.
Keep the runtime prefix and its receipt together. The temporary build workspace
is not a runtime dependency after a successful build.

## Register durable Drive storage before materialization

Corpus materialization may not use Colab's ephemeral `/content` filesystem as
its output backing. After Drive is mounted, create one fresh project directory
and register its observed `fuse.drive` identity:

```bash
mkdir -p /content/drive/MyDrive/WEFT1
/content/weft1-pa-runtime/bin/python3.11 -I -B \
  scripts/run_weft1_corpus_pa.py register-durable-storage \
  --durable-mount-root /content/drive \
  --durable-storage-root /content/drive/MyDrive/WEFT1 \
  --marker-out /content/drive/MyDrive/WEFT1/durable-storage-marker-v3.json \
  --receipt-out /content/drive/MyDrive/WEFT1/durable-storage-registration-v3.json
```

Registration fails closed unless the storage root is on the exact observed
`fuse.drive*` mount. Production re-observes that mount before each worker and
before minting its parent receipt. Shards, manifests, child receipts, and the
parent receipt are written directly to a registered durable output directory
and physically re-read there. Local `/content` storage is reserved for
rebuildable SQLite, spool, code-snapshot, and temporary files.

Copy the runtime build receipt to durable storage as soon as registration is
available and verify its SHA-256 after the copy. The runtime prefix is
reconstructible from its pins; corpus shards and manifests are not treated as
durable until their post-write hashes have been verified on Drive.

## What is pinned and proven

- CPython comes from the official `Python-3.11.9.tar.xz` archive and is checked
  for exact byte count and SHA-256 before extraction.
- SQLite comes from the official 3.45.1 amalgamation archive. The archive is
  checked by byte count, SHA-256, and SHA3-256; extracted `sqlite3.c` must match
  SQLite's published SHA3-256
  `0474604df9e1b69a5544295dd046aad954749279780d557da80f44b958100295`.
- SQLite is built as a shared library inside the runtime prefix before CPython.
  The final probe requires both `sqlite3` and `_sqlite3` to import, checks the
  exact SQLite source ID, parses the exact `ldd` SONAME rows, and resolves them
  to the governed versioned SQLite and libpython files. It repeats startup with
  `LD_LIBRARY_PATH` absent and requires the installed executable and `_sqlite3`
  extension to carry the governed library directory in RUNPATH/RPATH.
- CPython 3.11.9 installs one non-wheel file in `site-packages`: a 119-byte
  `README.txt`. The builder requires its pinned source hash, removes it, and
  directory-fsyncs that removal before the complete RECORD-owned tree audit.
- The requirements lock is copied through one stable file handle into the
  fresh build workspace, verified against its authority SHA-256, and used for
  both wheel resolution and offline installation with `--require-hashes`.
- The receipt includes every selected wheel's exact filename, byte count, and
  SHA-256; source rows; lock and recipe identities; builder and runtime-contract
  hashes; compiler/package provenance; runtime artifacts; the full repository
  runtime attestation; and a canonical identity over all evidence.
- Installation uses isolated Python startup, a fresh offline `pip --report`, and
  exact name/version/archive-SHA coverage against the selected wheels. The
  complete installed tree is hashed from RECORD and rechecked before each
  production worker and final receipt. The explicit threat model trusts the
  CPython-pinned bootstrap pip; coherent malicious file-and-RECORD rewriting
  before the first receipt is outside scope.

The OS build packages are not claimed to be timeless archive pins. Their exact
installed versions and the resolved build-tool hashes are recorded in each
receipt, while the governed runtime inputs and Python wheels are content-pinned.

## Fail-closed behavior

No `PASS` receipt is written unless every source, build, linkage, dependency,
and repository attestation check succeeds. A failure emits a canonical JSON
failure record on stderr and exits nonzero. The builder does not mint a corpus
gate, consume sealed data, train a model, or authorize downstream execution.

For a non-authoritative audit of the exact recipe without network, compilation,
or filesystem writes, run:

```bash
!python scripts/build_weft1_pa_runtime.py --dry-run
```
