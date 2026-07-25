# T1-lite-R Launch Hash Correction

**Date:** 2026-07-25  
**Classification:** infrastructure-only correction before model load or training

The first launch from implementation commit `368bd2a0` stopped during the
replication-basis preflight. No model was loaded, no optimizer was constructed,
no training step ran, and the registered seed-1 attempt was not consumed.

The locked amendment records
`4e55e946a8019d2c0c278bfaff2e76cd97b3efb7822b954b2cb74a539c037cba`
for the original T1-lite preregistration JSON. That value is the SHA-256 of the
Windows CRLF checkout. Git stores and Colab checks out the identical committed
JSON with LF endings, whose SHA-256 is
`69cc659b89f0e582641ba0d52a371723a96a8c80dafa483b11c1f4e76fc4ca09`.

The correction retains the registered hash as historical lock metadata and
adds the canonical LF hash for execution-integrity checking. Before hashing,
the launcher normalizes CRLF to LF. A regression test checks that LF and CRLF
copies have the same canonical hash and that the committed original lock
matches the canonical value.

No experimental factor changed: seed, endpoint policy, curriculum, optimizer,
loss, datasets, gates, evaluation policy, and stage-manifest requirements are
identical to lock commit `ae2793ac`.
