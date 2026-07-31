# DC1 Stage A EVAL-C Freeze Reconciliation

Date: 2026-07-30

The governing Stage A preregistration was verified against Drive file
`1o-RtPRHS5F7aMsKHmBahVc1jTiRDlkdH`: 14,333 bytes, SHA-256
`bd834c42d92b559dabd638c326dd76724f24adba6ade27bcdd4adb32703dc581`.

During coding-lane reconciliation, the two EVAL-C hashes named as existing in
Appendix A could not be found in the repository or Drive receipts. Existing
DC1 receipts instead uniformly state that EVAL-C remained untouched. The
machine-readable lock therefore cannot be completed truthfully until EVAL-C
is frozen and its sole teacher cache is built.

The added `paper2_dc1_eval_c_freeze` target is pre-lock infrastructure only.
It performs no scoring, training, optimization, or model mutation. It:

1. freezes exactly 200,000 tokens at the registered 50/50 source mix;
2. excludes all D0, EVAL-B, and DEV-C document IDs;
3. performs the sole pinned Qwen2.5-7B teacher-cache pass;
4. stores EVAL-C rows, document IDs, and teacher outputs only in the private
   Drive artifact tree; and
5. publishes only data, manifest, and teacher-cache hashes plus provenance and
   explicit `read_once_scoring_spent=false` status.

No Stage A training launcher is created by this reconciliation. After the
hash-only receipt lands, the coding lane will transcribe the two hashes, fill
the bridge allowlist and verdict-script name, commit `stage_a_prereg.json` with
`locked_before_training=true`, and only then create the registered runner.
