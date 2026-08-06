# Phase-2 A2 Step-237 Hash Portability Erratum

**Date:** 2026-08-06
**Scope:** Receipt verification only. No training, data, optimizer, guardrail, resume, or verdict field changes.

## Incident

The first Colab launch of the locked step-237 continuation stopped before staging or GPU training. A preflight test compared the raw SHA-256 of two tracked JSON receipts against hashes recorded from a Windows checkout. Git had normalized those receipts from CRLF to LF in the Linux checkout, so their raw byte hashes differed even though their UTF-8 JSON content was identical.

The three governing markdown documents already had LF line endings and matched their locked byte counts and hashes on both systems. The failure was limited to:

- `outputs/stage5/stage5_paper2_phase2_a2_resume_20260805/summary.json`
- `outputs/stage5/stage5_paper2_phase2_a2_tripwire_audit_20260806/summary.json`

No optimizer step ran. The A100 remained unused after the hardware preflight.

## Corrected Contract

For these two repository-tracked text receipts, the lock uses SHA-256 over UTF-8 text after universal-newline normalization to LF. No whitespace, key order, numeric representation, or trailing-newline normalization is permitted. Governing documents and binary checkpoints continue to use raw-byte SHA-256.

The normalized receipt hashes are:

- source summary: `243d49f0a70995456f3e6a011c648fffc9c9c99514aa3a8175c31fadde6b7500`
- tripwire audit summary: `1e2c064a028511116d006274516bdd673f26e7216777676367f099c1f9f85ba6`

## Scientific Effect

None. This correction changes only how Git checkout line endings are removed from the receipt identity calculation. The exact four checkpoints, step 237, attempt-238 row hash, guardrail inventory, relative-explosion rule, datasets, and all endpoint criteria remain locked as before.
