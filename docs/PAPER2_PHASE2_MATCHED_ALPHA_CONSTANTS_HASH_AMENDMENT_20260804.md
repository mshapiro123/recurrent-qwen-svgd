# Phase-2 Matched Alpha Constants Hash Amendment

Date: 2026-08-04

Status: `locked_before_training`

## Scope

This is a clerical hash correction only. No optimizer step had run when the
checkout-dependent mismatch was discovered. It changes no constant, arm,
seed, data row, loss, threshold, budget, stopping rule, or interpretation.

The preregistration field `constants_lf_sha256` was populated from a Windows
CRLF working-tree copy of `training/paper2_phase2_dc2_constants.json`. Its raw
digest was:

`87965cf6ab1768962fccac6d2598477832b95e1ab9128c465afa23da68d4f076`

The byte-identical Git blob uses LF line endings. After the registered LF
normalization convention, both checkouts have digest:

`4e56a43a6692a4c88e60c17cd5e12076f1a2f0c3c65b3027dfc3f0800ef558fc`

The runner must normalize CRLF and lone CR to LF before checking this one
explicitly named LF digest. All artifact and checkpoint hashes remain raw-byte
SHA-256 values. The corrected digest and normalization behavior are locked by
the commit containing this document before the first training step.
