# Phase 2 Option B Teacher-Summary Transport Erratum

Date: 2026-08-07

## Scope

This pre-training erratum corrects a platform-sensitive byte hash for the
public Option B teacher-cache summary. It changes no parsed JSON value, model,
data population, checkpoint, optimizer, schedule, gate, threshold, or analysis.
No Option B optimizer update occurred before this erratum.

## Diagnosis

The post-generation amendment recorded SHA-256
`04bf069d50753475c03dd817424eae657d27a43458b61c9178249b6f1b7d45e0`
from a Windows checkout containing CRLF line endings. That representation is
6,358 bytes. Git stores and checks out the same tracked JSON on Colab with LF
line endings. The Git-LF representation is 6,216 bytes and has SHA-256
`038bf9d1cda762b5107e7f6e45353a38aec5f3af8cbdcf3d1521081af49fee51`.
The 142-byte difference is exactly one carriage return for each of 142
newlines. Parsing either representation produces the same JSON object.

## Executable Integrity Contract

The original CRLF hash remains recorded as provenance. Runtime authorization
now requires both of the following:

1. SHA-256 after normalizing CRLF to Git-LF equals
   `038bf9d1cda762b5107e7f6e45353a38aec5f3af8cbdcf3d1521081af49fee51`.
2. SHA-256 of canonical JSON encoded as UTF-8 with sorted keys, ASCII escaping,
   and compact separators equals
   `e650f06f266d6b7d61e4a9ac1a67cb5625d2a05843ea3c5d6c4ae04eddc4780b`.

The existing field-level population, partition, cache-ledger, audit-set, and
exclusion checks remain mandatory. Thus line-ending transport is tolerated,
while any semantic mutation still fails before cache construction or training.

## Authorization State

The failed launch had already verified all four source endpoints and staged the
teacher cache, but stopped before derived-cache construction and before any
optimizer update. Option B training remains authorized under the original
locked protocol plus the endpoint reserialization erratum and this transport
erratum.
