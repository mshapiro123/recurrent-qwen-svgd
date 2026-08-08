# Phase-2 Option B Endpoint Reserialization Erratum

Date: 2026-08-07

Status: locked before the first Option B optimizer update

## Scope

The Option B launcher initially refused to stage `seed_0_full_a2` because the
source checkpoint bytes no longer matched the hashes transcribed from the
GitHub A2 receipt. The refusal occurred before cache construction and before
any Option B optimizer update.

The four current Drive checkpoints were audited against the canonical Drive A2
receipt. For every arm:

- checkpoint kind, seed, arm, step 2,000, target step 2,000, and null abort
  status match;
- the saved history ends at step 2,000;
- exactly 2,000 optimizer-update row hashes are present;
- the executed row-schedule SHA-256 is
  `a2718f46a22ff47a91f14fac2bb1fb38719fa29c4edb9663cab5143f139524c6`;
- the checkpoint byte hash equals the hash recorded in the canonical Drive
  `receipts/summary.json`.

The mismatch was systematic across all four arms. A completed A2 rerun loaded
each endpoint, executed no additional optimizer step, and nevertheless called
`torch.save` at function exit. PyTorch reserialization changed archive bytes.
The GitHub receipt retained the earlier hashes while the canonical Drive
receipt and checkpoint files retained the later hashes.

This is a provenance and idempotence defect, not an endpoint substitution and
not a scientific amendment. Option B remains authorized under the original
protocol. The source byte hashes are replaced by the canonical Drive hashes
below, and an independent digest of every trainable tensor is added as a
second startup assertion.

## Canonical endpoints

| Arm | Canonical checkpoint SHA-256 | Trainable-state digest |
|---|---|---|
| seed 0, full A2 | `5ebc1ec1f2299344b24fb055799c5e35a8236982a4840f2013418fd7513a6373` | `21616182b5f33036f4120ad192c7c56ceda2877d4827ad29d6a934f5d781f02c` |
| seed 0, draft-only control | `69f0b3970dd1de174d728ce062ceba242a55d9ae9c670e4c5dd0d27ad9249b1a` | `a1a7264eab4bc23ebc981233231bc822539b83f2fdba7f4da5cb92f199a55afe` |
| seed 1, full A2 | `5960ef967f3834db0c83eef26a2d9c896e43cc4f07f6a6d6047700dbcf5d4e76` | `ef9750b259fce4fb6ae5fa095f5621235f06dff99d870c53f2ed6aef4c146ef8` |
| seed 1, draft-only control | `691e102c0dd258f543e55aee291ac9d05675a9a8e8e8b170f47015e4782d1760` | `6f21e853ddfae61031bde1a47d97db059eb88dc1434c209fc75262a95f9278c2` |

## Implementation correction

1. Option B asserts both the canonical checkpoint byte hash and the
   trainable-state digest before constructing an optimizer.
2. A2 no-op resumes no longer rewrite a completed checkpoint when no state or
   optimizer update occurred.
3. The Colab wrapper prints the durable failure receipt when a nested launcher
   exits unexpectedly.

No result is reinterpreted. No source state is reconstructed, converted, or
modified. No Option B constant changes.
