# Paper One Hugging Face Converter Fix — 2026-08-16

The first private-release run stopped before packaging or upload. The exact
full-block keeper passed its source SHA-256 check, then reported 182,163,457
persisted trainable-state parameters rather than the 180,556,929 forward-active
release budget.

The difference is the already-receipted, forward-bypassed legacy concatenation
projection:

- `bridge.proj.weight`: 1,605,632 parameters, shape `[896, 1792]`
- `bridge.proj.bias`: 896 parameters, shape `[896]`
- total excluded: 1,606,528 parameters

The remaining state reconciles exactly:

- recurrent layers 6–17: 178,948,608 parameters
- active split bridge: 1,608,321 parameters
- safetensors release delta: 180,556,929 parameters

The release loader intentionally has no legacy `bridge.proj` module because its
forward path uses `prelude_proj` and `state_proj`. The converter now excludes
only manifest-listed compatibility tensors, requires both their names and their
aggregate parameter count to match the receipt, and records the excluded tensor
shapes and counts in each conversion receipt. It does not lower or bypass the
release parameter-count gate.

At discovery time no Hugging Face repository had been created or uploaded.
