# Phase-2 Matched Alpha Pilot Resource And Recovery Note

Date: 2026-08-04

- Runtime: NVIDIA A100 40 GB or larger; CUDA required. The original 80 GB
  estimate was amended before training after the cached-state sparse-logit
  implementation was measured structurally. See
  `PAPER2_PHASE2_MATCHED_ALPHA_RUNTIME_AMENDMENT_20260804.md`. The runner
  requires at least 35 GiB visible VRAM and rejects smaller or CPU-only
  runtimes before restoring private tensors.
- Expected peak: well below 35 GiB. This launcher never loads a transformer
  backbone and scores only cached sparse candidate sets. Cached teacher states
  remain on local scratch/CPU and are staged to GPU by batch.
- Expected wall time: approximately 4-8 hours for six 1,000-step arms plus
  evaluations. The registered one-time 2,000-step extension can add a similar
  amount. This is an estimate, not a gate.
- Local scratch: use `/content/local-scratch` when present, otherwise
  `/content`. Large immutable Drive artifacts are copied once and hash-checked.
- Durable storage: per-arm checkpoints, status JSON, and private row metrics
  write to the Pharma Initiatives Drive artifact folder. Public summaries and
  receipt tables alone are committed to GitHub.
- Resume: each arm persists model, optimizer, step, RNG states, batch-stream
  state, and evaluation history every 100 steps. Resume verifies the protocol
  lock, frozen hashes, alpha, seed, and non-alpha initialization hash before
  continuing. A mismatch aborts; no automatic restart is allowed.
- Publication: the launcher may commit completed arm receipts incrementally,
  but the cross-arm decision is emitted only after all six arms have terminal
  receipts or a registered abort.
- Runtime loss: relaunching the same target resumes completed arms and the most
  recent valid checkpoint. It never repeats completed optimizer steps.
