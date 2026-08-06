# Option B Teacher-Pass Resource Note

Date: 2026-08-06. Status: **draft for strategy lock; no launch authority**.

## 1. Banked reference

The only direct capacity reference is the completed Stage 0A pass:

- 50,000 anchors and 200,000 horizon samples;
- Qwen2.5 0.5B, 7B, 14B, and cascaded 32B at pinned revisions;
- Qwen2.5-14B hidden states from layers 16, 32, and 44;
- one A100 80GB-class runtime with a 368GiB local-scratch mount;
- 375,039,270,912 scratch bytes free at startup;
- durable resume shards on Drive;
- one forward-cache pass per model and no optimizer updates.

The public Stage 0A receipt does not record enough timing detail for a defensible
wall-clock forecast. Option B must measure throughput during a bounded preflight
rather than inherit an invented ETA.

## 2. Expansion size

The authorized target is 140,000 new anchors, with 100,000 as the minimum
admissible pass. Each anchor carries four horizon samples, so the pass creates:

| New anchors | Horizon samples | Post-splice training anchors |
|---:|---:|---:|
| 100,000 floor | 400,000 | 141,969 |
| 140,000 target | 560,000 | 181,969 |

The 14B state payload alone is approximately 120KiB per anchor at four horizons,
three layers, width 5,120, and bf16. That is about 11.4GiB at the floor and
16.0GiB at the target, before lattice, audit, indices, model caches, and resume
overhead. The runner must compute actual projected bytes from its selected
manifest and refuse to start unless scratch and Drive headroom exceed the
projection plus a 25-percent reserve.

This estimate assumes 14B states for every anchor. If strategy locks
threshold-only 14B state collection, the runner recomputes the estimate from the
measured threshold rate; it may not silently change coverage to fit hardware.

## 3. Runtime class and ordering

- Teacher/cache pass: A100 80GB class, high system RAM, writable local scratch
  of at least 300GiB, with Drive used only for durable resumable shards.
- Segment training: A100 40GB or 80GB subject to a measured one-batch memory
  preflight; it does not need the 32B teacher resident.
- One A100 cannot run the teacher pass and Segment 1 concurrently. With one
  available A100, run the teacher pass first, validate and hash it, then start
  Segment 1. With two independent runtimes, they may run in parallel only after
  the protocol lock and with disjoint run directories.

The target splice remains step 4,000. Completing the cache before Segment 1
starts does not permit an early splice: the pre-splice dose window is part of
the experiment.

## 4. Required throughput preflight

Before labeling proper, run a fixed, document-isolated pilot shard through each
model route and record:

- anchors and horizon samples per second by model;
- 14B state bytes and lattice bytes per anchor;
- 32B cascade rate;
- peak GPU memory, peak system RAM, peak scratch use, and Drive write rate;
- projected target and floor runtimes with uncertainty from shard variation.

The preflight performs no training and touches no existing evaluation document
or confirmatory partition. If the projected target is infeasible, the pass may
use the preregistered 100,000-anchor floor. It may not go below that floor or
silently drop teacher-state, tail-mass, audit, or cascade requirements.

## 5. Durability and admission

Every shard is written atomically, hashed, and read back from Drive before its
local copy can be deleted. Before the splice, a hash-only amendment must record:

- landed new-anchor and horizon-sample counts;
- source-document and exclusion hashes;
- sample-manifest, position-key, audit-subset, model-cache, lattice-ledger, and
  teacher-state-ledger hashes;
- fixed new-train diagnostic-subset hash;
- zero overlap with the existing 8,031-anchor evaluation slice and all frozen
  confirmatory partitions.

This note sizes and sequences the pass. It does not authorize the pass or any
training before strategy locks the protocol.
