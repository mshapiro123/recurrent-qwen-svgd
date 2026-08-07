# Option B A100-40GB Execution Amendment

Date: 2026-08-06. Status: implementation-only amendment after the teacher-pass
lock and before any teacher forward.

The available runtime is an A100-SXM4-40GB. A pinned Qwen2.5-32B bf16 model
cannot be fully resident because its weights alone exceed that memory. The
teacher/cache pass therefore adds one explicit hardware mode:

- the 0.5B, 7B, and 14B routes remain fully resident on CUDA;
- the pinned 32B bf16 route uses Accelerate big-model dispatch with CPU and, if
  needed, local-scratch backing;
- offloaded modules execute on the main CUDA device through Accelerate hooks;
- model IDs, revisions, bf16 dtype, top-K, cascade admission, state coverage,
  audit rows, and cache formats are unchanged;
- the receipt records the hardware mode and complete `hf_device_map`;
- no quantization, optimizer, training, or threshold change is authorized.

This amendment changes runtime and likely wall clock only. It does not change
the scientific intervention or permit a silent lower-capacity teacher. The
40GB mode requires at least 200GiB total local disk with 150GiB free at launch;
the 80GB mode retains the original 300GiB-total, 250GiB-free profile. The lower
40GB threshold is permitted because its 83GiB system-memory budget keeps the
offloaded 32B weights in GPU plus CPU memory, while local disk holds streaming
corpus data, model downloads, staging shards, and a reserve. The post-pilot
projection still checks measured scratch and Drive headroom with the locked
25-percent reserve before the full pass. The 32B cascade is expected to be
materially slower than the 80GB fully resident route; the bounded pilot remains
the source of the runtime projection.

Given a 79.99-compute-unit balance at an observed 5.3 units per hour, the first
40GB launch is preflight-only. It publishes the measured target/floor runtime
and storage receipt and exits before the full cache begins. Starting the full
cache requires review of that receipt; this prevents an automatically launched
multi-day pass from consuming a roughly 15-hour current credit budget.
