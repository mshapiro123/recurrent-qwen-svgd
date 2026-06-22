# Reasoning Trace Dataset Plan

This note keeps Hugging Face trace discovery separate from A100 training. The
working rule is simple: no dataset becomes training data just because it is
large, new, or interesting. It first has to pass a CPU audit, convert cleanly to
the recurrent training format, and serve the current gate.

## Current Training Gate

The active blocker is deterministic recurrent competence recovery. The current
selected recurrent checkpoint trails base Qwen 0.5B on balanced ARC:

| Benchmark | Base Qwen | Recurrent | Delta |
|---|---:|---:|---:|
| ARC-Easy | `421/570` | `415/570` | `-6` |
| ARC-Challenge | `167/299` | `164/299` | `-3` |
| Combined | `588/869` | `579/869` | `-9` |

That means the near-term dataset question is not "what is the richest trace
corpus?" It is "what data closes the answer-calibration and competence gap
without damaging ARC-style multiple choice behavior?"

## Recommended Order

1. **Train-now / bounded pilot:** `lordx64/reasoning-distill-opus-4-7-max-sft`
   after length filtering, mixed with benchmark-style ARC rows and response
   distillation.
2. **Audit-next:** raw Opus 4.7 and TraceInversion Opus sources. These are close
   enough to the current reasoning objective to plausibly improve recovery.
3. **Hold for filters:** Fable, Mythos, and agent/tool traces. These are valuable
   for the later particle/trajectory story, but they are not a clean fix for the
   current ARC regression.
4. **Large mining only after filters:** GLM/Kimi/combined million-row corpora.
   These are candidate reservoirs, not direct 0.5B recovery data.

## Generated Wide/Deep Curriculum Lane

Hugging Face trace corpora are only one source. The separate strong-model API
lane is documented in
[`CURRICULUM_DATA_GENERATION_PIPELINE.md`](CURRICULUM_DATA_GENERATION_PIPELINE.md).
That lane uses diverse non-student models to generate and judge candidate
problems, but labels every trace from independent answer verification,
method-constrained naturalness checks, depth decomposition, weak-reference
pass rate, and adversarial perturbation sorting.

The two lanes should not be confused:

- HF trace datasets are audited reservoirs.
- Generated curriculum records are typed supervision with measured
  `direct`, `deep_narrow`, `wide`, and `both` modes.
- Positive recurrent SFT consumes only `positive_*` roles from verified
  curriculum records.
- Rationalizations, wrong traces, and detector traces stay in verifier or
  selector training.

## Registry

The authoritative local registry is
[`config/reasoning_dataset_registry.yaml`](../config/reasoning_dataset_registry.yaml).
It now includes:

| Key | Dataset | Status |
|---|---|---|
| `opus47_sft` | `lordx64/reasoning-distill-opus-4-7-max-sft` | immediate small-train mix |
| `opus47_raw` | `lordx64/reasoning-distill-claude-opus-4-7-max` | audit/filter |
| `jackrong_opus47_trace_inversion` | `Jackrong/Claude-opus-4.7-TraceInversion-5000x` | immediate audit candidate |
| `jackrong_opus46_trace_inversion` | `Jackrong/Claude-opus-4.6-TraceInversion-9000x` | audit |
| `gryphe_opus46_reasoning_24k` | `Gryphe/Opus-4.6-Reasoning-24k` | audit |
| `angrygiraffe_opus46_47_reasoning_87k` | `angrygiraffe/claude-opus-4.6-4.7-reasoning-8.7k` | audit |
| `fable5_pi_agent` | `Glint-Research/Fable-5-traces`, `pi_agent` | hold for agent/tool filter |
| `fable5_flat` | `Glint-Research/Fable-5-traces`, `fable5_cot_merged.jsonl` | hold for filter |
| `fable5_agentic_sft` | `lordx64/agentic-distill-fable-5-sft` | later audit |
| `fable5_complete_2m` | `Glint-Research/Complete-FABLE.5-traces-2M` | streaming audit only |
| `withinus_claude_mythos_25k` | `WithinUsAI/claude_mythos_distilled_25k` | later audit |
| `jackrong_glm51_reasoning_1m` | `Jackrong/GLM-5.1-Reasoning-1M-Cleaned` | later audit |
| `jackrong_kimi25_reasoning_1m` | `Jackrong/Kimi-K2.5-Reasoning-1M-Cleaned` | later audit |
| `avtrkrb_combined_reasoning_1m` | mixed Opus/Kimi/GLM corpus | later mining only |

## CPU Audit Command

Run this on CPU or a free Colab runtime, not on A100:

```bash
STAGE5_DATASET_AUDIT_LIMIT=250 \
STAGE5_DATASET_AUDIT_PUSH=0 \
python colab/run_stage5_reasoning_dataset_audit.py
```

The default audit intentionally covers the core sources:

```text
opus47_sft, opus47_raw, fable5_pi_agent, fable5_flat,
jackrong_opus47_trace_inversion
```

For a broader but still CPU-only audit:

```bash
STAGE5_DATASET_AUDIT_LIMIT=100 \
STAGE5_DATASET_AUDIT_PUSH=0 \
STAGE5_DATASET_AUDIT_KEYS=opus47_sft,opus47_raw,jackrong_opus47_trace_inversion,jackrong_opus46_trace_inversion,gryphe_opus46_reasoning_24k,angrygiraffe_opus46_47_reasoning_87k,fable5_agentic_sft \
python colab/run_stage5_reasoning_dataset_audit.py
```

Only promote a source when the audit reports `promote_to_small_train_mix`.
Fable-style sources that convert successfully are still held unless a
task-specific filter removes tool/session artifacts and the experiment is
explicitly about agent or coding trajectory diversity.

## Why Fable Is Not Next

`Glint-Research/Fable-5-traces` is important for the program, but it is not the
right first answer to the current regression. It is tagged and structured as
agent traces, with Pi-agent sessions, tool/coding traces, and a flat
`fable5_cot_merged.jsonl` projection. That is aligned with the later thesis:
multiple latent trajectories and selector-convertible diversity. It is less
aligned with the immediate need to preserve multiple-choice answer calibration.

So the project should use Fable in this order:

1. CPU audit and sample inspection.
2. Build filters for ordinary text answer rows, coding rows, and tool/session
   rows.
3. Use filtered ordinary rows only after deterministic recurrent recovery is
   base-competitive.
4. Use tool/coding rows in a separate trajectory-diversity experiment, measured
   against a matching coding/tool benchmark, not ARC-Easy.

## A100 Rule

Dataset discovery, registry edits, schema inspection, row conversion, and
sample review are all no-GPU tasks. A100 time starts only after a concrete
training mix has been chosen and bounded by an explicit Stage 5 gate.
