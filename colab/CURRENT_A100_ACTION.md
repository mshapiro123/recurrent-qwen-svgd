# Current A100 Action

## Run This Notebook

Open the current single-purpose Colab notebook:

[09_stage5_arc_mix_recovery_once.ipynb](https://colab.research.google.com/github/mshapiro123/recurrent-qwen-svgd/blob/main/colab/09_stage5_arc_mix_recovery_once.ipynb)

Use an A100 runtime only when you intentionally want to spend one bounded proxy
run. The notebook should:

1. pull the latest `main` from GitHub;
2. authenticate GitHub and Hugging Face from Colab secrets;
3. mount Google Drive;
4. print the A100 go/no-go report;
5. run `colab/run_stage5_arc_mix_recovery_once.py`;
6. push safe text artifacts through the delegated runner;
7. disconnect the runtime when complete.

## Source Summary

```text
outputs/stage5/stage5_full_assessment_once_20260622_005522/summary.json
```

Current source result:

```text
status = needs_competence_recovery
ARC-Easy:       recurrent 415/570, base 421/570, delta -6
ARC-Challenge:  recurrent 164/299, base 167/299, delta -3
Combined:       recurrent 579/869, base 588/869, delta -9
```

## Experiment

Run exactly one ARC-mix recovery proxy:

```text
STAGE5_ARC_MIX_ARMS=arc_mix_response_w01_lr2e6
STAGE5_ARC_MIX_ARC_CHALLENGE_REPEAT=2
STAGE5_ARC_MIX_ARC_EASY_REPEAT=4
STAGE5_ARC_MIX_ARC_EVAL_LIMIT=128
STAGE5_ARC_MIX_OPUS_LIMIT=3000
STAGE5_ARC_MIX_MIN_MARGIN_DELTA=-0.05
STAGE5_ARC_MIX_MAX_PREDICTION_SHIFT=16
```

## How To Interpret The Result

New ARC-mix summaries include a machine-readable `decision` field.

| Decision | Meaning | Next action |
|---|---|---|
| `run_full_balanced_assessment` | Proxy improved or matched base while preserving calibration. | Run exactly one full balanced ARC confirmation. |
| `stop_for_calibration_repair` | Proxy accuracy moved but answer calibration degraded. | Stop A100; revise objective/data locally. |
| `stop_and_revise_objective` | Proxy did not improve the recurrent start or close the base gap. | Stop A100; revise objective/data locally. |

Do not run GPQA, Phase 2/SVGD, dataset audits, or model scaling from this
state.
