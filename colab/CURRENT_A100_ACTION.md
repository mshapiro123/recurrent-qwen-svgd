# Current A100 Action

## Run This Notebook

Open the current single-purpose Colab notebook:

[09_stage5_arc_mix_recovery_once.ipynb](https://colab.research.google.com/github/mshapiro123/recurrent-qwen-svgd/blob/main/colab/09_stage5_arc_mix_recovery_once.ipynb)

If Colab shows `Colab is waiting for authorization from GitHub`, a blocked
popup, or any GitHub-private-repo authorization problem, **do not connect an
A100 runtime**. Use the fallback below from any normal Drive or blank Colab
notebook instead.

Use an A100 runtime only when you intentionally want to spend one bounded proxy
run. The notebook should:

1. pull the latest `main` from GitHub;
2. authenticate GitHub and Hugging Face from Colab secrets;
3. mount Google Drive;
4. print the A100 go/no-go report;
5. install dependencies only after go/no-go allows the proxy;
6. run `colab/run_stage5_arc_mix_recovery_once.py`;
7. print the planner's next action from the new ARC-mix `summary.json`;
8. push safe text artifacts through the delegated runner;
9. disconnect the runtime when complete or when setup fails.

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

The notebook output should now include a final ARC-mix result review after the
proxy runner finishes. If that section says anything other than
`Next A100 spend: YES: run exactly one full balanced ARC confirmation.`, keep
the A100 shut down and do the next repair locally.

## Fallback When GitHub-Colab Auth Blocks The Notebook

This fallback avoids opening a private GitHub notebook through Colab. It still
uses GitHub, but only from inside a normal Colab Python cell via the
`GH_TOKEN`/`GITHUB_TOKEN` Colab secret.

1. Open any trusted Drive-backed Colab notebook or a blank notebook.
2. Keep the runtime disconnected while editing the cell.
3. Copy the single cell from the plain Python mirror
   [`colab/STAGE5_ARC_MIX_RECOVERY_CELL.py`](STAGE5_ARC_MIX_RECOVERY_CELL.py),
   or from the fenced code block in
   [`colab/STAGE5_ARC_MIX_RECOVERY_CELL.md`](STAGE5_ARC_MIX_RECOVERY_CELL.md).
4. Select an A100 runtime only immediately before running that single cell.
5. Run no other cells in that runtime.

The fallback cell performs the same no-waste sequence: clone or update the repo,
mount Drive, run the A100 go/no-go check before installing dependencies, run
exactly one `arc_mix_response_w01_lr2e6` proxy when allowed, print the
post-proxy review, push safe text artifacts, and disconnect on failure or
completion.

For a clean local review after the notebook pushes artifacts, run:

```bash
python colab/review_stage5_arc_mix_result.py \
  --summary outputs/stage5/<arc_mix_run_id>/summary.json
```

It prints whether another paid full balanced assessment is justified. The only
acceptable "continue spending" answer is:

```text
Next A100 spend: YES: run exactly one full balanced ARC confirmation.
```
