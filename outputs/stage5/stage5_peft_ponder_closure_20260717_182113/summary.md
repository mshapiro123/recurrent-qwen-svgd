# PEFT + Ponder Closure - stage5_peft_ponder_closure_20260717_182113

- Status: `finished_p2_failed`
- Next action: `strategy_review`
- Historical repaired-loop PEFT arm found: `False`

## P1

| Arm | Rank | Steps | Gate | Depth counts | Base hash |
|---|---:|---:|---|---|---|
| R16 | 16 | 6000 | True | {'1': {'correct': 64, 'total': 64, 'accuracy': 1.0}, '2': {'correct': 64, 'total': 64, 'accuracy': 1.0}, '3': {'correct': 60, 'total': 64, 'accuracy': 0.9375}, '4': {'correct': 53, 'total': 64, 'accuracy': 0.828125}} | True |

## P2

- Gate: `{'passed': False, 'loss_decreased': True, 'kl_stable': False, 'loss_first': 0.20813658833503723, 'loss_last': 0.06218121200799942, 'kl_tail': [0.6788765788078308, 0.3531549870967865, 0.6473495960235596, 0.674172043800354, 0.35868924856185913, 0.2971128821372986, 0.7494633793830872, 0.7412436008453369, 0.3336111307144165, 0.7278441190719604, 0.3101021647453308, 0.055848367512226105, 0.06015992909669876, 0.6631739735603333, 0.294019877910614, 0.28833305835723877, 0.2901185154914856, 0.04894305765628815, 0.3044099807739258, 0.039312802255153656], 'mean_expected_loops': 2.2547843595966697, 'depth_nontrivial': True, 'learned_depth_accuracy': 0.74609375, 'forced_depth_accuracy': 0.94140625, 'accuracy_gap': 0.1953125, 'accuracy_preserved': False}`
