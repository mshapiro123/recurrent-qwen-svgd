# Rescue Predictability - stage5_forced_depth_arc_challenge_loop123_20260625_194738

- Source: `outputs/stage5/stage5_forced_depth_arc_challenge_loop123_20260625_194738/summary.json`
- Score target: `content_question_only`
- Aggregate: `mean`
- Loops: `[1, 2, 3]`

This is the cheap precursor: it asks whether deeper-loop rescue is predictable enough to justify a selector before spending GPU on more recurrence or particle work.

## arc_challenge

- Loop-1 correct: `89/256`
- Oracle correct over loop-1/deeper: `117/256`
- Oracle gain vs loop-1: `28`
- Categories: `{'stable_wrong': 139, 'stable_correct': 58, 'rescuable': 28, 'harmable': 31}`

### Rescue Predictability

- `base_predicted_margin`: oriented AUC `0.7191416040100251`, raw AUC `0.2808583959899749`, direction `low_predicts_positive`, Fisher `0.32166033962291823`
- `loop1_predicted_margin`: oriented AUC `0.6953320802005012`, raw AUC `0.30466791979949875`, direction `low_predicts_positive`, Fisher `0.2509543406533825`
- `loop1_score_entropy`: oriented AUC `0.6608709273182958`, raw AUC `0.6608709273182958`, direction `high_predicts_positive`, Fisher `0.2232989352157145`
- `loop1_answer_halt_entropy_gold`: oriented AUC `0.5673558897243107`, raw AUC `0.5673558897243107`, direction `high_predicts_positive`, Fisher `0.042829759339725236`
- `loop1_answer_expected_loops_gold`: oriented AUC `0.5651629072681704`, raw AUC `0.5651629072681704`, direction `high_predicts_positive`, Fisher `0.006722044183930782`
- `loop1_mean_halt_entropy`: oriented AUC `0.5646929824561403`, raw AUC `0.5646929824561403`, direction `high_predicts_positive`, Fisher `0.03450356546676855`
- `loop1_mean_expected_loops`: oriented AUC `0.5595238095238095`, raw AUC `0.5595238095238095`, direction `high_predicts_positive`, Fisher `0.004639836714605114`
- `loop1_prediction_halt_entropy`: oriented AUC `0.5587406015037594`, raw AUC `0.5587406015037594`, direction `high_predicts_positive`, Fisher `0.030171201267021577`

### Conservative Binary Gates

- `base_predicted_margin` low threshold `0.19658637046813965` -> loop `3`: correct `97/256`, delta `8`, gap capture `0.3333333333333333`, routed `65`, W/L `14/6`, rescued/harmed `14/6`
- `base_predicted_margin` low threshold `0.3647139072418213` -> loop `3`: correct `97/256`, delta `8`, gap capture `0.3333333333333333`, routed `103`, W/L `17/9`, rescued/harmed `17/9`
- `base_predicted_margin` low threshold `0.4698638916015625` -> loop `3`: correct `96/256`, delta `7`, gap capture `0.2916666666666667`, routed `129`, W/L `20/13`, rescued/harmed `20/13`
- `loop1_predicted_margin` low threshold `0.5569872856140137` -> loop `3`: correct `96/256`, delta `7`, gap capture `0.2916666666666667`, routed `129`, W/L `20/13`, rescued/harmed `20/13`
- `base_predicted_margin` low threshold `0.2922482490539551` -> loop `3`: correct `95/256`, delta `6`, gap capture `0.25`, routed `85`, W/L `14/8`, rescued/harmed `14/8`
- `loop1_mean_expected_loops` high threshold `1.0929814279079437` -> loop `3`: correct `95/256`, delta `6`, gap capture `0.25`, routed `172`, W/L `19/13`, rescued/harmed `19/13`
- `loop1_mean_halt_entropy` high threshold `0.30811385810375214` -> loop `3`: correct `95/256`, delta `6`, gap capture `0.25`, routed `172`, W/L `19/13`, rescued/harmed `19/13`
- `base_predicted_margin` low threshold `0.5840404033660889` -> loop `3`: correct `95/256`, delta `6`, gap capture `0.25`, routed `154`, W/L `21/15`, rescued/harmed `21/15`
- `loop1_predicted_margin` low threshold `0.6948881149291992` -> loop `3`: correct `95/256`, delta `6`, gap capture `0.25`, routed `154`, W/L `22/16`, rescued/harmed `22/16`
- `loop1_prediction_expected_loops` high threshold `1.0599623918533325` -> loop `3`: correct `95/256`, delta `6`, gap capture `0.25`, routed `192`, W/L `22/16`, rescued/harmed `22/16`

### Harm Predictability

- `loop1_answer_margin_gold`: oriented AUC `0.8544802867383513`, direction `high_predicts_positive`, Fisher `0.6023698737461355`
- `loop1_prediction_expected_loops`: oriented AUC `0.5898207885304659`, direction `low_predicts_positive`, Fisher `0.05004328730702194`
- `loop1_margin_minus_base_margin`: oriented AUC `0.5863799283154122`, direction `high_predicts_positive`, Fisher `0.022323184788844437`
- `loop1_mean_expected_loops`: oriented AUC `0.5848028673835126`, direction `low_predicts_positive`, Fisher `0.0498541433337837`
- `loop1_answer_expected_loops_gold`: oriented AUC `0.5769175627240144`, direction `low_predicts_positive`, Fisher `0.04406826468181094`
