# MCQ Regression Diagnosis - stage5_routing_repair_answer_prior_drift_20260622_1403

- Candidate: `routing_repair_phase1_step_50`
- Benchmarks: `arc_easy`

| Benchmark | Base | Candidate | Delta | W/L/Tie-correct/Tie-wrong | Mean margin delta |
|---|---:|---:|---:|---|---:|
| arc_easy | 87/128 | 84/128 | -3 | 14/17/70/27 | -1.5560 |

## Feature Buckets

### arc_easy

| Feature | n yes | delta yes | n no | delta no |
|---|---:|---:|---:|---:|
| `asks_best` | 35 | +1 | 93 | -4 |
| `asks_which` | 82 | -1 | 46 | -2 |
| `asks_why` | 4 | +0 | 124 | -3 |
| `diagram_like` | 4 | -2 | 124 | -1 |
| `has_math_word` | 14 | -1 | 114 | -2 |
| `has_negation` | 2 | +1 | 126 | -4 |
| `has_number` | 8 | +0 | 120 | -3 |

## Routing Buckets

### arc_easy

| Bucket | n | delta | wins | losses | mean margin delta | mean loops |
|---|---:|---:|---:|---:|---:|---:|
| `ambiguous_proxy` | 35 | +8 | 10 | 2 | 0.6196 | 3.0530 |
| `base_confident_direct_proxy` | 76 | -12 | 0 | 12 | -2.9321 | 3.0557 |
| `conceptual_reasoning_proxy` | 10 | +1 | 2 | 1 | 0.1007 | 3.0682 |
| `deep_numeric_proxy` | 7 | +0 | 2 | 2 | 0.1399 | 3.0592 |

## Answer Prior Drift

### arc_easy

- Max abs candidate-base prediction count delta: `23`
- Max abs candidate-answer prediction count delta: `29`
- Changed predictions: `38/128`

| Label | Base count | Candidate count | Answer count | Candidate-base | Candidate-answer |
|---|---:|---:|---:|---:|---:|
| `A` | 36 | 59 | 30 | +23 | +29 |
| `B` | 33 | 17 | 27 | -16 | -10 |
| `C` | 42 | 22 | 38 | -20 | -16 |
| `D` | 17 | 30 | 33 | +13 | -3 |

Top prediction transitions:
- `A->A`: 36
- `C->C`: 22
- `B->B`: 17
- `D->D`: 15
- `C->A`: 12
- `B->A`: 9
- `C->D`: 8
- `B->D`: 7

## Largest Regressions

### arc_easy
- `Mercury_SC_LBS10170` answer `B`, base `B`, recurrent `D`, margin delta -5.1129: An anemometer is a tool that measures
- `MDSA_2010_4_7` answer `C`, base `C`, recurrent `A`, margin delta -2.4223: Weather patterns sometimes result in drought. Which activity would be most negatively affected during a drought year?
- `Mercury_7013073` answer `B`, base `B`, recurrent `A`, margin delta -2.3899: Which step of the scientific method will follow after a student graphs collected data during a lab experiment?
- `NCEOGA_2013_5_51` answer `B`, base `B`, recurrent `A`, margin delta -2.2134: A scientist is trying to decide whether an organism is unicellular or multicellular. Which information would help the scientist most to make her decision?
- `Mercury_7008435` answer `C`, base `C`, recurrent `A`, margin delta -2.2078: What contributes the most to beach erosion?
- `Mercury_SC_401652` answer `C`, base `C`, recurrent `A`, margin delta -2.1960: Before large trees could grow on Earth, what had to happen first?
- `VASoL_2007_3_33` answer `C`, base `C`, recurrent `A`, margin delta -2.1588: Which of these causes the MOST evaporation of water from a lake?
- `Mercury_SC_400840` answer `C`, base `C`, recurrent `D`, margin delta -1.8183: Which tools are needed to measure the length and mass of a seashell?

## Largest Wins

### arc_easy
- `Mercury_SC_402067` answer `D`, base `B`, recurrent `D`, margin delta 2.3369: Which method is the best safety procedure when working around open flames?
- `Mercury_7007630` answer `A`, base `C`, recurrent `A`, margin delta 2.2353: Where are electrons in an atom located?
- `Mercury_SC_400175` answer `D`, base `C`, recurrent `D`, margin delta 1.4152: After a rainfall, which process in the water cycle draws the water back up into the air?
- `TIMSS_2007_4_pg110` answer `A`, base `B`, recurrent `A`, margin delta 1.3063: Scientists believe that oceans once covered much of what is now land. which of these things found on land led scientists to believe this?
- `Mercury_7004165` answer `D`, base `B`, recurrent `D`, margin delta 1.1526: Which is most responsible for reflecting incoming solar heat back to space?
- `Mercury_7091928` answer `A`, base `B`, recurrent `A`, margin delta 1.0490: All nations need to import and export goods for their economic survival. As a result, many island nations have developed advanced technology for transporting goods by
- `Mercury_7112788` answer `D`, base `C`, recurrent `D`, margin delta 1.0005: A student conducts an experiment to determine the average size of acorn that squirrels eat. The student gave several different sizes of acorns to a squirrel. Which action would most likely improve the results?
- `CSZ20680` answer `A`, base `B`, recurrent `A`, margin delta 0.8896: An object composed mainly of ice is orbiting the Sun in an elliptical path. This object is most likely
