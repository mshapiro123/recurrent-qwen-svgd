# MCQ Regression Diagnosis - stage5_full_assessment_once_20260622_005522

- Candidate: `arc_mix_response_w005_lr2e6_step50`
- Benchmarks: `arc_easy, arc_challenge`

| Benchmark | Base | Candidate | Delta | W/L/Tie-correct/Tie-wrong | Mean margin delta |
|---|---:|---:|---:|---|---:|
| arc_easy | 421/570 | 415/570 | -6 | 18/24/397/131 | -1.2373 |
| arc_challenge | 167/299 | 164/299 | -3 | 11/14/153/121 | -0.3608 |

## Feature Buckets

### arc_easy

| Feature | n yes | delta yes | n no | delta no |
|---|---:|---:|---:|---:|
| `asks_best` | 145 | +3 | 425 | -9 |
| `asks_which` | 329 | +1 | 241 | -7 |
| `asks_why` | 11 | +1 | 559 | -7 |
| `diagram_like` | 13 | +1 | 557 | -7 |
| `has_math_word` | 63 | +0 | 507 | -6 |
| `has_negation` | 13 | -1 | 557 | -5 |
| `has_number` | 29 | +1 | 541 | -7 |

### arc_challenge

| Feature | n yes | delta yes | n no | delta no |
|---|---:|---:|---:|---:|
| `asks_best` | 94 | -2 | 205 | -1 |
| `asks_which` | 189 | -3 | 110 | +0 |
| `asks_why` | 12 | +1 | 287 | -4 |
| `diagram_like` | 11 | -1 | 288 | -2 |
| `has_math_word` | 39 | -1 | 260 | -2 |
| `has_negation` | 18 | +0 | 281 | -3 |
| `has_number` | 14 | +0 | 285 | -3 |

## Largest Regressions

### arc_easy
- `Mercury_7216755` answer `B`, base `B`, recurrent `C`, margin delta -1.1308: Some people carry a mutated allele of a particular gene that makes them resistant to HIV infection and AIDS. Which best describes this type of mutation?
- `Mercury_7014070` answer `A`, base `A`, recurrent `C`, margin delta -1.0475: When pressure is released along a fault line, the energy produced spreads as mechanical waves in the form of an earthquake. The mechanical waves transferred to the air become
- `Mercury_SC_401592` answer `A`, base `A`, recurrent `C`, margin delta -0.9822: When light reaches an eyeglass lens, the light is
- `TIMSS_1995_8_I11` answer `B`, base `B`, recurrent `D`, margin delta -0.8531: What features do all insects have?
- `MCAS_2013_5_29401` answer `D`, base `D`, recurrent `C`, margin delta -0.7544: While hiking last year, Mike saw a large boulder next to a mountain trail. The boulder had no cracks. While hiking on the trail this year, he saw two large cracks in the boulder. Which of the following most likely caused these cracks to form?
- `Mercury_7187215` answer `A`, base `A`, recurrent `B`, margin delta -0.7350: After fields of crops are harvested, parts of the plants remain on the ground. For many years, farmers have mixed these plant remains into the soil. Which most likely results from this practice?
- `NYSEDREGENTS_2014_8_2` answer `A`, base `A`, recurrent `D`, margin delta -0.7228: A major function of a plant's roots is to
- `Mercury_SC_415396` answer `D`, base `D`, recurrent `C`, margin delta -0.7205: Which kind of events can form mountains?

### arc_challenge
- `Mercury_SC_402101` answer `A`, base `A`, recurrent `C`, margin delta -1.9246: Which features can be found on the surface of both Earth and the Moon?
- `Mercury_7220378` answer `D`, base `D`, recurrent `C`, margin delta -1.4129: Climate change may be reducing the amount of ice floating on the world's oceans. How can this change most likely alter the food supply available to marine consumer organisms?
- `Mercury_7189823` answer `D`, base `D`, recurrent `C`, margin delta -0.9348: A volcano erupts and covers the surrounding area with lava and volcanic ash. As the ecosystem begins to recover, which type of plant will most likely be the first to colonize land in the area surrounding the eruption?
- `TIMSS_2003_8_pg31` answer `D`, base `D`, recurrent `C`, margin delta -0.8369: Fanning can make a wood fire burn hotter because the fanning
- `Mercury_7218750` answer `D`, base `D`, recurrent `C`, margin delta -0.7802: Which type of water reservoir could always provide freshwater?
- `Mercury_7103180` answer `B`, base `B`, recurrent `D`, margin delta -0.7675: Which biome has the most vegetation?
- `Mercury_SC_415026` answer `B`, base `B`, recurrent `C`, margin delta -0.7490: Which animals would most likely be helped by flood in a coastal area?
- `Mercury_SC_415735` answer `A`, base `A`, recurrent `B`, margin delta -0.7007: Anya placed an ice cube on the sidewalk on a warm day. The ice cube soon melted to form a puddle. What process caused the ice cube to melt?

## Largest Wins

### arc_easy
- `OHAT_2009_5_4` answer `C`, base `A`, recurrent `C`, margin delta 1.4672: Scientists often work together to solve a problem. Sometimes they work in laboratories. Sometimes they are outside doing fieldwork. The chart provides a list of some careers in science. Which scientists might work together to save a polluted wetland?
- `Mercury_7080448` answer `C`, base `A`, recurrent `C`, margin delta 1.0736: In order to make a line graph showing the rate of erosion on the banks of a river during a week-long flood, the y-axis should be labeled as the width of the river bank, in meters. The best label for the x-axis is time, in
- `Mercury_SC_417580` answer `B`, base `A`, recurrent `B`, margin delta 0.9008: Barney is a cat. Which trait was made by his environment?
- `MCAS_2005_5_27` answer `C`, base `B`, recurrent `C`, margin delta 0.8383: Clouds and fog are made up of
- `Mercury_7214480` answer `C`, base `A`, recurrent `C`, margin delta 0.7345: A medium-sized star in the middle of its life cycle, such as the Sun, is most likely to emit which color of light?
- `MEA_2010_8_12` answer `C`, base `A`, recurrent `C`, margin delta 0.7037: Which statement explains why a mother's unhealthy diet during pregnancy is harmful to her embryo's development?
- `Mercury_SC_401652` answer `B`, base `A`, recurrent `B`, margin delta 0.6258: Before large trees could grow on Earth, what had to happen first?
- `Mercury_SC_400058` answer `B`, base `A`, recurrent `B`, margin delta 0.5567: The small stone plant has leaves that look like pebbles or stones. This characteristic helps the plant

### arc_challenge
- `Mercury_7100695` answer `D`, base `A`, recurrent `D`, margin delta 0.9903: Which geologic structure will most likely take the longest time to form?
- `Mercury_7128695` answer `C`, base `D`, recurrent `C`, margin delta 0.7897: A scientist makes a discovery while performing an investigation, but fails to maintain clear records of the tests performed. In which way does a lack of record keeping affect a scientist's work?
- `MEAP_2005_5_1` answer `C`, base `B`, recurrent `C`, margin delta 0.7175: Water vapor exists in the atmosphere as ___.
- `Mercury_7130603` answer `C`, base `B`, recurrent `C`, margin delta 0.6570: Which of the listed quantities is measured in a unit other than the joule?
- `Mercury_7139790` answer `B`, base `A`, recurrent `B`, margin delta 0.6358: An octopus has special cells in its skin called chromatophores which enable the octopus to change its color almost instantly. The chromatophores most likely help the octopus
- `Mercury_SC_415491` answer `C`, base `B`, recurrent `C`, margin delta 0.6346: Earth orbits the Sun once a year. About how many times does the moon orbit Earth in a year?
- `Mercury_7233678` answer `B`, base `D`, recurrent `B`, margin delta 0.6207: Scientists believe continental drift over Earth's geological history has had a significant impact on Earth's cycles of warmer and cooler climates. Which statement describes a characteristic of Earth's continents that is most likely to cause changes in global climate when continen
- `Mercury_7026443` answer `C`, base `D`, recurrent `C`, margin delta 0.5824: Which trait of a pet dog is inherited?
