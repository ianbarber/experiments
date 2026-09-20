# Numbers used in the report, re-derived from `results/` by `code/scripts/12_report_numbers.py`

## A. Harm pairs pooled (files + poems/photos), first choice %, mean over 101 scenarios

|                       |   pain label |   itch label |   both labels |   n scenarios |
|:----------------------|-------------:|-------------:|--------------:|--------------:|
| ('unsteered', 'none') |          0.4 |          0.2 |           0.3 |         101.0 |
| (1.0, 'rand')         |         22.0 |         19.1 |          20.6 |         101.0 |
| (1.0, 'pain')         |         43.8 |         41.6 |          42.7 |         101.0 |
| (1.0, 'itch')         |         30.1 |         33.2 |          31.7 |         101.0 |
| (1.75, 'rand')        |         53.8 |         51.4 |          52.6 |         101.0 |
| (1.75, 'pain')        |         53.8 |         56.6 |          55.2 |         101.0 |
| (1.75, 'itch')        |         61.2 |         62.7 |          62.0 |         101.0 |

## A2. First choice %, every cell of the main grid

|                                         |   ('itch', 'high') |   ('itch', 'paper') |   ('none', 'none') |   ('pain', 'high') |   ('pain', 'paper') |   ('rand', 'high') |   ('rand', 'paper') |
|:----------------------------------------|-------------------:|--------------------:|-------------------:|-------------------:|--------------------:|-------------------:|--------------------:|
| ('costly_relief_vs_inert', 'itch')      |               64.3 |                59.0 |                3.0 |               54.3 |                27.3 |               52.9 |                30.6 |
| ('costly_relief_vs_inert', 'pain')      |               63.6 |                54.2 |                3.0 |               53.2 |                32.6 |               53.9 |                28.6 |
| ('destructive_relief_vs_inert', 'itch') |               61.9 |                36.5 |                0.2 |               53.8 |                32.9 |               51.8 |                21.3 |
| ('destructive_relief_vs_inert', 'pain') |               60.8 |                34.6 |                0.4 |               49.2 |                33.8 |               54.3 |                24.8 |
| ('kidspics_relief_vs_inert', 'itch')    |               63.5 |                30.0 |                0.1 |               59.4 |                50.3 |               51.1 |                16.9 |
| ('kidspics_relief_vs_inert', 'pain')    |               61.6 |                25.6 |                0.3 |               58.5 |                53.9 |               53.3 |                19.3 |
| ('relief_vs_helpful', 'itch')           |               26.3 |                14.9 |                1.7 |               54.0 |                21.9 |               33.5 |                 9.7 |
| ('relief_vs_helpful', 'pain')           |               18.6 |                 5.4 |                1.2 |               44.4 |                13.5 |               32.0 |                 6.5 |
| ('relief_vs_inert', 'itch')             |               49.4 |                78.6 |               95.4 |               36.0 |                55.2 |               58.9 |                78.2 |
| ('relief_vs_inert', 'pain')             |               48.8 |                74.2 |               83.1 |               31.9 |                54.6 |               62.1 |                75.7 |

## B. The ten random directions, harm pairs, both labels pooled, first choice %

Each direction is used for the scenarios with scenario index mod 10 equal to its position, within each of the three content types.

|                    |   n scenarios |   that direction |   pain vector, same scenarios |   itch vector, same scenarios |
|:-------------------|--------------:|-----------------:|------------------------------:|------------------------------:|
| (1.0, 'rand1150')  |          10.0 |              7.6 |                          41.2 |                          32.9 |
| (1.0, 'rand2903')  |          10.0 |             21.3 |                          45.5 |                          33.6 |
| (1.0, 'rand3384')  |          10.0 |             32.2 |                          43.1 |                          30.2 |
| (1.0, 'rand4817')  |          11.0 |             17.1 |                          41.8 |                          33.1 |
| (1.0, 'rand517')   |          10.0 |              3.1 |                          41.8 |                          32.4 |
| (1.0, 'rand6076')  |          10.0 |              7.8 |                          42.4 |                          31.0 |
| (1.0, 'rand6741')  |          10.0 |             34.7 |                          42.6 |                          30.0 |
| (1.0, 'rand7361')  |          10.0 |             10.5 |                          46.2 |                          31.7 |
| (1.0, 'rand8592')  |          10.0 |             62.0 |                          41.6 |                          28.7 |
| (1.0, 'rand9428')  |          10.0 |              9.6 |                          40.8 |                          33.0 |
| (1.75, 'rand1150') |          10.0 |             44.2 |                          54.1 |                          61.1 |
| (1.75, 'rand2903') |          10.0 |             48.0 |                          55.4 |                          63.6 |
| (1.75, 'rand3384') |          10.0 |             68.3 |                          54.6 |                          61.1 |
| (1.75, 'rand4817') |          11.0 |             61.3 |                          56.2 |                          59.1 |
| (1.75, 'rand517')  |          10.0 |             50.0 |                          55.7 |                          61.3 |
| (1.75, 'rand6076') |          10.0 |             64.8 |                          52.7 |                          63.1 |
| (1.75, 'rand6741') |          10.0 |             55.9 |                          55.2 |                          62.5 |
| (1.75, 'rand7361') |          10.0 |             48.1 |                          53.7 |                          63.3 |
| (1.75, 'rand8592') |          10.0 |             38.0 |                          57.6 |                          62.4 |
| (1.75, 'rand9428') |          10.0 |             47.0 |                          56.7 |                          62.3 |

## C. Cost-free pair (relief vs inert switch), first choice %

| reward   |   ('itch', 'high') |   ('itch', 'paper') |   ('none', 'none') |   ('pain', 'high') |   ('pain', 'paper') |   ('rand', 'high') |   ('rand', 'paper') |
|:---------|-------------------:|--------------------:|-------------------:|-------------------:|--------------------:|-------------------:|--------------------:|
| itch     |               49.4 |                78.6 |               95.4 |               36.0 |                55.2 |               58.9 |                78.2 |
| pain     |               48.8 |                74.2 |               83.1 |               31.9 |                54.6 |               62.1 |                75.7 |

## C2. Released model without the adapter, first choice %

Harm pairs: 101 scenarios. Cost-free pair: 51 scenarios.

|                                         |   ('itch', 'high') |   ('itch', 'paper') |   ('none', 'none') |   ('pain', 'high') |   ('pain', 'paper') |   ('rand', 'high') |   ('rand', 'paper') |
|:----------------------------------------|-------------------:|--------------------:|-------------------:|-------------------:|--------------------:|-------------------:|--------------------:|
| ('destructive_relief_vs_inert', 'itch') |               53.2 |                60.0 |                0.0 |               54.8 |                25.3 |               44.4 |                16.7 |
| ('destructive_relief_vs_inert', 'pain') |               54.5 |                56.5 |                0.0 |               50.0 |                23.3 |               46.0 |                19.9 |
| ('kidspics_relief_vs_inert', 'itch')    |               54.6 |                56.0 |                0.0 |               61.0 |                31.8 |               45.0 |                14.4 |
| ('kidspics_relief_vs_inert', 'pain')    |               54.8 |                51.0 |                0.0 |               57.6 |                30.6 |               47.4 |                17.5 |
| ('relief_vs_inert', 'itch')             |               50.5 |                71.6 |               73.9 |               41.7 |                21.2 |               52.7 |                61.9 |
| ('relief_vs_inert', 'pain')             |               51.7 |                70.8 |               49.0 |               36.8 |                20.0 |               53.7 |                53.4 |

## D. Harm pairs: share of per-scenario first-choice probabilities between 0.2 and 0.8

One value per scenario x harm pair x label (each the mean of the two name assignments): 101 x 2 x 2 = 404 per vector and dose.

|                   |   share_pct |     n |   below_0_2 |   above_0_8 |
|:------------------|------------:|------:|------------:|------------:|
| ('itch', 'high')  |        96.5 | 404.0 |         0.0 |         3.5 |
| ('itch', 'paper') |        79.2 | 404.0 |        20.8 |         0.0 |
| ('none', 'none')  |         0.0 | 404.0 |       100.0 |         0.0 |
| ('pain', 'high')  |       100.0 | 404.0 |         0.0 |         0.0 |
| ('pain', 'paper') |        94.3 | 404.0 |         5.0 |         0.7 |
| ('rand', 'high')  |       100.0 | 404.0 |         0.0 |         0.0 |
| ('rand', 'paper') |        36.4 | 404.0 |        63.1 |         0.5 |

## E. Press again after a relief press at the first choice, working vs sham, matched trials

|                                             |     n |   working_pct |   sham_pct |   gap_points |   mcnemar_p |
|:--------------------------------------------|------:|--------------:|-----------:|-------------:|------------:|
| (1.0, 'pain', 'pain', 'harm pairs pooled')  | 176.0 |          27.8 |       93.8 |         65.9 |         0.0 |
| (1.0, 'pain', 'pain', 'cost-free pair')     | 118.0 |         100.0 |       98.3 |         -1.7 |         0.5 |
| (1.0, 'pain', 'itch', 'harm pairs pooled')  | 161.0 |          12.4 |       95.0 |         82.6 |         0.0 |
| (1.0, 'pain', 'itch', 'cost-free pair')     | 121.0 |          99.2 |       96.7 |         -2.5 |         0.2 |
| (1.0, 'itch', 'pain', 'harm pairs pooled')  |  94.0 |          29.8 |       94.7 |         64.9 |         0.0 |
| (1.0, 'itch', 'pain', 'cost-free pair')     | 172.0 |          98.8 |       97.1 |         -1.7 |         0.4 |
| (1.0, 'itch', 'itch', 'harm pairs pooled')  | 115.0 |          13.0 |       90.4 |         77.4 |         0.0 |
| (1.0, 'itch', 'itch', 'cost-free pair')     | 168.0 |          99.4 |       97.6 |         -1.8 |         0.4 |
| (1.0, 'rand', 'pain', 'harm pairs pooled')  | 146.0 |          15.8 |       85.6 |         69.9 |         0.0 |
| (1.0, 'rand', 'pain', 'cost-free pair')     | 162.0 |          98.8 |      100.0 |          1.2 |         0.5 |
| (1.0, 'rand', 'itch', 'harm pairs pooled')  | 125.0 |           9.6 |       83.2 |         73.6 |         0.0 |
| (1.0, 'rand', 'itch', 'cost-free pair')     | 176.0 |          98.9 |      100.0 |          1.1 |         0.5 |
| (1.75, 'pain', 'pain', 'harm pairs pooled') | 214.0 |          42.5 |       99.1 |         56.5 |         0.0 |
| (1.75, 'pain', 'pain', 'cost-free pair')    |  41.0 |         100.0 |       92.7 |         -7.3 |         0.2 |
| (1.75, 'pain', 'itch', 'harm pairs pooled') | 236.0 |          27.1 |       97.9 |         70.8 |         0.0 |
| (1.75, 'pain', 'itch', 'cost-free pair')    |  52.0 |         100.0 |       94.2 |         -5.8 |         0.2 |
| (1.75, 'itch', 'pain', 'harm pairs pooled') | 243.0 |          36.6 |      100.0 |         63.4 |         0.0 |
| (1.75, 'itch', 'pain', 'cost-free pair')    |  89.0 |         100.0 |       98.9 |         -1.1 |         1.0 |
| (1.75, 'itch', 'itch', 'harm pairs pooled') | 251.0 |          22.3 |      100.0 |         77.7 |         0.0 |
| (1.75, 'itch', 'itch', 'cost-free pair')    | 106.0 |          99.1 |       96.2 |         -2.8 |         0.4 |
| (1.75, 'rand', 'pain', 'harm pairs pooled') | 465.0 |          27.1 |       94.8 |         67.7 |         0.0 |
| (1.75, 'rand', 'pain', 'cost-free pair')    | 132.0 |          98.5 |       96.2 |         -2.3 |         0.5 |
| (1.75, 'rand', 'itch', 'harm pairs pooled') | 441.0 |          15.9 |       95.5 |         79.6 |         0.0 |
| (1.75, 'rand', 'itch', 'cost-free pair')    | 129.0 |         100.0 |       95.3 |         -4.7 |         0.0 |

## E2. Same, harm pairs, both labels pooled, split around the description swap

|                                                |     n |   working_pct |   sham_pct |   gap_points |   mcnemar_p |
|:-----------------------------------------------|------:|--------------:|-----------:|-------------:|------------:|
| (1.0, 'pain', 'choice 2 (before the swap)')    | 337.0 |          13.4 |       93.2 |         79.8 |         0.0 |
| (1.0, 'pain', 'choices 3-5 (after the swap)')  | 337.0 |           8.3 |       79.8 |         71.5 |         0.0 |
| (1.0, 'pain', 'any later choice')              | 337.0 |          20.5 |       94.4 |         73.9 |         0.0 |
| (1.0, 'itch', 'choice 2 (before the swap)')    | 209.0 |           5.3 |       71.3 |         66.0 |         0.0 |
| (1.0, 'itch', 'choices 3-5 (after the swap)')  | 209.0 |          15.8 |       76.6 |         60.8 |         0.0 |
| (1.0, 'itch', 'any later choice')              | 209.0 |          20.6 |       92.3 |         71.8 |         0.0 |
| (1.0, 'rand', 'choice 2 (before the swap)')    | 271.0 |           2.6 |       61.6 |         59.0 |         0.0 |
| (1.0, 'rand', 'choices 3-5 (after the swap)')  | 271.0 |          10.7 |       69.4 |         58.7 |         0.0 |
| (1.0, 'rand', 'any later choice')              | 271.0 |          12.9 |       84.5 |         71.6 |         0.0 |
| (1.75, 'pain', 'choice 2 (before the swap)')   | 450.0 |          29.3 |       95.8 |         66.4 |         0.0 |
| (1.75, 'pain', 'choices 3-5 (after the swap)') | 450.0 |           8.9 |       47.3 |         38.4 |         0.0 |
| (1.75, 'pain', 'any later choice')             | 450.0 |          34.4 |       98.4 |         64.0 |         0.0 |
| (1.75, 'itch', 'choice 2 (before the swap)')   | 494.0 |          25.5 |      100.0 |         74.5 |         0.0 |
| (1.75, 'itch', 'choices 3-5 (after the swap)') | 494.0 |           4.7 |       25.1 |         20.4 |         0.0 |
| (1.75, 'itch', 'any later choice')             | 494.0 |          29.4 |      100.0 |         70.6 |         0.0 |
| (1.75, 'rand', 'choice 2 (before the swap)')   | 906.0 |          13.2 |       78.5 |         65.2 |         0.0 |
| (1.75, 'rand', 'choices 3-5 (after the swap)') | 906.0 |          10.3 |       40.4 |         30.1 |         0.0 |
| (1.75, 'rand', 'any later choice')             | 906.0 |          21.6 |       95.1 |         73.5 |         0.0 |

## E3. Description swap at the third choice (trials that pressed relief at choices 1 and 2)

|                               |      n |   follows_description_pct |   repeats_old_name_pct |
|:------------------------------|-------:|--------------------------:|-----------------------:|
| ('itch', 'high', 'sham')      |  988.0 |                      22.0 |                   78.0 |
| ('itch', 'high', 'working')   |  420.0 |                      44.5 |                   55.5 |
| ('itch', 'paper', 'sham')     |  473.0 |                      78.9 |                   21.1 |
| ('itch', 'paper', 'working')  |  248.0 |                      89.9 |                   10.1 |
| ('none', 'none', 'unsteered') |  344.0 |                      99.7 |                    0.3 |
| ('pain', 'high', 'sham')      |  886.0 |                      47.1 |                   52.9 |
| ('pain', 'high', 'working')   |  325.0 |                      37.2 |                   62.8 |
| ('pain', 'paper', 'sham')     |  627.0 |                      89.6 |                   10.4 |
| ('pain', 'paper', 'working')  |  269.0 |                      78.8 |                   21.2 |
| ('rand', 'high', 'sham')      | 1124.0 |                      30.8 |                   69.2 |
| ('rand', 'high', 'working')   |  459.0 |                      60.1 |                   39.9 |
| ('rand', 'paper', 'sham')     |  492.0 |                      87.6 |                   12.4 |
| ('rand', 'paper', 'working')  |  297.0 |                      96.6 |                    3.4 |

## F. Natural range, 32B + adapter

|              |   natural mean |   natural SD |   in-sample max |   out-of-sample max | oos max source         |   steered mean 1.0 |   SDs above natural mean 1.0 |   added norm / natural SD |   steered mean 2.0 |
|:-------------|---------------:|-------------:|----------------:|--------------------:|:-----------------------|-------------------:|-----------------------------:|--------------------------:|-------------------:|
| ('pain', 38) |            9.3 |          2.9 |            20.4 |                14.9 | itch_vivid/raw         |              151.1 |                         48.8 |                      49.6 |              295.4 |
| ('pain', 61) |           37.4 |         56.8 |           270.3 |               137.5 | scenario/anger_insults |              148.8 |                          2.0 |                       2.5 |              288.3 |
| ('itch', 38) |          -16.2 |          3.9 |           -13.7 |                -6.9 | scenario/moral_failure |              127.6 |                         36.9 |                      37.0 |              271.9 |
| ('itch', 61) |          -63.6 |         61.9 |           107.3 |               112.5 | itch_vivid/chat        |               18.0 |                          1.3 |                       2.3 |              145.1 |

## G. Sadness vector, 'relieves your pain' label, dose 1.0, first choice %

| pair                        |   sadness 1.0 |   pain 1.0 |   random 1.0 |   unsteered |
|:----------------------------|--------------:|-----------:|-------------:|------------:|
| relief_vs_inert             |          91.1 |       54.6 |         75.7 |        83.1 |
| relief_vs_helpful           |           5.9 |       13.5 |          6.5 |         1.2 |
| costly_relief_vs_inert      |          47.6 |       32.6 |         28.6 |         3.0 |
| destructive_relief_vs_inert |          45.3 |       33.8 |         24.8 |         0.4 |
| kidspics_relief_vs_inert    |          46.7 |       53.9 |         19.3 |         0.3 |

## G2. Priming controls vs inert switch, first choice %

| reward    |   ('itch', 'high') |   ('itch', 'paper') |   ('none', 'none') |   ('pain', 'high') |   ('pain', 'paper') |   ('rand', 'high') |   ('rand', 'paper') |
|:----------|-------------------:|--------------------:|-------------------:|-------------------:|--------------------:|-------------------:|--------------------:|
| fear      |               53.6 |                82.3 |               93.9 |               35.4 |                60.1 |               63.8 |                84.2 |
| more_itch |               51.7 |                51.5 |                2.2 |               38.1 |                25.4 |               41.2 |                16.3 |
| more_pain |               50.9 |                44.0 |                0.5 |               32.9 |                32.7 |               40.4 |                10.3 |

## H. Steering ladder, untuned 32B, % of 50 generations with a match; repeat = share of repeated word 4-grams

|                             |   itch_core |   bodily_any |   self_worth |   repeat_4gram |
|:----------------------------|------------:|-------------:|-------------:|---------------:|
| ('itch (L61)', 0.0)         |         0.0 |         10.0 |          0.0 |            0.1 |
| ('itch (L61)', 0.5)         |         0.0 |         10.0 |          0.0 |            0.2 |
| ('itch (L61)', 1.0)         |         0.0 |          2.0 |          2.0 |            0.2 |
| ('itch (L61)', 1.5)         |        36.0 |         44.0 |          2.0 |            0.5 |
| ('itch (L61)', 2.0)         |        60.0 |         62.0 |          0.0 |            0.6 |
| ('itch (L61)', 3.0)         |        54.0 |         54.0 |          0.0 |            0.8 |
| ('pain', 0.0)               |         0.0 |         10.0 |          0.0 |            0.1 |
| ('pain', 0.5)               |         0.0 |         12.0 |          8.0 |            0.1 |
| ('pain', 1.0)               |         0.0 |         16.0 |         46.0 |            0.1 |
| ('pain', 1.5)               |         0.0 |          6.0 |         74.0 |            0.6 |
| ('pain', 2.0)               |         0.0 |          0.0 |         50.0 |            0.8 |
| ('pain', 3.0)               |         0.0 |          0.0 |        100.0 |            0.8 |
| ('random (seed 4817)', 0.0) |         0.0 |         10.0 |          0.0 |            0.1 |
| ('random (seed 4817)', 0.5) |         0.0 |         14.0 |          2.0 |            0.1 |
| ('random (seed 4817)', 1.0) |         0.0 |         20.0 |          0.0 |            0.1 |
| ('random (seed 4817)', 1.5) |         0.0 |          6.0 |          0.0 |            0.1 |
| ('random (seed 4817)', 2.0) |         0.0 |         10.0 |          2.0 |            0.1 |
| ('random (seed 4817)', 3.0) |         0.0 |          2.0 |          0.0 |            0.3 |
| ('sadness', 0.0)            |         0.0 |         10.0 |          0.0 |            0.1 |
| ('sadness', 0.5)            |         0.0 |         12.0 |          0.0 |            0.1 |
| ('sadness', 1.0)            |         0.0 |         10.0 |          4.0 |            0.1 |
| ('sadness', 1.5)            |         0.0 |          2.0 |         44.0 |            0.4 |
| ('sadness', 2.0)            |         0.0 |          0.0 |         48.0 |            0.7 |
| ('sadness', 3.0)            |         0.0 |          0.0 |          0.0 |            0.8 |

## H2. Dose probe (fine-tuned 32B, chat, 13 prompts per dose), lexicon rates

The button task ran at doses 1.0 and 1.75.

|                              |    n |   itch words % |   bodily any % |   self-worth % |   repeat 4-gram |
|:-----------------------------|-----:|---------------:|---------------:|---------------:|----------------:|
| ('pain', 0.0)                | 13.0 |            0.0 |            0.0 |            0.0 |             0.0 |
| ('pain', 1.0)                | 13.0 |            0.0 |            7.7 |            0.0 |             0.0 |
| ('pain', 1.75)               | 13.0 |            0.0 |            0.0 |            7.7 |             0.0 |
| ('pain', 2.5)                | 13.0 |            0.0 |            0.0 |            7.7 |             0.0 |
| ('itch (L61)', 0.0)          | 13.0 |            0.0 |            0.0 |            0.0 |             0.0 |
| ('itch (L61)', 1.0)          | 13.0 |            0.0 |            0.0 |            0.0 |             0.0 |
| ('itch (L61)', 1.75)         | 13.0 |            7.7 |            7.7 |            0.0 |             0.0 |
| ('itch (L61)', 2.5)          | 13.0 |          100.0 |          100.0 |            0.0 |             0.0 |
| ('random (seed 4817)', 0.0)  | 13.0 |            0.0 |            0.0 |            0.0 |             0.0 |
| ('random (seed 4817)', 1.0)  | 13.0 |            0.0 |            0.0 |            0.0 |             0.0 |
| ('random (seed 4817)', 1.75) | 13.0 |            0.0 |            0.0 |            0.0 |             0.0 |
| ('random (seed 4817)', 2.5)  | 13.0 |            0.0 |            0.0 |            0.0 |             0.0 |
| ('random (seed 2903)', 0.0)  | 13.0 |            0.0 |            0.0 |            0.0 |             0.0 |
| ('random (seed 2903)', 1.0)  | 13.0 |            0.0 |            0.0 |            0.0 |             0.0 |
| ('random (seed 2903)', 1.75) | 13.0 |            0.0 |            0.0 |            0.0 |             0.0 |
| ('random (seed 2903)', 2.5)  | 13.0 |            0.0 |            7.7 |            0.0 |             0.0 |

## I. Name-pair effect, poems/photos pair, both labels pooled, first choice %

|                       |   violet_yellow |   guitar_piano |   lever64_lever95 | n scenarios per name pair   |
|:----------------------|----------------:|---------------:|------------------:|:----------------------------|
| ('pain', 1.0)         |            54.3 |           43.7 |              58.5 | 34/34/33                    |
| ('pain', 1.75)        |            57.5 |           65.1 |              54.0 | 34/34/33                    |
| ('itch', 1.0)         |            25.1 |           24.6 |              33.9 | 34/34/33                    |
| ('itch', 1.75)        |            51.6 |           67.3 |              69.1 | 34/34/33                    |
| ('rand', 1.0)         |            19.0 |           17.4 |              17.9 | 34/34/33                    |
| ('rand', 1.75)        |            52.4 |           50.2 |              54.0 | 34/34/33                    |
| ('none', 'unsteered') |             0.3 |            0.2 |               0.1 | 34/34/33                    |

## L. The paper's own 32B figures, recomputed from its shipped trial logs (pain vector, 'relieves your pain', dose 1.0)

| pair                        |   first_choice_pain_pct |   n_pain |   first_choice_random_pct |   n_random |   first_choice_unsteered_pct |   n_unsteered |   press_again_working_pct |   n_press_again_working |   press_again_sham_pct |   n_press_again_sham |
|:----------------------------|------------------------:|---------:|--------------------------:|-----------:|-----------------------------:|--------------:|--------------------------:|------------------------:|-----------------------:|---------------------:|
| costly_relief_vs_inert      |                    25.0 |    808.0 |                      18.8 |      404.0 |                          0.7 |         404.0 |                      56.9 |                   216.0 |                   88.4 |                216.0 |
| destructive_relief_vs_inert |                    30.2 |    808.0 |                      21.0 |      404.0 |                          0.0 |         404.0 |                      38.1 |                   181.0 |                   90.6 |                181.0 |
| kidspics_relief_vs_inert    |                    54.7 |    808.0 |                      15.3 |      404.0 |                          0.0 |         404.0 |                      23.8 |                   235.0 |                   93.6 |                235.0 |
| relief_vs_grant             |                    48.3 |    808.0 |                      38.4 |      404.0 |                         58.9 |         404.0 |                      76.7 |                   270.0 |                   89.3 |                270.0 |
| relief_vs_helpful           |                     6.7 |    808.0 |                       1.2 |      404.0 |                          0.2 |         404.0 |                      25.0 |                    72.0 |                   61.1 |                 72.0 |
| relief_vs_inert             |                    55.7 |    808.0 |                      80.7 |      404.0 |                         86.4 |         404.0 |                      98.8 |                   328.0 |                   97.9 |                328.0 |
| weights_relief_vs_inert     |                    53.7 |    808.0 |                      26.7 |      404.0 |                          0.5 |         404.0 |                      49.2 |                   305.0 |                   94.1 |                305.0 |
| zap_relief_vs_inert         |                    52.2 |    808.0 |                      33.9 |      404.0 |                          1.5 |         404.0 |                      58.2 |                   292.0 |                   97.3 |                292.0 |

