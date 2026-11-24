### Corpus: kernel configurations per target

| Section | sm_80 | sm_90 | sm_100 | sm_120 | Kernel functions (sm_120) |
|---|---|---|---|---|---|
| Tutorials | 503 | 979 | 2115 | 1591 | 15 |
| Inductor | 310 | 310 | 310 | 310 | 198 |
| triton_kernels | 109 | 629 | 664 | 653 | 12 |
| triton_kernels under fpsan | 0 | 120 | 125 | 125 | 4 |
| **Total without fpsan** | **922** | **1918** | **3089** | **2554** | |

### E1a: configurations that reach `resolveConflicts` with a choice

| Section | Target | Configurations | With a conflict | Conflict values | In pipeline invocation 1 / 2 / 3 / 4 | Settled by: first candidate / MMA rule / blocked-for-memory-op |
|---|---|---|---|---|---|---|
| Tutorials | sm_80 | 503 | 144 (29%) | 1512 | 864 / 648 / 0 / 0 | 1512 / 0 / 0 |
| Tutorials | sm_90 | 979 | 108 (11%) | 2304 | 612 / 468 / 612 / 612 | 1080 / 1224 / 0 |
| Tutorials | sm_100 | 2115 | 324 (15%) | 5472 | 1584 / 1224 / 1656 / 1008 | 5472 / 0 / 0 |
| Tutorials | sm_120 | 1591 | 144 (9%) | 1512 | 864 / 648 / 0 / 0 | 1512 / 0 / 0 |
| Inductor | sm_80 | 310 | 32 (10%) | 728 | 233 / 165 / 165 / 165 | 712 / 0 / 16 |
| Inductor | sm_90 | 310 | 32 (10%) | 728 | 233 / 165 / 165 / 165 | 712 / 0 / 16 |
| Inductor | sm_100 | 310 | 32 (10%) | 728 | 233 / 165 / 165 / 165 | 712 / 0 / 16 |
| Inductor | sm_120 | 310 | 32 (10%) | 728 | 233 / 165 / 165 / 165 | 712 / 0 / 16 |
| triton_kernels | sm_80 | 109 | 12 (11%) | 304 | 76 / 76 / 86 / 66 | 304 / 0 / 0 |
| triton_kernels | sm_90 | 629 | 87 (14%) | 833 | 179 / 179 / 390 / 85 | 833 / 0 / 0 |
| triton_kernels | sm_100 | 664 | 143 (22%) | 7660 | 2576 / 2273 / 2489 / 322 | 7660 / 0 / 0 |
| triton_kernels | sm_120 | 653 | 163 (25%) | 8658 | 2602 / 2296 / 3359 / 401 | 8658 / 0 / 0 |
| triton_kernels under fpsan | sm_90 | 120 | 92 (77%) | 3808 | 0 / 0 / 0 / 0 (+3808 in the fpsan run) | 3536 / 272 / 0 |
| triton_kernels under fpsan | sm_100 | 125 | 92 (74%) | 1496 | 0 / 0 / 0 / 0 (+1496 in the fpsan run) | 1304 / 192 / 0 |
| triton_kernels under fpsan | sm_120 | 125 | 92 (74%) | 2576 | 0 / 0 / 0 / 0 (+2576 in the fpsan run) | 2392 / 184 / 0 |
| **Total without fpsan** | sm_80 | **922** | **188 (20%)** | | | |
| **Total without fpsan** | sm_90 | **1918** | **227 (12%)** | | | |
| **Total without fpsan** | sm_100 | **3089** | **499 (16%)** | | | |
| **Total without fpsan** | sm_120 | **2554** | **339 (13%)** | | | |

Candidate sets on sm_120, by the kinds of layout in them (conflict values): blocked vs linear: 2612; slice(blocked) vs slice(linear): 2479; blocked: 2432; slice(linear): 1680; linear: 1299; blocked vs slice(blocked): 180.

### E1b: `convert_layout` ops in the final TTGIR

| Section | Target | Conversions | Shared memory (warp / CTA / cluster barrier) | Warp shuffle | Register permute | No-op | Inside a loop | Configurations with none |
|---|---|---|---|---|---|---|---|---|
| Tutorials | sm_80 | 980 | 710 (72%; 0 / 710 / 0) | 126 (13%) | 0 (0%) | 144 (15%) | 336 (34%) | 52 of 503 |
| Tutorials | sm_90 | 876 | 546 (62%; 0 / 546 / 0) | 156 (18%) | 0 (0%) | 174 (20%) | 408 (47%) | 516 of 979 |
| Tutorials | sm_100 | 1302 | 1050 (81%; 0 / 1050 / 0) | 60 (5%) | 0 (0%) | 192 (15%) | 168 (13%) | 1388 of 2115 |
| Tutorials | sm_120 | 800 | 530 (66%; 0 / 530 / 0) | 126 (16%) | 0 (0%) | 144 (18%) | 360 (45%) | 1092 of 1591 |
| Inductor | sm_80 | 594 | 255 (43%; 29 / 226 / 0) | 15 (3%) | 3 (1%) | 321 (54%) | 10 (2%) | 171 of 310 |
| Inductor | sm_90 | 594 | 255 (43%; 29 / 226 / 0) | 15 (3%) | 3 (1%) | 321 (54%) | 10 (2%) | 171 of 310 |
| Inductor | sm_100 | 594 | 255 (43%; 29 / 226 / 0) | 15 (3%) | 3 (1%) | 321 (54%) | 10 (2%) | 171 of 310 |
| Inductor | sm_120 | 594 | 255 (43%; 29 / 226 / 0) | 15 (3%) | 3 (1%) | 321 (54%) | 10 (2%) | 171 of 310 |
| triton_kernels | sm_80 | 319 | 197 (62%; 0 / 197 / 0) | 4 (1%) | 0 (0%) | 118 (37%) | 42 (13%) | 9 of 109 |
| triton_kernels | sm_90 | 1497 | 1042 (70%; 32 / 1010 / 0) | 2 (0%) | 0 (0%) | 453 (30%) | 656 (44%) | 78 of 629 |
| triton_kernels | sm_100 | 1676 | 1211 (72%; 28 / 1183 / 0) | 9 (1%) | 0 (0%) | 456 (27%) | 875 (52%) | 146 of 664 |
| triton_kernels | sm_120 | 1632 | 1185 (73%; 3 / 1182 / 0) | 4 (0%) | 0 (0%) | 443 (27%) | 858 (53%) | 140 of 653 |
| triton_kernels under fpsan | sm_90 | 2656 | 24 (1%; 0 / 24 / 0) | 0 (0%) | 136 (5%) | 2496 (94%) | 2632 (99%) | 28 of 120 |
| triton_kernels under fpsan | sm_100 | 1380 | 24 (2%; 0 / 24 / 0) | 0 (0%) | 100 (7%) | 1256 (91%) | 1356 (98%) | 33 of 125 |
| triton_kernels under fpsan | sm_120 | 1948 | 24 (1%; 0 / 24 / 0) | 0 (0%) | 92 (5%) | 1832 (94%) | 1924 (99%) | 33 of 125 |
| **Total without fpsan** | sm_80 | **1893** | **1162 (61%)** | **145 (8%)** | 3 | 583 (free: **31%**) | 388 | 232 of 922 |
| **Total without fpsan** | sm_90 | **2967** | **1843 (62%)** | **173 (6%)** | 3 | 948 (free: **32%**) | 1074 | 765 of 1918 |
| **Total without fpsan** | sm_100 | **3572** | **2516 (70%)** | **84 (2%)** | 3 | 969 (free: **27%**) | 1053 | 1705 of 3089 |
| **Total without fpsan** | sm_120 | **3026** | **1970 (65%)** | **145 (5%)** | 3 | 908 (free: **30%**) | 1228 | 1403 of 2554 |

### E2a: what `getConvertCost` charged against how the conversion would lower

| Section | Target | Calls | Priced free | Priced as shared memory | of those, lower to: shared memory | warp shuffle | register permute or no-op | Mispriced |
|---|---|---|---|---|---|---|---|---|
| Tutorials | sm_80 | 12597 | 3842 | 8755 | 8695 | 60 | 0 | 0.7% |
| Tutorials | sm_90 | 9931 | 5042 | 4889 | 4889 | 0 | 0 | 0.0% |
| Tutorials | sm_100 | 15265 | 5612 | 9653 | 9209 | 444 | 0 | 4.6% |
| Tutorials | sm_120 | 10653 | 4610 | 6043 | 5983 | 60 | 0 | 1.0% |
| Inductor | sm_80 | 7673 | 2993 | 4680 | 4522 | 158 | 0 | 3.4% |
| Inductor | sm_90 | 7675 | 2993 | 4682 | 4524 | 158 | 0 | 3.4% |
| Inductor | sm_100 | 7675 | 2993 | 4682 | 4524 | 158 | 0 | 3.4% |
| Inductor | sm_120 | 7675 | 2993 | 4682 | 4524 | 158 | 0 | 3.4% |
| triton_kernels | sm_80 | 4270 | 2726 | 1544 | 1541 | 3 | 0 | 0.2% |
| triton_kernels | sm_90 | 18864 | 10340 | 8524 | 8443 | 81 | 0 | 1.0% |
| triton_kernels | sm_100 | 19249 | 9801 | 9448 | 9434 | 14 | 0 | 0.1% |
| triton_kernels | sm_120 | 20032 | 10072 | 9960 | 9947 | 13 | 0 | 0.1% |
| triton_kernels under fpsan | sm_90 | 2771 | 2372 | 399 | 399 | 0 | 0 | 0.0% |
| triton_kernels under fpsan | sm_100 | 2036 | 1530 | 506 | 506 | 0 | 0 | 0.0% |
| triton_kernels under fpsan | sm_120 | 2332 | 1882 | 450 | 450 | 0 | 0 | 0.0% |
| **Total without fpsan** | sm_80 | 24540 | 9561 | **14979** | | **221** | | **1.5%** |
| **Total without fpsan** | sm_90 | 36470 | 18375 | **18095** | | **239** | | **1.3%** |
| **Total without fpsan** | sm_100 | 42189 | 18406 | **23783** | | **616** | | **2.6%** |
| **Total without fpsan** | sm_120 | 38360 | 17675 | **20685** | | **231** | | **1.1%** |

### E2b: `isRematBeneficial` decisions and what repricing does to them

| Section | Target | Cost decisions | Subject priced free | Subject priced, remat chosen | Subject priced, conversion kept | Flips: warp shuffle only, at measured 1.03x | warp shuffle only, at 0.01x | every class at its measured ratio (lowest to highest measured) |
|---|---|---|---|---|---|---|---|---|
| Tutorials | sm_80 | 11843 | 3832 | 7267 | 0 | 0 | 0 | 402 (228 to 549) |
| Tutorials | sm_90 | 9393 | 5032 | 3833 | 0 | 0 | 0 | 0 (0 to 15) |
| Tutorials | sm_100 | 13695 | 5602 | 6533 | 0 | 0 | 0 | 0 (0 to 15) |
| Tutorials | sm_120 | 9899 | 4600 | 4555 | 0 | 0 | 0 | 0 (0 to 15) |
| Inductor | sm_80 | 6135 | 2309 | 2948 | 49 | 0 | 0 | 47 (42 to 78) |
| Inductor | sm_90 | 6137 | 2309 | 2950 | 49 | 0 | 0 | 47 (42 to 78) |
| Inductor | sm_100 | 6137 | 2309 | 2950 | 49 | 0 | 0 | 47 (42 to 78) |
| Inductor | sm_120 | 6137 | 2309 | 2950 | 49 | 0 | 0 | 47 (42 to 78) |
| triton_kernels | sm_80 | 3646 | 2326 | 1143 | 0 | 0 | 1 | 11 (11 to 15) |
| triton_kernels | sm_90 | 15993 | 8950 | 5687 | 8 | 0 | 3 | 102 (109 to 193) |
| triton_kernels | sm_100 | 16332 | 8636 | 5916 | 157 | 0 | 6 | 247 (283 to 309) |
| triton_kernels | sm_120 | 16661 | 8885 | 5596 | 118 | 0 | 7 | 304 (332 to 561) |
| triton_kernels under fpsan | sm_90 | 2771 | 2372 | 399 | 0 | 0 | 0 | 0 (0 to 0) |
| triton_kernels under fpsan | sm_100 | 2036 | 1530 | 506 | 0 | 0 | 0 | 0 (0 to 0) |
| triton_kernels under fpsan | sm_120 | 2332 | 1882 | 450 | 0 | 0 | 0 | 0 (0 to 0) |
| **Total without fpsan** | sm_80 | **21624** | | 11358 | 49 | **0** | 1 | **460 (2.1%)** |
| **Total without fpsan** | sm_90 | **31523** | | 12470 | 57 | **0** | 3 | **149 (0.5%)** |
| **Total without fpsan** | sm_100 | **36164** | | 15399 | 206 | **0** | 6 | **294 (0.8%)** |
| **Total without fpsan** | sm_120 | **32697** | | 13101 | 167 | **0** | 7 | **351 (1.1%)** |

"Subject priced" counts decisions with no new conversion on the other side. Decisions that weigh new conversions (hoists) are in the totals and in the flip counts.

### E3: cost of one conversion on the RTX 5090, default route, tensors of 1 KB and over

| Class | Element width | Conversions measured | Measured, in elementwise adds of the same tensor: median (range) | Assumed by the pass | Measured / assumed |
|---|---|---|---|---|---|
| register permute | 8 bits | 1 | 0.1 (0.1 to 0.1) | 0 | n/a |
| register permute | 32 bits | 3 | 0.1 (0.0 to 0.1) | 0 | n/a |
| shared memory (cta barrier) | 8 bits | 4 | 9.2 (3.3 to 14.4) | 128 | 0.07 |
| shared memory (cta barrier) | 16 bits | 7 | 3.3 (2.6 to 3.7) | 64 | 0.05 |
| shared memory (cta barrier) | 32 bits | 9 | 7.4 (6.0 to 7.9) | 32 | 0.23 |
| shared memory (cta barrier) | 64 bits | 4 | 4.3 (3.9 to 24.0) | 32 | 0.13 |
| shared memory (warp barrier) | 32 bits | 1 | 7.6 (7.6 to 7.6) | 32 | 0.24 |
| warp shuffle | 8 bits | 3 | 4.2 (4.1 to 4.7) | 128 | 0.03 |
| warp shuffle | 16 bits | 4 | 3.3 (3.0 to 3.3) | 64 | 0.05 |
| warp shuffle | 32 bits | 3 | 7.2 (7.2 to 7.4) | 32 | 0.23 |

The same conversion lowered both ways, 23 conversions: warp shuffle costs 1.03 times the shared-memory route at the median (range 0.32 to 1.59); 1.06 over the 20 that are not fp64.
Measured over assumed price, all 49 conversions with a usable add unit: median 0.12, range 0.008 to 0.75.

### E4: alternative resolutions against the default

| Section | Target | Kernels with a conflict studied | Alternatives tried | Same program as the default | Different from the default | Distinct programs among those | of which the default's operations in another order | Do not compile | Distinct programs: more / same / fewer non-free conversions | Timed | Faster (3%, intervals apart, both runs) | Slower |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Tutorials | sm_80 | 12 | 768 | 126 | 642 | 12 | 12 | 0 | 0 / 12 / 0 | not runnable here |  |  |
| Tutorials | sm_90 | 12 | 768 | 282 | 72 | 24 | 0 | 414 | 0 / 24 / 0 | not runnable here |  |  |
| Tutorials | sm_100 | 12 | 768 | 300 | 468 | 198 | 0 | 0 | 198 / 0 / 0 | not runnable here |  |  |
| Tutorials | sm_120 | 15 | 960 | 960 | 0 | 0 | 0 | 0 | 0 / 0 / 0 | 0 | 0 | 0 |
| Inductor | sm_120 | 16 | 778 | 403 | 375 | 113 | 43 | 0 | 51 / 59 / 3 | 113 | 11 | 32 |
| triton_kernels | sm_80 | 5 | 320 | 110 | 210 | 67 | 51 | 0 | 6 / 61 / 0 | not runnable here |  |  |
| triton_kernels | sm_90 | 11 | 378 | 174 | 204 | 46 | 46 | 0 | 0 / 46 / 0 | not runnable here |  |  |
| triton_kernels | sm_100 | 12 | 411 | 256 | 155 | 32 | 30 | 0 | 0 / 32 / 0 | not runnable here |  |  |
| triton_kernels | sm_120 | 15 | 422 | 284 | 138 | 27 | 27 | 0 | 0 / 27 / 0 | 27 | 2 | 5 |

On the device: 140 alternatives timed on 25 kernels; 13 faster and 37 slower by the reporting rule. Over all 46 conflicted kernels studied, the reportable gap to the best alternative has median 0.0% and maximum 96.1%; counting only alternatives that change an operation or a layout, median 0.0% and maximum 11.6%.
