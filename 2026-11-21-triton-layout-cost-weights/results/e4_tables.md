## E4: alternative conflict resolutions

| Section | Target | Kernels studied | Alternatives that exist (median per kernel) | Tried | Sampled kernels | Same program as the default | Different from the default | Distinct programs among those | of which the default's operations in another order | Fail to compile | Wrong output | Launch error | Output not checkable |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| inductor | sm_120 | 16 | 1,023 | 778 | 10 | 403 | 375 | 113 | 43 | 0 | 0 | 0 | 0 |
| triton_kernels | sm_80 | 5 | 268,435,455 | 320 | 5 | 110 | 210 | 67 | 51 | 0 | 0 | 0 | 0 |
| triton_kernels | sm_90 | 11 | 15 | 378 | 5 | 174 | 204 | 46 | 46 | 0 | 0 | 0 | 0 |
| triton_kernels | sm_100 | 12 | 39 | 411 | 5 | 256 | 155 | 32 | 30 | 0 | 0 | 0 | 0 |
| triton_kernels | sm_120 | 15 | 15 | 422 | 5 | 284 | 138 | 27 | 27 | 0 | 0 | 0 | 0 |
| tutorials | sm_80 | 12 | 2,303 | 768 | 12 | 126 | 642 | 12 | 12 | 0 | 0 | 0 | 0 |
| tutorials | sm_90 | 12 | 8,519,679 | 768 | 12 | 282 | 72 | 24 | 0 | 414 | 0 | 0 | 0 |
| tutorials | sm_100 | 12 | 33,816,575 | 768 | 12 | 300 | 468 | 198 | 0 | 0 | 0 | 0 | 0 |
| tutorials | sm_120 | 15 | 511 | 960 | 15 | 960 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

A build is the same program as another when their final TTGIR is equal up to debug locations and the names of SSA values. "Different from the default" counts alternatives; several of them usually produce the same program, and each distinct program is checked and timed once.

### Kernels where no alternative changed the code

| Section | Target | Kernels | of which: every alternative tried is the same program as the default |
|---|---|---|---|
| inductor | sm_120 | 16 | 2 |
| triton_kernels | sm_80 | 5 | 0 |
| triton_kernels | sm_90 | 11 | 6 |
| triton_kernels | sm_100 | 12 | 6 |
| triton_kernels | sm_120 | 15 | 4 |
| tutorials | sm_80 | 12 | 0 |
| tutorials | sm_90 | 12 | 0 |
| tutorials | sm_100 | 12 | 0 |
| tutorials | sm_120 | 15 | 15 |

### Different programs: conversions left in the final TTGIR against the default

| Section | Target | Distinct programs other than the default | fewer non-free conversions | same | more | fewer inside a loop | more inside a loop |
|---|---|---|---|---|---|---|---|
| inductor | sm_120 | 113 | 3 | 59 | 51 | 0 | 15 |
| triton_kernels | sm_80 | 67 | 0 | 61 | 6 | 0 | 0 |
| triton_kernels | sm_90 | 46 | 0 | 46 | 0 | 0 | 0 |
| triton_kernels | sm_100 | 32 | 0 | 32 | 0 | 0 | 0 |
| triton_kernels | sm_120 | 27 | 0 | 27 | 0 | 0 | 0 |
| tutorials | sm_80 | 12 | 0 | 12 | 0 | 0 | 0 |
| tutorials | sm_90 | 24 | 0 | 24 | 0 | 0 | 0 |
| tutorials | sm_100 | 198 | 0 | 0 | 198 | 0 | 126 |

### Timing on the device

| Section | Kernel | Conflicts | Different-code alternatives timed | Default, ms per launch [95% interval] | Best alternative vs default | Worst | Reportably faster | Reportably slower |
|---|---|---|---|---|---|---|---|---|
| inductor | `per_fused_add_embe…norm_backward_view_10` | 4 | 2 | 0.0059 [0.0059, 0.0059] | +13.1% | +13.1% | 0 | 2 |
| inductor | `per_fused_embeddin…dense_backward_sum_12` | 4 | 3 | 0.0019 [0.0019, 0.0019] | -0.9% | -0.9% | 0 | 0 |
| inductor | `per_fused__native_…t_functional_copy__13` | 10 | 2 | 0.0008 [0.0008, 0.0008] | +3.2% | +3.3% | 0 | 2 |
| inductor | `per_fused__native_…t_functional_copy__17` | 10 | 2 | 0.0010 [0.0010, 0.0010] | -0.8% | +0.3% | 0 | 0 |
| inductor | `per_fused__native_…t_functional_copy__22` | 10 | 2 | 0.0009 [0.0009, 0.0009] | +1.3% | +1.3% | 0 | 0 |
| inductor | `per_fused__native_…t_functional_copy__45` | 10 | 2 | 0.0011 [0.0011, 0.0011] | +1.3% | +3.9% | 0 | 1 |
| inductor | `per_fused__native_…t_functional_copy__48` | 10 | 2 | 0.0011 [0.0011, 0.0011] | +1.3% | +3.9% | 0 | 1 |
| inductor | `poi_fused__native_…threshold_backward_66` | 5 | 8 | 0.0114 [0.0114, 0.0115] | -0.2% | +5.1% | 0 | 4 |
| inductor | `poi_fused__native_…atch_norm_backward_71` | 13 | 12 | 0.0118 [0.0118, 0.0119] | -0.1% | +5.5% | 0 | 6 |
| inductor | `poi_fused_add_max_…h_indices_backward_65` | 4 | 1 | 0.0216 [0.0214, 0.0288] | +24.9% | +24.9% | 0 | 0 |
| inductor | `red_fused__native_…atch_norm_backward_69` | 27 | 30 | 0.0169 [0.0169, 0.0170] | -3.1% | +3.8% | 5 | 3 |
| inductor | `red_fused_add_div_…shold_backward_view_7` | 78 | 8 | 0.0047 [0.0047, 0.0047] | -0.1% | +0.1% | 0 | 0 |
| inductor | `red_fused_div_expa…shold_backward_view_0` | 39 | 24 | 0.0025 [0.0024, 0.0025] | -11.6% | +12.7% | 6 | 6 |
| inductor | `red_fused_div_expa…shold_backward_view_2` | 57 | 15 | 0.0038 [0.0038, 0.0038] | -0.0% | +53.5% | 0 | 7 |
| triton_kernels | `_matmul fp8e5×mxfp4 M16 #1` | 4 | 1 | 0.1857 [0.1852, 0.1858] | -3.9% | -3.9% | 1 | 0 |
| triton_kernels | `_matmul mxfp8×mxfp4 M128 #6` | 4 | 1 | 0.0209 [0.0209, 0.0209] | +492.5% | +492.5% | 0 | 1 |
| triton_kernels | `_matmul fp8e5×mxfp4 M128 #8` | 4 | 1 | 3.1686 [3.1670, 3.1701] | -96.1% | -96.1% | 1 | 0 |
| triton_kernels | `_p_matmul mxfp8×fp16 M16 #10` | 3 | 1 | 0.0028 [0.0028, 0.0028] | -0.3% | -0.3% | 0 | 0 |
| triton_kernels | `_p_matmul fp8e5×mxfp4 M128 #11` | 3 | 1 | 0.1131 [0.1131, 0.1140] | +14.0% | +14.0% | 0 | 1 |
| triton_kernels | `_p_matmul fp8e5×mxfp4 M128 #12` | 3 | 1 | 0.0501 [0.0501, 0.0501] | +10.6% | +10.6% | 0 | 1 |
| triton_kernels | `_p_matmul mxfp8×mxfp4 M128 #13` | 3 | 1 | 0.1140 [0.1140, 0.1142] | +19.2% | +19.2% | 0 | 1 |
| triton_kernels | `_p_matmul mxfp8×mxfp4 M128 #14` | 3 | 1 | 0.1467 [0.1466, 0.1468] | +11.6% | +11.6% | 0 | 1 |
| triton_kernels | `_p_matmul mxfp8×bf16 M128 #15` | 3 | 1 | 0.0084 [0.0083, 0.0084] | +0.4% | +0.4% | 0 | 0 |
| triton_kernels | `_upcast_from_mxfp fp8e5×mxfp4 #2` | 21 | 10 | 0.0083 [0.0082, 0.0083] | -0.3% | -0.0% | 0 | 0 |
| triton_kernels | `_upcast_from_mxfp bf16×mxfp8 #5` | 21 | 8 | 0.0045 [0.0045, 0.0045] | -0.0% | -0.0% | 0 | 0 |

140 alternatives with different code were timed on 25 kernels. 13 are faster than the default by 3% or more with non-overlapping intervals, on 4 kernels; 37 are slower by the same test. Best alternative per kernel: median -0.0%, most favourable -96.1%.
Of the faster ones, 11 change an operation or a layout (on 2 kernels): most favourable -11.6%, inductor `red_fused_div_expa…shold_backward_view_0`.
Of the faster ones, 2 hold the default's operations on the same layouts in another order (on 2 kernels): most favourable -96.1%, triton_kernels `_matmul fp8e5×mxfp4 M128 #8`.

Over all 46 conflicted kernels studied on the device (identical-code alternatives count as no gap): reportable gap to the best alternative, median 0.0%, maximum 96.1%. Counting only alternatives that change an operation or a layout: median 0.0%, maximum 11.6%.

### Replication and null control

Studies on the device were run twice in separate processes. Only differences both runs report are counted, at the smaller of the two sizes.

| Study | Builds timed in both runs | Faster: first run | both runs | Slower: first run | both runs |
|---|---|---|---|---|---|
| inductor | 112 | 13 | 11 | 32 | 32 |
| inductor_reprice | 14 | 4 | 4 | 6 | 6 |
| triton_kernels | 27 | 2 | 2 | 5 | 5 |
| triton_kernels_reprice | 7 | 0 | 0 | 2 | 2 |

Null control (the default built a second time and timed as if it were an alternative), 98 kernels: median absolute difference 0.01%, largest 6.71%; 0 would have been reported as a difference.

A second null control: 11 builds that the harness timed as different code are the default with its values renamed. Largest difference from the default 0.37%; 0 would have been reported.

Kernels left out because their null control moved by 3% or more in either run: 1 (`triton_poi_fused_add_max_pool2d_with_indices_max_pool2d_with_indices_backward_65`), in 2 studies (alternative resolutions and repriced builds are separate studies).

### Distinct programs: what differs from the default, and the timing verdict

| Section | Against the default | Faster | No reportable difference | Slower |
|---|---|---|---|---|
| inductor | same operations on the same layouts, in a different order | 0 | 43 | 0 |
| inductor | different operations, fewer non-free conversions | 0 | 0 | 2 |
| inductor | different operations, as many non-free conversions | 5 | 8 | 3 |
| inductor | different operations, more non-free conversions | 6 | 18 | 27 |
| triton_kernels | same operations on the same layouts, in a different order | 2 | 20 | 5 |

### Every reportable difference

| Section | Kernel | Alternative | vs default (smaller of two runs) | Default, µs | Alternative, µs | Non-free conversions, default → alternative (in a loop) | Same operations reordered | Registers per thread | Stack frame, bytes |
|---|---|---|---|---|---|---|---|---|---|
| inductor | `red_fused_div_expa…shold_backward_view_0` | all | -11.6% | 2.46 | 2.15 | 2 → 3 (1 → 2) | no | 40 → 40 | 0 → 0 |
| inductor | `red_fused_div_expa…shold_backward_view_0` | inv4 | -11.6% | 2.46 | 2.16 | 2 → 3 (1 → 2) | no | 40 → 40 | 0 → 0 |
| inductor | `red_fused_div_expa…shold_backward_view_0` | rand8 | -10.8% | 2.46 | 2.19 | 2 → 3 (1 → 2) | no | 40 → 40 | 0 → 0 |
| inductor | `red_fused_div_expa…shold_backward_view_0` | one_4_8_1 | -7.2% | 2.46 | 2.27 | 2 → 3 (1 → 2) | no | 40 → 40 | 0 → 0 |
| inductor | `red_fused_div_expa…shold_backward_view_0` | rand10 | -6.2% | 2.46 | 2.26 | 2 → 3 (1 → 2) | no | 40 → 40 | 0 → 0 |
| inductor | `red_fused_div_expa…shold_backward_view_0` | rand5 | -5.7% | 2.46 | 2.32 | 2 → 3 (1 → 2) | no | 40 → 40 | 0 → 0 |
| inductor | `red_fused__native_…atch_norm_backward_69` | rand19 | -3.1% | 16.94 | 16.41 | 2 → 2 (0 → 0) | no | 40 → 40 | 0 → 0 |
| inductor | `red_fused__native_…atch_norm_backward_69` | rand4 | -3.1% | 16.94 | 16.42 | 2 → 2 (0 → 0) | no | 40 → 40 | 0 → 0 |
| inductor | `red_fused__native_…atch_norm_backward_69` | rand17 | -3.1% | 16.94 | 16.42 | 2 → 2 (0 → 0) | no | 40 → 40 | 0 → 0 |
| inductor | `red_fused__native_…atch_norm_backward_69` | rand14 | -3.0% | 16.94 | 16.41 | 2 → 2 (0 → 0) | no | 40 → 40 | 0 → 0 |
| inductor | `red_fused__native_…atch_norm_backward_69` | rand30 | -3.0% | 16.94 | 16.42 | 2 → 2 (0 → 0) | no | 40 → 40 | 0 → 0 |
| inductor | `per_fused__native_…t_functional_copy__13` | all | +3.2% | 0.81 | 0.84 | 4 → 5 (0 → 0) | no | 35 → 29 | 0 → 0 |
| inductor | `per_fused__native_…t_functional_copy__13` | inv4 | +3.3% | 0.81 | 0.84 | 4 → 5 (0 → 0) | no | 35 → 29 | 0 → 0 |
| inductor | `poi_fused__native_…atch_norm_backward_71` | rand2 | +3.4% | 11.85 | 12.24 | 1 → 2 (0 → 0) | no | 40 → 40 | 0 → 0 |
| inductor | `poi_fused__native_…atch_norm_backward_71` | rand14 | +3.4% | 11.85 | 12.25 | 1 → 2 (0 → 0) | no | 40 → 40 | 0 → 0 |
| inductor | `poi_fused__native_…atch_norm_backward_71` | rand34 | +3.4% | 11.85 | 12.26 | 1 → 2 (0 → 0) | no | 40 → 40 | 0 → 0 |
| inductor | `red_fused__native_…atch_norm_backward_69` | rand11 | +3.7% | 16.94 | 17.57 | 2 → 2 (0 → 0) | no | 40 → 43 | 0 → 0 |
| inductor | `red_fused__native_…atch_norm_backward_69` | rand21 | +3.7% | 16.94 | 17.57 | 2 → 2 (0 → 0) | no | 40 → 43 | 0 → 0 |
| inductor | `red_fused__native_…atch_norm_backward_69` | rand3 | +3.8% | 16.94 | 17.59 | 2 → 2 (0 → 0) | no | 40 → 43 | 0 → 0 |
| inductor | `per_fused__native_…t_functional_copy__45` | inv4 | +3.9% | 1.08 | 1.12 | 4 → 5 (0 → 0) | no | 40 → 40 | 0 → 0 |
| inductor | `per_fused__native_…t_functional_copy__48` | inv4 | +3.9% | 1.10 | 1.14 | 4 → 5 (0 → 0) | no | 40 → 40 | 0 → 0 |
| inductor | `poi_fused__native_…threshold_backward_66` | inv4 | +4.5% | 11.40 | 12.00 | 1 → 2 (0 → 0) | no | 40 → 40 | 0 → 0 |
| inductor | `poi_fused__native_…threshold_backward_66` | all | +5.0% | 11.40 | 12.01 | 1 → 2 (0 → 0) | no | 40 → 40 | 0 → 0 |
| inductor | `poi_fused__native_…threshold_backward_66` | rand4 | +5.0% | 11.40 | 12.00 | 1 → 2 (0 → 0) | no | 40 → 40 | 0 → 0 |
| inductor | `poi_fused__native_…threshold_backward_66` | rand22 | +5.1% | 11.40 | 12.01 | 1 → 2 (0 → 0) | no | 40 → 40 | 0 → 0 |
| inductor | `poi_fused__native_…atch_norm_backward_71` | inv4 | +5.2% | 11.85 | 12.49 | 1 → 2 (0 → 0) | no | 40 → 40 | 0 → 0 |
| inductor | `red_fused_div_expa…shold_backward_view_0` | rand15 | +5.2% | 2.46 | 2.59 | 2 → 4 (1 → 3) | no | 40 → 40 | 0 → 0 |
| inductor | `red_fused_div_expa…shold_backward_view_0` | rand3 | +5.4% | 2.46 | 2.59 | 2 → 4 (1 → 3) | no | 40 → 40 | 0 → 0 |
| inductor | `poi_fused__native_…atch_norm_backward_71` | rand30 | +5.5% | 11.85 | 12.50 | 1 → 2 (0 → 0) | no | 40 → 40 | 0 → 0 |
| inductor | `poi_fused__native_…atch_norm_backward_71` | all | +5.5% | 11.85 | 12.51 | 1 → 2 (0 → 0) | no | 40 → 40 | 0 → 0 |
| inductor | `red_fused_div_expa…shold_backward_view_0` | rand17 | +5.7% | 2.46 | 2.60 | 2 → 3 (1 → 2) | no | 40 → 40 | 0 → 0 |
| inductor | `red_fused_div_expa…shold_backward_view_0` | rand7 | +5.8% | 2.46 | 2.63 | 2 → 3 (1 → 2) | no | 40 → 40 | 0 → 0 |
| inductor | `red_fused_div_expa…shold_backward_view_2` | one_4_3_1 | +6.7% | 3.84 | 4.10 | 3 → 4 (0 → 0) | no | 48 → 52 | 0 → 0 |
| inductor | `red_fused_div_expa…shold_backward_view_0` | rand9 | +12.3% | 2.46 | 2.77 | 2 → 4 (1 → 3) | no | 40 → 40 | 0 → 0 |
| inductor | `red_fused_div_expa…shold_backward_view_0` | rand1 | +12.7% | 2.46 | 2.77 | 2 → 4 (1 → 3) | no | 40 → 40 | 0 → 0 |
| inductor | `per_fused_add_embe…norm_backward_view_10` | inv4 | +13.1% | 5.90 | 6.67 | 3 → 2 (0 → 0) | no | 167 → 153 | 0 → 0 |
| inductor | `per_fused_add_embe…norm_backward_view_10` | all | +13.1% | 5.90 | 6.67 | 3 → 2 (0 → 0) | no | 167 → 153 | 0 → 0 |
| inductor | `red_fused_div_expa…shold_backward_view_2` | one_4_4_1 | +17.1% | 3.84 | 4.52 | 3 → 4 (0 → 0) | no | 48 → 52 | 0 → 0 |
| inductor | `red_fused_div_expa…shold_backward_view_2` | rand2 | +37.5% | 3.84 | 5.31 | 3 → 6 (0 → 0) | no | 48 → 64 | 0 → 0 |
| inductor | `red_fused_div_expa…shold_backward_view_2` | rand1 | +38.5% | 3.84 | 5.35 | 3 → 5 (0 → 0) | no | 48 → 59 | 0 → 0 |
| inductor | `red_fused_div_expa…shold_backward_view_2` | one_4_7_1 | +45.4% | 3.84 | 5.62 | 3 → 5 (0 → 0) | no | 48 → 54 | 0 → 0 |
| inductor | `red_fused_div_expa…shold_backward_view_2` | inv4 | +53.5% | 3.84 | 5.92 | 3 → 5 (0 → 0) | no | 48 → 72 | 0 → 0 |
| inductor | `red_fused_div_expa…shold_backward_view_2` | all | +53.5% | 3.84 | 5.92 | 3 → 5 (0 → 0) | no | 48 → 72 | 0 → 0 |
| triton_kernels | `_matmul fp8e5×mxfp4 M128 #8` | all | -96.1% | 3168.59 | 122.77 | 3 → 3 (1 → 1) | yes | 40 → 255 | 7,880 → 408 |
| triton_kernels | `_matmul fp8e5×mxfp4 M16 #1` | all | -3.9% | 185.69 | 178.52 | 3 → 3 (1 → 1) | yes | 255 → 255 | 696 → 384 |
| triton_kernels | `_p_matmul fp8e5×mxfp4 M128 #12` | all | +10.6% | 50.14 | 55.44 | 1 → 1 (1 → 1) | yes | 168 → 168 | 256 → 432 |
| triton_kernels | `_p_matmul mxfp8×mxfp4 M128 #14` | all | +11.6% | 146.72 | 163.70 | 3 → 3 (3 → 3) | yes | 168 → 168 | 448 → 616 |
| triton_kernels | `_p_matmul fp8e5×mxfp4 M128 #11` | all | +14.0% | 113.12 | 129.11 | 1 → 1 (1 → 1) | yes | 168 → 168 | 248 → 432 |
| triton_kernels | `_p_matmul mxfp8×mxfp4 M128 #13` | all | +19.2% | 114.01 | 135.93 | 2 → 2 (2 → 2) | yes | 168 → 168 | 352 → 560 |
| triton_kernels | `_matmul mxfp8×mxfp4 M128 #6` | all | +492.5% | 20.94 | 124.34 | 5 → 5 (2 → 2) | yes | 255 → 40 | 632 → 9,544 |

Registers and stack frame are what the CUDA driver reports when the binary is loaded: registers per thread, and the per-thread local stack frame in bytes (Triton's `n_spills` times four).

### Rebuilt with repriced conversions, on the device

| Section | Kernels rebuilt | Same code | Different code | Timed (stable timing) | Faster (3%, intervals apart, both runs) | Slower | Median difference | Range |
|---|---|---|---|---|---|---|---|---|
| inductor | 198 | 183 | 15 | 14 | 4 | 6 | +0.0% | -8.5% to +45.3% |
| triton_kernels | 47 | 40 | 7 | 7 | 0 | 2 | +2.4% | -0.0% to +7.9% |

