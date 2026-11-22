## E3: measured cost of one conversion

Device: NVIDIA GeForce RTX 5090, 580.95.05, 3180 MHz, compute capability 12.0, CUDA runtime 13.2, 170 SMs. 25 timed runs per variant after 5 warm-up, 32 CTAs per SM.

### The same conversion through warp shuffles and through shared memory

| Conversion | bytes | warps | default route | shuffle: SM-ns | adds | shared memory: SM-ns (barrier) | adds | shuffle / shared memory | assumed (adds) |
|---|---|---|---|---|---|---|---|---|---|
| `survive00_inductor_shared_memory_16_f32` | 64 | 4 | shared memory | 10.4 | 9.7 | 9.6 (cta) | 9.0 | 1.09 | 64 |
| `survive01_inductor_shared_memory_1x1024_f32` | 4096 | 1 | shared memory | 22.9 | 8.3 | 21.1 (warp) | 7.6 | 1.09 | 32 |
| `survive02_inductor_shared_memory_8_f32` | 32 | 4 | shared memory | 10.4 | 9.8 | 8.9 (cta) | 8.3 | 1.17 | 128 |
| `priced06_inductor_shared_memory_16x1_f32` | 64 | 4 | shared memory | 10.5 | 9.7 | 9.6 (cta) | 8.9 | 1.09 | 64 |
| `survive07_inductor_warp_shuffle_2_f32` | 8 | 4 | warp shuffle | 2.3 | 4.3 | 7.3 (cta) | 13.8 | 0.32 | 512 |
| `priced11_inductor_warp_shuffle_2x1_f32` | 8 | 4 | warp shuffle | 2.3 | 4.3 | 7.3 (cta) | 13.8 | 0.32 | 512 |
| `priced22_triton_kernels_warp_shuffle_16x4_i8` | 64 | 4 | warp shuffle | 10.6 | 4.2 | 9.2 (cta) | 3.6 | 1.15 | 128 |
| `priced23_triton_kernels_warp_shuffle_16x8_i8` | 128 | 8 | warp shuffle | 36.5 | 4.1 | 31.9 (cta) | 3.6 | 1.14 | 128 |
| `priced24_triton_kernels_warp_shuffle_16x4_i8` | 64 | 4 | warp shuffle | 10.6 | 4.2 | 9.2 (cta) | 3.6 | 1.15 | 128 |
| `priced33_tutorials_warp_shuffle_64x1_f32` | 256 | 8 | warp shuffle | 18.3 | 4.0 | 23.4 (warp) | 5.1 | 0.78 | 32 |
| `canon_mma_to_dotop_64x32_f16` | 4096 | 4 | warp shuffle | 16.8 | 3.0 | 18.6 (warp) | 3.3 | 0.90 | 64 |
| `syn_inwarp_32x32_i8` | 1024 | 1 | warp shuffle | 14.2 | 4.7 | 12.1 (warp) | 4.0 | 1.17 | 128 |
| `syn_inwarp_32x64_i8` | 2048 | 1 | warp shuffle | 23.8 | 4.1 | 19.9 (warp) | 3.4 | 1.19 | 128 |
| `syn_inwarp_32x128_i8` | 4096 | 1 | warp shuffle | 49.8 | 4.2 | 41.9 (warp) | 3.5 | 1.19 | 128 |
| `syn_inwarp_32x32_f16` | 2048 | 1 | warp shuffle | 8.6 | 3.3 | 10.4 (warp) | 3.9 | 0.83 | 64 |
| `syn_inwarp_32x64_f16` | 4096 | 1 | warp shuffle | 17.4 | 3.3 | 19.0 (warp) | 3.6 | 0.92 | 64 |
| `syn_inwarp_32x128_f16` | 8192 | 1 | warp shuffle | 34.3 | 3.3 | 35.9 (warp) | 3.4 | 0.96 | 64 |
| `syn_inwarp_32x32_f32` | 4096 | 1 | warp shuffle | 20.2 | 7.2 | 21.3 (warp) | 7.6 | 0.95 | 32 |
| `syn_inwarp_32x64_f32` | 8192 | 1 | warp shuffle | 41.6 | 7.4 | 41.0 (warp) | 7.3 | 1.01 | 32 |
| `syn_inwarp_32x128_f32` | 16384 | 1 | warp shuffle | 83.6 | 7.2 | 81.5 (warp) | 7.0 | 1.03 | 32 |
| `syn_inwarp_32x32_f64` | 8192 | 1 | warp shuffle | 2.8 | n/a | 2.8 (warp) | n/a | 1.02 | 32 |
| `syn_inwarp_32x64_f64` | 16384 | 1 | warp shuffle | 5.2 | n/a | 8.5 (warp) | n/a | 0.62 | 32 |
| `syn_inwarp_32x128_f64` | 32768 | 1 | warp shuffle | 31.3 | n/a | 19.7 (warp) | n/a | 1.59 | 32 |

Shuffle cost over shared-memory cost for the same conversion, 23 conversions: median 1.03, range 0.32 to 1.59. Without the three fp64 cases, where the conversion is a small difference of two long loops: median 1.06, range 0.32 to 1.19.

### Default route, by class

| Conversion | source | dtype | bytes | warps | class (barrier, rounds) | SM-ns per conversion [95% interval] | SM cycles | adds [95% interval] | assumed (adds) | assumed (SM cycles) | measured / assumed |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `survive00_inductor_shared_memory_16_f32` | survive | f32 | 64 | 4 | shared memory (cta, 1+1) | 9.6 [9.5, 9.6] | 25 | 9.0 [8.9, 9.0] | 64 | 4 | 0.140 |
| `survive01_inductor_shared_memory_1x1024_f32` | survive | f32 | 4096 | 1 | shared memory (warp, 8+8) | 21.1 [21.1, 21.1] | 62 | 7.6 [7.6, 7.6] | 32 | 131 | 0.238 |
| `survive02_inductor_shared_memory_8_f32` | survive | f32 | 32 | 4 | shared memory (cta, 1+1) | 8.9 [8.9, 8.9] | 26 | 8.3 [8.3, 8.3] | 128 | 4 | 0.065 |
| `survive03_inductor_shared_memory_64_f32` | survive | f32 | 256 | 4 | shared memory (cta, 1+1) | 9.6 [9.6, 9.6] | 28 | 9.0 [9.0, 9.0] | 32 | 8 | 0.280 |
| `priced04_inductor_shared_memory_512_i64` | priced | i64 | 4096 | 8 | shared memory (cta, 1+1) | 18.6 [18.4, 18.7] | 55 | 4.4 [4.4, 4.4] | 32 | 131 | 0.138 |
| `priced05_inductor_shared_memory_1024_i64` | priced | i64 | 8192 | 4 | shared memory (cta, 1+1) | 34.9 [34.8, 35.0] | 100 | 4.1 [4.1, 4.1] | 32 | 262 | 0.128 |
| `priced06_inductor_shared_memory_16x1_f32` | priced | f32 | 64 | 4 | shared memory (cta, 1+1) | 9.6 [9.6, 9.6] | 28 | 8.9 [8.9, 9.0] | 64 | 4 | 0.140 |
| `survive07_inductor_warp_shuffle_2_f32` | survive | f32 | 8 | 4 | warp shuffle | 2.3 [2.3, 2.3] | 7 | 4.3 [4.3, 4.3] | 512 | 4 | 0.008 |
| `survive08_inductor_warp_shuffle_64_f32` | survive | f32 | 256 | 2 | mixed (cta, 1) | 4.3 [4.3, 4.3] | 12 | 15.6 [15.6, 15.7] | 32 | 8 | 0.489 |
| `survive09_inductor_warp_shuffle_64x1_f32` | survive | f32 | 256 | 2 | mixed (cta, 1) | 4.3 [4.3, 4.3] | 12 | 15.6 [15.6, 15.6] | 32 | 8 | 0.487 |
| `survive10_inductor_warp_shuffle_128_f32` | survive | f32 | 512 | 2 | mixed (cta, 1) | 5.8 [5.8, 5.8] | 17 | 9.3 [9.3, 9.3] | 32 | 16 | 0.290 |
| `priced11_inductor_warp_shuffle_2x1_f32` | priced | f32 | 8 | 4 | warp shuffle | 2.3 [2.3, 2.3] | 7 | 4.3 [4.3, 4.3] | 512 | 4 | 0.008 |
| `priced12_inductor_warp_shuffle_128x1_f32` | priced | f32 | 512 | 2 | mixed (cta, 1) | 5.8 [5.8, 5.8] | 17 | 9.3 [9.3, 9.3] | 32 | 16 | 0.290 |
| `survive13_triton_kernels_shared_memory_256_f32` | survive | f32 | 1024 | 8 | shared memory (cta, 1+1) | 47.0 [47.0, 47.0] | 138 | 7.9 [7.9, 7.9] | 32 | 33 | 0.248 |
| `survive14_triton_kernels_shared_memory_128x8_f32` | survive | f32 | 4096 | 8 | shared memory (cta, 2+2) | 96.0 [95.9, 96.1] | 283 | 7.4 [7.4, 7.4] | 32 | 131 | 0.232 |
| `survive15_triton_kernels_shared_memory_128x8_f32` | survive | f32 | 4096 | 8 | shared memory (cta, 1+1) | 22.7 [22.7, 22.8] | 67 | 6.4 [6.3, 6.4] | 32 | 131 | 0.199 |
| `survive16_triton_kernels_shared_memory_128x4_bf16` | survive | bf16 | 1024 | 8 | shared memory (cta, 1+1) | 30.7 [30.6, 30.7] | 90 | 2.6 [2.6, 2.6] | 64 | 66 | 0.040 |
| `priced17_triton_kernels_shared_memory_32x128_i64` | priced | i64 | 32768 | 4 | shared memory (cta, 1+1) | 20895.9 [20828.2, 20962.6] | 54998 | 24.0 [23.8, 24.2] | 32 | 1049 | 0.750 |
| `priced18_triton_kernels_shared_memory_64x128_i64` | priced | i64 | 65536 | 4 | shared memory (cta, 8+8) | 275.6 [273.8, 276.3] | 750 | 3.9 [3.9, 3.9] | 32 | 2097 | 0.122 |
| `priced19_triton_kernels_shared_memory_64x128_i8` | priced | i8 | 8192 | 4 | shared memory (cta, 4+4) | 83.6 [83.6, 83.8] | 246 | 3.3 [3.3, 3.3] | 128 | 1049 | 0.026 |
| `survive20_triton_kernels_warp_shuffle_16x8_f16` | survive | f16 | 256 | 2 | mixed (cta, 1) | 4.1 [4.1, 4.1] | 12 | 7.7 [7.7, 7.7] | 64 | 16 | 0.120 |
| `priced22_triton_kernels_warp_shuffle_16x4_i8` | priced | i8 | 64 | 4 | warp shuffle | 10.6 [10.6, 10.6] | 31 | 4.2 [4.2, 4.2] | 128 | 8 | 0.033 |
| `priced23_triton_kernels_warp_shuffle_16x8_i8` | priced | i8 | 128 | 8 | warp shuffle | 36.5 [36.5, 36.5] | 108 | 4.1 [4.1, 4.1] | 128 | 16 | 0.032 |
| `priced24_triton_kernels_warp_shuffle_16x4_i8` | priced | i8 | 64 | 4 | warp shuffle | 10.6 [10.6, 10.6] | 31 | 4.2 [4.2, 4.2] | 128 | 8 | 0.033 |
| `survive25_tutorials_shared_memory_128_f32` | survive | f32 | 512 | 4 | shared memory (cta, 1+1) | 10.3 [10.3, 10.3] | 30 | 9.6 [9.6, 9.7] | 32 | 16 | 0.301 |
| `survive26_tutorials_shared_memory_128_f32` | survive | f32 | 512 | 8 | shared memory (cta, 1+1) | 12.5 [12.5, 12.5] | 37 | 11.9 [11.9, 12.0] | 32 | 16 | 0.372 |
| `survive27_tutorials_shared_memory_64_f32` | survive | f32 | 256 | 4 | shared memory (cta, 1+1) | 12.1 [12.1, 12.1] | 36 | 7.7 [7.7, 7.7] | 32 | 8 | 0.240 |
| `survive28_tutorials_shared_memory_64_f32` | survive | f32 | 256 | 8 | shared memory (cta, 1+1) | 21.1 [21.1, 21.2] | 62 | 6.7 [6.7, 6.8] | 32 | 8 | 0.211 |
| `priced29_tutorials_shared_memory_128x128_i32` | priced | i32 | 65536 | 8 | shared memory (cta, 1+1) | 471.3 [470.9, 471.5] | 1375 | 6.0 [6.0, 6.0] | 32 | 2097 | 0.187 |
| `priced30_tutorials_shared_memory_128x128_i32` | priced | i32 | 65536 | 4 | shared memory (cta, 1+1) | 295.6 [295.4, 295.9] | 796 | 6.0 [6.0, 6.0] | 32 | 2097 | 0.187 |
| `priced31_tutorials_shared_memory_128x64_f16` | priced | f16 | 16384 | 4 | shared memory (cta, 4+4) | 81.2 [81.2, 81.3] | 239 | 3.7 [3.7, 3.7] | 64 | 1049 | 0.058 |
| `priced32_tutorials_shared_memory_128x64_f16` | priced | f16 | 16384 | 8 | shared memory (cta, 2+2) | 74.5 [74.5, 74.5] | 220 | 3.3 [3.3, 3.3] | 64 | 1049 | 0.052 |
| `priced33_tutorials_warp_shuffle_64x1_f32` | priced | f32 | 256 | 8 | warp shuffle | 18.3 [18.3, 18.3] | 54 | 4.0 [4.0, 4.0] | 32 | 8 | 0.125 |
| `canon_mma_to_blocked_128x128_f16` | canonical | f16 | 32768 | 4 | shared memory (cta, 4+4) | 151.9 [151.7, 152.0] | 415 | 3.4 [3.4, 3.4] | 64 | 2097 | 0.053 |
| `canon_mma_to_blocked_128x128_f32` | canonical | f32 | 65536 | 4 | shared memory (cta, 4+4) | 337.2 [337.0, 337.4] | 885 | 7.4 [7.3, 7.4] | 32 | 2097 | 0.230 |
| `canon_mma_to_dotop_64x32_f16` | canonical | f16 | 4096 | 4 | warp shuffle | 16.8 [16.8, 16.8] | 50 | 3.0 [3.0, 3.0] | 64 | 262 | 0.047 |
| `syn_inwarp_32x32_i8` | synthetic | i8 | 1024 | 1 | warp shuffle | 14.2 [14.2, 14.2] | 42 | 4.7 [4.7, 4.7] | 128 | 131 | 0.037 |
| `syn_inwarp_32x64_i8` | synthetic | i8 | 2048 | 1 | warp shuffle | 23.8 [23.7, 23.8] | 65 | 4.1 [4.1, 4.1] | 128 | 262 | 0.032 |
| `syn_inwarp_32x128_i8` | synthetic | i8 | 4096 | 1 | warp shuffle | 49.8 [49.7, 49.9] | 131 | 4.2 [4.2, 4.2] | 128 | 524 | 0.033 |
| `syn_inwarp_32x32_f16` | synthetic | f16 | 2048 | 1 | warp shuffle | 8.6 [8.6, 8.6] | 25 | 3.3 [3.3, 3.3] | 64 | 131 | 0.051 |
| `syn_inwarp_32x64_f16` | synthetic | f16 | 4096 | 1 | warp shuffle | 17.4 [17.4, 17.4] | 51 | 3.3 [3.3, 3.3] | 64 | 262 | 0.052 |
| `syn_inwarp_32x128_f16` | synthetic | f16 | 8192 | 1 | warp shuffle | 34.3 [34.3, 34.4] | 99 | 3.3 [3.3, 3.3] | 64 | 524 | 0.051 |
| `syn_inwarp_32x32_f32` | synthetic | f32 | 4096 | 1 | warp shuffle | 20.2 [20.2, 20.2] | 59 | 7.2 [7.2, 7.2] | 32 | 131 | 0.226 |
| `syn_inwarp_32x64_f32` | synthetic | f32 | 8192 | 1 | warp shuffle | 41.6 [41.5, 41.6] | 112 | 7.4 [7.4, 7.4] | 32 | 262 | 0.231 |
| `syn_inwarp_32x128_f32` | synthetic | f32 | 16384 | 1 | warp shuffle | 83.6 [83.3, 83.9] | 223 | 7.2 [7.1, 7.2] | 32 | 524 | 0.225 |
| `syn_inwarp_32x32_f64` | synthetic | f64 | 8192 | 1 | warp shuffle | 2.8 [2.8, 3.0] | 8 | n/a | 32 | 262 | n/a |
| `syn_inwarp_32x64_f64` | synthetic | f64 | 16384 | 1 | warp shuffle | 5.2 [4.2, 5.4] | 15 | n/a | 32 | 524 | n/a |
| `syn_inwarp_32x128_f64` | synthetic | f64 | 32768 | 1 | warp shuffle | 31.3 [30.9, 33.3] | 92 | n/a | 32 | 1049 | n/a |
| `syn_crosswarp_32x32_i8` | synthetic | i8 | 1024 | 4 | shared memory (cta, 1+1) | 163.6 [163.4, 163.8] | 482 | 14.4 [14.3, 14.4] | 128 | 131 | 0.112 |
| `syn_crosswarp_64x64_i8` | synthetic | i8 | 4096 | 4 | shared memory (cta, 1+1) | 228.3 [228.2, 228.5] | 664 | 9.7 [9.7, 9.7] | 128 | 524 | 0.076 |
| `syn_crosswarp_128x128_i8` | synthetic | i8 | 16384 | 4 | shared memory (cta, 1+1) | 401.6 [401.4, 401.8] | 1051 | 8.7 [8.7, 8.7] | 128 | 2097 | 0.068 |
| `syn_crosswarp_32x32_f16` | synthetic | f16 | 2048 | 4 | shared memory (cta, 1+1) | 38.6 [38.5, 38.6] | 112 | 3.5 [3.5, 3.5] | 64 | 131 | 0.055 |
| `syn_crosswarp_64x64_f16` | synthetic | f16 | 8192 | 4 | shared memory (cta, 1+1) | 72.4 [72.3, 72.4] | 213 | 3.3 [3.3, 3.3] | 64 | 524 | 0.051 |
| `syn_crosswarp_128x128_f16` | synthetic | f16 | 32768 | 4 | shared memory (cta, 1+1) | 141.6 [141.3, 141.9] | 371 | 3.2 [3.2, 3.3] | 64 | 2097 | 0.051 |
| `syn_crosswarp_32x32_f32` | synthetic | f32 | 4096 | 4 | shared memory (cta, 1+1) | 82.5 [82.5, 82.6] | 243 | 7.4 [7.4, 7.4] | 32 | 131 | 0.232 |
| `syn_crosswarp_64x64_f32` | synthetic | f32 | 16384 | 4 | shared memory (cta, 1+1) | 160.8 [160.5, 161.1] | 474 | 7.3 [7.3, 7.3] | 32 | 524 | 0.227 |
| `syn_crosswarp_128x128_f32` | synthetic | f32 | 65536 | 4 | shared memory (cta, 1+1) | 336.5 [335.7, 337.7] | 939 | 7.5 [7.5, 7.5] | 32 | 2097 | 0.234 |
| `syn_crosswarp_32x32_f64` | synthetic | f64 | 8192 | 4 | shared memory (cta, 1+1) | 3.6 [2.6, 5.3] (not separated) | 11 | n/a | 32 | 262 | n/a |
| `syn_crosswarp_64x64_f64` | synthetic | f64 | 32768 | 4 | shared memory (cta, 1+1) | 33.1 [32.3, 34.0] | 97 | n/a | 32 | 1049 | n/a |
| `survive_regperm_64x64_i32` | survive | i32 | 16384 | 4 | register permute | 0.8 [0.7, 0.9] (not separated) | 2 | 0.1 [0.1, 0.1] | 0 | 0 | n/a |
| `survive_regperm_64x64_f32` | survive | f32 | 16384 | 4 | register permute | 0.0 [0.0, 0.0] (not separated) | 0 | 0.0 [0.0, 0.0] | 0 | 0 | n/a |
| `survive_regperm_32x64_i32` | survive | i32 | 8192 | 2 | register permute | 0.4 [0.4, 0.5] (not separated) | 1 | 0.1 [0.1, 0.1] | 0 | 0 | n/a |
| `survive_regperm_2x512_i8` | survive | i8 | 1024 | 8 | register permute | 0.4 [0.4, 0.5] (not separated) | 1 | 0.1 [0.1, 0.1] | 0 | 0 | n/a |

### Summary: conversions of at least 1 KB, by class and element width

| Class | Element width | Cases | adds per conversion: median (range) | assumed | measured / assumed: median |
|---|---|---|---|---|---|
| register permute | 8 bits | 1 | 0.1 (0.1 to 0.1) | 0 | n/a |
| register permute | 32 bits | 3 | 0.1 (0.0 to 0.1) | 0 | n/a |
| shared memory (cta barrier) | 8 bits | 4 | 9.2 (3.3 to 14.4) | 128 | 0.072 |
| shared memory (cta barrier) | 16 bits | 7 | 3.3 (2.6 to 3.7) | 64 | 0.052 |
| shared memory (cta barrier) | 32 bits | 9 | 7.4 (6.0 to 7.9) | 32 | 0.230 |
| shared memory (cta barrier) | 64 bits | 4 | 4.3 (3.9 to 24.0) | 32 | 0.133 |
| shared memory (warp barrier) | 32 bits | 1 | 7.6 (7.6 to 7.6) | 32 | 0.238 |
| warp shuffle | 8 bits | 3 | 4.2 (4.1 to 4.7) | 128 | 0.033 |
| warp shuffle | 16 bits | 4 | 3.3 (3.0 to 3.3) | 64 | 0.051 |
| warp shuffle | 32 bits | 3 | 7.2 (7.2 to 7.4) | 32 | 0.226 |

### Measured over assumed price, by element width and size (input to the E2 reprice)

| Element width | Size | Cases | median | min | max |
|---|---|---|---|---|---|
| 16 bits | 1 KB and over | 11 | 0.051 | 0.040 | 0.058 |
| 32 bits | 1 KB and over | 13 | 0.230 | 0.187 | 0.248 |
| 32 bits | 32 elements to 1 KB | 6 | 0.260 | 0.125 | 0.372 |
| 32 bits | under 32 elements | 5 | 0.065 | 0.008 | 0.140 |
| 64 bits | 1 KB and over | 4 | 0.133 | 0.122 | 0.750 |
| 8 bits | 1 KB and over | 7 | 0.037 | 0.026 | 0.112 |
| 8 bits | 32 elements to 1 KB | 3 | 0.033 | 0.032 | 0.033 |

All 49 measured conversions: median 0.122, range 0.008 to 0.750.

### Cases that could not be measured

- `survive03_inductor_shared_memory_64_f32` (shuffle route): compile_error: RuntimeError: Parse MLIR file failed.
- `priced21_triton_kernels_warp_shuffle_128x2x1_f32`: ValueError: the harness handles 1-D and 2-D tensors only
- `syn_crosswarp_128x128_f64`: OutOfResources: out of resource: shared memory, Required: 131072, Hardware limit: 101376. Reducing block sizes or `num_stages` may help.
