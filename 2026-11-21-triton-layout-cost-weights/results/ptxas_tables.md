## The same builds under other ptxas releases

Final TTGIR of 29 builds (12 configurations of `_matmul` and `_p_matmul`, the default and each distinct alternative), compiled again from the saved TTGIR with each release as Triton's assembler and assembled with Triton's flags for sm_120a. Nothing is launched. 0 builds failed to assemble.

Builds with a stack frame of 4 KB or more per thread: 12.8.93: 2; 12.9.86: 2; 13.0.88: 2; 13.1.80: 2; 13.3.73: 2; 13.5.37: 2. They are the same 2 builds under every release: `_matmul mxfp8×mxfp4 M128 #6` alternative, `_matmul fp8e5×mxfp4 M128 #8` default. Each is given 40 registers; every other build gets 164 to 255.

Registers per thread / stack frame per thread in bytes, for the configurations where two builds differ in either under some release:

| Configuration | Build | 12.8.93 | 12.9.86 | 13.0.88 | 13.1.80 | 13.3.73 | 13.5.37 |
|---|---|---|---|---|---|---|---|
| `_matmul fp8e5×mxfp4 M16 #1` | default | 255 / 584 | 255 / 584 | 255 / 584 | 255 / 592 | 255 / 696 | 255 / 696 |
| `_matmul fp8e5×mxfp4 M16 #1` | alternative all | 255 / 704 | 255 / 704 | 255 / 704 | 255 / 696 | 255 / 384 | 255 / 384 |
| `_matmul mxfp8×mxfp4 M128 #6` | default | 255 / 2,064 | 255 / 1,976 | 255 / 1,976 | 255 / 2,008 | 255 / 632 | 255 / 632 |
| `_matmul mxfp8×mxfp4 M128 #6` | alternative all | 40 / 10,904 | 40 / 10,944 | 40 / 10,944 | 40 / 11,080 | 40 / 9,544 | 40 / 9,544 |
| `_matmul fp8e5×mxfp4 M128 #8` | default | 40 / 9,648 | 40 / 8,352 | 40 / 8,352 | 40 / 8,392 | 40 / 7,880 | 40 / 7,880 |
| `_matmul fp8e5×mxfp4 M128 #8` | alternative all | 255 / 1,480 | 255 / 1,480 | 255 / 1,480 | 255 / 1,488 | 255 / 408 | 255 / 408 |
| `_p_matmul fp8e5×mxfp4 M128 #11` | default | 168 / 352 | 168 / 352 | 168 / 352 | 168 / 400 | 168 / 248 | 168 / 248 |
| `_p_matmul fp8e5×mxfp4 M128 #11` | alternative all | 168 / 1,040 | 168 / 1,040 | 168 / 1,040 | 168 / 1,128 | 168 / 432 | 168 / 432 |
| `_p_matmul fp8e5×mxfp4 M128 #12` | default | 168 / 384 | 168 / 384 | 168 / 384 | 168 / 408 | 168 / 256 | 168 / 256 |
| `_p_matmul fp8e5×mxfp4 M128 #12` | alternative all | 168 / 1,064 | 168 / 1,064 | 168 / 1,064 | 168 / 1,128 | 168 / 432 | 168 / 432 |
| `_p_matmul mxfp8×mxfp4 M128 #13` | default | 168 / 408 | 168 / 408 | 168 / 408 | 168 / 456 | 168 / 352 | 168 / 352 |
| `_p_matmul mxfp8×mxfp4 M128 #13` | alternative all | 168 / 1,072 | 168 / 1,072 | 168 / 1,072 | 168 / 1,128 | 168 / 560 | 168 / 560 |
| `_p_matmul mxfp8×mxfp4 M128 #14` | default | 168 / 440 | 168 / 440 | 168 / 440 | 168 / 480 | 168 / 448 | 168 / 448 |
| `_p_matmul mxfp8×mxfp4 M128 #14` | alternative all | 168 / 1,000 | 168 / 1,000 | 168 / 1,000 | 168 / 1,024 | 168 / 616 | 168 / 616 |
| `_p_matmul mxfp8×bf16 M128 #15` | default | 255 / 456 | 255 / 456 | 255 / 456 | 255 / 528 | 255 / 448 | 255 / 448 |
| `_p_matmul mxfp8×bf16 M128 #15` | alternative all | 255 / 432 | 255 / 432 | 255 / 432 | 255 / 464 | 255 / 432 | 255 / 432 |

Triton always passes `--regAllocOptLevel=2`. Without it, 10 of 174 assemblies get a different register count, and 0 move into or out of the group with a frame of 4 KB or more.

## Timed on the device with ptxas 12.9.86 against 13.5.37

The same harness, two runs each, the same reporting rule (3% or more, non-overlapping intervals, both runs, null control under 3%). Rows are the builds either release reports as different from the default; microseconds are the first run's medians and the percentage is the smaller of the two runs'.

### triton_kernels

15 conflicted kernels studied under 13.5.37, 15 under 12.9.86; 15 are the same launches.

| Kernel | Build | 13.5.37: default, µs | alternative, µs | against default | registers | 12.9.86: default, µs | alternative, µs | against default | registers |
|---|---|---|---|---|---|---|---|---|---|
| `_matmul fp8e5×mxfp4 M16 #1` | all | 185.69 | 178.52 | -3.9% | 255 → 255 | 186.76 | 218.65 | +16.8% | 255 → 255 |
| `_matmul mxfp8×mxfp4 M128 #6` | all | 20.94 | 124.34 | +492.5% | 255 → 40 | 30.35 | 151.79 | +400.2% | 255 → 40 |
| `_matmul fp8e5×mxfp4 M128 #8` | all | 3,168.59 | 122.77 | -96.1% | 40 → 255 | 3,691.17 | 244.41 | -93.4% | 40 → 255 |
| `_p_matmul fp8e5×mxfp4 M128 #11` | all | 113.12 | 129.11 | +14.0% | 168 → 168 | 126.49 | 275.93 | +117.1% | 168 → 168 |
| `_p_matmul fp8e5×mxfp4 M128 #12` | all | 50.14 | 55.44 | +10.6% | 168 → 168 | 54.65 | 89.25 | +63.3% | 168 → 168 |
| `_p_matmul mxfp8×mxfp4 M128 #13` | all | 114.01 | 135.93 | +19.2% | 168 → 168 | 128.29 | 256.82 | +100.2% | 168 → 168 |
| `_p_matmul mxfp8×mxfp4 M128 #14` | all | 146.72 | 163.70 | +11.6% | 168 → 168 | 145.52 | 268.12 | +83.8% | 168 → 168 |

Builds by what each release reports (13.5.37 / 12.9.86): faster / faster: 1; faster / slower: 1; none / none: 20; slower / slower: 5.
Null control under 13.5.37: largest 0.08% over 13 kernels.
Null control under 12.9.86: largest 0.57% over 13 kernels.

### Inductor

16 conflicted kernels studied under 13.5.37, 16 under 12.9.86; 15 are the same launches.
Not compared, because it was launched with another specialization under 12.9.86: `per_fused_embeddin…dense_backward_sum_12` (num_warps 2 then 8); its alternatives are within 0.9% of the default under both.

| Kernel | Build | 13.5.37: default, µs | alternative, µs | against default | registers | 12.9.86: default, µs | alternative, µs | against default | registers |
|---|---|---|---|---|---|---|---|---|---|
| `per_fused_add_embe…norm_backward_view_10` | all | 5.90 | 6.67 | +13.1% | 167 → 153 | 5.70 | 6.77 | +18.7% | 165 → 147 |
| `per_fused_add_embe…norm_backward_view_10` | inv4 | 5.90 | 6.67 | +13.1% | 167 → 153 | 5.70 | 6.77 | +18.7% | 165 → 147 |
| `per_fused__native_…t_functional_copy__13` | all | 0.81 | 0.84 | +3.2% | 35 → 29 | 0.81 | 0.84 | +3.2% | 35 → 29 |
| `per_fused__native_…t_functional_copy__13` | inv4 | 0.81 | 0.84 | +3.3% | 35 → 29 | 0.81 | 0.84 | +3.3% | 35 → 29 |
| `per_fused__native_…t_functional_copy__45` | inv4 | 1.08 | 1.12 | +3.9% | 40 → 40 | 1.08 | 1.12 | +3.9% | 40 → 40 |
| `per_fused__native_…t_functional_copy__48` | inv4 | 1.10 | 1.14 | +3.9% | 40 → 40 | 1.10 | 1.15 | +4.0% | 40 → 40 |
| `poi_fused__native_…threshold_backward_66` | all | 11.40 | 12.01 | +5.0% | 40 → 40 | 11.36 | 11.97 | +5.2% | 40 → 40 |
| `poi_fused__native_…threshold_backward_66` | inv4 | 11.40 | 12.00 | +4.5% | 40 → 40 | 11.36 | 11.97 | +5.1% | 40 → 40 |
| `poi_fused__native_…threshold_backward_66` | rand22 | 11.40 | 12.01 | +5.1% | 40 → 40 | 11.36 | 11.97 | +5.2% | 40 → 40 |
| `poi_fused__native_…threshold_backward_66` | rand4 | 11.40 | 12.00 | +5.0% | 40 → 40 | 11.36 | 11.97 | +5.2% | 40 → 40 |
| `poi_fused__native_…atch_norm_backward_71` | all | 11.85 | 12.51 | +5.5% | 40 → 40 | 12.10 | 12.53 | +3.5% | 40 → 40 |
| `poi_fused__native_…atch_norm_backward_71` | inv4 | 11.85 | 12.49 | +5.2% | 40 → 40 | 12.10 | 12.53 | +3.5% | 40 → 40 |
| `poi_fused__native_…atch_norm_backward_71` | rand14 | 11.85 | 12.25 | +3.4% | 40 → 40 | 12.10 | 12.30 | +1.6% (not reportable) | 40 → 40 |
| `poi_fused__native_…atch_norm_backward_71` | rand2 | 11.85 | 12.24 | +3.4% | 40 → 40 | 12.10 | 12.36 | +2.1% (not reportable) | 40 → 40 |
| `poi_fused__native_…atch_norm_backward_71` | rand30 | 11.85 | 12.50 | +5.5% | 40 → 40 | 12.10 | 12.53 | +3.5% | 40 → 40 |
| `poi_fused__native_…atch_norm_backward_71` | rand34 | 11.85 | 12.26 | +3.4% | 40 → 40 | 12.10 | 12.29 | +1.6% (not reportable) | 40 → 40 |
| `red_fused__native_…atch_norm_backward_69` | all | 16.94 | 16.42 | -3.1% (not reportable) | 40 → 40 | 16.96 | 16.32 | -3.8% | 40 → 40 |
| `red_fused__native_…atch_norm_backward_69` | inv4 | 16.94 | 16.45 | -2.9% (not reportable) | 40 → 40 | 16.96 | 16.33 | -3.7% | 40 → 40 |
| `red_fused__native_…atch_norm_backward_69` | rand11 | 16.94 | 17.57 | +3.7% | 40 → 43 | 16.96 | 17.59 | +3.5% | 40 → 43 |
| `red_fused__native_…atch_norm_backward_69` | rand14 | 16.94 | 16.41 | -3.0% | 40 → 40 | 16.96 | 16.36 | -3.6% | 40 → 40 |
| `red_fused__native_…atch_norm_backward_69` | rand17 | 16.94 | 16.42 | -3.1% | 40 → 40 | 16.96 | 16.38 | -3.5% | 40 → 40 |
| `red_fused__native_…atch_norm_backward_69` | rand19 | 16.94 | 16.41 | -3.1% | 40 → 40 | 16.96 | 16.37 | -3.5% | 40 → 40 |
| `red_fused__native_…atch_norm_backward_69` | rand21 | 16.94 | 17.57 | +3.7% | 40 → 43 | 16.96 | 17.59 | +3.4% | 40 → 43 |
| `red_fused__native_…atch_norm_backward_69` | rand22 | 16.94 | 16.41 | -3.1% (not reportable) | 40 → 40 | 16.96 | 16.33 | -3.7% | 40 → 40 |
| `red_fused__native_…atch_norm_backward_69` | rand3 | 16.94 | 17.59 | +3.8% | 40 → 43 | 16.96 | 17.60 | +3.5% | 40 → 43 |
| `red_fused__native_…atch_norm_backward_69` | rand30 | 16.94 | 16.42 | -3.0% | 40 → 40 | 16.96 | 16.32 | -3.8% | 40 → 40 |
| `red_fused__native_…atch_norm_backward_69` | rand4 | 16.94 | 16.42 | -3.1% | 40 → 40 | 16.96 | 16.35 | -3.6% | 40 → 40 |
| `red_fused__native_…atch_norm_backward_69` | rand9 | 16.94 | 16.44 | -3.0% (not reportable) | 40 → 40 | 16.96 | 16.37 | -3.5% | 40 → 40 |
| `red_fused_div_expa…shold_backward_view_0` | all | 2.46 | 2.15 | -11.6% | 40 → 40 | 2.34 | 2.05 | -11.0% | 40 → 40 |
| `red_fused_div_expa…shold_backward_view_0` | inv4 | 2.46 | 2.16 | -11.6% | 40 → 40 | 2.34 | 2.05 | -12.3% | 40 → 40 |
| `red_fused_div_expa…shold_backward_view_0` | one_4_8_1 | 2.46 | 2.27 | -7.2% | 40 → 40 | 2.34 | 2.18 | -6.8% | 40 → 40 |
| `red_fused_div_expa…shold_backward_view_0` | rand1 | 2.46 | 2.77 | +12.7% | 40 → 40 | 2.34 | 2.61 | +11.6% | 40 → 40 |
| `red_fused_div_expa…shold_backward_view_0` | rand10 | 2.46 | 2.26 | -6.2% | 40 → 40 | 2.34 | 2.18 | -6.8% (not reportable) | 40 → 40 |
| `red_fused_div_expa…shold_backward_view_0` | rand15 | 2.46 | 2.59 | +5.2% | 40 → 40 | 2.34 | 2.47 | +5.8% | 40 → 40 |
| `red_fused_div_expa…shold_backward_view_0` | rand17 | 2.46 | 2.60 | +5.7% | 40 → 40 | 2.34 | 2.48 | +6.0% (not reportable) | 40 → 40 |
| `red_fused_div_expa…shold_backward_view_0` | rand3 | 2.46 | 2.59 | +5.4% | 40 → 40 | 2.34 | 2.47 | +4.4% | 40 → 40 |
| `red_fused_div_expa…shold_backward_view_0` | rand5 | 2.46 | 2.32 | -5.7% | 40 → 40 | 2.34 | 2.18 | -6.8% (not reportable) | 40 → 40 |
| `red_fused_div_expa…shold_backward_view_0` | rand7 | 2.46 | 2.63 | +5.8% | 40 → 40 | 2.34 | 2.47 | +5.6% | 40 → 40 |
| `red_fused_div_expa…shold_backward_view_0` | rand8 | 2.46 | 2.19 | -10.8% | 40 → 40 | 2.34 | 2.05 | -7.8% | 40 → 40 |
| `red_fused_div_expa…shold_backward_view_0` | rand9 | 2.46 | 2.77 | +12.3% | 40 → 40 | 2.34 | 2.61 | +11.7% | 40 → 40 |
| `red_fused_div_expa…shold_backward_view_2` | all | 3.84 | 5.92 | +53.5% | 48 → 72 | 3.96 | 6.05 | +52.9% | 49 → 66 |
| `red_fused_div_expa…shold_backward_view_2` | inv4 | 3.84 | 5.92 | +53.5% | 48 → 72 | 3.96 | 6.05 | +52.9% | 49 → 66 |
| `red_fused_div_expa…shold_backward_view_2` | one_4_3_1 | 3.84 | 4.10 | +6.7% | 48 → 52 | 3.96 | 4.08 | +3.1% | 49 → 55 |
| `red_fused_div_expa…shold_backward_view_2` | one_4_4_1 | 3.84 | 4.52 | +17.1% | 48 → 52 | 3.96 | 4.53 | +14.5% | 49 → 58 |
| `red_fused_div_expa…shold_backward_view_2` | one_4_7_1 | 3.84 | 5.62 | +45.4% | 48 → 54 | 3.96 | 5.53 | +39.9% | 49 → 54 |
| `red_fused_div_expa…shold_backward_view_2` | rand1 | 3.84 | 5.35 | +38.5% | 48 → 59 | 3.96 | 5.31 | +34.2% | 49 → 56 |
| `red_fused_div_expa…shold_backward_view_2` | rand2 | 3.84 | 5.31 | +37.5% | 48 → 64 | 3.96 | 5.34 | +34.9% | 49 → 64 |

Builds by what each release reports (13.5.37 / 12.9.86): faster / faster: 9; faster / none: 2; none / faster: 4; none / none: 63; slower / none: 4; slower / slower: 28.
Null control under 13.5.37: largest 0.75% over 14 kernels.
Null control under 12.9.86: largest 6.29% over 14 kernels, 1 at 3% or more and left out.


## Other targets, assembled only

The builds saved by the code-only E4 runs, assembled the same way. Nothing here can be launched on this card. A cell is the number of builds with a stack frame of 4 KB or more, of those that assembled.

| Target | Configurations | Builds | 12.8.93 | 12.9.86 | 13.0.88 | 13.1.80 | 13.3.73 | 13.5.37 |
|---|---|---|---|---|---|---|---|---|
| sm_80 | 4 | 67 | 42 of 67 | 42 of 67 | 42 of 67 | 42 of 67 | 42 of 67 | 42 of 67 |
| sm_90 | 8 | 31 | 0 of 31 | 0 of 31 | 0 of 31 | 0 of 31 | 0 of 31 | 0 of 31 |
| sm_100 | 8 | 16 | 1 of 16 | 1 of 16 | 1 of 16 | 1 of 16 | 0 of 16 | 0 of 16 |

Which builds:

- sm_80 `_matmul mxfp4×bf16 M128 #4`: every build, the default and all alternatives, under every release (32 registers, frame 5,248 to 5,880 bytes)
- sm_80 `_matmul mxfp4×fp16 M16 #5`: every build, the default and all alternatives, under every release (32 registers, frame 7,624 to 7,744 bytes)
- sm_100 `_matmul fp8e5×mxfp4 M16 #1`: the default only, under 12.8.93, 12.9.86, 13.0.88, 13.1.80 (32 registers, frame 7,680 to 7,752 bytes; otherwise 255 registers)
