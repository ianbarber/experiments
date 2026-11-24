## Recompiled with repriced conversions (reprice_0.12)

| Section | Target | Configurations | Final TTGIR identical | Changed | of which more non-free conversions | same | fewer | more inside a loop | fewer inside a loop |
|---|---|---|---|---|---|---|---|---|---|
| triton_kernels | sm_120 | 653 | 516 | 137 | 78 | 19 | 40 | 5 | 40 |
| triton_kernels_fpsan | sm_120 | 125 | 125 | 0 | 0 | 0 | 0 | 0 | 0 |
| tutorials | sm_120 | 1591 | 1591 | 0 | 0 | 0 | 0 | 0 | 0 |

Kernels that changed: `_convert_blackwell_mx_scale` (66), `_p_matmul` (59), `_matmul` (12).
