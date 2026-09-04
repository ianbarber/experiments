# Background: TLX and the Triton plugin ABI

Snapshot of four documents from `ianbarber/tritonext` at commit `2d0435d` (2026-07-13),
the investigation that established what the extension architecture delivers on upstream
Triton and which the kernel in `../code/` builds on. tritonext remains a live repo; these
copies are for reading this entry standalone.

| File | What |
|---|---|
| `WRITEUP.md` | Consolidated narrative: the plugin ABI is real; `async_dot` runs on upstream 3.8 *source* + `triton-utlx`, still not the stock pip wheel; warp-spec is broken on upstream (the gap `../code/kernel/tlx_upstream_patch.py` bridges); the dead ends, recorded |
| `HOWTO.md` | The two install paths (fbtriton wheel vs upstream source + plugin) and the hardware-gate table |
| `REPORT.md` | The original deep-dive: PR archaeology of the extension ABI, the µTLX install-over-Triton mechanism (namespace hijack, stage-replacement hook, compiler-dispatch monkey-patch), critical analysis, the 5090 dense-GEMM baseline |
| `docs/utlx-upstream-state.md` | June 2026 op-by-op / pass-by-pass inventory of µTLX against genuinely stock upstream, with the July status banner |

Not copied: the fbtriton 5090 GEMM demos and plugin probes (`benchmarks/`), the recovered
`triton-tlx-core-changes` fork patch (`patches/`, 184 KB), and the H100 runbook and
open-threads notes (`docs/`). Links from these files into those paths point at the
tritonext repo.
