# Code — NVFP4 GEMM on upstream Triton

The kernel and its support stack as packaged in the original repo
(`ianbarber/nvfp4-triton-extensions`, archived). Paths in `../REPORT.md` are relative to
this directory.

```
kernel/nvfp4_ws_ksplit.py     the GEMM: persistent, TMA producer (4 warps, 24 regs),
                              two N-split consumers (4 warps each, 232 regs), K-split
                              dot_scaled, explicit blocked layout for the B staging load
kernel/tlx_upstream_patch.py  runtime patches: visit_With dispatch for tlx.async_tasks,
                              GluonOpBuilder swap, WS codegen for ttg.warp_specialize,
                              explicit-layout local_load, compat bridges
patches/triton-ext-nvfp4.patch  triton-ext: upstream-op TMA wrappers + the warp-spec
                              register-restore pass
scripts/build_plugin.sh       clone triton-ext, apply the patch, build lib/libutlx.so
scripts/fetch_utlx.py         pull the pure-Python DSL out of the triton-utlx wheel
                              into utlx_py/ and apply two small patches
scripts/env.sh                TRITON_PLUGIN_PATHS / PYTHONPATH for a run
verify.py                     bit-exactness vs plain tl.dot_scaled
bench.py                      shape table (2048³, 4096³, 8192×8192×4096) or one shape
```

Prereqs: CUDA 13 toolkit, PyTorch (cu13x build), `pip install cmake ninja lit`, an sm_12x
GPU. Block sizes are tuned for GB10 / sm_121.

1. Build upstream Triton with the plugin ABI enabled (once):
   ```bash
   git clone https://github.com/triton-lang/triton && cd triton
   git checkout release/3.8.x           # or main; both validated
   pip install -r python/requirements.txt
   TRITON_EXT_ENABLED=ON pip install -e . --no-build-isolation
   ```
2. Build the plugin:
   ```bash
   export TRITON_SOURCE_DIR=/path/to/triton
   export TRITON_BUILD_DIR=$TRITON_SOURCE_DIR/build/cmake.linux-<arch>-cpython-3.12
   export LLVM_INSTALL_DIR=$HOME/.triton/llvm/llvm-<hash>-<platform>
   scripts/build_plugin.sh              # -> lib/libutlx.so
   ```
3. Fetch the TLX DSL: `python scripts/fetch_utlx.py` (→ `utlx_py/`).
4. Run:
   ```bash
   . scripts/env.sh
   python verify.py
   python bench.py                      # or: python bench.py 4096 4096 4096
   ```

Import order in your own drivers: `import utlx_plugin`, then `import tlx_upstream_patch`,
then the kernel. `build/`, `lib/`, `utlx_py/` and `third_party/` are gitignored.
