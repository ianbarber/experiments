# Code

Everything needed to rebuild the instrumented compiler and rerun E1 to E4. Paths below are
relative to this directory. `$TRITON_ROOT` is a checkout of Triton and `$RUNS` any scratch
directory for raw output (several GB; not committed).

## Environment

- One NVIDIA GPU. The numbers in this entry are from an RTX 5090 (compute capability 12.0),
  driver 580.95.05, CUDA runtime 13.2.
- Python 3.12, `uv`, CMake 3.20 to 3.x, Ninja, a C++17 compiler.
- Triton at `359c5be728f73c4cbdf51d71e516f6502f8e29a8`, built from source with its pinned
  prebuilt LLVM, plus the two patches in `patches/`. The assembler is the ptxas that commit
  bundles, release 13.5 (V13.5.37).
- torch 2.15.0 (cu132), torchvision, transformers (models are built from configs, no weights
  are downloaded), numpy, matplotlib, pytest, pytest-timeout.

```bash
git clone https://github.com/triton-lang/triton.git $TRITON_ROOT
cd $TRITON_ROOT && git checkout 359c5be728f73c4cbdf51d71e516f6502f8e29a8
git apply /path/to/patches/0001-layout-logging.patch
git apply /path/to/patches/0002-layout-overrides.patch

uv venv --python 3.12 venv && source venv/bin/activate
uv pip install "cmake>=3.20,<4.0" "ninja>=1.11.1" "nanobind==2.10.2" setuptools wheel pybind11 lit pip
TRITON_BUILD_PROTON=OFF python -m pip install -e . --no-build-isolation      # about 16 minutes on 16 cores
uv pip install torch torchvision transformers numpy matplotlib pytest pytest-timeout
ninja -C build/cmake.linux-x86_64-cpython-3.12 triton-opt                    # for the lit sanity set
```

torch 2.15 pins `triton==3.9.0`, which the source build satisfies, so installing torch
after Triton keeps the source build. Do not run Python from the directory that contains
the `triton` checkout: Python will import the checkout's root as a namespace package.

## The patches

`patches/0001-layout-logging.patch` is read-only. With `TRITON_LAYOUT_LOG=<file>` set, the
compiler appends one JSON object per line: `conflict`, `convert_cost`, `remat_decision`,
`remat_skip`, `uncosted_hoist`, `census`, `lowered`, `lowered_smem_detail`. Each carries
`TRITON_LAYOUT_TAG` (set a new value before every compile) and the index of the
`RemoveLayoutConversions` invocation within that compile. `ir.module.log_layout_census(stage)`
is exposed to Python for the census of the final TTGIR.

`patches/0002-layout-overrides.patch` adds three switches, each inert unless set:

| Variable | Effect |
|---|---|
| `TRITON_LAYOUT_RESOLVE_OVERRIDE="1:0=2,3:1=0"` | at pass invocation 1, conflict 0 takes candidate 2; at invocation 3, conflict 1 takes candidate 0 |
| `TRITON_LAYOUT_DISABLE_WARP_SHUFFLE=1` | conversions that would lower to warp shuffles go through shared memory, unless `force_warp_shuffle` is on the op |
| `TRITON_LAYOUT_REPRICE="shared_memory=0:3.84,warp_shuffle=0:3.84"` | `getConvertCost` for that class (or `class:barrier`) becomes `fixed + perByte * pricedBytes` |

The invocation index restarts when the tag changes, so a driver that uses the conflict
override must set a fresh `TRITON_LAYOUT_TAG` for each compile. None of these variables is
in Triton's cache key: compile with `TRITON_ALWAYS_COMPILE=1` or a fresh cache directory.

## E1 and E2: the census

```bash
export TRITON_ROOT=...            # the patched checkout
./sweep_tutorials.sh $RUNS/census                        # tutorials, four targets, about 12 minutes with JOBS=8

# Inductor: run on the device, recompile every kernel for the other targets
for m in resnet50 vit_b_16 bert_base gpt2; do
  TORCHINDUCTOR_COMPILE_THREADS=1 TORCHINDUCTOR_FX_GRAPH_CACHE=0 TORCHINDUCTOR_AUTOGRAD_CACHE=0 \
    python run_census.py --workload inductor:$m --launch --replay 80,90,100 --out $RUNS/census/inductor_$m
done

# triton_kernels matmul suite, one process per target, compile only
for arch in 80 90 100 120; do
  (cd $TRITON_ROOT/python/triton_kernels && \
   PYTHONPATH=/path/to/this/dir:$TRITON_ROOT/python/triton_kernels \
   LAYOUTLAB_WORKLOAD=tk LAYOUTLAB_OUT=$RUNS/census/tk_sm$arch LAYOUTLAB_ARCH=$arch LAYOUTLAB_SAMPLE=test_op=12 \
   python -m pytest -p layoutlab.pytest_plugin tests/test_matmul.py -q --timeout=300 -p no:cacheprovider)
done

# lit sanity set
python run_lit_census.py --triton-root $TRITON_ROOT \
  --triton-opt $TRITON_ROOT/build/cmake.linux-x86_64-cpython-3.12/bin/triton-opt --out $RUNS/census/lit

python analyze_census.py --runs $RUNS/census --out ../results --ratios ../results/e3_ratios.json \
  [--vanilla $RUNS/vanilla] > ../results/e1_e2_tables.md
```

In census mode a kernel is compiled as far as LLVM IR and never assembled or launched, and
the compiler and `torch.cuda.get_device_capability` are pointed at the requested target.
Test assertions fail in that mode because outputs are never written; that is expected.

The interference check needs a second, unpatched build. Copy the unpatched
`python/triton` tree somewhere before applying the patches, then
`PYTHONPATH=<copy's parent> ./sweep_tutorials.sh $RUNS/vanilla --no-events`.

## E3: cost per class

```bash
python make_e3_cases.py --runs $RUNS/census --arch 120 --out $RUNS/e3/cases.json
python e3_bench.py --cases $RUNS/e3/cases.json --out $RUNS/e3 --runs 25          # about 7 minutes
python analyze_e3.py --run $RUNS/e3 --out ../results > ../results/e3_tables.md
```

The four register-permute cases in `../results/e3_cases_register_permute.json` were run as
a second batch into the same output directory.

## Recompiling the corpus with the measured price

```bash
RP="shared_memory=0:3.84,warp_shuffle=0:3.84"      # 0.12 of the assumed 32 per byte
for w in tut02 tut03 tut05 tut06 tut08 tut09; do   # tutorial 10 has no sm_120 configurations
  TRITON_LAYOUT_REPRICE="$RP" python run_census.py --workload $w --arch 120 --out $RUNS/reprice_0.12/${w}_sm120
done
# and the triton_kernels census command above with TRITON_LAYOUT_REPRICE="$RP",
# LAYOUTLAB_ARCH=120 and LAYOUTLAB_OUT=$RUNS/reprice_0.12/tk_sm120
python compare_reprice.py --base $RUNS/census --repriced $RUNS/reprice_0.12 --out ../results \
  --name reprice_0.12 > ../results/reprice_0.12_tables.md
```

The Inductor kernels are repriced as part of E4, where they are also timed.

## E4: alternative resolutions

Every study on the device is run twice, in separate processes, into `<name>_sm120` and
`<name>_run2_sm120`. The analysis reports a difference only where both runs do.

```bash
RP="shared_memory=0:3.84,warp_shuffle=0:3.84"      # the measured price, 0.12 of the assumed one

python e4_bench.py --workload tut06 --out $RUNS/e4/tut06_sm120
# Inductor has to compile in this process and without its caches, as in the census: a kernel
# built in a compile worker or restored from a cache comes back without its Python source,
# and every study then fails with "'NoneType' object has no attribute '__closure__'"
export TORCHINDUCTOR_COMPILE_THREADS=1 TORCHINDUCTOR_FX_GRAPH_CACHE=0 TORCHINDUCTOR_AUTOGRAD_CACHE=0
for run in "" _run2; do                                # about 7 and 4 minutes a run
  python e4_bench.py --workload inductor --out $RUNS/e4/inductor${run}_sm120 --runs 30 --warmup 5
  python e4_bench.py --workload inductor --reprice "$RP" --out $RUNS/e4/inductor_reprice${run}_sm120 \
    --runs 30 --warmup 5
done

# triton_kernels: the first 430 sampled tests reach six conflicted configurations each of
# _matmul and _p_matmul and three of _upcast_from_mxfp, which is the six-per-kernel limit
# or all there are. Repriced builds go to tk<run>_reprice_sm120. About 11 minutes a run.
for run in "" _run2; do
  (cd $TRITON_ROOT/python/triton_kernels && \
   PYTHONPATH=/path/to/this/dir:$TRITON_ROOT/python/triton_kernels \
   LAYOUTLAB_E4_OUT=$RUNS/e4/tk${run}_sm120 LAYOUTLAB_SAMPLE=test_op=12 LAYOUTLAB_E4_PER_KERNEL=6 \
   LAYOUTLAB_E4_MAX_TESTS=430 LAYOUTLAB_E4_REPRICE="$RP" \
   python -m pytest -p layoutlab.e4_plugin tests/test_matmul.py -q --timeout=1800 -p no:cacheprovider)
done

# other targets: nothing is launched, the generated code is compared
for arch in 80 90 100; do
  python e4_bench.py --workload tut06 --arch $arch --out $RUNS/e4/tut06_sm$arch
  # and the pytest command above with LAYOUTLAB_ARCH=$arch LAYOUTLAB_E4_OUT=$RUNS/e4/tk_sm$arch
done

python census_from_ttgir.py --runs $RUNS/e4       # conversions left in each saved TTGIR
python analyze_e4.py --runs $RUNS/e4 --out ../results > ../results/e4_tables.md
```

The arguments of a launch are captured after the kernel has run, so an output tensor may
already hold the right answer. Before an alternative is compared with the default, the
harness finds the tensors the kernel writes. A tensor that changes when the kernel is
launched from the captured state checks itself. A tensor that does not, but changes when
it is overwritten first, is overwritten before every launch (floating-point and 8-bit
tensors only: wider integers may be indices). A study with neither is marked
`unverified_output` and not timed. The harness compares TTGIR as text; `analyze_e4.py`
then treats builds that differ only in the names of SSA values as the same program, using
the digests `census_from_ttgir.py` writes.

Run E3 and the device-target E4 with nothing else on the GPU.

## The same builds under other ptxas releases

`$PTXAS` holds one directory per release, named by its version, each with that release's
`ptxas` in it. The six used here: 12.9.86 from the CUDA 12.9 toolkit, 12.8.93 and 13.1.80
from the Triton 3.7.0 and 3.6.0 wheels (`ptxas` and `ptxas-blackwell`), 13.0.88 from the
torch 2.12.0+cu130 wheel, 13.3.73 from the `nvidia-cuda-nvcc` wheel of that version, and
the bundled 13.5.37.

```bash
# registers and stack frame of every saved _matmul and _p_matmul build; nothing is launched
for arch in 80 90 100 120; do
  python ptxas_versions.py --ttgir $RUNS/e4/tk_sm$arch/ttgir --ptxas-root $PTXAS --match _matmul_ _p_matmul_ \
    --arch $arch --jobs 6 $([ $arch = 120 ] && echo --no-regalloc-flag) --out $RUNS/ptxas_versions/tk_sm$arch.json
done

# the device studies again with 12.9.86 as Triton's assembler: the triton_kernels and
# Inductor commands of the E4 section, without the repriced builds, with
export TRITON_PTXAS_PATH=$PTXAS/12.9.86/ptxas TRITON_CACHE_DIR=/path/to/an/empty/directory
# and output in $RUNS/e4_ptxas12.9/{tk,tk_run2,inductor,inductor_run2}_sm120

python census_from_ttgir.py --runs $RUNS/e4_ptxas12.9
python analyze_e4.py --runs $RUNS/e4_ptxas12.9 --out /tmp/ptxas12.9 > /dev/null
python analyze_ptxas.py --versions $RUNS/ptxas_versions/tk_sm120.json \
  --base ../results/e4_summary.json --base-version 13.5.37 --other /tmp/ptxas12.9/e4_summary.json --other-version 12.9.86 \
  --other-targets $RUNS/ptxas_versions/tk_sm80.json $RUNS/ptxas_versions/tk_sm90.json $RUNS/ptxas_versions/tk_sm100.json \
  --json ../results/ptxas_summary.json > ../results/ptxas_tables.md
```

## Everything from the raw runs

```bash
RUNS=/path/to/runs ./assemble_results.sh
```

rebuilds every table, summary, archive and figure under `../results` and `../images`.

## From the committed archives, without a GPU or a Triton build

The analysis scripts need only Python and matplotlib. The two archives unpack into runs
directories the scripts read as they read the raw ones.

```bash
mkdir -p /tmp/census /tmp/e4 /tmp/out
tar -xJf ../results/census_evidence_sm120.tar.xz -C /tmp/census
python analyze_census.py --runs /tmp/census --out /tmp/out --ratios ../results/e3_ratios.json > /tmp/out/e1_e2_tables.md
tar -xzf ../results/e4_results.tar.gz -C /tmp/e4
python analyze_e4.py --runs /tmp/e4 --out /tmp/out > /tmp/out/e4_tables.md

mkdir -p /tmp/e4_ptxas /tmp/out_ptxas                 # the ptxas 12.9.86 runs
tar -xzf ../results/e4_ptxas12.9_results.tar.gz -C /tmp/e4_ptxas
python analyze_e4.py --runs /tmp/e4_ptxas --out /tmp/out_ptxas > /dev/null
python analyze_ptxas.py --versions ../results/ptxas_versions_sm120.json \
  --base ../results/e4_summary.json --base-version 13.5.37 --other /tmp/out_ptxas/e4_summary.json --other-version 12.9.86 \
  --other-targets ../results/ptxas_versions_sm80.json ../results/ptxas_versions_sm90.json ../results/ptxas_versions_sm100.json \
  > /tmp/out_ptxas/ptxas_tables.md
```

The E4 summary and tables, and `ptxas_tables.md`, come out identical to the committed ones.
The census archive
holds sm_120 only, so every sm_120 row of `e1_e2_summary.json` comes out identical and the
other targets are empty. The patch-interference check and the classifier total over all
four targets need the raw runs.

## Files

| File | What it does |
|---|---|
| `layoutlab/instrument.py` | wraps `triton.compile`: one tagged compile per (kernel, specialization, options, target), saves TTGIR, emulates other targets |
| `layoutlab/tutorials.py`, `inductor.py`, `pytest_plugin.py` | the three corpus sources |
| `layoutlab/events.py` | loads runs into per-configuration records |
| `layoutlab/gluon_layouts.py` | turns a printed layout attribute into the Gluon layout object |
| `layoutlab/variants.py`, `e4_plugin.py` | capture a launch, rebuild under overrides, compare, verify, time |
| `run_census.py`, `run_lit_census.py`, `sweep_tutorials.sh` | census drivers |
| `make_e3_cases.py`, `e3_bench.py` | E3 case selection and microbenchmark |
| `e4_bench.py` | E4 driver for the tutorial and Inductor kernels |
| `census_from_ttgir.py` | counts the conversions in the TTGIR saved by the device E4 runs |
| `compare_reprice.py` | compares a census recompiled with a different conversion price against the base census |
| `analyze_census.py`, `analyze_e3.py`, `analyze_e4.py`, `make_figures.py` | tables and figures |
| `report_tables.py`, `headline.py` | the tables and the headline block as they appear in the report |
| `ptxas_versions.py`, `analyze_ptxas.py` | the saved builds under other ptxas releases: registers and stack frame per release, and the device timing under two releases |
| `pack_evidence.py` | reduces the event log of one target to what the tables read, in a form `analyze_census.py` loads |
| `pack_ttgir.py` | packs the TTGIR saved by the device E4 runs, with debug locations removed |
| `assemble_results.sh` | runs all of the analysis in order |
