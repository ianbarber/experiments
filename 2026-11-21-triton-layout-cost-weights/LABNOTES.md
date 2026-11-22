# Lab notes: does Triton's layout-conversion pass leave performance behind?

Condensed on 2026-11-22 from the notes I kept while running on 2026-11-21. The order is
unchanged and every correction and failure is kept; tables the report repeats are gone.
Where a later entry corrects an earlier one, the earlier numbers stand as first written.
Orchestrated with Claude Code on `worker-b` (one RTX 5090, 32 GB, compute capability 12.0).

## Build, patch, first events

Built Triton at `359c5be7` with its prebuilt LLVM and installed torch 2.15.0+cu132 over it.
torch pins `triton==3.9.0`, which the source build satisfies, so Inductor uses the pinned
compiler. The LLVM is a release build without assertions, so `LLVM_DEBUG` prints nothing
and the log is a patch that writes JSON lines. I kept a copy of the unpatched
`python/triton` tree to run both compilers on the same inputs.

**A wrong reading of the lowering.** From the generic lowering file I concluded that shared
memory was only ever the fall-through when the other three tests fail. My ground-truth
event there never fired on tutorial 03, whose only surviving conversion is a shared-memory
one. The NVIDIA backend registers a second pattern with higher benefit that takes every
conversion for which `cvtNeedsSharedMemory(op)` is true. The four classes are unchanged.
The patch now logs from both patterns.

**First look at raw events** (tutorial 03, sm_120, one configuration, 60-odd events). Most
cost comparisons are for conversions that are no-ops priced at zero, so the test is
`0 >= remat`. The ones with a real price are index, pointer and mask tensors priced at 1 to
4 million units against rematerialization costs of tens to tens of thousands. None is
close. The same conversion is re-examined in every backward round and every invocation.
So, before collecting anything:

- decisions are deduplicated by the conversion op and the two costs;
- E2 reports the break-even ratio of each decision (`remat_cost / convert_cost`) as well as
  flip counts, because a flip count at one ratio would hide how lopsided the comparisons
  are;
- pointer and `i1` tensors are priced as 32-bit, a second pricing assumption besides the
  class, which E3's width sweep covers.

## E1 and E2: the census

Two things went wrong on the first sweep. My stand-in for the profiler module (switched
off in this build) did not accept `scope(name, metadata)`, so tutorial 09 stopped after its
first kernel: 72 configurations instead of several hundred. Then the full sweep was heading
for two and a half hours on tutorial 09 alone, because the autotuner's benchmarker was
timing thousands of no-op launches per configuration. With a single call in census mode
all four targets finish in twelve minutes. 193 autotune configurations are rejected by
their kernel's own `static_assert` before reaching the pass and are not counted.

Configurations as collected:

| Section | sm_80 | sm_90 | sm_100 | sm_120 |
|---|---|---|---|---|
| Tutorials | 503 | 979 | 2,115 | 1,591 |
| Inductor | 310 | 310 | 310 | 310 |
| triton_kernels | 109 | 749 | 789 | 778 |
| Total | 922 | 2,038 | 3,214 | 2,679 |

**Patch interference.** My first comparison of final TTGIR between the patched and
unpatched builds said 24 identical and 5,164 different. The text carries debug locations
with the path of Triton's own standard library, and the two builds live in different
directories. With paths normalised: 5,166 identical, 22 different. The 22 are all
`matmul_kernel_descriptor_persistent` on sm_90 with warp specialization, each pair
differing only in the order of two `ttg.memdesc_index` ops on barrier arrays. I compiled
tutorial 09 for sm_90 twice more with the unpatched build, and it disagrees with itself on
16 to 17 of the same configurations. That is nondeterminism in the pinned compiler, not the
patch. Inductor and triton_kernels were not run on the unpatched build.

**Classifier.** 20,024 conversions reached the lowering. The class I compute statically and
the branch the lowering took agree on every one (10,145 shared memory, 8,992 no-op, 547
warp shuffle, 340 register permute). On the lit tests the logger sees what they were built
to create: 338 conflict values in `combine.mlir`, 313 resolved by first candidate and 25 by
the MMA rule.

**First numbers** (corrected further down). On sm_120, 431 of 2,679 configurations (16%)
have a conflict; 188 of 922 on sm_80, 319 of 2,038 on sm_90, 591 of 3,214 on sm_100. 4,974
conversions survive on sm_120: 40.1% shared memory, 2.9% warp shuffle, 57.0% free. Of
21,135 calls priced as shared memory, 231 (1.1%) would lower to a warp shuffle. 35,029 cost
decisions; repricing the warp-shuffle ones at any ratio down to a hundredth flips at most 7.

Three things I did not expect. The conflicts are not the "blocked versus MMA" choice the
rule is written for: on sm_120 every candidate set is blocked or linear layouts, and every
one is settled by the first-candidate fallback. Inductor has no `tl.dot`, and a tenth of
its kernels have a conflict anyway. And the comparisons are not close: when the subject has
a non-zero price and no new conversion is involved, rematerialization wins 13,551 times and
loses 167. The cost model is rarely the thing deciding; whether the slice can be
rematerialized at all is.

## E3: what a conversion costs

Two things I got wrong first.

- The loop count was specialized away. Triton specializes integer arguments equal to 1,
  and I warmed the kernel up with 1, so the "loop" ran once. The correctness check caught
  it.
- **Amendment.** The first pass had baselines in the source layout only, so a layout in
  which an add is dearer (more elements per thread, or data repeated across warps) leaked
  into the conversion estimate. I added baselines in the destination layout and subtract
  one add of each. For most cases nothing moved. For the triton_kernels cases whose
  destination repeats data across eight warps, an add is sixteen times dearer in the
  destination and the estimate changed materially.

61 cases, 59 measured: one needs a 3-D tensor my kernel does not build, one runs out of
shared memory, and on a third the verifier rejects the forced-shuffle route, so only the
default route is measured. Later I added four register-permute conversions from the census,
because the headline figure has a "free" class and I had not measured it: 0.0 to 0.1 adds
each. 65 cases, 63 measured.

Sanity check on the unit: an add of 1,024 fp32 elements in one warp takes 2.78 ns of SM
time, 8 cycles at the 2.95 GHz the card was running at, or 128 lanes per cycle.

Result: the same conversion through both routes costs the same (1.03 at the median over 23
conversions; 20.2 ns by shuffle and 21.3 ns through shared memory for 32 by 32 fp32 in one
warp), and both cost far less than assumed: 7.2 to 7.9 adds for fp32 against 32, 2.6 to 3.7
for fp16 against 64. Measured over assumed across 49 conversions: median 0.12, range 0.008
to 0.75. fp64 is excluded from the add comparison (207 ns for an add against 2.78 ns in
fp32).

I had expected the opposite split: the shuffle well below shared memory, and shared memory
near its price. What I measured is both classes together, at a fifth to a thirtieth of the
assumption. So "priced alike but cost differently" is wrong, and "priced wrong relative to
arithmetic" holds.

Plugging the measured ratio into every conversion price flips 351 of 35,029 decisions on
sm_120 (1.0%): none in the tutorials, 47 in Inductor, 304 in triton_kernels. First order
only: the pass is not re-run and only the conversion side is repriced.

## Two corrections to the census

**The pass runs more than four times on some kernels.** Some configurations logged six or
eight runs within one compile. I first read anything past the fourth as the fpsan mode. It
is not. On the Blackwell pipeline (sm_100 and sm_120) `OptimizePartitionWarps` builds its
own pass manager and runs `RemoveLayoutConversions` twice for each warp-specialized
partition it re-lays out, between the third and fourth pipeline invocations. The log tells
the fpsan run apart by its `disable-remat-splitting` flag, so invocations are now labelled
1, 2, 3, nested, 4, fpsan. 78 tutorial configurations on sm_100 and 67 to 81 triton_kernels
configurations on sm_100 and sm_120 have nested runs. No conflict was seen inside one in
the tutorials.

**fpsan configurations are instrumentation, not kernels.** The triton_kernels sample
includes 120 to 125 configurations per target compiled under the fpsan sanitizer. 92 have
conflicts, every one in the extra fpsan run. They are now a separate section and left out
of the headline counts. On sm_120:

| | Before | Corrected |
|---|---|---|
| Configurations | 2,679 | 2,554 |
| With a conflict | 431 (16%) | 339 (13%) |
| Conversions in the final TTGIR | 4,974 | 3,026 |
| Shared memory / warp shuffle / free | 40% / 3% / 57% | 65% / 5% / 30% |
| Cost decisions | 35,029 | 32,697 |

The sanitizer's configurations carry many no-op conversions, which is why the free share
drops when they go.

## E4: alternative resolutions, first attempts

- The tutorial's backward attention kernel does not run on this card: it needs 107 KB of
  shared memory and the 5090 allows 99 KB. The census found no conflict in the backward
  kernels, so E4 uses the forward kernel.
- My compile-only runs for the other targets found nothing to study the first time. The
  census wrapper was overwriting the tag the variant builder had set, so conflicts were
  filed under the wrong compile.
- **Amendment to timing.** The first Inductor kernels came back at 7 µs per launch, which
  is the host's launch overhead. Wall-clock timing of repeated launches cannot see a
  code-generation difference in a kernel that finishes before the next launch is issued. I
  stopped the run and changed to timing replays of a CUDA graph of the launches, which is
  what Triton's own `do_bench_cudagraph` does.

**Tutorial 06.** 15 conflicted kernels on sm_120, 64 alternatives each, and all 960 compile
to identical TTGIR. The overrides did take effect (the logs show 1 to 11 applied per
variant). `AccelerateMatmul` assigns MMA layouts after the conflicts, and the third
invocation starts again from those.

**Inductor, first runs (superseded).** 778 alternatives, 119 with different code. The first
run reported 14 faster than the default, the best by 22%, and 26 slower. A 22% gain on a
three-microsecond kernel needed checking, so I ran the study again in a fresh process with
a null control: the default built a second time and timed as an alternative. All 26 slower
findings repeated. Only 6 of the 14 faster ones did, all on the batch-norm backward
reduction, at 10% to 15.5%.

**Amendment to the reporting rule**, made after seeing a first run, in the conservative
direction: a difference counts only if both runs report it, at the smaller of the two
sizes, and a kernel whose null control moves by 3% or more is left out.

I also misread my own records here. They showed an empty conversion census for every
Inductor variant, which I took to mean no conversions. The hook that logs the census was
not installed by the device driver. Recomputed from the saved TTGIR, the kernel with the
gain has one shared-memory conversion in its loop by default and two in the faster build.

None of these numbers is used: the output check behind them was vacuous (below).

## Recompiling with the measured price

E2's flip count is first order, so I recompiled the sm_120 corpus with both non-free
classes at 0.12 of the pass's price (3.84 per priced byte instead of 32) and compared final
TTGIR configuration by configuration. The logged prices confirm the override took effect.
Tutorials: 0 of 1,591 change. triton_kernels: 137 of 653 change.

First timing of the Inductor kernels: 15 of 198 compile differently, six slower by 10% to
46%, four faster by 5% to 8%. One max-pool kernel shows -33%, and so does its null control:
its timing swings between 22 and 38 µs from run to run. That is where the rule about
unstable controls came from.

The price is measured correctly for a conversion on its own. The comparison it feeds is
with a rematerialization cost that is also by the byte and also wrong, and a small
conversion behind a CTA barrier in the middle of a pointwise kernel is not the saturated
loop E3 timed.

## triton_kernels on the device, and a hole in my output check

The first native run over the triton_kernels suite came back with numbers I did not
believe. One `_matmul` configuration (mxfp4 weights, fp8 activations) had an alternative
96% faster than the default, 3.17 ms against 0.12 ms. A sibling (mxfp8 activations) had an
alternative 493% slower, 21 µs against 125 µs. Both alternatives landed on nearly the same
time, which looked like a kernel that had stopped doing its work.

My output check could not have caught that. Arguments are captured after the kernel has
already run once, so the output tensor already holds the right answer, and I restore every
tensor from that snapshot before each launch. A rebuilt kernel that wrote nothing would
pass. The records confirmed it: no tensor argument changed across a default launch. The
check was vacuous for these runs and for the Inductor ones before them.

Getting a check that is not vacuous, and does not crash kernels, took four rounds of fixes.

1. Overwrite each tensor argument with a byte pattern, launch the default, and call a
   tensor an output if it comes back equal to the snapshot; then overwrite outputs before
   every launch. This crashed the GPU context: it scrambled the int32 slice offsets of a
   ragged batch and the kernel read out of bounds. Only floating-point and 8-bit tensors
   are overwritten now.
2. On the Inductor rerun, kernel after kernel came back "output not checkable". Inductor
   kernels write part of a tensor, or read and write the same tensor, so a second launch
   from the snapshot does not reproduce the snapshot. The probe now sorts written tensors
   into two kinds. A tensor that changes when the kernel is launched from the snapshot
   checks itself, provided the change is larger than the comparison's tolerance. A tensor
   that does not change, but does when overwritten first, is an output and is overwritten
   before every launch. A kernel with neither is not timed.
3. Reading the comparison again, I found it dropped non-finite differences before taking
   the maximum, so an infinity against a finite value counted as no difference. And the
   overwrite pattern read as 1.5e16 in fp32, which would have set the scale for the
   relative tolerance in a partly written tensor. Infinities now compare as values and the
   pattern is 0x3C bytes (0.01 in fp32 and bf16, 1.06 in fp16). I tested probe and
   comparison on fake launches: a build that writes nothing fails, a build that writes a
   wrong value fails, and a correct build passes, for a fully written output, a
   half-written one and an in-place update.
4. The restarted Inductor run died at resnet50's `max_pool2d_with_indices_backward` with a
   device-side assert, and every study after it failed on the dead context. The probe had
   overwritten an int8 tensor, which in Inductor is the max-pool offset. Same mistake as
   the int32 offsets, one dtype down. 8-bit tensors are now overwritten only for kernels
   launched through Triton's own JIT, where they are mxfp values and scales.

With the check enforced the two `_matmul` results stand: output identified, default
repeatable, alternative bit-identical, and plain launches without the CUDA graph give the
same times (3.18 ms against 0.135 ms).

**The cause.** The default and the fast alternative are the same program rearranged: same
ops, same layouts, the same three `convert_layout` ops. They differ in where one
`ttg.local_load` sits. ptxas keeps 40 registers and a 7,880-byte frame for the slow build,
and 255 registers and 408 bytes for the fast one. For the sibling configuration it is the
alternative that gets 40 registers and 9,544 bytes. I first wrote these frames down as
spill counts. Triton's `n_spills` is the function's local stack size in bytes divided by
four, not a count of spill instructions.

Is the slow default something my patched build does? I compiled the cached TTIR of both
configurations with the unpatched Python tree and compiler: same program, same 40 registers
and 7,880-byte frame. ptxas is the one bundled with the pinned Triton, release 13.5
(V13.5.37), invoked once with `--regAllocOptLevel=2`. Does it repeat? A second run in a
fresh process with a fresh cache gives -96.1% and +493.5%, against -96.1% and +493.1% in
the first.

## "Same code" was counting two things

The harness keys builds by the text of their TTGIR and marks an alternative "identical" if
that text has been seen before in the same study, from the default or from an earlier
alternative. So my "same code as the default" column included alternatives that were only
the same as each other. Counted properly, my earlier note that on sm_80 "1 alternative in
64 produces different code" was wrong: 642 of 768 differ from the default, and they
collapse to one program per kernel.

In the other direction, text comparison overcounts: some builds differ from the default
only in the names of SSA values. The analysis now renames values in order of first
appearance before comparing. The harness had already timed eleven such builds as if they
were different (nine triton_kernels, two Inductor), which makes them an unplanned second
null control: all within 0.4% of the default. Two `_p_matmul` builds still looked different
until I noticed that only layout alias numbers (`#linear5`, `#linear6`) had been
renumbered, so the comparison of operations now writes aliases out.

## Final runs

Everything on the device was run again from scratch with the corrected probe and
comparison, twice each. The tables are in the report. Against the earlier notes:

- **Inductor.** 113 distinct programs on 14 kernels (119 texts before renaming), all with
  the default's output. Every one of the 14 kernels had something the check could see: ten
  update a tensor between launches, four have outputs that were overwritten first. 32
  builds are slower in both runs and 11 faster in both; the first run alone said 13 faster.
- **The gain shrank.** On the batch-norm backward reduction it is 5.7% to 11.6%, against
  the 10% to 15.5% above. The default timed 2.57 µs then and 2.46 µs now, in another
  process at another hour. The ordering of the builds is the same. A second reduction now
  shows five builds 3.0% to 3.1% faster; in the unverified runs they sat just under the
  line.
- **Inductor, repriced.** Of 15 changed kernels one is the unstable max-pool kernel. Of the
  other 14: six slower (10.5% to 45.3%), four faster (4.7% to 8.5%), four the same.
- **triton_kernels.** 27 distinct programs on 11 configurations, all bit-exact and all the
  default's operations in another order. Seven time differently in both runs. In the three
  `_matmul` pairs the operation that moves is the `local_load` of the activation tile:
  directly before its first use in the faster build, six to eight operations earlier in the
  slower one.
- **Null control.** Over 98 studies: median 0.01%, none flagged. The largest, 6.7%, is the
  max-pool kernel that is left out.

A faster alternative was the result I expected least. There is one on four of the 46
conflicted kernels, and none of the four wins by removing a conversion.
