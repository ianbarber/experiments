#!/usr/bin/env python
# Warp-specialized pipelined fp16 GEMM for NVIDIA GB10 (sm_121) written with
# the CUTLASS 4.7 CuTe DSL *Task Scheduling* (TS) framework.
#
# This is (to our knowledge) the first TS GEMM that runs on the sm_12x class:
# every TS GEMM tutorial shipped with CUTLASS 4.7 uses tcgen05/TMEM, which this
# architecture does not have.  Here the MMA consumer instead uses the
# sm_120-class mma.sync tensor-core path (cute.nvgpu.warp.MmaF16BF16Op with
# ldmatrix SMEM->RF copies), with the accumulator in registers.
#
# Architecture
# ------------
#   Grid:   one output tile (128x128) per CTA, grid = (M/128, N/128)
#   Block:  256 threads = 8 warps
#     warps 0-3 : MmaTask   (consumer)  ldmatrix + mma.sync.m16n8k16, fp32 acc
#     warp  4   : LoadTask  (producer)  TMA loads of A and B tiles
#     warps 5-7 : PaddingTask (register-budget padding for the warp group)
#   Resources:
#     GmemAbResource  --(k_tile token)-->  SmemAbResource  --->  GmemCResource
#     SmemAbResource: 3-stage SMEM ring holding a 128x64 A tile + 128x64 B tile
#     per stage, guarded by a TmaAsync pipeline (TMA transaction mbarriers).
#   Schedules (checked statically by the TaskManager at compile time):
#     LoadTask : for k: coords -> try_acquire/acquire -> tma_load -> commit
#     MmaTask  : init(frags+acc) -> for k: try_wait/wait -> mma_step -> release
#                -> tail: store C (RF -> GMEM epilogue)
#
# Broken-variant support (for demonstrating TS static analysis):
#   --variant no_release   consumer never releases stages  -> deadlock at compile
#   --variant double_wait  consumer waits twice per produce -> deadlock at compile
#
# Run with the CuTeDSL venv python:
#   .venv-cutedsl/bin/python gemm_ws_ts.py --verify          # small-shape check
#   .venv-cutedsl/bin/python gemm_ws_ts.py --bench           # full benchmark
#   .venv-cutedsl/bin/python gemm_ws_ts.py --diagnose        # static analysis dump
#   .venv-cutedsl/bin/python gemm_ws_ts.py --variant no_release   # broken compile

import argparse
import json
import statistics
from dataclasses import dataclass, field
from typing import Any

import torch

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass.cute.runtime import from_dlpack

from cutlass.experimental.task_scheduling.resources import (
    WorkAttr,
    MemoryResource,
    StageInfo,
    TaskLocalVariable,
    PipelineConfig,
    consumer_work,
    producer_work,
)
from cutlass.experimental.task_scheduling.memory import SmemAllocation, SmemAllocator
from cutlass.experimental.task_scheduling.schedule_builder import schedule, domain_loop
from cutlass.experimental.task_scheduling.task import Task
from cutlass.experimental.task_scheduling.task_manager import TaskManager
from cutlass.experimental import primitives as prims

# ----------------------------------------------------------------------------
# Static configuration
# ----------------------------------------------------------------------------
TILE_M, TILE_N, TILE_K = 128, 128, 64
STAGES = 3                       # SMEM ring depth (3 * 32KB = 96KB <= 99KB)
NUM_MMA_WARPS = 4                # warps 0-3
NUM_LOAD_WARPS = 1               # warp 4
ATOM_LAYOUT_MNK = (2, 2, 1)      # 2x2 warp tiling of the tile
MMA_INST_MNK = (16, 8, 16)       # mma.sync.aligned.m16n8k16
IO_DTYPE = cutlass.Float16
ACC_DTYPE = cutlass.Float32
SMEM_CAPACITY_BYTES = 101376     # GB10 usable SMEM per CTA (NOT the 232KB default)
MMA_REGISTERS = 232
LOAD_REGISTERS = 40
RASTER_GROUP_M = 8               # grouped rasterization for L2 reuse


@cute.jit
def _swizzled_tile_coords(grid_m: cutlass.Constexpr, grid_n: cutlass.Constexpr):
    """Map the 1D block index to supertiled (pid_m, pid_n) tile coordinates.

    Groups RASTER_GROUP_M CTA rows so consecutive CTAs share B tiles and
    nearby A tiles (Triton-style grouped rasterization).  Requires
    grid_m % group == 0 (enforced host-side).
    """
    group = min(RASTER_GROUP_M, grid_m)
    bidx, _, _ = cute.arch.block_idx()
    num_pid_in_group = group * grid_n
    group_id = bidx // num_pid_in_group
    pid_in_group = bidx % num_pid_in_group
    pid_m = group_id * group + (pid_in_group % group)
    pid_n = pid_in_group // group
    return pid_m, pid_n


# ----------------------------------------------------------------------------
# Resources
# ----------------------------------------------------------------------------
@dataclass
class GmemAbResource(MemoryResource):
    """Coordinate-only GMEM source: emits the K-tile index consumed by TMA."""

    k_tile: cutlass.Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()

    def __post_init__(self) -> None:
        self.k_tile = TaskLocalVariable(
            dtype=cutlass.Int32,
            default=cutlass.Int32(0),
            docs="K tile index consumed by the SMEM TMA load.",
        )

    @consumer_work(returns=k_tile)
    @cute.jit
    def compute_coords(self, stage_info: StageInfo) -> cutlass.Int32:
        return cutlass.Int32(stage_info.loop_offset)


@dataclass
class SmemAbResource(MemoryResource):
    """Staged SMEM ring for the A and B operand tiles (TmaAsync pipeline).

    Producer side (LoadTask): TMA-fills the current stage via cute TMA copy
    atoms, arming the stage's transaction mbarrier.
    Consumer side (MmaTask): ldmatrix SMEM->RF copies + mma.sync accumulation
    into a register accumulator that lives across the whole K loop.
    """

    # Host/compile-time objects (set in __init__)
    tma_atom_a: Any = field(init=False, default=None)
    tma_atom_b: Any = field(init=False, default=None)
    mA: Any = field(init=False, default=None)          # TMA coord tensor of A
    mB: Any = field(init=False, default=None)          # TMA coord tensor of B
    mC: Any = field(init=False, default=None)          # output C (for acc shape)
    a_layout_enum: cutlass.Constexpr = field(init=False, default=None)
    b_layout_enum: cutlass.Constexpr = field(init=False, default=None)
    tiled_mma: Any = field(init=False, default=None)
    acc: Any = field(init=False, default=None)         # register accumulator
    grid_m: cutlass.Constexpr = field(init=False, default=None)
    grid_n: cutlass.Constexpr = field(init=False, default=None)
    _alloc_a: cutlass.Constexpr = field(init=False, default=None)
    _alloc_b: cutlass.Constexpr = field(init=False, default=None)

    def __init__(
        self,
        tma_atom_a: Any,
        mA: cute.Tensor,
        tma_atom_b: Any,
        mB: cute.Tensor,
        mC: cute.Tensor,
        a_layout_enum: Any,
        b_layout_enum: Any,
        tiled_mma: Any,
        a_smem_bytes: int,
        b_smem_bytes: int,
        grid_m: int,
        grid_n: int,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self.tma_atom_a = tma_atom_a
        self.tma_atom_b = tma_atom_b
        self.mA = mA
        self.mB = mB
        self.mC = mC
        # Store the *Python-level* layout enums; the staged ComposedLayouts are
        # rebuilt fresh inside each work body.  Storing MLIR layout values on
        # the resource breaks dominance when the framework threads the resource
        # through the per-task warp-gate regions.
        self.a_layout_enum = a_layout_enum
        self.b_layout_enum = b_layout_enum
        self.tiled_mma = tiled_mma
        self.grid_m = grid_m
        self.grid_n = grid_n
        self._alloc_a = SmemAllocation("smem_a", a_smem_bytes, alignment=1024)
        self._alloc_b = SmemAllocation("smem_b", b_smem_bytes, alignment=1024)
        # The register accumulator is allocated HERE (kernel top-level trace,
        # outside the per-task warp gates) so its alloca dominates the MMA
        # loop and the tail epilogue.  Work bodies only mutate it in place;
        # resource fields are never reassigned inside task regions (the DSL
        # forbids carrying new Python values across the warp-gate `if`s).
        gC0 = cute.local_tile(mC, (TILE_M, TILE_N), (0, 0))
        tCgC0 = self.tiled_mma.get_slice(0).partition_C(gC0)
        self.acc = cute.make_rmem_tensor(tCgC0.shape, ACC_DTYPE)

    def get_smem_requirements(self):
        return [self._alloc_a, self._alloc_b]

    @cute.jit
    def _make_smem_tensors(self, stage_info: StageInfo):
        """Build the staged (swizzled) SMEM tensors over the allocator block.

        The staged layouts are reconstructed here (compile-time layout algebra)
        rather than stored on the resource; see __init__ for why.
        """
        tile_mnk = (TILE_M, TILE_N, TILE_K)
        a_layout_staged = sm90_utils.make_smem_layout_a(
            self.a_layout_enum, tile_mnk, IO_DTYPE, STAGES
        )
        b_layout_staged = sm90_utils.make_smem_layout_b(
            self.b_layout_enum, tile_mnk, IO_DTYPE, STAGES
        )
        smem_base = stage_info.context.smem_base
        a_ptr = cute.make_ptr(
            IO_DTYPE, smem_base.data_ptr(self._alloc_a.offset), assumed_align=1024
        )
        b_ptr = cute.make_ptr(
            IO_DTYPE, smem_base.data_ptr(self._alloc_b.offset), assumed_align=1024
        )
        sA = cute.make_tensor(
            cute.recast_ptr(a_ptr, a_layout_staged.inner, dtype=IO_DTYPE),
            a_layout_staged.outer,
        )
        sB = cute.make_tensor(
            cute.recast_ptr(b_ptr, b_layout_staged.inner, dtype=IO_DTYPE),
            b_layout_staged.outer,
        )
        return sA, sB

    # ------------------------------------------------------------------ #
    # Producer (LoadTask) work                                           #
    # ------------------------------------------------------------------ #
    @producer_work
    @cute.jit
    def tma_load(self, stage_info: StageInfo, *, k_tile: cutlass.Int32) -> None:
        sA, sB = self._make_smem_tensors(stage_info)
        pid_m, pid_n = _swizzled_tile_coords(self.grid_m, self.grid_n)
        # (TILE_M, TILE_K, loopK) tile of A for this CTA's M block
        gA = cute.local_tile(self.mA, (TILE_M, TILE_K), (pid_m, None))
        # (TILE_N, TILE_K, loopK) tile of B for this CTA's N block
        gB = cute.local_tile(self.mB, (TILE_N, TILE_K), (pid_n, None))
        tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
            self.tma_atom_a,
            0,
            cute.make_layout(1),
            cute.group_modes(sA, 0, 2),
            cute.group_modes(gA, 0, 2),
        )
        tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
            self.tma_atom_b,
            0,
            cute.make_layout(1),
            cute.group_modes(sB, 0, 2),
            cute.group_modes(gB, 0, 2),
        )
        # TS's acquire() already armed the stage barrier with the expected
        # transaction byte count; we only issue the TMA copies at it.
        bar = self.pipeline.sync_object_full.get_barrier(stage_info.stage_idx)
        cute.copy(
            self.tma_atom_a,
            tAgA[(None, k_tile)],
            tAsA[(None, stage_info.stage_idx)],
            tma_bar_ptr=bar,
        )
        cute.copy(
            self.tma_atom_b,
            tBgB[(None, k_tile)],
            tBsB[(None, stage_info.stage_idx)],
            tma_bar_ptr=bar,
        )

    # ------------------------------------------------------------------ #
    # Consumer (MmaTask) work                                            #
    # ------------------------------------------------------------------ #
    @consumer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_mma_state(self, stage_info: StageInfo) -> None:
        # Zero the K-loop register accumulator (allocated in __init__).
        self.acc.fill(0.0)

    @cute.jit
    def _mma_partitions(self, stage_info: StageInfo):
        """(Re)build the thread-level SMEM views and RF fragments.

        Pure layout algebra over compile-time layouts; runtime cost is a few
        pointer computations, largely hoisted by the compiler.
        """
        sA, sB = self._make_smem_tensors(stage_info)
        tidx, _, _ = cute.arch.thread_idx()
        tiled_mma = self.tiled_mma
        thr_mma = tiled_mma.get_slice(tidx)

        # MMA fragments over one pipeline stage
        tCsA = thr_mma.partition_A(sA)  # (MMA, MMA_M, MMA_K, STAGE)
        tCsB = thr_mma.partition_B(sB)  # (MMA, MMA_N, MMA_K, STAGE)
        tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
        tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])

        # ldmatrix SMEM->RF tiled copies (both operands K-major -> no transpose)
        atom_ld = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 4), IO_DTYPE
        )
        copy_A = cute.make_tiled_copy_A(atom_ld, tiled_mma)
        copy_B = cute.make_tiled_copy_B(atom_ld, tiled_mma)
        thr_copy_A = copy_A.get_slice(tidx)
        thr_copy_B = copy_B.get_slice(tidx)
        tCsA_cv = thr_copy_A.partition_S(sA)
        tCrA_cv = thr_copy_A.retile(tCrA)
        tCsB_cv = thr_copy_B.partition_S(sB)
        tCrB_cv = thr_copy_B.retile(tCrB)
        return copy_A, copy_B, tCsA_cv, tCrA_cv, tCsB_cv, tCrB_cv, tCrA, tCrB

    @consumer_work
    @cute.jit
    def mma_step(self, stage_info: StageInfo) -> None:
        (copy_A, copy_B, tCsA_cv, tCrA_cv, tCsB_cv, tCrB_cv, tCrA, tCrB) = (
            self._mma_partitions(stage_info)
        )
        stage = stage_info.stage_idx
        num_k_blocks = cute.size(tCrA, mode=[2])
        tCsA_p = tCsA_cv[None, None, None, stage]
        tCsB_p = tCsB_cv[None, None, None, stage]
        # SMEM -> RF for k-block 0, then overlap copy(kb+1) with gemm(kb)
        cute.copy(copy_A, tCsA_p[None, None, 0], tCrA_cv[None, None, 0])
        cute.copy(copy_B, tCsB_p[None, None, 0], tCrB_cv[None, None, 0])
        for kb in cutlass.range_constexpr(num_k_blocks):
            if cutlass.const_expr(kb + 1 < num_k_blocks):
                cute.copy(
                    copy_A, tCsA_p[None, None, kb + 1], tCrA_cv[None, None, kb + 1]
                )
                cute.copy(
                    copy_B, tCsB_p[None, None, kb + 1], tCrB_cv[None, None, kb + 1]
                )
            cute.gemm(
                self.tiled_mma,
                self.acc,
                tCrA[None, None, kb],
                tCrB[None, None, kb],
                self.acc,
            )


@dataclass
class GmemCResource(MemoryResource):
    """GMEM sink: the MmaTask's tail stores the register accumulator to C."""

    mma_state: Any = field(init=False, default=None)

    def __init__(self, mma_state: SmemAbResource, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.mma_state = mma_state  # Python ref: accumulator + C partition live there

    @producer_work
    @cute.jit
    def store_c(self, stage_info: StageInfo) -> None:
        src = self.mma_state
        tidx, _, _ = cute.arch.thread_idx()
        pid_m, pid_n = _swizzled_tile_coords(src.grid_m, src.grid_n)
        gC = cute.local_tile(src.mC, (TILE_M, TILE_N), (pid_m, pid_n))
        tCgC = src.tiled_mma.get_slice(tidx).partition_C(gC)  # (MMA, MMA_M, MMA_N)
        tCrC = cute.make_rmem_tensor(tCgC.shape, IO_DTYPE)
        tCrC.store(src.acc.load().to(IO_DTYPE))
        cute.autovec_copy(tCrC, tCgC)


# ----------------------------------------------------------------------------
# Kernel
# ----------------------------------------------------------------------------
@cute.kernel
def gemm_ws_ts_kernel(
    tma_atom_a: Any,
    mA_tma: cute.Tensor,
    tma_atom_b: Any,
    mB_tma: cute.Tensor,
    mC: cute.Tensor,
    tiled_mma: Any,
    a_layout_enum: cutlass.Constexpr,
    b_layout_enum: cutlass.Constexpr,
    a_smem_bytes: cutlass.Constexpr,
    b_smem_bytes: cutlass.Constexpr,
    grid_m: cutlass.Constexpr,
    grid_n: cutlass.Constexpr,
    k_tiles: cutlass.Constexpr,
    tma_bytes_per_stage: cutlass.Constexpr,
    variant: cutlass.Constexpr,
    exhaustive_check: cutlass.Constexpr,
):
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    if warp_idx == 0:
        cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_a)
        cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_b)

    # ------------------------------------------------------------------ #
    # Resources                                                          #
    # ------------------------------------------------------------------ #
    smem_ab_cfg = PipelineConfig.create_tma_async_pipeline_cfg(
        num_stages=STAGES,
        num_bytes=tma_bytes_per_stage,
        # LoadTask has 1 warp; one elected thread arms the barrier per stage.
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        # MmaTask has 4 warps (128 threads) releasing each stage.
        consumer_group=pipeline.CooperativeGroup(
            pipeline.Agent.Thread, NUM_MMA_WARPS * 32
        ),
    )

    gmem_ab_resource = GmemAbResource(name="GmemAB")
    smem_ab_resource = SmemAbResource(
        name="SmemAB",
        tma_atom_a=tma_atom_a,
        mA=mA_tma,
        tma_atom_b=tma_atom_b,
        mB=mB_tma,
        mC=mC,
        a_layout_enum=a_layout_enum,
        b_layout_enum=b_layout_enum,
        tiled_mma=tiled_mma,
        a_smem_bytes=a_smem_bytes,
        b_smem_bytes=b_smem_bytes,
        grid_m=grid_m,
        grid_n=grid_n,
        pipeline_config=smem_ab_cfg,
    )
    gmem_c_resource = GmemCResource(name="GmemC", mma_state=smem_ab_resource)

    allocator = SmemAllocator()
    allocator.add_resource(smem_ab_resource)
    allocator.compute_layout()

    # ------------------------------------------------------------------ #
    # Schedules                                                          #
    # ------------------------------------------------------------------ #
    @schedule
    def load_schedule(gmem_ab: MemoryResource, smem_ab: MemoryResource) -> None:
        with domain_loop(0, k_tiles, 1):
            k_tile = gmem_ab.compute_coords()
            smem_ab.try_acquire()
            smem_ab.acquire()
            smem_ab.tma_load(k_tile=k_tile)
            smem_ab.commit()

    @schedule
    def mma_schedule(smem_ab: MemoryResource, gmem_c: MemoryResource) -> None:
        smem_ab.init_mma_state()
        with domain_loop(0, k_tiles, 1):
            smem_ab.try_wait()
            smem_ab.wait()
            smem_ab.mma_step()
            if cutlass.const_expr(variant != "no_release"):
                smem_ab.release()
            if cutlass.const_expr(variant == "double_wait"):
                # BROKEN: consume two producer commits per produced stage
                smem_ab.wait()
                smem_ab.release()
        # Tail: epilogue store of the register accumulator
        gmem_c.store_c()

    @schedule
    def padding_schedule() -> None:
        with domain_loop(0, k_tiles, 1):
            pass

    load_result = load_schedule(gmem_ab_resource, smem_ab_resource)
    mma_result = mma_schedule(smem_ab_resource, gmem_c_resource)
    padding_result = padding_schedule()

    # ------------------------------------------------------------------ #
    # Tasks                                                              #
    # ------------------------------------------------------------------ #
    mma_task = Task(
        name="MmaTask",
        src_resources=[smem_ab_resource],
        dst_resources=[gmem_c_resource],
        warp_idx=0,
        num_warps=NUM_MMA_WARPS,
        schedule=mma_result,
        num_registers=MMA_REGISTERS,
    )
    load_task = Task(
        name="LoadTask",
        src_resources=[gmem_ab_resource],
        dst_resources=[smem_ab_resource],
        warp_idx=NUM_MMA_WARPS,
        num_warps=NUM_LOAD_WARPS,
        schedule=load_result,
        num_registers=LOAD_REGISTERS,
    )
    padding_task = Task(
        name="PaddingTask",
        src_resources=[],
        dst_resources=[],
        warp_idx=NUM_MMA_WARPS + NUM_LOAD_WARPS,
        num_warps=3,
        schedule=padding_result,
        num_registers=LOAD_REGISTERS,
    )

    resource_dependency_graph = {
        smem_ab_resource: [gmem_ab_resource],
        gmem_c_resource: [smem_ab_resource],
    }

    task_manager = TaskManager(
        tasks=[mma_task, load_task, padding_task],
        resource_dependency_graph=resource_dependency_graph,
        smem_allocator=allocator,
        smem_capacity_bytes=SMEM_CAPACITY_BYTES,
        exhaustive_deadlock_race_check=exhaustive_check,
    )
    task_manager.setup_resources_and_tasks()

    prims.fence_mbarrier_init()
    prims.barrier_cta_sync(0)

    task_manager.run()


# ----------------------------------------------------------------------------
# Host wrapper
# ----------------------------------------------------------------------------
@cute.jit
def gemm_ws_ts_host(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor,
    variant: cutlass.Constexpr,
    exhaustive_check: cutlass.Constexpr,
):
    tile_mnk = (TILE_M, TILE_N, TILE_K)

    a_layout_enum = utils.LayoutEnum.from_tensor(mA)
    b_layout_enum = utils.LayoutEnum.from_tensor(mB)

    # Staged, swizzled SMEM layouts (same helpers as the sm_120 dense GEMM)
    a_layout_staged = sm90_utils.make_smem_layout_a(
        a_layout_enum, tile_mnk, IO_DTYPE, STAGES
    )
    b_layout_staged = sm90_utils.make_smem_layout_b(
        b_layout_enum, tile_mnk, IO_DTYPE, STAGES
    )

    # Tiled MMA: 4 warps in a 2x2 layout of m16n8k16 mma.sync atoms
    op = cute.nvgpu.warp.MmaF16BF16Op(IO_DTYPE, ACC_DTYPE, MMA_INST_MNK)
    permutation_mnk = (
        ATOM_LAYOUT_MNK[0] * MMA_INST_MNK[0],
        ATOM_LAYOUT_MNK[1] * MMA_INST_MNK[1] * 2,
        ATOM_LAYOUT_MNK[2] * MMA_INST_MNK[2],
    )
    tiled_mma = cute.make_tiled_mma(
        op, cute.make_layout(ATOM_LAYOUT_MNK), permutation_mnk=permutation_mnk
    )

    # TMA copy atoms + coordinate tensors
    g2s = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
    a_stage_layout = cute.slice_(a_layout_staged, (None, None, 0))
    b_stage_layout = cute.slice_(b_layout_staged, (None, None, 0))
    tma_atom_a, mA_tma = cute.nvgpu.cpasync.make_tiled_tma_atom(
        g2s, mA, a_stage_layout, (TILE_M, TILE_K)
    )
    tma_atom_b, mB_tma = cute.nvgpu.cpasync.make_tiled_tma_atom(
        g2s, mB, b_stage_layout, (TILE_N, TILE_K)
    )
    tma_bytes = cute.size_in_bytes(IO_DTYPE, a_stage_layout) + cute.size_in_bytes(
        IO_DTYPE, b_stage_layout
    )
    elem_bytes = IO_DTYPE.width // 8
    a_smem_bytes = cute.cosize(a_layout_staged) * elem_bytes
    b_smem_bytes = cute.cosize(b_layout_staged) * elem_bytes

    k_tiles = mA.shape[1] // TILE_K
    grid_m = mA.shape[0] // TILE_M
    grid_n = mB.shape[0] // TILE_N

    gemm_ws_ts_kernel(
        tma_atom_a,
        mA_tma,
        tma_atom_b,
        mB_tma,
        mC,
        tiled_mma,
        a_layout_enum,
        b_layout_enum,
        a_smem_bytes,
        b_smem_bytes,
        grid_m,
        grid_n,
        k_tiles,
        tma_bytes,
        variant,
        exhaustive_check,
    ).launch(
        grid=(grid_m * grid_n, 1, 1),
        block=((NUM_MMA_WARPS + NUM_LOAD_WARPS + 3) * 32, 1, 1),
    )


# ----------------------------------------------------------------------------
# Compile / verify / benchmark driver
# ----------------------------------------------------------------------------
def compile_for(m, n, k, variant="good", exhaustive_check=True):
    assert m % TILE_M == 0 and n % TILE_N == 0 and k % TILE_K == 0
    grid_m = m // TILE_M
    assert grid_m % min(RASTER_GROUP_M, grid_m) == 0, "raster group must divide grid_m"
    torch.manual_seed(2026)
    a = torch.randn(m, k, dtype=torch.float16, device="cuda")
    b = torch.randn(n, k, dtype=torch.float16, device="cuda")
    c = torch.zeros(m, n, dtype=torch.float16, device="cuda")
    mA, mB, mC = from_dlpack(a), from_dlpack(b), from_dlpack(c)
    fn = cute.compile(gemm_ws_ts_host, mA, mB, mC, variant, exhaustive_check)
    return fn, (a, b, c), (mA, mB, mC)


def verify(m, n, k, variant="good", exhaustive_check=True):
    print(f"=== verify M={m} N={n} K={k} (variant={variant}) ===")
    fn, (a, b, c), (mA, mB, mC) = compile_for(m, n, k, variant, exhaustive_check)
    fn(mA, mB, mC)
    torch.cuda.synchronize()
    ref = (a.float() @ b.float().t()).to(torch.float16)
    torch.testing.assert_close(c, ref, atol=1e-2, rtol=1e-2)
    print(f"verify M={m} N={n} K={k}: PASS")
    return fn, (a, b, c), (mA, mB, mC)


def bench_shape(m, n, k, warmup=25, iters=100):
    fn, (a, b, c), (mA, mB, mC) = verify(m, n, k, variant="good",
                                         exhaustive_check=(k // TILE_K) <= 16)

    def time_median(call):
        for _ in range(warmup):
            call()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        stops = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for i in range(iters):
            starts[i].record()
            call()
            stops[i].record()
        torch.cuda.synchronize()
        return statistics.median(
            starts[i].elapsed_time(stops[i]) for i in range(iters)
        )

    flops = 2.0 * m * n * k
    yours_ms = time_median(lambda: fn(mA, mB, mC))
    bt = b.t()  # (K, N) view so torch.matmul computes the same GEMM
    torch_ms = time_median(lambda: torch.matmul(a, bt))
    result = {
        "yours_ms": yours_ms,
        "yours_tflops": flops / (yours_ms * 1e-3) / 1e12,
        "torch_ms": torch_ms,
        "torch_tflops": flops / (torch_ms * 1e-3) / 1e12,
    }
    print(f"M=N=K={m}: ts={yours_ms:.3f} ms ({result['yours_tflops']:.2f} TFLOP/s)  "
          f"torch={torch_ms:.3f} ms ({result['torch_tflops']:.2f} TFLOP/s)")
    return result


def main():
    parser = argparse.ArgumentParser(description="TS warp-specialized fp16 GEMM (sm_121)")
    parser.add_argument("--variant", default="good",
                        choices=["good", "no_release", "double_wait"])
    parser.add_argument("--verify", action="store_true", help="small-shape verify only")
    parser.add_argument("--diagnose", action="store_true",
                        help="small-shape compile with full static-analysis output")
    parser.add_argument("--bench", action="store_true", help="run the full benchmark")
    parser.add_argument("--mnk", type=int, default=512,
                        help="M=N=K for --verify/--diagnose")
    parser.add_argument("--json", default="bench_results.json")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("GPU required")

    if args.variant != "good":
        # Broken variants are expected to FAIL at compile time (static analysis).
        m = n = k = 512
        verify(m, n, k, variant=args.variant, exhaustive_check=True)
        raise SystemExit(
            f"ERROR: variant {args.variant} unexpectedly compiled and verified"
        )

    if args.diagnose or args.verify:
        verify(args.mnk, args.mnk, args.mnk, exhaustive_check=True)
        return

    if args.bench:
        results = {}
        for s in (1024, 2048, 4096):
            results[str(s)] = bench_shape(s, s, s)
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"wrote {args.json}")
        return

    # default: verify small, then bench
    verify(512, 512, 512, exhaustive_check=True)


if __name__ == "__main__":
    main()
