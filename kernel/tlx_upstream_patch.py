"""Warp-specialization for uTLX on *upstream* Triton — runtime patch.

Two monkeypatches, applied on import:

1. `visit_With` dispatch: upstream `CodeGenerator.visit_With` has no
   extension hook, so uTLX's `TLX_WITH_DISPATCH` table is never consulted.
   We patch it to check the table first.
2. WS codegen: reimplements `visit_withAsyncTasks` against the *current*
   upstream `ttg.warp_specialize` IR shape (default region + partition-op
   holder + nested `warp_specialize_partitions` carrying explicit captures),
   creating the structural ops through a `GluonOpBuilder` that shares the
   MLIRContext/insertion point with the regular builder. Upstream fully
   lowers `ttg.warp_specialize` (including `setmaxnreg` from
   `set_requested_registers`), so no fork is involved anywhere.

Adapted from wychi/wheels `runner/tlx_patches.py`
(https://github.com/wychi/wheels), extended with the WS codegen fixes,
the layout-hint primitive, and the register-restore stage described in
the README. Validated against upstream Triton 3.8 (release/3.8.x) +
the triton-utlx 3.7.1 DSL + the patched triton-ext plugin.

Usage: import AFTER `utlx_plugin`:
    import utlx_plugin
    import utlx_ws_patch  # applies on import

Retire when: upstream gains a with-statement dispatch hook (the fork's
`WITH_DISPATCH` comment says "will be done later").
"""

import ast


def _apply_dispatch_visit_with():
    from triton.compiler import code_generator as cg
    from utlx_plugin.compiler.dispatch import TLX_WITH_DISPATCH

    if getattr(cg.CodeGenerator.visit_With, "_utlx_ws_patched", False):
        return
    _orig = cg.CodeGenerator.visit_With

    def _patched(self, node):
        if len(node.items) == 1:
            ctx = node.items[0].context_expr
            if isinstance(ctx, ast.Call):
                try:
                    fn = self.visit(ctx.func)
                except Exception:
                    fn = None
                if fn is not None and fn in TLX_WITH_DISPATCH:
                    return TLX_WITH_DISPATCH[fn](self, node)
        return _orig(self, node)

    _patched._utlx_ws_patched = True
    cg.CodeGenerator.visit_With = _patched


def _apply_broadcast_shape_overload():
    """Gluon shadows `create_broadcast(value, type)`; TritonSemantic calls
    `(value, shape_list)`. Route list-shape calls to the base-class binding
    (works unbound on the subclass instance)."""
    from triton._C.libtriton import gluon_ir as _gluon_ir
    from triton._C.libtriton import ir as _ir

    _base = _ir.builder.create_broadcast
    _orig = _gluon_ir.GluonOpBuilder.create_broadcast

    def _patched(self, value, arg):
        if isinstance(arg, _ir.type):
            return _orig(self, value, arg)
        return _base(self, value, arg)

    _gluon_ir.GluonOpBuilder.create_broadcast = _patched


def _apply_gluon_op_builder_swap():
    """Swap `CodeGenerator.builder` to `GluonOpBuilder` after `__init__`.

    Most TLX-relevant ops (`create_memdesc_subslice/trans/reinterpret/index`,
    `create_async_tma_*`, `create_warp_specialize`, …) are bound only on
    `GluonOpBuilder`, which is a pybind *subclass* of `ir.builder` — swapping
    it in keeps every regular `tl.*` op (and the plugin's `utlx_*` methods,
    inherited) working while exposing the gluon-only `create_*` surface.
    """
    from triton._C.libtriton import gluon_ir
    from triton.compiler import code_generator as cg

    if getattr(cg.CodeGenerator.__init__, "_utlx_ws_patched", False):
        return
    _orig_init = cg.CodeGenerator.__init__

    def _target_name(options):
        arch = str(options.arch)
        if options.backend_name == "cuda" and arch.startswith("sm"):
            return f"cuda:{int(arch[2:])}"
        return f"{options.backend_name}:{arch}"

    def _patched_init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        if kwargs.get("is_gluon"):
            return
        context = args[0]
        options = kwargs["options"]
        module_map = kwargs.get("module_map")
        new_builder = gluon_ir.GluonOpBuilder(context)
        new_builder.set_loc(kwargs.get("file_name"),
                            kwargs.get("begin_line", 0),
                            kwargs.get("begin_col", 1))
        new_builder.options = options
        new_builder.codegen_fns = kwargs.get("codegen_fns")
        new_builder.module_map = {} if module_map is None else module_map
        self.semantic.builder = new_builder
        self.builder = new_builder
        if kwargs.get("module") is None:
            self.module = new_builder.create_module()
            # Set ttg attrs eagerly so layout verifiers that resolve
            # warps-per-CTA succeed mid-codegen (mirrors GluonASTSource).
            self.module.set_attr(
                "ttg.target",
                new_builder.get_string_attr(_target_name(options)))
            self.module.set_attr("ttg.num-warps",
                                 new_builder.get_int32_attr(options.num_warps))
            self.module.set_attr("ttg.num-ctas",
                                 new_builder.get_int32_attr(options.num_ctas))
            self.module.set_attr("ttg.threads-per-warp",
                                 new_builder.get_int32_attr(options.warp_size))

    _patched_init._utlx_ws_patched = True
    cg.CodeGenerator.__init__ = _patched_init


def _apply_warp_specialize_codegen():
    from triton._C.libtriton import gluon_ir
    import utlx_plugin.compiler.code_generator as _ucg
    from utlx_plugin.compiler.dispatch import TLX_WITH_DISPATCH

    def _new_visit_withAsyncTasks(self, node):
        from triton.compiler.code_generator import (
            enter_sub_region,
            _is_list_like,
            _is_constexpr,
        )

        builder = self.builder
        gb = gluon_ir.GluonOpBuilder(self.context)

        def _sync_gb():
            gb.restore_insertion_point(builder.get_insertion_point())

        def _flatten_value_handles(val):
            handles = []
            if hasattr(val, "_flatten_ir"):
                val._flatten_ir(handles)
            else:
                handles.append(val.handle)
            return handles

        with enter_sub_region(self) as sr:
            liveins, _ = sr
            ip, last_loc = self._get_insertion_point_and_loc()

            region_replica_id_stack = _ucg._get_region_replica_id_stack()

            stmts = node.body
            if not _is_list_like(stmts):
                stmts = [stmts]
            stmts = _ucg._resolve_async_task_stmts(self, stmts)

            has_non_default = False
            for stmt in stmts:
                task_check = _ucg._get_async_task(self, stmt)
                if not task_check.is_default:
                    has_non_default = True
                    break

            if not has_non_default:
                for stmt in stmts:
                    self.visit(stmt)
                return

            with _ucg.tlx_enter_sub_region():
                tmp_block = builder.create_block()
                builder.set_insertion_point_to_start(tmp_block)
                taskNumWarps = []
                taskNumRegs = []
                taskReplica = []
                taskWarpGroupStartIds = []
                perTaskNumWarps = []
                perTaskStartIds = []
                perTaskReplicates = []
                region_replica_id_stack.append(-1)
                num_default = 0
                for stmt in stmts:
                    task = _ucg._get_async_task(self, stmt)
                    assert task.is_explict
                    assert task.replicate is not None
                    if task.is_default:
                        num_default += 1
                        if task.replicate > 1:
                            taskReplica.append(task.replicate - 1)
                            taskNumWarps.extend([builder.options.num_warps] *
                                                (task.replicate - 1))
                            if task.num_regs:
                                taskNumRegs.extend([task.num_regs] *
                                                   (task.replicate - 1))
                            if task.warp_group_start_id is not None:
                                taskWarpGroupStartIds.extend(
                                    [task.warp_group_start_id] *
                                    (task.replicate - 1))
                    else:
                        taskReplica.append(task.replicate)
                        taskNumWarps.extend([task.num_warps] * task.replicate)
                        if task.num_regs:
                            taskNumRegs.extend([task.num_regs] *
                                               task.replicate)
                        if task.warp_group_start_id is not None:
                            for r in range(task.replicate):
                                taskWarpGroupStartIds.append(
                                    task.warp_group_start_id +
                                    r * task.num_warps)
                            perTaskNumWarps.append(task.num_warps)
                            perTaskStartIds.append(task.warp_group_start_id)
                            perTaskReplicates.append(task.replicate)
                region_replica_id_stack.pop()

                assert num_default == 1, "Default task must be one and only one"
                tmp_block.erase()

                assert len(taskNumRegs) in [0, len(taskNumWarps)]
                assert len(taskWarpGroupStartIds) in [0, len(taskNumWarps)]
                if len(perTaskStartIds) > 0:
                    _ucg._validate_warp_group_start_ids(
                        perTaskStartIds, perTaskNumWarps, perTaskReplicates,
                        builder.options.num_warps)

                # First pass: emit partition bodies into scratch blocks (then
                # erase) so any deferred semantic state (constexpr folding,
                # fn specialization) is discovered before capture collection.
                for stmt in stmts:
                    task = _ucg._get_async_task(self, stmt)
                    task_replicate = (task.replicate -
                                      1) if task.is_default else task.replicate
                    if task_replicate > 0:
                        # Restore a valid insertion point every iteration:
                        # after scratch.erase() the builder IP dangles on the
                        # erased block and the next create_block() would
                        # dereference it (latent in the wychi original, which
                        # only ever ran one worker stmt).
                        self._set_insertion_point_and_loc(ip, last_loc)
                        scratch = builder.create_block()
                        region_replica_id_stack.append(0)
                        builder.set_insertion_point_to_start(scratch)
                        with enter_sub_region(self):
                            self.visit(stmt)
                        region_replica_id_stack.pop()
                        scratch.erase()

                # Over-capture all non-constexpr liveins: unused block args
                # are DCE'd, but a missing capture breaks IsolatedFromAbove.
                captures = sorted(name for name, val in liveins.items()
                                  if not _is_constexpr(val))
                capture_handles = []
                for name in captures:
                    val = liveins[name]
                    if getattr(val, "__triton_aggregate__", False):
                        for field in val.type.fields:
                            v = getattr(val, field[0])
                            capture_handles.extend(_flatten_value_handles(v))
                    else:
                        capture_handles.extend(_flatten_value_handles(val))
                arg_types = [h.get_type() for h in capture_handles]

                self._set_insertion_point_and_loc(ip, last_loc)
                _sync_gb()
                ws_op = gb.create_warp_specialize([], list(taskNumWarps))
                if len(taskNumRegs) > 0:
                    ws_op.set_requested_registers(list(taskNumRegs))

                for stmt in stmts:
                    task = _ucg._get_async_task(self, stmt)
                    if not task.is_default:
                        continue
                    region_replica_id_stack.append(0)
                    default_block = builder.create_block_with_parent(
                        ws_op.get_default_region(), [])
                    builder.set_insertion_point_to_start(default_block)
                    with enter_sub_region(self):
                        self.visit(stmt)
                    _sync_gb()
                    gb.create_warp_yield([])
                    region_replica_id_stack.pop()
                    break

                holder_block = builder.create_block_with_parent(
                    ws_op.get_partition_op_holder(), [])
                builder.set_insertion_point_to_start(holder_block)
                _sync_gb()
                partitions_op = gb.create_warp_specialize_partitions(
                    capture_handles, sum(taskReplica))

                index = 0
                for stmt in stmts:
                    task = _ucg._get_async_task(self, stmt)
                    replicate_start = 1 if task.is_default else 0
                    for i in range(replicate_start, task.replicate):
                        region_replica_id_stack.append(i)
                        partition_region = partitions_op.get_region(index)
                        index += 1
                        block = builder.create_block_with_parent(
                            partition_region, arg_types)
                        builder.set_insertion_point_to_start(block)
                        with enter_sub_region(self):
                            self.visit(stmt)
                        arg_idx = 0
                        for name in captures:
                            val = liveins[name]
                            if getattr(val, "__triton_aggregate__", False):
                                for field in val.type.fields:
                                    v = getattr(val, field[0])
                                    for h in _flatten_value_handles(v):
                                        arg = block.get_argument(arg_idx)
                                        arg_idx += 1
                                        block.replace_use_in_block_with(
                                            h, arg)
                            else:
                                for h in _flatten_value_handles(val):
                                    arg = block.get_argument(arg_idx)
                                    arg_idx += 1
                                    block.replace_use_in_block_with(h, arg)
                        _sync_gb()
                        gb.create_warp_return()
                        region_replica_id_stack.pop()

                builder.set_insertion_point_after(ws_op.get_operation())

    _patched = _ucg.tlx_enter_sub_region()(_new_visit_withAsyncTasks)
    _ucg.visit_withAsyncTasks = _patched

    import triton.language.extra.tlx as _tlx_extra
    TLX_WITH_DISPATCH.clear()
    TLX_WITH_DISPATCH[_tlx_extra.async_tasks] = _patched
    TLX_WITH_DISPATCH[_tlx_extra.async_task] = _ucg.visit_withAsyncTask
    TLX_WITH_DISPATCH._initialized = True


import triton  # noqa: E402  (utlx_plugin is imported before this module)
import triton.language as tl  # noqa: E402  (must be module-level: jit globals)


@triton.jit
def _get_bufidx_phase(accum_cnt, NUM_BUFFERS: tl.constexpr):
    buf_idx = accum_cnt % NUM_BUFFERS
    phase = (accum_cnt // NUM_BUFFERS) & 1
    return buf_idx, phase


def _apply_make_tensor_descriptor_gluon():
    """Route `tl.make_tensor_descriptor` through Gluon's 5-arg typed binding.

    `GluonOpBuilder` shadows `create_make_tensor_descriptor` with a
    different signature (explicit result type carrying the shared layout),
    so the stock semantic call breaks after the builder swap. Rebuild the
    descriptor with an `NVMMASharedLayout`-encoded tensordesc type
    (mirroring `NVMMASharedLayout.get_default_for`, which also matches what
    µTLX `local_alloc` + `require_nv_mma_shared_layout` produce) — this
    doubles as the descriptor-layout stamp the
    `ttng.async_tma_copy_global_to_local` verifier requires.
    """
    from triton._C.libtriton import gluon_ir
    import triton.language as _tl
    from triton.language import core as _tl_core
    from triton.language import semantic as _sem_mod

    _orig = _sem_mod.TritonSemantic.make_tensor_descriptor

    def _default_nvmma_swizzle(block_shape, element_bitwidth):
        contig_bytes = block_shape[-1] * element_bitwidth // 8
        if contig_bytes >= 128 and contig_bytes % 128 == 0:
            swizzle = 128
        elif contig_bytes >= 64 and contig_bytes % 64 == 0:
            swizzle = 64
        elif contig_bytes >= 32 and contig_bytes % 32 == 0:
            swizzle = 32
        else:
            swizzle = 0
        flatten_outer = 1
        for s in block_shape[:-1]:
            flatten_outer *= s
        if len(block_shape) < 2 or flatten_outer < 8:
            swizzle = 0
        return swizzle

    def _patched(self, base, shape, strides, block_shape,
                 padding_option="zero"):
        if not isinstance(self.builder, gluon_ir.GluonOpBuilder):
            return _orig(self, base, shape, strides, block_shape,
                         padding_option)
        shape_vals = [self.make_scalar(x, _tl.int32) for x in shape]
        strides_vals = [
            self.make_scalar(_tl_core._unwrap_if_constexpr(x), _tl.int64)
            for x in strides
        ]
        block_shape_ints = [_tl_core._unwrap_if_constexpr(x) for x in block_shape]
        elt_ty = base.dtype.element_ty
        block_type = _tl.block_type(elt_ty, block_shape_ints)
        is_signed_int = elt_ty.is_int_signed()
        padding = self._str_to_padding_option(padding_option)
        swizzle = _default_nvmma_swizzle(block_shape_ints,
                                         elt_ty.primitive_bitwidth)
        layout_attr = self.builder.get_nvmma_shared_layout(
            swizzle, elt_ty.primitive_bitwidth, False, False, [],
            len(block_shape_ints))
        result_ty = self.builder.get_tensor_descriptor_layout_type(
            block_type.to_ir(self.builder), is_signed_int, layout_attr)
        handle = self.builder.create_make_tensor_descriptor(
            result_ty, base.handle, [s.handle for s in shape_vals],
            [s.handle for s in strides_vals], padding)
        return _tl.tensor_descriptor(handle, shape_vals, strides_vals,
                                     block_type)

    _sem_mod.TritonSemantic.make_tensor_descriptor = _patched


def _apply_local_slice_fix():
    """Fix `utlx_plugin.mem_ops.local_slice` for the current
    `create_memdesc_subslice` binding: `(result_type, source, offsets)` —
    result memdesc type comes first (the wheel's 3-arg call targets a stale
    binding)."""
    from utlx_plugin import mem_ops as _mem_ops
    from utlx_plugin.types import (buffered_tensor, buffered_tensor_type,
                                   storage_kind)
    import triton.language.core as _tl_core

    @_tl_core.builtin
    def _local_slice_patched(buffer, offset, shape, _semantic=None):
        offset = [int(_tl_core._unwrap_if_constexpr(o)) for o in offset]
        shape = [int(_tl_core._unwrap_if_constexpr(s)) for s in shape]
        if buffer.type.storage == storage_kind.tmem:
            assert len(offset) == 2 and len(shape) == 2
            assert offset[0] == 0
            assert shape[0] == buffer.type.shape[0]
            return _mem_ops.subslice(buffer, offset[1], shape[1],
                                     _semantic=_semantic)
        builder = _semantic.builder
        # Build the slice memdesc directly with the parent view's shape as
        # allocShape — the swizzle constraint validates against the
        # allocation, not the view (fork emits e.g.
        # `memdesc<128x64xi8, #shared, #smem, mutable, 128x128>`).
        elem_ir = buffer.type.element_ty.to_ir(builder)
        layout_attr = buffer.type.layout.to_ir(builder)
        slice_ty_ir = builder.get_shared_mem_desc_ty(
            elem_ir, list(shape), layout_attr, list(buffer.type.shape))
        slice_handle = builder.create_memdesc_subslice(
            slice_ty_ir, buffer.handle, offset)
        return buffered_tensor(
            slice_handle,
            buffer.type.element_ty,
            list(shape),
            buffer.type.num,
            buffer.type.storage,
            buffer.type.layout,
        )

    _mem_ops.local_slice = _local_slice_patched
    import utlx_plugin as _tlx
    _tlx.local_slice = _local_slice_patched


def _apply_nv_mma_shared_layout_to_ir_fix():
    """Fix `nv_mma_shared_layout_encoding.to_ir` for GluonOpBuilder: the
    wheel calls the stale `make_nv_mma_shared_encoding_attr`; current method
    is `get_nvmma_shared_layout(swizzle_bw, elem_bw, transposed, fp4_padded,
    cga_layout, rank)`."""
    from utlx_plugin.types import nv_mma_shared_layout_encoding

    def _swizzle_byte_width(shape, element_bitwidth):
        contig_dim_bytes = shape[-1] * element_bitwidth // 8
        if contig_dim_bytes >= 128 and contig_dim_bytes % 128 == 0:
            sw = 128
        elif contig_dim_bytes >= 64 and contig_dim_bytes % 64 == 0:
            sw = 64
        elif contig_dim_bytes >= 32 and contig_dim_bytes % 32 == 0:
            sw = 32
        else:
            sw = 0
        flatten_outer = 1
        for s in shape[:-1]:
            flatten_outer *= s
        if len(shape) < 2 or flatten_outer < 8:
            sw = 0
        return sw

    def _is_natural_order(order):
        return list(order) == list(reversed(range(len(order))))

    def _patched_to_ir(self, builder):
        element_bitwidth = self.elemType.primitive_bitwidth
        rank = len(self.shape)
        swizzle_bw = (_swizzle_byte_width(self.shape, element_bitwidth)
                      if self.swizzled else 0)
        transposed = not _is_natural_order(self.order)
        is_default_cta = (list(self.numCTAsPerCGA) == [1] * rank
                          and list(self.numCTASplit) == [1] * rank)
        if not is_default_cta:
            raise NotImplementedError(
                "non-default CTA layout not mapped to gluon cga_layout")
        return builder.get_nvmma_shared_layout(swizzle_bw, element_bitwidth,
                                               transposed,
                                               bool(self.fp4Padded), [], rank)

    nv_mma_shared_layout_encoding.to_ir = _patched_to_ir


def _apply_hinted_local_load():
    """`tlx.local_load_blocked(src, spt, tpw, wpc, order)` — local_load with
    an explicit blocked register layout (gluon typed `create_local_load`).
    Upstream's pipeline anchors some staging loads on scalar layouts (the
    641×`ld.shared.b8` B-restage pathology); this lets the kernel pin the
    fork's vectorized choice."""
    import triton.language as _tl2
    import triton.language.core as _tl_core
    import utlx_plugin as _up

    @_tl_core.builtin
    def local_load_blocked(src, size_per_thread, threads_per_warp,
                           warps_per_cta, order, _semantic=None):
        b = _semantic.builder
        unwrap = _tl_core._unwrap_if_constexpr
        spt = [int(unwrap(x)) for x in size_per_thread]
        tpw = [int(unwrap(x)) for x in threads_per_warp]
        wpc = [int(unwrap(x)) for x in warps_per_cta]
        odr = [int(unwrap(x)) for x in order]
        layout = b.get_blocked_layout(spt, tpw, wpc, odr, [])
        ty = b.get_distributed_ty(src.type.element_ty.to_ir(b),
                                  list(src.type.shape), layout)
        handle = b.create_local_load(ty, src.handle)
        return _tl2.tensor(
            handle, _tl2.block_type(src.type.element_ty,
                                    list(src.type.shape)))

    _up.local_load_blocked = local_load_blocked
    import triton.language.extra.tlx as _tlx
    _tlx.local_load_blocked = local_load_blocked


def _apply_warp_spec_helpers():
    """Provide `triton.language.extra.tlx.warp_spec` (fork-only submodule
    with the pure-`tl` `get_bufidx_phase` helper) on the µTLX alias."""
    import sys
    import types

    import triton.language.extra.tlx as _tlx

    if hasattr(_tlx, "warp_spec"):
        return

    mod = types.ModuleType("triton.language.extra.tlx.warp_spec")
    mod.get_bufidx_phase = _get_bufidx_phase
    sys.modules[mod.__name__] = mod
    _tlx.warp_spec = mod


def _apply_cuda_layout_propagation():
    """Run the plugin's ported TLX layout-propagation passes on CUDA.

    The utlx stages hook runs `utlx_insert_and_propagate_layout` only on the
    AMD branch; on CUDA the standard pipeline anchors e.g. the B-restage
    staging load on a pathological blocked layout (sizePerThread=[1,1],
    order=[0,1] → 641 scalar ld.shared.b8), where the fork's propagation
    picks sizePerThread=[1,16], order=[1,0] (16B vectorized). Append the
    propagation + cleanup to the ttgir stage for CUDA targets.
    """
    from triton import knobs
    from triton._C.libtriton import ir, passes

    orig_hook = knobs.runtime.add_stages_inspection_hook
    if orig_hook is None or getattr(orig_hook, "_utlx_cuda_layout", False):
        return

    def hook(self=None, stages=None, options=None, language=None,
             capability=None):
        if all(a is None for a in (stages, options, language, capability)):
            import os as _os
            key, hsh = orig_hook()
            # UTLX_WS_REGS changes codegen — it must be part of the cache key.
            regs = _os.environ.get("UTLX_WS_REGS", "")
            return str(key) + "+cudalayout+regs:" + regs, hsh
        orig_hook(self, stages, options, language, capability)
        backend_name = getattr(self, "name", "") if self else ""
        target = getattr(options, "arch", "") if options else ""
        if backend_name == "amd" or str(target).startswith("gfx"):
            return
        orig_ttgir = stages.get("ttgir")
        if orig_ttgir is None:
            return

        def ttgir_with_layout(src, metadata):
            import os as _os
            mod = orig_ttgir(src, metadata)
            pm = ir.pass_manager(mod.context)
            pm.enable_debug()
            passes.plugin.utlx_insert_and_propagate_layout(pm, [])
            passes.ttgpuir.add_remove_layout_conversions(pm)
            passes.common.add_canonicalizer(pm)
            passes.common.add_cse(pm)
            # Upstream's OptimizePartitionWarps clobbers explicit
            # requestedRegisters with `tensorRegs ? 88 : 24` guesses —
            # re-stamp the kernel's true requests so AllocateWarpGroups
            # computes the intended economy. UTLX_WS_REGS="232,24".
            ws_regs = _os.environ.get("UTLX_WS_REGS")
            if ws_regs:
                passes.plugin.utlx_set_ws_requested_regs(
                    pm, [s.strip() for s in ws_regs.split(",")])
            pm.run(mod, "utlx_cuda_layout")
            return mod

        stages["ttgir"] = ttgir_with_layout

    hook._utlx_cuda_layout = True
    knobs.runtime.add_stages_inspection_hook = hook


_apply_dispatch_visit_with()
_apply_broadcast_shape_overload()
_apply_gluon_op_builder_swap()
_apply_make_tensor_descriptor_gluon()
_apply_local_slice_fix()
_apply_nv_mma_shared_layout_to_ir_fix()
_apply_warp_specialize_codegen()
_apply_hinted_local_load()
_apply_warp_spec_helpers()
_apply_cuda_layout_propagation()
