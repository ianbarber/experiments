#!/usr/bin/env python3
"""Fetch the triton-utlx DSL (pure Python) and apply two small patches.

The PyPI `triton-utlx` wheel is x86_64-only, but its Python packages are
architecture-independent — this script pip-downloads the pinned wheel,
extracts `utlx_plugin`/`tlx`/`utlx` into ./utlx_py, drops the bundled x86
libutlx.so (we build our own — see scripts/build_plugin.sh), and applies:

  1. TMA reroute: `async_descriptor_load/store/store_wait` call the
     plugin-owned upstream-op wrappers (utlx_async_TMA_*) instead of the
     Gluon-only ir.builder bindings that upstream never exposes.
  2. fp8 dtype support in local_alloc's type-carrier map (needed for the
     UE4M3 scale buffers).

Usage:  python scripts/fetch_utlx.py   (from the repo root)
"""
import pathlib
import subprocess
import sys
import tempfile
import zipfile

PIN = "triton-utlx==3.7.1"
ROOT = pathlib.Path(__file__).resolve().parent.parent
DEST = ROOT / "utlx_py"

MEM_OPS_PATCHES = [
    # 1a. TMA load
    (
        """    multicast = len(multicast_targets) > 0
    # Use gluon: create_async_tma_copy_global_to_local(desc, coord, barrier, result, pred, multicast, offsets)
    _semantic.builder.create_async_tma_copy_global_to_local(
        desc.handle, offsets, barrier.handle, result_handle, pred_handle,
        multicast, None)""",
        """    multicast = len(multicast_targets) > 0
    # Patched: plugin-owned upstream-op wrapper
    # (utlx_async_TMA_load -> ttng.async_tma_copy_global_to_local).
    _semantic.builder.utlx_async_TMA_load(
        [desc.handle, barrier.handle, result_handle, pred_handle] +
        list(offsets))""",
    ),
    # 1b. TMA store
    (
        """    if store_reduce == "":
        # Use gluon: create_async_tma_copy_local_to_global(desc, coord, src)
        _semantic.builder.create_async_tma_copy_local_to_global(
            desc.handle, offsets, source_handle)""",
        """    if store_reduce == "":
        # Patched: plugin-owned upstream-op wrapper
        # (utlx_async_TMA_store -> ttng.async_tma_copy_local_to_global).
        _semantic.builder.utlx_async_TMA_store(
            [desc.handle, source_handle] + list(offsets))""",
    ),
    # 1c. TMA store wait
    (
        """    pendings = tl._unwrap_if_constexpr(pendings)
    # Use gluon: create_async_tma_store_wait(pendings)
    _semantic.builder.create_async_tma_store_wait(pendings)""",
        """    pendings = tl._unwrap_if_constexpr(pendings)
    # Patched: plugin-owned upstream-op wrapper
    # (utlx_async_TMA_store_wait -> ttng.async_tma_store_wait).
    _semantic.builder.utlx_async_TMA_store_wait(
        [_semantic.builder.get_int32(int(pendings))])""",
    ),
    # 2. fp8 type carriers
    (
        """def _make_type_carrier(builder, dtype):
    \"\"\"Create a type-carrier scalar constant of the desired element type.\"\"\"
    builder_method = _DTYPE_TO_BUILDER_METHOD.get(dtype)
    if builder_method is None:
        raise ValueError(f"Unsupported dtype: {dtype}")""",
        """# Patched: fp8 dtypes carried via get_null_value(<fp8 ty>)
_DTYPE_TO_TY_METHOD = {
    tl.float8e4nv: "get_fp8e4nv_ty",
    tl.float8e4b8: "get_fp8e4b8_ty",
    tl.float8e5: "get_fp8e5_ty",
    tl.float8e5b16: "get_fp8e5b16_ty",
}


def _make_type_carrier(builder, dtype):
    \"\"\"Create a type-carrier scalar constant of the desired element type.\"\"\"
    ty_method = _DTYPE_TO_TY_METHOD.get(dtype)
    if ty_method is not None:
        return builder.get_null_value(getattr(builder, ty_method)())
    builder_method = _DTYPE_TO_BUILDER_METHOD.get(dtype)
    if builder_method is None:
        raise ValueError(f"Unsupported dtype: {dtype}")""",
    ),
]


def main():
    with tempfile.TemporaryDirectory() as td:
        subprocess.run(
            [sys.executable, "-m", "pip", "download", PIN, "--no-deps",
             "--platform", "manylinux_2_34_x86_64", "--only-binary", ":all:",
             "-d", td],
            check=True,
        )
        wheel = next(pathlib.Path(td).glob("triton_utlx-*.whl"))
        with zipfile.ZipFile(wheel) as z:
            for name in z.namelist():
                if name.split("/")[0] in ("utlx_plugin", "tlx", "utlx"):
                    z.extract(name, DEST)
    # Drop the bundled x86_64 plugin binary — we build a matched one.
    so = DEST / "utlx_plugin" / "libutlx.so"
    if so.exists():
        so.unlink()

    mem_ops = DEST / "utlx_plugin" / "mem_ops.py"
    src = mem_ops.read_text()
    for old, new in MEM_OPS_PATCHES:
        if new.splitlines()[1] in src:
            continue  # already applied
        assert old in src, (
            f"patch anchor not found in {mem_ops} — wheel drift? "
            f"(pinned {PIN}); first missing anchor line: {old.splitlines()[0]!r}")
        src = src.replace(old, new)
    mem_ops.write_text(src)
    print(f"OK: DSL extracted+patched at {DEST}")


if __name__ == "__main__":
    main()
