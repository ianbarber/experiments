#!/usr/bin/env python
# BROKEN VARIANT 1: the consumer schedule never calls release() on the SMEM
# ring.  A hand-written kernel with this bug compiles fine and HANGS at
# runtime (the producer blocks forever on stage empty barriers).  With TS the
# TaskManager's static analysis catches it at cute.compile time, before any
# kernel is launched.
#
# The kernel is identical to gemm_ws_ts.py; the only difference is the
# `variant="no_release"` switch, which removes the `smem_ab.release()` line
# from the MMA consumer schedule:
#
#     with domain_loop(0, k_tiles, 1):
#         smem_ab.try_wait()
#         smem_ab.wait()
#         smem_ab.mma_step()
#         # smem_ab.release()        <-- REMOVED
#
# Expected result: compile-time schedule-analysis failure (no launch, no hang).
#
# Run:  timeout 90 .venv-cutedsl/bin/python gemm_ws_ts_broken1.py

import sys
import traceback

from gemm_ws_ts import verify

if __name__ == "__main__":
    try:
        verify(512, 512, 512, variant="no_release", exhaustive_check=True)
    except Exception as e:
        print("\n" + "=" * 72)
        print("TS STATIC ANALYSIS REJECTED THE SCHEDULE AT COMPILE TIME (expected):")
        print("=" * 72)
        traceback.print_exc()
        sys.exit(0)
    print("ERROR: broken schedule unexpectedly compiled and ran")
    sys.exit(1)
