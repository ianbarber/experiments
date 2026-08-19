#!/usr/bin/env python
# BROKEN VARIANT 2: credit-model deadlock.  The consumer waits on (and
# releases) TWO SMEM stages per K iteration while the producer only commits
# one, so the consumer eventually blocks on a stage that will never be filled.
# A hand-written kernel with this bug HANGS at runtime; TS's credit-model
# schedule simulator / exhaustive interleaving checker refuses to compile it.
#
# The kernel is identical to gemm_ws_ts.py; the only difference is the
# `variant="double_wait"` switch, which turns the MMA consumer schedule into:
#
#     with domain_loop(0, k_tiles, 1):
#         smem_ab.try_wait()
#         smem_ab.wait()
#         smem_ab.mma_step()
#         smem_ab.release()
#         smem_ab.wait()             <-- ADDED: second wait per produce
#         smem_ab.release()          <-- ADDED
#
# Expected result: compile-time deadlock detection (no launch, no hang).
#
# Run:  timeout 90 .venv-cutedsl/bin/python gemm_ws_ts_broken2.py

import sys
import traceback

from gemm_ws_ts import verify

if __name__ == "__main__":
    try:
        verify(512, 512, 512, variant="double_wait", exhaustive_check=True)
    except Exception as e:
        print("\n" + "=" * 72)
        print("TS STATIC ANALYSIS REJECTED THE SCHEDULE AT COMPILE TIME (expected):")
        print("=" * 72)
        traceback.print_exc()
        sys.exit(0)
    print("ERROR: broken schedule unexpectedly compiled and ran")
    sys.exit(1)
