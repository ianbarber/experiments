#!/bin/bash
# Build LSP-enabled variants of ProMax instance images (run on an x86 worker).
# Usage: build_layers.sh <instance_id> [<instance_id> ...]
# Produces promax-lsp:<id>, archives to NAS, then removes local images to bound disk.
set -uo pipefail
CTX="$HOME/lsp-layer-build"        # build context: contains lsp-tool/ and Dockerfile
NAS_BASE=/mnt/nas/refactorbench/images
NAS_LSP=/mnt/nas/refactorbench/images-lsp
mkdir -p "$NAS_LSP"
FAILED=()
for IID in "$@"; do
  BASE="key4127/refactor-dockerhub:${IID}"
  OUT_TAR="$NAS_LSP/${IID}.tar.zst"
  if [[ -f "$OUT_TAR" ]]; then echo "[layer] $IID: already on NAS"; continue; fi
  echo "[layer] $IID: fetching base"
  if ! docker image inspect "$BASE" >/dev/null 2>&1; then
    if [[ -f "$NAS_BASE/${IID}.tar.zst" ]]; then zstd -dc "$NAS_BASE/${IID}.tar.zst" | docker load >/dev/null
    else docker pull -q "$BASE" >/dev/null || { echo "[layer] $IID: PULL FAILED"; FAILED+=("$IID"); continue; }
    fi
  fi
  echo "[layer] $IID: building"
  if docker build -q --build-arg BASE="$BASE" -t "promax-lsp:${IID}" -f "$CTX/Dockerfile" "$CTX" > /dev/null; then
    echo "[layer] $IID: verifying lsp inside image"
    if docker run --rm "promax-lsp:${IID}" bash -c 'cd /testbed && f=$(git ls-files "*.py" | xargs -r ls -S 2>/dev/null | head -1) && [ -n "$f" ] && timeout 240 lsp outline "$f" >/dev/null'; then
      docker save "promax-lsp:${IID}" | zstd -T0 -o "${OUT_TAR}.tmp" && mv "${OUT_TAR}.tmp" "$OUT_TAR"
      echo "[layer] $IID: OK ($(du -h "$OUT_TAR" | cut -f1))"
    else
      echo "[layer] $IID: VERIFY FAILED"; FAILED+=("$IID")
    fi
  else
    echo "[layer] $IID: BUILD FAILED"; FAILED+=("$IID")
  fi
  docker rmi "promax-lsp:${IID}" "$BASE" >/dev/null 2>&1
  docker builder prune -f >/dev/null 2>&1
done
echo "[layer] complete. failed: ${FAILED[*]:-none}"
