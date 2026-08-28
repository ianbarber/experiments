#!/bin/bash
# Fetch a ProMax instance image: local docker -> NAS tar cache -> Docker Hub (and seed NAS).
set -euo pipefail
IID="$1"
IMG="key4127/refactor-dockerhub:${IID}"
NAS_TAR="/mnt/nas/refactorbench/images/${IID}.tar.zst"
if docker image inspect "$IMG" >/dev/null 2>&1; then
  echo "[fetch] $IID: already local"
elif [[ -f "$NAS_TAR" ]]; then
  echo "[fetch] $IID: loading from NAS cache"
  zstd -dc "$NAS_TAR" | docker load >/dev/null
else
  echo "[fetch] $IID: pulling from Docker Hub"
  docker pull -q "$IMG" >/dev/null
  mkdir -p /mnt/nas/refactorbench/images
  docker save "$IMG" | zstd -T0 -o "${NAS_TAR}.tmp" && mv "${NAS_TAR}.tmp" "$NAS_TAR"
fi
