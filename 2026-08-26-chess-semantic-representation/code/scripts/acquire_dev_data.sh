#!/usr/bin/env bash
set -euo pipefail

# Research-only inputs. ChessSight is CC-BY-NC-4.0; ChessReD is
# CC-BY-NC-SA-4.0. Review those licenses before redistributing derivatives.

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"

mkdir -p data/raw/chessred/chessred2k data/raw/chesssight

download() {
  local url="$1"
  local destination="$2"
  curl -L --fail --retry 3 --continue-at - --output "$destination" "$url"
}

verify_sha256() {
  local destination="$1"
  local expected="$2"
  local observed
  observed="$(sha256sum "$destination" | cut -d' ' -f1)"
  if [[ "$observed" != "$expected" ]]; then
    echo "SHA-256 mismatch for $destination" >&2
    return 1
  fi
}

download \
  "https://data.4tu.nl/file/99b5c721-280b-450b-b058-b2900b69a90f/3cae6364-daca-4967-b426-1e4b68cdb64c" \
  data/raw/chessred/annotations.json
download \
  "https://data.4tu.nl/file/99b5c721-280b-450b-b058-b2900b69a90f/410e41c7-dde5-413f-8722-3e112363a1a2" \
  data/raw/chessred/ChessReD2K.zip
download \
  "https://huggingface.co/datasets/tchauffi/chesssight-synthetic-40k/resolve/main/data/train-00000-of-00006.parquet?download=true" \
  data/raw/chesssight/train-00000-of-00006.parquet
download \
  "https://huggingface.co/datasets/tchauffi/chesssight-synthetic-40k/resolve/main/data/validation-00000-of-00001.parquet?download=true" \
  data/raw/chesssight/validation-00000-of-00001.parquet

verify_sha256 data/raw/chessred/annotations.json \
  16e99d7e8535c0fc56507caa2aa5f7594d7ca076ebab3f5a432cfd4aa10668cc
verify_sha256 data/raw/chessred/ChessReD2K.zip \
  3e63d0c1fcf0f2af6598836c68a373ad12390d056cad6b5e46bcafb59551ad62
verify_sha256 data/raw/chesssight/train-00000-of-00006.parquet \
  005c8313f658431bb0ffa7effe8f2c11d974ed55f5a6a40dc487b7cadfbfcbac
verify_sha256 data/raw/chesssight/validation-00000-of-00001.parquet \
  1cbcbf8f00a69f71a34922402c6e5f8019a1379c990c4397c2a35c48294cec4b

unzip -q -n data/raw/chessred/ChessReD2K.zip -d data/raw/chessred/chessred2k

uv run python scripts/build_chessred_manifest.py
uv run python scripts/build_chesssight_manifest.py \
  data/raw/chesssight/train-00000-of-00006.parquet
uv run python scripts/build_chesssight_manifest.py \
  data/raw/chesssight/validation-00000-of-00001.parquet \
  --split validation \
  --output data/manifests/chesssight_validation.jsonl
uv run python scripts/audit_manifest.py \
  data/manifests/chesssight_train0.jsonl \
  data/manifests/chesssight_validation.jsonl \
  data/manifests/chessred2k.jsonl \
  --output runs/E0002/audit.json
uv run python scripts/build_position_disjoint_validation.py \
  --train-manifest data/manifests/chesssight_train0.jsonl \
  --train-manifest data/manifests/chessred2k.jsonl \
  --validation-manifest data/manifests/chesssight_validation.jsonl \
  --validation-manifest data/manifests/chessred2k.jsonl \
  --output data/manifests/position_disjoint_validation.jsonl
uv run python scripts/build_position_disjoint_validation.py \
  --train-manifest data/manifests/chesssight_train0.jsonl \
  --train-manifest data/manifests/chessred2k.jsonl \
  --validation-manifest data/manifests/chesssight_validation.jsonl \
  --validation-manifest data/manifests/chessred2k.jsonl \
  --output data/manifests/position_disjoint_validation_balanced.jsonl \
  --balanced-per-source 256
