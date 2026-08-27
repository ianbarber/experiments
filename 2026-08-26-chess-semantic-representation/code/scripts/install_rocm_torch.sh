#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${project_root}/.venv/bin/python"

if [[ ! -x "${python_bin}" ]]; then
  echo "Missing ${python_bin}; run 'uv venv --python 3.12 .venv' first." >&2
  exit 1
fi

wheel_root="https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2"
torch_wheel="${wheel_root}/torch-2.9.1%2Brocm7.2.0.lw.git7e1940d4-cp312-cp312-linux_x86_64.whl"
torchvision_wheel="${wheel_root}/torchvision-0.24.0%2Brocm7.2.0.gitb919bd0c-cp312-cp312-linux_x86_64.whl"
triton_wheel="${wheel_root}/triton-3.5.1%2Brocm7.2.0.gita272dfa8-cp312-cp312-linux_x86_64.whl"

uv pip install --python "${python_bin}" --upgrade \
  "numpy==1.26.4" \
  "${torch_wheel}" \
  "${torchvision_wheel}" \
  "${triton_wheel}"

"${python_bin}" -c 'import torch, torchvision; print(torch.__version__); print(torchvision.__version__)'

