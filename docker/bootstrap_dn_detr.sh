#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/NYCU_VRDL_HW2}"

mkdir -p "${HOME:-/tmp}" "${PYTHONUSERBASE:-/tmp/python-user-base}"
export PYTHONPATH="${PYTHONUSERBASE}/lib/python3.10/site-packages${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="/usr/local/lib/python3.10/dist-packages/torch/lib:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

cd "$PROJECT_ROOT"

echo "[bootstrap] Verifying CUDA and PyTorch versions"
python -c "import torch; print(torch.__version__, torch.version.cuda)"
nvcc --version

echo "[bootstrap] Installing DN-DETR Python requirements into the user site"
python -m pip install --user --no-build-isolation -r DN-DETR/requirements.txt

echo "[bootstrap] Building MultiScaleDeformableAttention"
cd DN-DETR/models/dn_dab_deformable_detr/ops
rm -rf build MultiScaleDeformableAttention.egg-info
python setup.py build install --user

echo "[bootstrap] Verifying compiled extension import"
python -c "import torch; import MultiScaleDeformableAttention; print('MultiScaleDeformableAttention import ok')"
