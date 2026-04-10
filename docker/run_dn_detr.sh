#!/usr/bin/env bash

set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-nycu-vrdl-hw2-dn-detr:cu128}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST_USER="${USER:-$(id -un)}"
GPU_DEVICE="${GPU_DEVICE:-1}"

mkdir -p "$PROJECT_ROOT/.docker-home" "$PROJECT_ROOT/.docker-python"

if [ "$#" -eq 0 ]; then
  set -- bash
fi

docker run --rm -it \
  --gpus "device=${GPU_DEVICE}" \
  --ipc=host \
  -p 6006:6006 \
  --user "$(id -u):$(id -g)" \
  -e HOME=/workspace/NYCU_VRDL_HW2/.docker-home \
  -e PYTHONUSERBASE=/workspace/NYCU_VRDL_HW2/.docker-python \
  -e PYTHONPATH=/workspace/NYCU_VRDL_HW2/.docker-python/lib/python3.10/site-packages \
  -e LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/torch/lib:/usr/local/cuda/lib64 \
  -e USER="$HOST_USER" \
  -e LOGNAME="$HOST_USER" \
  -e LNAME="$HOST_USER" \
  -e USERNAME="$HOST_USER" \
  -e NVIDIA_VISIBLE_DEVICES="${GPU_DEVICE}" \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e TORCHINDUCTOR_CACHE_DIR=/workspace/NYCU_VRDL_HW2/.docker-home/.cache/torch/inductor \
  -e PROJECT_ROOT=/workspace/NYCU_VRDL_HW2 \
  -v "$PROJECT_ROOT":/workspace/NYCU_VRDL_HW2 \
  -w /workspace/NYCU_VRDL_HW2 \
  "$IMAGE_NAME" "$@"
