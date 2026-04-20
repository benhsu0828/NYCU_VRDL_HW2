# DN-Deformable-DETR-R50

This project uses Docker for `DN-Deformable-DETR-R50` so the custom CUDA operator is built against CUDA `12.8` and PyTorch `cu128` inside the container instead of the host machine's older toolkit.

The PyTorch install command in the Docker image follows the official CUDA 12.8 wheel listed on the PyTorch previous versions page:
https://pytorch.org/get-started/previous-versions/

## Files

- [Dockerfile.dn_detr](/home/ben/nycu_hw/NYCU_VRDL_HW2/Dockerfile.dn_detr): CUDA 12.8 + PyTorch cu128 image for DN-DETR
- [docker/bootstrap_dn_detr.sh](/home/ben/nycu_hw/NYCU_VRDL_HW2/docker/bootstrap_dn_detr.sh): installs DN-DETR requirements and builds the deformable attention operator inside the container
- [docker/run_dn_detr.sh](/home/ben/nycu_hw/NYCU_VRDL_HW2/docker/run_dn_detr.sh): starts the container with GPU access, mounts this repo, and keeps generated files owned by your user

## Build the image

Run this once from the repo root:

```bash
docker build -t nycu-vrdl-hw2-dn-detr:cu128 -f Dockerfile.dn_detr .
```

If you already built the old `cu130` image, you can remove it:

```bash
docker rmi nycu-vrdl-hw2-dn-detr:cu130
```

## Start the container

To open an interactive shell:

```bash
./docker/run_dn_detr.sh
```

To run a single command directly:

```bash
./docker/run_dn_detr.sh python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

The helper script already does these things for you:
- mounts `/home/ben/nycu_hw/NYCU_VRDL_HW2` into the container
- exposes only one host GPU to the container by default
- maps TensorBoard port `6006`
- runs as your user instead of root, so generated files stay writable on the host

By default the script uses host GPU `0`. If your administrator gave you a different visible GPU index, you can override it:

```bash
GPU_DEVICE=0 ./docker/run_dn_detr.sh python -c "import torch; print(torch.cuda.device_count()); print(torch.cuda.get_device_name(0))"
```

Inside the container, the selected host GPU is remapped to `cuda:0`, so the training code still uses a single visible device.

## Bootstrap DN-DETR inside the container

After the image is built, run this once:

```bash
./docker/run_dn_detr.sh bootstrap-dn-detr
```

This step is intentionally run inside `docker run`, not during `docker build`, because DN-DETR's setup script checks CUDA availability before compiling the custom operator.

That command will:
- install `DN-DETR/requirements.txt`
- build `MultiScaleDeformableAttention`
- install the compiled extension into your project-local user site under `.docker-python/`

## Verify the environment

Check PyTorch and CUDA:

```bash
./docker/run_dn_detr.sh python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.is_available())"
./docker/run_dn_detr.sh nvcc --version
```

The expected versions are:

```text
torch 2.10.0+cu128
CUDA toolkit 12.8
```

Check the compiled operator:

```bash
./docker/run_dn_detr.sh python -c "import torch; import MultiScaleDeformableAttention; print('ok')"
```


## Train and Predict

```bash
# This is data augmentation version
# ./docker/run_dn_detr.sh python src/main_dn_deformable.py train \
#   --tensorboard \
#   --batch-size 24 \
#   --predict-after-train \
#   --epochs 100 \
#   --eval-interval 3 \
#   --train-dir nycu-hw2-data/train_canvas_offline \
#   --train-json nycu-hw2-data/train_canvas_offline.json \
#   --scalar 4 \
#   --label-noise-scale 0.05 \
#   --box-noise-scale 0.1 \
#   --dropout 0.1 \
#   --lr 3e-4 \
#   --canvas-scale-factor 1.0 \
#   --random-expand-noise-std 0.0 \
#   --lr-gamma 0.5 \
#   --plateau-patience 6 \
#   --plateau-threshold 5e-4 \
#   --image-size 720 \
#   --num-queries 250 \
#   --checkpoint-dir checkpoints/dn_deformable_aug \
#   --tensorboard-dir tensorboard_dn_deformable/data_aug3 \
#   --output predict_result/pred_dn_deformable_aug3.json

    # --eval-every-epoch \
    # --resume /home/ben/nycu_hw/NYCU_VRDL_HW2/checkpoints/dn_deformable_digits_V3/best_map.pth \

  ./docker/run_dn_detr.sh python src/main_dn_deformable.py train \
  --tensorboard \
  --batch-size 32 \
  --predict-after-train \
  --no-class-balance \
  --epochs 70 \
  --eval-interval 3 \
  --scalar 5 \
  --canvas-scale-factor 1.0 \
  --random-expand-noise-std 0.0 \
  --dropout 0.1 \
  --label-noise-scale 0.2 \
  --box-noise-scale 0.4 \
  --lr 2e-4 \
  --lr-backbone 1e-5 \
  --lr-gamma 0.5 \
  --plateau-patience 6 \
  --plateau-threshold 5e-4 \
  --image-size 720 \
  --num-queries 300 \
  --no-pretrained-checkpoint \
  --checkpoint-dir checkpoints/dn_deformable_final \
  --tensorboard-dir tensorboard_dn_deformable/final \
  --output predict_result/pred_dn_deformable_final.json
```

For this homework dataset, start without `--freeze-transformer`.
That flag tends to fail here because the COCO pretrained checkpoint cannot directly reuse the DN classification head after switching from COCO classes to 10 digit classes.

## Predict

```bash
./docker/run_dn_detr.sh python src/main_dn_deformable.py predict \
  --checkpoint checkpoints/dn_deformable_aug/best.pth \
  --output predict_result/pred_dn_deformable_aug.json \
  --nms-iou-threshold 0 \
  --score-threshold 0.1 \
  --top-k 10
```

## TensorBoard

Start TensorBoard in the container:

```bash
./docker/run_dn_detr.sh tensorboard --logdir tensorboard_dn_deformable/run1 --host 0.0.0.0 --port 6006
```

Then open `http://localhost:6006` on the host.
