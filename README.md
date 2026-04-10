# Visual Recognition using Deep Learning 2026 Spring, Homework 2


Student Name: HSU, PAO-HUA
Student ID: 314581025

## Introduction

This homework includes two Python scripts:
- `data_augmentation.py`: offline augmentation with 1 image -> 3 images.
- `main.py`: two-stage training/validation with optional inference.

## Conda Environment

```bash
conda create -n nycu-cv-hw2 python=3.10 -y
conda activate nycu-cv-hw2
pip install torch torchvision pillow tqdm scipy tensorboard
pip install pycocotools
pip install --user gdown
```

### get the dataset

```bash
gdown https://drive.google.com/uc?id=13JXJ_hIdcloC63sS-vF3wFQLsUP1sMz5
tar -xvf nycu-hw2-data.tar.gz
```

## Train And Predict

If you do not already have a trained checkpoint, use `train` first. You can also train and then automatically run inference in the same command with `--predict-after-train`.

### Train and Predict

```bash
  python src/main.py train \
  --tensorboard \
  --batch-size 32 \
  --epochs 80 \
  --lr 1e-4 \
  --lr-backbone 1e-5 \
  --weight-decay 1e-4 \
  --lr-drop 70 \
  --lr-gamma 0.1 \
  --clip-max-norm 0.1 \
  --cost-class 4.0 \
  --cost-bbox 5.0 \
  --cost-giou 2.0 \
  --ce-loss-coef 2.0 \
  --eos-coef 0.1 \
  --label-smoothing 0.05 \
  --amp \
  --freeze-transformer \
  --early-stop-patience 10 \
  --checkpoint-dir checkpoints/detr_onPretrain \
  --output pred_freeze_transformer.json
```

`train` 目前預設會在結束後自動做一次 inference。若你只想訓練、不想自動輸出預測，可以加上 `--no-predict-after-train`。

如果訓練結束後顯示 `Saved 0 predictions`，通常代表 checkpoint 雖然有成功載入，但目前分數都低於 `--score-threshold`。現在 predict 會額外列出兩組統計：
- threshold 過濾後真正輸出的 boxes 分數平均值
- 每張圖在過濾前的最佳 query 分數平均值

如果後者也很低，通常代表模型本身還沒學起來；如果後者不低、但前者是 0，通常只是 threshold 太高，這時可以改成 `0.2` 或 `0.1` 再重跑 predict。

### Open tensorboard

```bash
tensorboard --logdir tensorboard_baseline/run1 --port 6006
```

## DN-Deformable-DETR-R50

This repo also includes a wrapper that keeps the homework baseline data augmentation policy but swaps the detector to official `DN-Deformable-DETR-R50`.

The DN-DETR setup, Docker image, CUDA 12.8 notes, custom op build steps, and training commands have been moved to [README-DN-DETR.md](/home/ben/nycu_hw/NYCU_VRDL_HW2/README-DN-DETR.md).
### Direct Predict With An Existing Checkpoint

If you already have a trained checkpoint, you can run inference directly:

```bash
python src/main.py predict --checkpoint /home/ben/nycu_hw/NYCU_VRDL_HW2/checkpoints/detr_onPretrain/best.pt --output pred_baseline_onPretrainv2.json --score-threshold 0.1 --amp
```
