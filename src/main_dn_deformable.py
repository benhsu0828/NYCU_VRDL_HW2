#!/usr/bin/env python3
"""Train and predict with DN-Deformable-DETR-R50 on NYCU HW2.

This wrapper keeps the homework baseline data pipeline and augmentation policy
from `src/main.py`, while swapping the detector backbone/transformer stack to
the official DN-Deformable-DETR-R50 implementation.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from types import MethodType
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.ops import batched_nms

import main as baseline

PROJECT_ROOT = Path(__file__).resolve().parent.parent
THIRD_PARTY_DN_DETR = PROJECT_ROOT / "DN-DETR"
DEFAULT_DATA_ROOT = PROJECT_ROOT / "nycu-hw2-data"
DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "dn_deformable_detr"
DEFAULT_PRED_PATH = PROJECT_ROOT / "pred_dn_deformable.json"

if str(THIRD_PARTY_DN_DETR) not in sys.path:
    sys.path.insert(0, str(THIRD_PARTY_DN_DETR))

DN_IMPORT_ERROR: Exception | None = None
try:
    from engine import train_one_epoch as dn_train_one_epoch
    from models.dn_dab_deformable_detr import build_dab_deformable_detr
    from util.misc import NestedTensor, accuracy
except Exception as exc:  # pragma: no cover - depends on runtime environment
    DN_IMPORT_ERROR = exc
    dn_train_one_epoch = None
    build_dab_deformable_detr = None
    NestedTensor = None
    accuracy = None


def require_dn_detr() -> None:
    if DN_IMPORT_ERROR is None:
        return
    raise RuntimeError(
        "Failed to import DN-Deformable-DETR. Make sure your training environment has "
        "the official dependencies installed and the deformable attention operator built.\n"
        f"Expected repo: {THIRD_PARTY_DN_DETR}\n"
        "Compile with:\n"
        f"  cd {THIRD_PARTY_DN_DETR / 'models' / 'dn_dab_deformable_detr' / 'ops'}\n"
        "  python setup.py build install\n"
        f"Original import error: {DN_IMPORT_ERROR}"
    )


def add_shared_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--image-size", type=int, default=720)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--tensorboard", action="store_true")
    parser.add_argument(
        "--tensorboard-dir",
        type=Path,
        default=PROJECT_ROOT / "tensorboard_dn_deformable" / "run1",
    )

    parser.add_argument("--backbone", type=str, default="resnet50")
    parser.add_argument("--no-pretrained-backbone", action="store_true")
    parser.add_argument("--dilation", action="store_true")
    parser.add_argument("--position-embedding", type=str, default="sine", choices=("sine", "learned"))
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dim-feedforward", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--nheads", type=int, default=8)
    parser.add_argument("--enc-layers", type=int, default=6)
    parser.add_argument("--dec-layers", type=int, default=6)
    parser.add_argument("--num-queries", type=int, default=300)
    parser.add_argument("--num-feature-levels", type=int, default=4)
    parser.add_argument("--enc-n-points", type=int, default=4)
    parser.add_argument("--dec-n-points", type=int, default=4)
    parser.add_argument("--num-patterns", type=int, default=0)
    parser.add_argument("--two-stage", action="store_true")
    parser.add_argument("--random-refpoints-xy", action="store_true")
    parser.add_argument("--no-aux-loss", dest="aux_loss", action="store_false")
    parser.set_defaults(aux_loss=True)

    parser.add_argument("--score-threshold", type=float, default=0.2)
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Optional max detections per image after thresholding and NMS; 0 disables the cap.",
    )
    parser.add_argument(
        "--nms-iou-threshold",
        type=float,
        default=0,
        help="Class-aware NMS IoU threshold after score filtering; set <= 0 to disable NMS.",
    )


def add_model_training_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--lr-backbone", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lr-gamma", type=float, default=0.1)
    parser.add_argument("--plateau-patience", type=int, default=2)
    parser.add_argument("--plateau-threshold", type=float, default=1e-4)
    parser.add_argument("--clip-max-norm", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=40)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument(
        "--pretrained-checkpoint",
        type=Path,
        default=PROJECT_ROOT / "R50_v2" / "checkpoint.pth",
        help="Optional pretrained DN-DETR checkpoint to warm-start model weights before fine-tuning.",
    )
    parser.add_argument(
        "--no-pretrained-checkpoint",
        action="store_true",
        help="Disable warm-start from DN-DETR checkpoint; keep only backbone pretrained initialization.",
    )
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--eval-every-epoch", action="store_true")
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=3,
        help="Run COCO mAP evaluation every N epochs; ignored when --eval-every-epoch is set.",
    )
    parser.add_argument("--predict-after-train", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument(
        "--freeze-transformer",
        action="store_true",
        help="Freeze the DN-DETR transformer block during training while keeping other modules trainable.",
    )

    parser.add_argument("--set-cost-class", type=float, default=2.0)
    parser.add_argument("--set-cost-bbox", type=float, default=5.0)
    parser.add_argument("--set-cost-giou", type=float, default=3.0)
    parser.add_argument("--cls-loss-coef", type=float, default=2.0)
    parser.add_argument("--bbox-loss-coef", type=float, default=5.0)
    parser.add_argument("--giou-loss-coef", type=float, default=3.0)
    parser.add_argument("--eos-coef", type=float, default=0.05)
    parser.add_argument("--focal-alpha", type=float, default=0.25)
    parser.add_argument("--no-class-balance", action="store_true")
    parser.add_argument("--class-balance-power", type=float, default=0.5)
    parser.add_argument("--class-balance-min", type=float, default=0.5)
    parser.add_argument("--class-balance-max", type=float, default=3.5)

    parser.add_argument("--use-dn", action="store_true", default=True)
    parser.add_argument("--scalar", type=int, default=2)
    parser.add_argument("--label-noise-scale", type=float, default=0.0)
    parser.add_argument("--box-noise-scale", type=float, default=0.2)
    parser.add_argument("--no-augmentation", action="store_true")
    parser.add_argument(
        "--canvas-scale-factor",
        type=float,
        default=1.2,
        help="Canvas expansion factor for train-time random placement augmentation.",
    )
    parser.add_argument(
        "--random-expand-noise-std",
        type=float,
        default=0.08,
        help="Std of Gaussian noise used to fill expanded canvas background (in normalized tensor space).",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train or run inference with DN-Deformable-DETR-R50.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train DN-Deformable-DETR-R50 on train/valid COCO annotations.")
    add_shared_arguments(train_parser)
    add_model_training_arguments(train_parser)
    train_parser.add_argument("--train-dir", type=Path, default=DEFAULT_DATA_ROOT / "train")
    train_parser.add_argument("--valid-dir", type=Path, default=DEFAULT_DATA_ROOT / "valid")
    train_parser.add_argument("--test-dir", type=Path, default=DEFAULT_DATA_ROOT / "test")
    train_parser.add_argument("--train-json", type=Path, default=DEFAULT_DATA_ROOT / "train.json")
    train_parser.add_argument("--valid-json", type=Path, default=DEFAULT_DATA_ROOT / "valid.json")
    train_parser.add_argument("--output", type=Path, default=DEFAULT_PRED_PATH)

    predict_parser = subparsers.add_parser("predict", help="Run inference with a trained DN-Deformable-DETR-R50 checkpoint.")
    add_shared_arguments(predict_parser)
    add_model_training_arguments(predict_parser)
    predict_parser.add_argument("--test-dir", type=Path, default=DEFAULT_DATA_ROOT / "test")
    predict_parser.add_argument("--checkpoint", type=Path, required=True)
    predict_parser.add_argument("--output", type=Path, default=DEFAULT_PRED_PATH)

    args = parser.parse_args()
    if getattr(args, "no_pretrained_checkpoint", False):
        args.pretrained_checkpoint = None
    args.dataset_file = "custom_coco"
    args.modelname = "dn_dab_deformable_detr"
    args.masks = False
    args.frozen_weights = None
    args.num_classes = 10
    args.pretrained_backbone = not args.no_pretrained_backbone
    args.contrastive = False
    args.use_mqs = False
    args.use_lft = False
    args.debug = False
    args.save_results = False
    args.save_log = False
    args.output_dir = str(args.checkpoint_dir)
    args.num_select = args.num_queries
    args.num_results = args.num_queries
    args.pe_temperatureH = 20
    args.pe_temperatureW = 20
    args.batch_norm_type = "FrozenBatchNorm2d"
    args.return_interm_layers = False
    args.backbone_freeze_keywords = None
    args.pre_norm = False
    args.transformer_activation = "relu"
    args.fix_size = False
    args.remove_difficult = False
    args.world_size = 1
    args.rank = 0
    args.local_rank = 0
    args.dist_url = "env://"
    args.find_unused_params = False
    if not args.device:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    return args


def dn_collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]]) -> tuple[Any, list[dict[str, torch.Tensor]]]:
    images, masks, targets = zip(*batch)
    max_height = max(int(image.shape[-2]) for image in images)
    max_width = max(int(image.shape[-1]) for image in images)

    padded_images: list[torch.Tensor] = []
    padded_masks: list[torch.Tensor] = []
    updated_targets: list[dict[str, torch.Tensor]] = []

    for image, mask, target in zip(images, masks, targets):
        height = int(image.shape[-2])
        width = int(image.shape[-1])
        padded_image = torch.zeros((image.shape[0], max_height, max_width), dtype=image.dtype)
        padded_image[:, :height, :width] = image
        padded_mask = torch.ones((max_height, max_width), dtype=mask.dtype)
        padded_mask[:height, :width] = mask

        updated_target = dict(target)
        updated_target["size"] = torch.tensor([max_height, max_width], dtype=torch.int64)
        if updated_target["boxes_abs"].numel() > 0:
            normalized_boxes = baseline.box_xyxy_to_cxcywh(updated_target["boxes_abs"])
            normalized_boxes[:, 0] /= float(max_width)
            normalized_boxes[:, 2] /= float(max_width)
            normalized_boxes[:, 1] /= float(max_height)
            normalized_boxes[:, 3] /= float(max_height)
            updated_target["boxes"] = normalized_boxes
        else:
            updated_target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)

        padded_images.append(padded_image)
        padded_masks.append(padded_mask)
        updated_targets.append(updated_target)

    samples = NestedTensor(torch.stack(padded_images, dim=0), torch.stack(padded_masks, dim=0))
    return samples, updated_targets


class BatchMaxPadDetrTransform:
    """Resize to fit within a max side length, then pad per batch in collate."""

    def __init__(
        self,
        image_size: int,
        augment: bool = False,
        canvas_scale_factor: float = 1.5,
        random_expand_noise_std: float = 0.08,
        mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> None:
        self.image_size = image_size
        self.augment = augment
        self.canvas_scale_factor = max(1.0, float(canvas_scale_factor))
        self.random_expand_noise_std = max(0.0, float(random_expand_noise_std))
        self.to_tensor = baseline.transforms.ToTensor()
        self.normalize = baseline.transforms.Normalize(mean=mean, std=std)
        self.color_jitter = baseline.transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.1,
            hue=0.02,
        )

    def __call__(
        self,
        image,
        boxes_xywh: torch.Tensor | None,
        image_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        image = image.convert("RGB")
        orig_width, orig_height = image.size
        if self.augment and torch.rand(1).item() < 0.8:
            image = self.color_jitter(image)

        if self.augment:
            # Keep original size (scale=1.0), then paste onto a larger noise canvas.
            scale = 1.0
            resized_width = max(1, int(orig_width))
            resized_height = max(1, int(orig_height))
            image_tensor = self.normalize(self.to_tensor(image))
            canvas_width = max(1, int(round(orig_width * self.canvas_scale_factor)))
            canvas_height = max(1, int(round(orig_height * self.canvas_scale_factor)))
            canvas_width = max(canvas_width, resized_width)
            canvas_height = max(canvas_height, resized_height)
            max_top = canvas_height - resized_height
            max_left = canvas_width - resized_width
            top = int(torch.randint(0, max_top + 1, (1,)).item()) if max_top > 0 else 0
            left = int(torch.randint(0, max_left + 1, (1,)).item()) if max_left > 0 else 0
            canvas = torch.randn((3, canvas_height, canvas_width), dtype=image_tensor.dtype) * self.random_expand_noise_std
            canvas[:, top : top + resized_height, left : left + resized_width] = image_tensor
            image_tensor = canvas
            mask = torch.ones((canvas_height, canvas_width), dtype=torch.bool)
            mask[top : top + resized_height, left : left + resized_width] = False
        else:
            # Validation/test: keep original resolution and rely on batch-max
            # padding in collate_fn for batching.
            scale = 1.0
            resized_width = max(1, int(orig_width))
            resized_height = max(1, int(orig_height))
            image_tensor = self.normalize(self.to_tensor(image))
            canvas_height = resized_height
            canvas_width = resized_width
            top = 0
            left = 0
            mask = torch.zeros((canvas_height, canvas_width), dtype=torch.bool)

        target: dict[str, torch.Tensor] = {
            "image_id": torch.tensor(image_id, dtype=torch.int64),
            "orig_size": torch.tensor([orig_height, orig_width], dtype=torch.int64),
            "size": torch.tensor([canvas_height, canvas_width], dtype=torch.int64),
            "resized_size": torch.tensor([resized_height, resized_width], dtype=torch.int64),
            "scale": torch.tensor(scale, dtype=torch.float32),
            "offset": torch.tensor([top, left], dtype=torch.int64),
        }

        if boxes_xywh is None:
            target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
            target["boxes_abs"] = torch.zeros((0, 4), dtype=torch.float32)
            target["labels"] = torch.zeros((0,), dtype=torch.int64)
            return image_tensor, mask, target

        scaled_boxes_xywh = boxes_xywh.clone().float()
        scaled_boxes_xywh[:, :4] *= scale
        boxes_abs = baseline.box_xywh_to_xyxy(scaled_boxes_xywh)
        if top != 0 or left != 0:
            boxes_abs[:, 0::2] += float(left)
            boxes_abs[:, 1::2] += float(top)
        boxes_abs = baseline.clip_xyxy_to_image(boxes_abs, float(canvas_width), float(canvas_height))
        boxes_norm = baseline.box_xyxy_to_cxcywh(boxes_abs)
        boxes_norm[:, 0] /= float(canvas_width)
        boxes_norm[:, 2] /= float(canvas_width)
        boxes_norm[:, 1] /= float(canvas_height)
        boxes_norm[:, 3] /= float(canvas_height)

        target["boxes_abs"] = boxes_abs
        target["boxes"] = boxes_norm
        return image_tensor, mask, target


def build_dataloaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader]:
    train_transform = BatchMaxPadDetrTransform(
        image_size=args.image_size,
        augment=not args.no_augmentation,
        canvas_scale_factor=args.canvas_scale_factor,
        random_expand_noise_std=args.random_expand_noise_std,
    )
    valid_transform = BatchMaxPadDetrTransform(image_size=args.image_size, augment=False)
    train_dataset = baseline.CocoDigitDataset(
        image_dir=args.train_dir,
        annotation_path=args.train_json,
        transform=train_transform,
    )
    valid_dataset = baseline.CocoDigitDataset(
        image_dir=args.valid_dir,
        annotation_path=args.valid_json,
        transform=valid_transform,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=dn_collate_fn,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=dn_collate_fn,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    return train_loader, valid_loader


def build_test_dataloader(args: argparse.Namespace) -> DataLoader:
    transform = BatchMaxPadDetrTransform(image_size=args.image_size, augment=False)
    dataset = baseline.TestDigitDataset(image_dir=args.test_dir, transform=transform)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=dn_collate_fn,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )


def build_model(args: argparse.Namespace):
    require_dn_detr()
    return build_dab_deformable_detr(args)


def class_balanced_sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_boxes: float,
    alpha: float = 0.25,
    gamma: float = 2.0,
    positive_class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1.0 - prob) * (1.0 - targets)
    loss = ce_loss * ((1.0 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
        loss = alpha_t * loss

    if positive_class_weights is not None:
        class_weights = positive_class_weights.view(*([1] * (targets.ndim - 1)), -1)
        loss = loss * torch.where(targets > 0, class_weights, torch.ones_like(loss))

    return loss.mean(1).sum() / max(float(num_boxes), 1.0)


def configure_dn_classification_loss(
    criterion: torch.nn.Module,
    class_weight: torch.Tensor | None,
) -> None:
    if class_weight is None:
        return
    if accuracy is None:
        raise RuntimeError("DN accuracy helper is unavailable; DN-DETR import did not complete successfully.")

    criterion.register_buffer("positive_class_weight", class_weight)

    def loss_labels(
        self,
        outputs: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
        indices: list[tuple[torch.Tensor, torch.Tensor]],
        num_boxes: float,
        log: bool = True,
    ) -> dict[str, torch.Tensor]:
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][j] for t, (_, j) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros(
            [src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
            dtype=src_logits.dtype,
            layout=src_logits.layout,
            device=src_logits.device,
        )
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
        target_classes_onehot = target_classes_onehot[:, :, :-1]

        loss_ce = class_balanced_sigmoid_focal_loss(
            src_logits,
            target_classes_onehot,
            num_boxes,
            alpha=self.focal_alpha,
            gamma=2.0,
            positive_class_weights=self.positive_class_weight,
        ) * src_logits.shape[1]
        losses = {"loss_ce": loss_ce}

        if log:
            losses["class_error"] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    criterion.loss_labels = MethodType(loss_labels, criterion)


def apply_training_freezes(model: torch.nn.Module, args: argparse.Namespace) -> list[str]:
    frozen_modules: list[str] = []
    if args.freeze_transformer:
        for name, parameter in model.named_parameters():
            if name.startswith("transformer."):
                parameter.requires_grad_(False)
        frozen_modules.append("transformer")
    return frozen_modules


def move_targets_to_device(targets: list[dict[str, torch.Tensor]], device: torch.device) -> list[dict[str, torch.Tensor]]:
    return [{key: value.to(device) for key, value in target.items()} for target in targets]


def forward_for_eval(model: torch.nn.Module, samples: Any, args: argparse.Namespace):
    with torch.amp.autocast("cuda", enabled=args.amp and torch.cuda.is_available()):
        if args.use_dn:
            outputs, _ = model(samples, dn_args=args.num_patterns)
        else:
            outputs = model(samples)
    return outputs


@torch.no_grad()
def evaluate_loss(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, float]:
    model.eval()
    criterion.eval()

    running: dict[str, float] = {}
    num_batches = 0
    for samples, targets in dataloader:
        samples = samples.to(device)
        targets = move_targets_to_device(targets, device)
        outputs = forward_for_eval(model, samples, args)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        loss_total = sum(loss_dict[name] * weight_dict[name] for name in loss_dict if name in weight_dict)

        stats = {
            "loss_total": float(loss_total.detach().cpu()),
            "loss_ce": float(loss_dict.get("loss_ce", torch.tensor(0.0, device=device)).detach().cpu()),
            "loss_bbox": float(loss_dict.get("loss_bbox", torch.tensor(0.0, device=device)).detach().cpu()),
            "loss_giou": float(loss_dict.get("loss_giou", torch.tensor(0.0, device=device)).detach().cpu()),
            "class_error": float(loss_dict.get("class_error", torch.tensor(0.0, device=device)).detach().cpu()),
        }
        for key, value in stats.items():
            running[key] = running.get(key, 0.0) + value
        num_batches += 1

    num_batches = max(num_batches, 1)
    return {key: value / num_batches for key, value in running.items()}


@torch.no_grad()
def generate_predictions(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[list[dict[str, float]], baseline.PredictionScoreSummary]:
    model.eval()
    predictions: list[dict[str, float]] = []
    raw_top_scores: list[float] = []

    for samples, targets in dataloader:
        samples = samples.to(device)
        outputs = forward_for_eval(model, samples, args)

        pred_logits = outputs["pred_logits"].sigmoid()
        pred_boxes = outputs["pred_boxes"]

        scores_per_query, labels_per_query = pred_logits.max(dim=-1)
        raw_top_scores.extend(scores_per_query.max(dim=-1).values.detach().cpu().tolist())
        for batch_idx in range(pred_boxes.shape[0]):
            scores = scores_per_query[batch_idx]
            labels = labels_per_query[batch_idx]
            padded_height = int(targets[batch_idx]["size"][0].item())
            padded_width = int(targets[batch_idx]["size"][1].item())
            scale_factors = pred_boxes[batch_idx].new_tensor([padded_width, padded_height, padded_width, padded_height])
            boxes_xyxy = baseline.box_cxcywh_to_xyxy(pred_boxes[batch_idx]) * scale_factors

            scale = float(targets[batch_idx]["scale"].item())
            orig_height = int(targets[batch_idx]["orig_size"][0].item())
            orig_width = int(targets[batch_idx]["orig_size"][1].item())
            resized_height = int(targets[batch_idx]["resized_size"][0].item())
            resized_width = int(targets[batch_idx]["resized_size"][1].item())
            top = int(targets[batch_idx]["offset"][0].item())
            left = int(targets[batch_idx]["offset"][1].item())

            boxes_xyxy[:, 0::2] -= float(left)
            boxes_xyxy[:, 1::2] -= float(top)
            boxes_xyxy = baseline.clip_xyxy_to_image(boxes_xyxy, float(resized_width), float(resized_height))
            boxes_xyxy[:, 0::2] /= max(scale, 1e-6)
            boxes_xyxy[:, 1::2] /= max(scale, 1e-6)
            boxes_xyxy = baseline.clip_xyxy_to_image(boxes_xyxy, float(orig_width), float(orig_height))

            keep = torch.where(scores >= args.score_threshold)[0]
            if len(keep) == 0:
                continue
            keep = keep[scores[keep].argsort(descending=True)]
            if args.nms_iou_threshold > 0.0:
                nms_keep = batched_nms(
                    boxes_xyxy[keep],
                    scores[keep],
                    labels[keep],
                    args.nms_iou_threshold,
                )
                keep = keep[nms_keep]
            if args.top_k > 0:
                keep = keep[: args.top_k]

            for idx in keep.tolist():
                x0, y0, x1, y1 = boxes_xyxy[idx].tolist()
                predictions.append(
                    {
                        "image_id": int(targets[batch_idx]["image_id"].item()),
                        "bbox": [
                            float(x0),
                            float(y0),
                            float(max(x1 - x0, 0.0)),
                            float(max(y1 - y0, 0.0)),
                        ],
                        "score": float(scores[idx].item()),
                        "category_id": int(labels[idx].item() + 1),
                    }
                )

    predictions.sort(key=lambda item: (item["image_id"], item["bbox"][0], item["bbox"][1]))
    return predictions, baseline.summarize_prediction_scores(predictions, raw_top_scores)


def save_checkpoint(
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    best_valid_loss: float,
    best_map: float,
    args: argparse.Namespace,
) -> None:
    serialized_args = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_valid_loss": best_valid_loss,
            "best_map": best_map,
            "args": serialized_args,
        },
        checkpoint_path,
    )


def load_checkpoint(
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    return checkpoint


def load_pretrained_model_weights(
    checkpoint_path: Path,
    model: torch.nn.Module,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model", checkpoint)
    model_state = model.state_dict()

    compatible_state: dict[str, torch.Tensor] = {}
    skipped_keys: list[str] = []
    for key, value in state_dict.items():
        if key not in model_state:
            skipped_keys.append(key)
            continue
        if model_state[key].shape != value.shape:
            skipped_keys.append(key)
            continue
        compatible_state[key] = value

    missing_keys, unexpected_keys = model.load_state_dict(compatible_state, strict=False)
    loaded_count = len(compatible_state)
    print(
        f"Loaded pretrained weights from {checkpoint_path}: "
        f"{loaded_count} tensors matched, {len(skipped_keys)} skipped by name/shape."
    )
    if missing_keys:
        preview = ", ".join(missing_keys[:8])
        suffix = "..." if len(missing_keys) > 8 else ""
        print(f"Missing model keys after warm start: {preview}{suffix}")
    if unexpected_keys:
        preview = ", ".join(unexpected_keys[:8])
        suffix = "..." if len(unexpected_keys) > 8 else ""
        print(f"Unexpected pretrained keys ignored: {preview}{suffix}")
    checkpoint["_codex_skipped_keys"] = skipped_keys
    checkpoint["_codex_missing_keys"] = list(missing_keys)
    checkpoint["_codex_unexpected_keys"] = list(unexpected_keys)
    return checkpoint


def run_prediction_with_model(model: torch.nn.Module, args: argparse.Namespace, device: torch.device) -> None:
    infer_start_time = time.perf_counter()
    test_loader = build_test_dataloader(args)
    predictions, score_summary = generate_predictions(model=model, dataloader=test_loader, device=device, args=args)
    infer_elapsed = time.perf_counter() - infer_start_time

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(predictions, f)
    baseline.print_prediction_score_summary(score_summary, args.score_threshold)
    print(f"Saved {len(predictions)} predictions to {args.output}")
    print(f"Total inference time: {baseline.format_duration(infer_elapsed)}")


def run_train(args: argparse.Namespace) -> None:
    require_dn_detr()
    train_start_time = time.perf_counter()
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    train_loader, valid_loader = build_dataloaders(args)
    writer = SummaryWriter(log_dir=str(args.tensorboard_dir)) if args.tensorboard else None

    model, criterion, _ = build_model(args)
    model = model.to(device)
    criterion = criterion.to(device)
    frozen_modules = apply_training_freezes(model, args)
    class_weight = None
    if not args.no_class_balance:
        class_weight = baseline.build_class_weight_from_counts(
            train_loader.dataset.class_counts,
            power=args.class_balance_power,
            min_weight=args.class_balance_min,
            max_weight=args.class_balance_max,
        ).to(device)
        configure_dn_classification_loss(criterion, class_weight)

    total_params, trainable_params = baseline.count_parameters(model)
    print(
        f"Model parameters: total={baseline.format_param_count(total_params)} "
        f"trainable={baseline.format_param_count(trainable_params)}"
    )
    if frozen_modules:
        print(f"Frozen modules: {', '.join(frozen_modules)}")
    if class_weight is not None:
        rounded = [round(value, 3) for value in class_weight.detach().cpu().tolist()]
        print(f"Using class-balanced DN focal weights: {rounded}")
    if args.freeze_transformer:
        print(
            "Warning: --freeze-transformer is a high-risk setting for this 10-class fine-tuning task, "
            "especially because the DN classification head is reinitialized."
        )

    optimizer = AdamW(
        [
            {
                "params": [
                    parameter
                    for name, parameter in model.named_parameters()
                    if parameter.requires_grad and "backbone" not in name
                ],
                "lr": args.lr,
            },
            {
                "params": [
                    parameter
                    for name, parameter in model.named_parameters()
                    if parameter.requires_grad and "backbone" in name
                ],
                "lr": args.lr_backbone,
            },
        ],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_gamma,
        patience=args.plateau_patience,
        threshold=args.plateau_threshold,
    )

    start_epoch = 1
    best_valid_loss = float("inf")
    best_map = float("-inf")
    if args.resume is not None and args.resume.exists():
        checkpoint = load_checkpoint(args.resume, model, optimizer, scheduler, device=device)
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_valid_loss = float(checkpoint.get("best_valid_loss", float("inf")))
        best_map = float(checkpoint.get("best_map", float("-inf")))
        print(f"Resumed from {args.resume} at epoch {start_epoch}.")
    elif args.pretrained_checkpoint is not None:
        if not args.pretrained_checkpoint.exists():
            raise FileNotFoundError(f"Pretrained checkpoint not found: {args.pretrained_checkpoint}")
        warm_start = load_pretrained_model_weights(args.pretrained_checkpoint, model, device=device)
        skipped_keys = warm_start.get("_codex_skipped_keys", [])
        skipped_classification_keys = [
            key for key in skipped_keys if "class_embed" in key or "label_enc" in key
        ]
        if skipped_classification_keys:
            preview = ", ".join(skipped_classification_keys[:6])
            suffix = "..." if len(skipped_classification_keys) > 6 else ""
            print(
                "Warm start skipped DN classification tensors due to shape mismatch: "
                f"{preview}{suffix}"
            )
            if args.freeze_transformer:
                print(
                    "This means the classifier is learning almost from scratch while the transformer is frozen; "
                    "prefer removing --freeze-transformer first."
                )

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            train_stats = dn_train_one_epoch(
                model=model,
                criterion=criterion,
                data_loader=train_loader,
                optimizer=optimizer,
                device=device,
                epoch=epoch,
                max_norm=args.clip_max_norm,
                wo_class_error=False,
                lr_scheduler=scheduler,
                args=args,
                logger=None,
            )
            valid_stats = evaluate_loss(model, criterion, valid_loader, device, args)
            eval_stats: dict[str, float] | None = None
            should_eval_map = args.eval_every_epoch or (
                args.eval_interval > 0 and epoch % args.eval_interval == 0
            )
            if should_eval_map:
                predictions, _ = generate_predictions(model, valid_loader, device, args)
                eval_stats = baseline.maybe_run_coco_eval(args.valid_json, predictions)
            scheduler.step(valid_stats["loss_total"])

            status_parts = [
                f"epoch={epoch:03d}",
                f"train_loss={train_stats.get('loss', 0.0):.4f}",
                f"valid_loss={valid_stats['loss_total']:.4f}",
                f"train_cls_err={train_stats.get('class_error', 0.0):.4f}",
                f"valid_cls_err={valid_stats.get('class_error', 0.0):.4f}",
            ]
            if args.use_dn:
                status_parts.append(f"train_dn_ce={train_stats.get('tgt_loss_ce', 0.0):.4f}")
            if eval_stats is not None:
                status_parts.append(f"valid_mAP={eval_stats['map']:.4f}")
                status_parts.append(f"valid_AP50={eval_stats['ap50']:.4f}")
            print(" ".join(status_parts))

            if writer is not None:
                writer.add_scalar("train/loss_total", train_stats.get("loss", 0.0), epoch)
                writer.add_scalar("train/loss_ce", train_stats.get("loss_ce", 0.0), epoch)
                writer.add_scalar("train/loss_bbox", train_stats.get("loss_bbox", 0.0), epoch)
                writer.add_scalar("train/loss_giou", train_stats.get("loss_giou", 0.0), epoch)
                writer.add_scalar("train/class_error", train_stats.get("class_error", 0.0), epoch)
                writer.add_scalar("valid/loss_total", valid_stats["loss_total"], epoch)
                writer.add_scalar("valid/loss_ce", valid_stats.get("loss_ce", 0.0), epoch)
                writer.add_scalar("valid/loss_bbox", valid_stats.get("loss_bbox", 0.0), epoch)
                writer.add_scalar("valid/loss_giou", valid_stats.get("loss_giou", 0.0), epoch)
                writer.add_scalar("valid/class_error", valid_stats.get("class_error", 0.0), epoch)
                writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch)
                if len(optimizer.param_groups) > 1:
                    writer.add_scalar("train/lr_backbone", optimizer.param_groups[1]["lr"], epoch)
                if args.use_dn:
                    writer.add_scalar("train/dn_loss_ce", train_stats.get("tgt_loss_ce", 0.0), epoch)
                    writer.add_scalar("train/dn_loss_bbox", train_stats.get("tgt_loss_bbox", 0.0), epoch)
                    writer.add_scalar("train/dn_loss_giou", train_stats.get("tgt_loss_giou", 0.0), epoch)
                if eval_stats is not None:
                    writer.add_scalar("valid/mAP", eval_stats["map"], epoch)
                    writer.add_scalar("valid/AP50", eval_stats["ap50"], epoch)
                    writer.add_scalar("valid/AP75", eval_stats["ap75"], epoch)
                    writer.add_scalar("valid/AR100", eval_stats["ar100"], epoch)

            if valid_stats["loss_total"] < best_valid_loss:
                best_valid_loss = valid_stats["loss_total"]

            if eval_stats is not None and eval_stats["map"] > best_map:
                best_map = eval_stats["map"]
                best_map_path = args.checkpoint_dir / "best_map.pth"
                save_checkpoint(best_map_path, model, optimizer, scheduler, epoch, best_valid_loss, best_map, args)

            last_path = args.checkpoint_dir / "last.pth"
            save_checkpoint(last_path, model, optimizer, scheduler, epoch, best_valid_loss, best_map, args)
    finally:
        if writer is not None:
            writer.close()

    train_elapsed = time.perf_counter() - train_start_time
    best_map_path = args.checkpoint_dir / "best_map.pth"
    last_path = args.checkpoint_dir / "last.pth"
    print(f"Total training time: {baseline.format_duration(train_elapsed)}")
    if best_map_path.exists():
        print(f"Best mAP checkpoint: {best_map_path}")
    if last_path.exists():
        print(f"Last checkpoint: {last_path}")
    if args.tensorboard:
        print(f"TensorBoard logdir: {args.tensorboard_dir}")

    if args.predict_after_train:
        predict_checkpoint = best_map_path if best_map_path.exists() else args.checkpoint_dir / "last.pth"
        load_checkpoint(predict_checkpoint, model, device=device)
        print(f"Running prediction with checkpoint {predict_checkpoint}")
        run_prediction_with_model(model, args, device)


def run_predict(args: argparse.Namespace) -> None:
    require_dn_detr()
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model, _, _ = build_model(args)
    model = model.to(device)
    total_params, trainable_params = baseline.count_parameters(model)
    print(
        f"Model parameters: total={baseline.format_param_count(total_params)} "
        f"trainable={baseline.format_param_count(trainable_params)}"
    )
    checkpoint = load_checkpoint(args.checkpoint, model, device=device)
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}.")
    run_prediction_with_model(model, args, device)


def main() -> None:
    args = parse_args()
    baseline.set_seed(getattr(args, "seed", 42))
    if args.command == "train":
        run_train(args)
    else:
        run_predict(args)


if __name__ == "__main__":
    main()
