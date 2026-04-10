#!/usr/bin/env python3
"""Custom DETR training and inference entry point for NYCU VRDL HW2.

This file intentionally keeps the DETR pipeline in one place so it is easy to
modify the backbone usage, transformer depth, query count, losses, and
post-processing while you experiment with your own model.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFile
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.ops import batched_nms, generalized_box_iou
from tqdm import tqdm

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
except ImportError:
    COCO = None
    COCOeval = None

if not hasattr(np, "float"):
    np.float = float  # type: ignore[attr-defined]

ImageFile.LOAD_TRUNCATED_IMAGES = True

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_ROOT = PROJECT_ROOT / "nycu-hw2-data"
DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
DEFAULT_PRED_PATH = PROJECT_ROOT / "pred.json"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def box_xywh_to_xyxy(boxes: Tensor) -> Tensor:
    x, y, w, h = boxes.unbind(-1)
    return torch.stack((x, y, x + w, y + h), dim=-1)


def box_xyxy_to_cxcywh(boxes: Tensor) -> Tensor:
    x0, y0, x1, y1 = boxes.unbind(-1)
    return torch.stack(((x0 + x1) / 2.0, (y0 + y1) / 2.0, x1 - x0, y1 - y0), dim=-1)


def box_cxcywh_to_xyxy(boxes: Tensor) -> Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack((cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h), dim=-1)


def clip_xyxy_to_image(boxes: Tensor, width: float, height: float) -> Tensor:
    boxes = boxes.clone()
    boxes[:, 0::2] = boxes[:, 0::2].clamp(0.0, width)
    boxes[:, 1::2] = boxes[:, 1::2].clamp(0.0, height)
    return boxes


def format_duration(seconds: float) -> str:
    total_seconds = max(int(round(seconds)), 0)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total_params, trainable_params


def format_param_count(num_params: int) -> str:
    return f"{num_params:,} ({num_params / 1_000_000:.2f}M)"


@dataclass
class PredictionScoreSummary:
    kept_count: int
    kept_mean: float | None
    kept_min: float | None
    kept_max: float | None
    raw_count: int
    raw_mean: float | None
    raw_min: float | None
    raw_max: float | None


def _summarize_scores(scores: list[float]) -> tuple[int, float | None, float | None, float | None]:
    if not scores:
        return 0, None, None, None
    return len(scores), float(sum(scores) / len(scores)), float(min(scores)), float(max(scores))


def summarize_prediction_scores(
    predictions: list[dict[str, float]],
    raw_top_scores: list[float],
) -> PredictionScoreSummary:
    kept_scores = [float(pred["score"]) for pred in predictions]
    kept_count, kept_mean, kept_min, kept_max = _summarize_scores(kept_scores)
    raw_count, raw_mean, raw_min, raw_max = _summarize_scores(raw_top_scores)
    return PredictionScoreSummary(
        kept_count=kept_count,
        kept_mean=kept_mean,
        kept_min=kept_min,
        kept_max=kept_max,
        raw_count=raw_count,
        raw_mean=raw_mean,
        raw_min=raw_min,
        raw_max=raw_max,
    )


def print_prediction_score_summary(summary: PredictionScoreSummary, score_threshold: float) -> None:
    if summary.kept_count > 0:
        print(
            "Prediction score stats after thresholding: "
            f"count={summary.kept_count} "
            f"mean={summary.kept_mean:.4f} "
            f"min={summary.kept_min:.4f} "
            f"max={summary.kept_max:.4f}"
        )
    else:
        print(
            "Prediction score stats after thresholding: "
            f"count=0 threshold={score_threshold:.3f}"
        )

    if summary.raw_count > 0:
        print(
            "Per-image best score before thresholding: "
            f"count={summary.raw_count} "
            f"mean={summary.raw_mean:.4f} "
            f"min={summary.raw_min:.4f} "
            f"max={summary.raw_max:.4f}"
        )

class FixedSizeDetrTransform:
    """Resize with aspect ratio preserved and pad to a fixed square canvas."""

    def __init__(
        self,
        image_size: int,
        augment: bool = False,
        mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> None:
        self.image_size = image_size
        self.augment = augment
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=mean, std=std)
        # Digits are not safely left-right flippable, so use photometric jitter
        # plus mild scale/translation augmentation instead of horizontal flips.
        self.color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02)

    def __call__(
        self,
        image: Image.Image,
        boxes_xywh: Tensor | None,
        image_id: int,
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        image = image.convert("RGB")
        orig_width, orig_height = image.size
        if self.augment and random.random() < 0.8:
            image = self.color_jitter(image)
        # Keep aspect ratio, then place the resized image on the top-left corner
        # of a fixed square canvas so every sample in a batch has the same shape.
        scale = min(self.image_size / max(orig_width, 1), self.image_size / max(orig_height, 1))
        if self.augment:
            scale *= random.uniform(0.8, 1.0)
        resized_width = max(1, int(round(orig_width * scale)))
        resized_height = max(1, int(round(orig_height * scale)))

        resized_image = image.resize((resized_width, resized_height), resample=Image.BILINEAR)
        image_tensor = self.normalize(self.to_tensor(resized_image))

        canvas = torch.zeros((3, self.image_size, self.image_size), dtype=image_tensor.dtype)
        max_top = max(self.image_size - resized_height, 0)
        max_left = max(self.image_size - resized_width, 0)
        if self.augment:
            top = random.randint(0, max_top) if max_top > 0 else 0
            left = random.randint(0, max_left) if max_left > 0 else 0
        else:
            top = 0
            left = 0
        canvas[:, top : top + resized_height, left : left + resized_width] = image_tensor

        mask = torch.ones((self.image_size, self.image_size), dtype=torch.bool)
        mask[top : top + resized_height, left : left + resized_width] = False

        target: dict[str, Tensor] = {
            "image_id": torch.tensor(image_id, dtype=torch.int64),
            "orig_size": torch.tensor([orig_height, orig_width], dtype=torch.int64),
            "size": torch.tensor([self.image_size, self.image_size], dtype=torch.int64),
            "resized_size": torch.tensor([resized_height, resized_width], dtype=torch.int64),
            "scale": torch.tensor(scale, dtype=torch.float32),
            "offset": torch.tensor([top, left], dtype=torch.int64),
        }

        if boxes_xywh is None:
            target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
            target["boxes_abs"] = torch.zeros((0, 4), dtype=torch.float32)
            target["labels"] = torch.zeros((0,), dtype=torch.int64)
            return canvas, mask, target

        scaled_boxes_xywh = boxes_xywh.clone().float()
        scaled_boxes_xywh[:, :4] *= scale
        boxes_abs = box_xywh_to_xyxy(scaled_boxes_xywh)
        boxes_abs[:, 0::2] += float(left)
        boxes_abs[:, 1::2] += float(top)
        boxes_abs = clip_xyxy_to_image(boxes_abs, float(self.image_size), float(self.image_size))
        # DETR predicts normalized cx, cy, w, h values, so ground-truth boxes are
        # converted to that format and divided by the fixed canvas size.
        boxes_norm = box_xyxy_to_cxcywh(boxes_abs) / float(self.image_size)

        target["boxes_abs"] = boxes_abs
        target["boxes"] = boxes_norm
        return canvas, mask, target


class CocoDigitDataset(Dataset):
    def __init__(self, image_dir: Path, annotation_path: Path, transform: FixedSizeDetrTransform) -> None:
        self.image_dir = image_dir
        self.annotation_path = annotation_path
        self.transform = transform

        with annotation_path.open("r", encoding="utf-8") as f:
            coco = json.load(f)

        self.categories = sorted(coco["categories"], key=lambda item: item["id"])
        self.num_classes = len(self.categories)
        self.id_to_label = {cat["id"]: idx for idx, cat in enumerate(self.categories)}
        self.class_counts = torch.zeros(self.num_classes, dtype=torch.float32)

        self.images = sorted(coco["images"], key=lambda item: item["id"])
        self.annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for ann in coco["annotations"]:
            self.annotations_by_image[ann["image_id"]].append(ann)
            self.class_counts[self.id_to_label[ann["category_id"]]] += 1.0

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        image_info = self.images[index]
        image_path = self.image_dir / image_info["file_name"]
        with Image.open(image_path) as img:
            image = img.copy()

        annotations = self.annotations_by_image.get(image_info["id"], [])
        if annotations:
            boxes_xywh = torch.tensor([ann["bbox"] for ann in annotations], dtype=torch.float32)
        else:
            boxes_xywh = torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.tensor(
            [self.id_to_label[ann["category_id"]] for ann in annotations],
            dtype=torch.int64,
        )

        image_tensor, mask, target = self.transform(image, boxes_xywh, image_info["id"])
        target["labels"] = labels
        return image_tensor, mask, target


class TestDigitDataset(Dataset):
    def __init__(self, image_dir: Path, transform: FixedSizeDetrTransform) -> None:
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = sorted(
            [path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS],
            key=lambda path: int(path.stem),
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        image_path = self.image_paths[index]
        with Image.open(image_path) as img:
            image = img.copy()
        image_id = int(image_path.stem)
        image_tensor, mask, target = self.transform(image, boxes_xywh=None, image_id=image_id)
        return image_tensor, mask, target


def detr_collate_fn(batch: list[tuple[Tensor, Tensor, dict[str, Tensor]]]) -> tuple[Tensor, Tensor, list[dict[str, Tensor]]]:
    images, masks, targets = zip(*batch)
    return torch.stack(images, dim=0), torch.stack(masks, dim=0), list(targets)


class Backbone(nn.Module):
    def __init__(self, pretrained: bool = True, train_backbone: bool = True) -> None:
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = resnet50(weights=weights)
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        if not train_backbone:
            for parameter in self.parameters():
                parameter.requires_grad = False

        self.num_channels = [512, 1024, 2048]

    def forward(self, x: Tensor) -> list[Tensor]:
        x = self.stem(x)
        x = self.layer1(x)
        c3 = self.layer2(x)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return [c3, c4, c5]


class PositionEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats: int = 128, temperature: int = 10000, normalize: bool = True) -> None:
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = 2.0 * math.pi

    def forward(self, mask: Tensor) -> Tensor:
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / self.num_pos_feats)

        pos_x = x_embed[..., None] / dim_t
        pos_y = y_embed[..., None] / dim_t
        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3)
        return pos.permute(0, 3, 1, 2)


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int) -> None:
        super().__init__()
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        self.layers = nn.ModuleList(
            nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)
        )

    def forward(self, x: Tensor) -> Tensor:
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            if idx < len(self.layers) - 1:
                x = F.relu(x)
        return x


class DetrModel(nn.Module):
    def __init__(
        self,
        num_classes: int,
        num_queries: int = 100,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        pretrained_backbone: bool = True,
        train_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = Backbone(pretrained=pretrained_backbone, train_backbone=train_backbone)
        self.input_proj = nn.ModuleList(
            [nn.Conv2d(in_channels, d_model, kernel_size=1) for in_channels in self.backbone.num_channels]
        )
        self.fpn_output = nn.ModuleList(
            [nn.Conv2d(d_model, d_model, kernel_size=3, padding=1) for _ in self.backbone.num_channels]
        )
        self.position_embedding = PositionEmbeddingSine(d_model // 2)
        self.query_embed = nn.Embedding(num_queries, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        self.class_embed = nn.Linear(d_model, num_classes + 1)
        self.bbox_embed = MLP(d_model, d_model, 4, 3)

        self.d_model = d_model
        self.num_queries = num_queries

    def forward(self, images: Tensor, masks: Tensor) -> dict[str, Tensor]:
        features = self.backbone(images)
        projected_features = [proj(feature) for proj, feature in zip(self.input_proj, features)]

        # Standard FPN fusion:
        # 1) use 1x1 conv to align channels at each scale
        # 2) upsample the deeper feature
        # 3) add it to the lateral feature
        # 4) smooth the fused map with a 3x3 conv
        fused_features: list[Tensor] = [torch.empty(0, device=images.device)] * len(projected_features)
        running = projected_features[-1]
        fused_features[-1] = self.fpn_output[-1](running)
        for level_idx in range(len(projected_features) - 2, -1, -1):
            lateral = projected_features[level_idx]
            upsampled = F.interpolate(running, size=lateral.shape[-2:], mode="nearest")
            running = lateral + upsampled
            fused_features[level_idx] = self.fpn_output[level_idx](running)

        # Feed only the highest-resolution fused map into the vanilla DETR
        # encoder so we keep the standard transformer cost while still letting
        # the backbone path benefit from FPN-style multi-scale fusion.
        src = fused_features[0]
        masks = F.interpolate(masks.unsqueeze(1).float(), size=src.shape[-2:]).to(torch.bool).squeeze(1)
        pos_embed = self.position_embedding(masks)

        batch_size = images.shape[0]
        src_flat = src.flatten(2).permute(2, 0, 1)
        pos_flat = pos_embed.flatten(2).permute(2, 0, 1)
        mask_flat = masks.flatten(1)

        memory = self.encoder(src_flat + pos_flat, src_key_padding_mask=mask_flat)
        # Query embeddings are learned object slots. The decoder refines these
        # slots by attending to the encoded image features in `memory`.
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, batch_size, 1)
        tgt = torch.zeros_like(query_embed)
        hs = self.decoder(
            tgt + query_embed,
            memory,
            memory_key_padding_mask=mask_flat,
        )

        hs = hs.transpose(0, 1)
        pred_logits = self.class_embed(hs)
        pred_boxes = self.bbox_embed(hs).sigmoid()
        return {"pred_logits": pred_logits, "pred_boxes": pred_boxes}


class HungarianMatcher(nn.Module):
    def __init__(self, cost_class: float = 1.0, cost_bbox: float = 5.0, cost_giou: float = 2.0) -> None:
        super().__init__()
        if linear_sum_assignment is None:
            raise ImportError("scipy is required for Hungarian matching. Install it with `pip install scipy`.")
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(self, outputs: dict[str, Tensor], targets: list[dict[str, Tensor]]) -> list[tuple[Tensor, Tensor]]:
        bs, num_queries = outputs["pred_logits"].shape[:2]
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)
        out_bbox = outputs["pred_boxes"].flatten(0, 1)

        if not torch.isfinite(out_prob).all():
            raise RuntimeError("HungarianMatcher received non-finite class probabilities from model outputs.")
        if not torch.isfinite(out_bbox).all():
            raise RuntimeError("HungarianMatcher received non-finite predicted boxes from model outputs.")

        sizes = [len(target["labels"]) for target in targets]
        if sum(sizes) == 0:
            return [
                (
                    torch.empty(0, dtype=torch.int64),
                    torch.empty(0, dtype=torch.int64),
                )
                for _ in range(bs)
            ]

        tgt_ids = torch.cat([target["labels"] for target in targets])
        tgt_bbox = torch.cat([target["boxes"] for target in targets])

        cost_class = -out_prob[:, tgt_ids]
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        cost_giou = -generalized_box_iou(
            box_cxcywh_to_xyxy(out_bbox),
            box_cxcywh_to_xyxy(tgt_bbox),
        )

        # Compute one global cost matrix first, then slice out each image's
        # targets before calling Hungarian matching independently per batch item.
        total_cost = (
            self.cost_class * cost_class
            + self.cost_bbox * cost_bbox
            + self.cost_giou * cost_giou
        )
        if not torch.isfinite(total_cost).all():
            invalid_count = int((~torch.isfinite(total_cost)).sum().item())
            raise RuntimeError(
                f"HungarianMatcher produced a cost matrix with {invalid_count} non-finite entries. "
                "This usually means the model outputs or targets became numerically unstable."
            )
        total_cost = total_cost.view(bs, num_queries, -1).cpu()

        indices: list[tuple[Tensor, Tensor]] = []
        start = 0
        for batch_idx, size in enumerate(sizes):
            if size == 0:
                indices.append(
                    (
                        torch.empty(0, dtype=torch.int64),
                        torch.empty(0, dtype=torch.int64),
                    )
                )
                continue
            batch_cost = total_cost[batch_idx, :, start : start + size]
            matched_queries, matched_targets = linear_sum_assignment(batch_cost.numpy())
            indices.append(
                (
                    torch.as_tensor(matched_queries, dtype=torch.int64),
                    torch.as_tensor(matched_targets, dtype=torch.int64),
                )
            )
            start += size
        return indices


class SetCriterion(nn.Module):
    def __init__(
        self,
        num_classes: int,
        matcher: HungarianMatcher,
        eos_coef: float = 0.1,
        bbox_loss_coef: float = 5.0,
        giou_loss_coef: float = 2.0,
        ce_loss_coef: float = 1.0,
        class_weight: Tensor | None = None,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.eos_coef = eos_coef
        self.bbox_loss_coef = bbox_loss_coef
        self.giou_loss_coef = giou_loss_coef
        self.ce_loss_coef = ce_loss_coef
        self.label_smoothing = label_smoothing

        empty_weight = torch.ones(num_classes + 1)
        if class_weight is not None:
            if class_weight.shape != (num_classes,):
                raise ValueError(
                    f"class_weight must have shape ({num_classes},), got {tuple(class_weight.shape)}"
                )
            empty_weight[:-1] = class_weight
        empty_weight[-1] = eos_coef
        self.register_buffer("empty_weight", empty_weight)

    def _get_src_permutation_idx(self, indices: list[tuple[Tensor, Tensor]]) -> tuple[Tensor, Tensor]:
        batch_idx = torch.cat(
            [torch.full_like(src, batch_id) for batch_id, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def loss_labels(
        self,
        outputs: dict[str, Tensor],
        targets: list[dict[str, Tensor]],
        indices: list[tuple[Tensor, Tensor]],
    ) -> dict[str, Tensor]:
        src_logits = outputs["pred_logits"]
        # Every query starts as "no object" (the extra last class). Only matched
        # queries are overwritten with a foreground category.
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )

        for batch_id, (src_idx, tgt_idx) in enumerate(indices):
            if len(src_idx) == 0:
                continue
            target_classes[batch_id, src_idx] = targets[batch_id]["labels"][tgt_idx]

        loss_ce = F.cross_entropy(
            src_logits.transpose(1, 2),
            target_classes,
            weight=self.empty_weight,
            label_smoothing=self.label_smoothing,
        )

        pred_classes = src_logits[..., :-1].argmax(dim=-1)
        fg_mask = target_classes != self.num_classes
        if fg_mask.any():
            class_error = 1.0 - (pred_classes[fg_mask] == target_classes[fg_mask]).float().mean()
        else:
            class_error = torch.tensor(0.0, device=src_logits.device)

        return {
            "loss_ce": loss_ce,
            "class_error": class_error,
        }

    def loss_boxes(
        self,
        outputs: dict[str, Tensor],
        targets: list[dict[str, Tensor]],
        indices: list[tuple[Tensor, Tensor]],
        num_boxes: float,
    ) -> dict[str, Tensor]:
        if num_boxes == 0:
            zero = outputs["pred_boxes"].sum() * 0.0
            return {"loss_bbox": zero, "loss_giou": zero}

        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat(
            [target["boxes"][tgt_idx] for target, (_, tgt_idx) in zip(targets, indices) if len(tgt_idx) > 0],
            dim=0,
        )

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none").sum() / num_boxes
        loss_giou = 1.0 - torch.diag(
            generalized_box_iou(
                box_cxcywh_to_xyxy(src_boxes),
                box_cxcywh_to_xyxy(target_boxes),
            )
        )
        loss_giou = loss_giou.sum() / num_boxes
        return {
            "loss_bbox": loss_bbox,
            "loss_giou": loss_giou,
        }

    def forward(self, outputs: dict[str, Tensor], targets: list[dict[str, Tensor]]) -> dict[str, Tensor]:
        indices = self.matcher(outputs, targets)
        num_boxes = float(sum(len(target["labels"]) for target in targets))
        num_boxes = max(num_boxes, 1.0)

        losses = {}
        losses.update(self.loss_labels(outputs, targets, indices))
        losses.update(self.loss_boxes(outputs, targets, indices, num_boxes))
        losses["loss_total"] = (
            self.ce_loss_coef * losses["loss_ce"]
            + self.bbox_loss_coef * losses["loss_bbox"]
            + self.giou_loss_coef * losses["loss_giou"]
        )
        return losses


def build_class_weight_from_counts(
    class_counts: Tensor,
    power: float = 0.5,
    min_weight: float = 0.5,
    max_weight: float = 3.0,
) -> Tensor:
    counts = class_counts.float().clamp_min(1.0)
    weights = counts.pow(-power)
    weights = weights / weights.mean().clamp_min(1e-6)
    return weights.clamp(min=min_weight, max=max_weight)


def freeze_module(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = False


def move_targets_to_device(targets: list[dict[str, Tensor]], device: torch.device) -> list[dict[str, Tensor]]:
    moved_targets = []
    for target in targets:
        moved_targets.append({key: value.to(device) for key, value in target.items()})
    return moved_targets


def train_one_epoch(
    model: DetrModel,
    criterion: SetCriterion,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    scaler: torch.amp.GradScaler,
    amp_enabled: bool,
    clip_max_norm: float = 0.0,
) -> dict[str, float]:
    model.train()
    criterion.train()

    running = defaultdict(float)
    skipped_batches = 0
    progress = tqdm(dataloader, desc=f"train {epoch:03d}", leave=False)
    for batch_idx, (images, masks, targets) in enumerate(progress, start=1):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        targets = move_targets_to_device(targets, device)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            outputs = model(images, masks)
            if not torch.isfinite(outputs["pred_logits"]).all():
                print(
                    f"Skipping train batch {batch_idx} in epoch {epoch:03d}: "
                    "model produced non-finite logits."
                )
                skipped_batches += 1
                continue
            if not torch.isfinite(outputs["pred_boxes"]).all():
                print(
                    f"Skipping train batch {batch_idx} in epoch {epoch:03d}: "
                    "model produced non-finite boxes."
                )
                skipped_batches += 1
                continue
            losses = criterion(outputs, targets)
        if not all(torch.isfinite(value).all() for value in losses.values()):
            print(
                f"Skipping train batch {batch_idx} in epoch {epoch:03d}: "
                "criterion produced non-finite losses."
            )
            skipped_batches += 1
            continue
        scaler.scale(losses["loss_total"]).backward()
        if clip_max_norm > 0.0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
            if not torch.isfinite(grad_norm):
                print(
                    f"Skipping optimizer step for train batch {batch_idx} in epoch {epoch:03d}: "
                    "gradient norm became non-finite."
                )
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                skipped_batches += 1
                continue
        scaler.step(optimizer)
        scaler.update()

        for key, value in losses.items():
            running[key] += float(value.detach().cpu())
        progress.set_postfix(loss=f"{losses['loss_total'].item():.4f}")

    num_batches = max(len(dataloader) - skipped_batches, 1)
    stats = {key: value / num_batches for key, value in running.items()}
    stats["skipped_batches"] = float(skipped_batches)
    return stats


@torch.no_grad()
def evaluate_loss(
    model: DetrModel,
    criterion: SetCriterion,
    dataloader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
) -> dict[str, float]:
    model.eval()
    criterion.eval()

    running = defaultdict(float)
    progress = tqdm(dataloader, desc="valid", leave=False)
    for images, masks, targets in progress:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        targets = move_targets_to_device(targets, device)

        with torch.amp.autocast("cuda", enabled=amp_enabled):
            outputs = model(images, masks)
            losses = criterion(outputs, targets)
        for key, value in losses.items():
            running[key] += float(value.detach().cpu())
        progress.set_postfix(loss=f"{losses['loss_total'].item():.4f}")

    num_batches = max(len(dataloader), 1)
    return {key: value / num_batches for key, value in running.items()}


@torch.no_grad()
def postprocess_predictions(
    outputs: dict[str, Tensor],
    targets: list[dict[str, Tensor]],
    score_threshold: float,
    top_k: int,
    image_size: int,
    nms_iou_threshold: float,
) -> list[list[dict[str, float]]]:
    pred_logits = outputs["pred_logits"].softmax(-1)
    pred_boxes = outputs["pred_boxes"]

    results: list[list[dict[str, float]]] = []
    for batch_idx in range(pred_logits.shape[0]):
        probs = pred_logits[batch_idx, :, :-1]
        scores, labels = probs.max(dim=-1)
        # Model outputs are normalized relative to the padded square canvas, so
        # decode them back to absolute xyxy coordinates on that canvas first.
        boxes_xyxy = box_cxcywh_to_xyxy(pred_boxes[batch_idx]) * float(image_size)

        scale = float(targets[batch_idx]["scale"].item())
        orig_height = int(targets[batch_idx]["orig_size"][0].item())
        orig_width = int(targets[batch_idx]["orig_size"][1].item())
        resized_height = int(targets[batch_idx]["resized_size"][0].item())
        resized_width = int(targets[batch_idx]["resized_size"][1].item())
        top = int(targets[batch_idx]["offset"][0].item())
        left = int(targets[batch_idx]["offset"][1].item())

        # Remove padding by clipping to the resized image area, then divide by
        # the stored resize scale to map boxes back to the original image size.
        boxes_xyxy[:, 0::2] -= float(left)
        boxes_xyxy[:, 1::2] -= float(top)
        boxes_xyxy = clip_xyxy_to_image(boxes_xyxy, float(resized_width), float(resized_height))
        boxes_xyxy[:, 0::2] /= max(scale, 1e-6)
        boxes_xyxy[:, 1::2] /= max(scale, 1e-6)
        boxes_xyxy = clip_xyxy_to_image(boxes_xyxy, float(orig_width), float(orig_height))

        # Keep only confident detections so each image can emit a variable
        # number of boxes, including zero boxes when nothing is reliable.
        keep = torch.where(scores >= score_threshold)[0]
        keep = keep[scores[keep].argsort(descending=True)]
        if nms_iou_threshold > 0.0 and len(keep) > 0:
            nms_keep = batched_nms(
                boxes_xyxy[keep],
                scores[keep],
                labels[keep],
                nms_iou_threshold,
            )
            keep = keep[nms_keep]
        if top_k > 0:
            keep = keep[:top_k]

        batch_results = []
        for idx in keep.tolist():
            box = boxes_xyxy[idx]
            x0, y0, x1, y1 = box.tolist()
            batch_results.append(
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
        results.append(batch_results)
    return results


@torch.no_grad()
def generate_predictions(
    model: DetrModel,
    dataloader: DataLoader,
    device: torch.device,
    score_threshold: float,
    top_k: int,
    image_size: int,
    nms_iou_threshold: float,
    amp_enabled: bool,
) -> tuple[list[dict[str, float]], PredictionScoreSummary]:
    model.eval()
    predictions: list[dict[str, float]] = []
    raw_top_scores: list[float] = []
    progress = tqdm(dataloader, desc="predict", leave=False)
    for images, masks, targets in progress:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            outputs = model(images, masks)
        probs = outputs["pred_logits"].softmax(-1)[..., :-1]
        raw_top_scores.extend(probs.max(dim=-1).values.max(dim=-1).values.detach().cpu().tolist())
        batch_predictions = postprocess_predictions(
            outputs=outputs,
            targets=targets,
            score_threshold=score_threshold,
            top_k=top_k,
            image_size=image_size,
            nms_iou_threshold=nms_iou_threshold,
        )
        for sample_preds in batch_predictions:
            predictions.extend(sample_preds)
    predictions.sort(key=lambda item: (item["image_id"], item["bbox"][0], item["bbox"][1]))
    return predictions, summarize_prediction_scores(predictions, raw_top_scores)


def maybe_run_coco_eval(annotation_path: Path, predictions: list[dict[str, float]]) -> dict[str, float] | None:
    if COCO is None or COCOeval is None:
        print("Skipping COCO eval because pycocotools is not installed.")
        return None
    if not predictions:
        print("Skipping COCO eval because there are no predictions.")
        return None

    coco_gt = COCO(str(annotation_path))
    coco_dt = coco_gt.loadRes(predictions)
    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    return {
        "map": float(evaluator.stats[0]),
        "ap50": float(evaluator.stats[1]),
        "ap75": float(evaluator.stats[2]),
        "ar100": float(evaluator.stats[8]),
    }


def save_checkpoint(
    checkpoint_path: Path,
    model: DetrModel,
    optimizer: torch.optim.Optimizer,
    scheduler: StepLR,
    epoch: int,
    best_valid_loss: float,
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
            "args": serialized_args,
        },
        checkpoint_path,
    )


def upgrade_legacy_state_dict(state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
    upgraded = dict(state_dict)
    if "input_proj.weight" in upgraded and "input_proj.2.weight" not in upgraded:
        upgraded["input_proj.2.weight"] = upgraded.pop("input_proj.weight")
    if "input_proj.bias" in upgraded and "input_proj.2.bias" not in upgraded:
        upgraded["input_proj.2.bias"] = upgraded.pop("input_proj.bias")
    return upgraded


def load_checkpoint(
    checkpoint_path: Path,
    model: DetrModel,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: StepLR | None = None,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_state = upgrade_legacy_state_dict(checkpoint["model"])
    missing_keys, unexpected_keys = model.load_state_dict(model_state, strict=False)
    if missing_keys:
        preview = ", ".join(missing_keys[:8])
        suffix = "..." if len(missing_keys) > 8 else ""
        print(f"Missing checkpoint keys initialized from current model defaults: {preview}{suffix}")
    if unexpected_keys:
        preview = ", ".join(unexpected_keys[:8])
        suffix = "..." if len(unexpected_keys) > 8 else ""
        print(f"Unexpected checkpoint keys ignored: {preview}{suffix}")
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    return checkpoint


def remap_pretrained_key(key: str) -> str | None:
    if key.startswith("module."):
        key = key[len("module.") :]

    if key.startswith("backbone.0.body."):
        suffix = key[len("backbone.0.body.") :]
        if suffix.startswith("conv1."):
            return "backbone.stem.0." + suffix[len("conv1.") :]
        if suffix.startswith("bn1."):
            return "backbone.stem.1." + suffix[len("bn1.") :]
        if suffix.startswith("layer1."):
            return "backbone.layer1." + suffix[len("layer1.") :]
        if suffix.startswith("layer2."):
            return "backbone.layer2." + suffix[len("layer2.") :]
        if suffix.startswith("layer3."):
            return "backbone.layer3." + suffix[len("layer3.") :]
        if suffix.startswith("layer4."):
            return "backbone.layer4." + suffix[len("layer4.") :]
        return None

    if key.startswith("transformer.encoder.layers."):
        return "encoder.layers." + key[len("transformer.encoder.layers.") :]
    if key.startswith("transformer.decoder.layers."):
        return "decoder.layers." + key[len("transformer.decoder.layers.") :]
    if key.startswith("transformer.encoder.norm.") or key.startswith("transformer.decoder.norm."):
        return None
    if key == "input_proj.weight":
        return "input_proj.2.weight"
    if key == "input_proj.bias":
        return "input_proj.2.bias"

    if key.startswith("backbone.1.") or key.startswith("transformer."):
        return None

    return key


def load_pretrained_model_weights(
    checkpoint_path: Path,
    model: DetrModel,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    raw_state_dict = checkpoint.get("model", checkpoint)
    model_state = model.state_dict()

    compatible_state: dict[str, Tensor] = {}
    skipped_keys: list[str] = []
    for original_key, value in raw_state_dict.items():
        mapped_key = remap_pretrained_key(original_key)
        if mapped_key is None:
            skipped_keys.append(original_key)
            continue
        if mapped_key not in model_state:
            skipped_keys.append(original_key)
            continue
        if model_state[mapped_key].shape != value.shape:
            skipped_keys.append(original_key)
            continue
        compatible_state[mapped_key] = value

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
    return checkpoint


def build_dataloaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader]:
    train_transform = FixedSizeDetrTransform(image_size=args.image_size, augment=not args.no_augmentation)
    valid_transform = FixedSizeDetrTransform(image_size=args.image_size, augment=False)
    train_dataset = CocoDigitDataset(
        image_dir=args.train_dir,
        annotation_path=args.train_json,
        transform=train_transform,
    )
    valid_dataset = CocoDigitDataset(
        image_dir=args.valid_dir,
        annotation_path=args.valid_json,
        transform=valid_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=detr_collate_fn,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=detr_collate_fn,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    return train_loader, valid_loader


def build_test_dataloader(args: argparse.Namespace) -> DataLoader:
    transform = FixedSizeDetrTransform(image_size=args.image_size)
    dataset = TestDigitDataset(image_dir=args.test_dir, transform=transform)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=detr_collate_fn,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )


def build_model(args: argparse.Namespace, num_classes: int = 10) -> DetrModel:
    return DetrModel(
        num_classes=num_classes,
        num_queries=args.num_queries,
        d_model=args.hidden_dim,
        nhead=args.nheads,
        num_encoder_layers=args.encoder_layers,
        num_decoder_layers=args.decoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        pretrained_backbone=not args.no_pretrained_backbone,
        train_backbone=not args.freeze_backbone,
    )


def run_prediction_with_model(
    model: DetrModel,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    infer_start_time = time.perf_counter()
    test_loader = build_test_dataloader(args)
    predictions, score_summary = generate_predictions(
        model=model,
        dataloader=test_loader,
        device=device,
        score_threshold=args.score_threshold,
        top_k=args.top_k,
        image_size=args.image_size,
        nms_iou_threshold=args.nms_iou_threshold,
        amp_enabled=args.amp and device.type == "cuda",
    )
    infer_elapsed = time.perf_counter() - infer_start_time

    if not predictions:
        print(
            "Prediction produced 0 boxes. "
            f"Current score_threshold={args.score_threshold:.3f}; "
            "if training already finished normally, try a lower value such as 0.2 or 0.1."
        )
    print_prediction_score_summary(score_summary, args.score_threshold)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        if args.output.is_dir():
            raise IsADirectoryError(f"Output path is a directory, cannot overwrite it: {args.output}")
        print(f"Overwriting existing prediction file: {args.output}")
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(predictions, f)
    print(f"Saved {len(predictions)} predictions to {args.output}")
    print(f"Total inference time: {format_duration(infer_elapsed)}")


def run_train(args: argparse.Namespace) -> None:
    train_start_time = time.perf_counter()
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    amp_enabled = args.amp and device.type == "cuda"
    train_loader, valid_loader = build_dataloaders(args)
    writer = SummaryWriter(log_dir=str(args.tensorboard_dir)) if args.tensorboard else None

    model = build_model(args).to(device)
    if args.freeze_transformer:
        freeze_module(model.encoder)
        freeze_module(model.decoder)
        freeze_module(model.query_embed)
        print("Freezing transformer blocks: encoder, decoder, query_embed")
    total_params, trainable_params = count_parameters(model)
    print(
        f"Model parameters: total={format_param_count(total_params)} "
        f"trainable={format_param_count(trainable_params)}"
    )
    matcher = HungarianMatcher(
        cost_class=args.cost_class,
        cost_bbox=args.cost_bbox,
        cost_giou=args.cost_giou,
    )
    class_weight = None
    if not args.no_class_balance:
        class_weight = build_class_weight_from_counts(
            train_loader.dataset.class_counts,
            power=args.class_balance_power,
            min_weight=args.class_balance_min,
            max_weight=args.class_balance_max,
        )
        print(f"Using class-balanced CE weights: {[round(value, 3) for value in class_weight.tolist()]}")
    criterion = SetCriterion(
        num_classes=10,
        matcher=matcher,
        eos_coef=args.eos_coef,
        bbox_loss_coef=args.bbox_loss_coef,
        giou_loss_coef=args.giou_loss_coef,
        ce_loss_coef=args.ce_loss_coef,
        class_weight=class_weight.to(device) if class_weight is not None else None,
        label_smoothing=args.label_smoothing,
    ).to(device)

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
    scheduler = StepLR(optimizer, step_size=args.lr_drop, gamma=args.lr_gamma)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    start_epoch = 1
    best_valid_loss = float("inf")
    epochs_without_improvement = 0
    if args.resume is not None and args.resume.exists():
        checkpoint = load_checkpoint(args.resume, model, optimizer, scheduler, device=device)
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_valid_loss = float(checkpoint.get("best_valid_loss", float("inf")))
        print(f"Resumed from {args.resume} at epoch {start_epoch}.")
    elif args.pretrained_checkpoint is not None:
        if not args.pretrained_checkpoint.exists():
            raise FileNotFoundError(f"Pretrained checkpoint not found: {args.pretrained_checkpoint}")
        load_pretrained_model_weights(args.pretrained_checkpoint, model, device=device)

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            train_stats = train_one_epoch(
                model,
                criterion,
                train_loader,
                optimizer,
                device,
                epoch,
                scaler=scaler,
                amp_enabled=amp_enabled,
                clip_max_norm=args.clip_max_norm,
            )
            valid_stats = evaluate_loss(model, criterion, valid_loader, device, amp_enabled=amp_enabled)
            eval_stats: dict[str, float] | None = None
            if args.eval_every_epoch:
                predictions, _ = generate_predictions(
                    model=model,
                    dataloader=valid_loader,
                    device=device,
                    score_threshold=args.score_threshold,
                    top_k=args.top_k,
                    image_size=args.image_size,
                    nms_iou_threshold=args.nms_iou_threshold,
                    amp_enabled=amp_enabled,
                )
                eval_stats = maybe_run_coco_eval(args.valid_json, predictions)
            scheduler.step()

            status_parts = [
                f"epoch={epoch:03d}",
                f"train_loss={train_stats['loss_total']:.4f}",
                f"valid_loss={valid_stats['loss_total']:.4f}",
                f"valid_cls_err={valid_stats.get('class_error', 0.0):.4f}",
            ]
            if eval_stats is not None:
                status_parts.append(f"valid_mAP={eval_stats['map']:.4f}")
                status_parts.append(f"valid_AP50={eval_stats['ap50']:.4f}")
            print(" ".join(status_parts))

            if writer is not None:
                writer.add_scalar("train/loss_total", train_stats["loss_total"], epoch)
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
                if eval_stats is not None:
                    writer.add_scalar("valid/mAP", eval_stats["map"], epoch)
                    writer.add_scalar("valid/AP50", eval_stats["ap50"], epoch)
                    writer.add_scalar("valid/AP75", eval_stats["ap75"], epoch)
                    writer.add_scalar("valid/AR100", eval_stats["ar100"], epoch)

            if valid_stats["loss_total"] < best_valid_loss - args.early_stop_min_delta:
                best_valid_loss = valid_stats["loss_total"]
                epochs_without_improvement = 0
                best_path = args.checkpoint_dir / "best.pt"
                save_checkpoint(best_path, model, optimizer, scheduler, epoch, best_valid_loss, args)
                print(f"Saved best checkpoint to {best_path}")
            else:
                epochs_without_improvement += 1
                if args.early_stop_patience > 0:
                    print(
                        f"Early-stop counter: {epochs_without_improvement}/{args.early_stop_patience} "
                        f"(best_valid_loss={best_valid_loss:.4f})"
                    )

            last_path = args.checkpoint_dir / "last.pt"
            save_checkpoint(last_path, model, optimizer, scheduler, epoch, best_valid_loss, args)

            if args.early_stop_patience > 0 and epochs_without_improvement >= args.early_stop_patience:
                print(
                    f"Early stopping triggered at epoch {epoch:03d}. "
                    f"No valid_loss improvement larger than {args.early_stop_min_delta:.6f} "
                    f"for {args.early_stop_patience} consecutive epoch(s)."
                )
                break

    finally:
        if writer is not None:
            writer.close()

    train_elapsed = time.perf_counter() - train_start_time
    print(f"Total training time: {format_duration(train_elapsed)}")

    if args.predict_after_train:
        best_path = args.checkpoint_dir / "best.pt"
        predict_checkpoint = best_path if best_path.exists() else args.checkpoint_dir / "last.pt"
        load_checkpoint(predict_checkpoint, model, device=device)
        print(f"Running prediction with checkpoint {predict_checkpoint}")
        run_prediction_with_model(model, args, device)


def run_predict(args: argparse.Namespace) -> None:
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model(args).to(device)
    total_params, trainable_params = count_parameters(model)
    print(
        f"Model parameters: total={format_param_count(total_params)} "
        f"trainable={format_param_count(trainable_params)}"
    )
    checkpoint = load_checkpoint(args.checkpoint, model, device=device)
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}.")
    run_prediction_with_model(model, args, device)


def add_shared_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision on CUDA for faster training/inference.")
    parser.add_argument("--tensorboard", action="store_true")
    parser.add_argument("--tensorboard-dir", type=Path, default=PROJECT_ROOT / "tensorboard_baseline" / "run_freeze_transformer")

    parser.add_argument("--num-queries", type=int, default=100)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--nheads", type=int, default=8)
    parser.add_argument("--encoder-layers", type=int, default=6)
    parser.add_argument("--decoder-layers", type=int, default=6)
    parser.add_argument("--dim-feedforward", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--freeze-transformer", action="store_true")
    parser.add_argument("--no-pretrained-backbone", action="store_true")

    parser.add_argument("--score-threshold", type=float, default=0.2)
    parser.add_argument("--top-k", type=int, default=4, help="Optional max detections per image after thresholding; 0 disables the cap.")
    parser.add_argument(
        "--nms-iou-threshold",
        type=float,
        default=0.3,
        help="Class-aware NMS IoU threshold after score filtering; set <= 0 to disable NMS.",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train or run inference with a custom DETR for digit detection.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train DETR on train/valid COCO annotations.")
    add_shared_arguments(train_parser)
    train_parser.add_argument("--train-dir", type=Path, default=DEFAULT_DATA_ROOT / "train")
    train_parser.add_argument("--valid-dir", type=Path, default=DEFAULT_DATA_ROOT / "valid")
    train_parser.add_argument("--train-json", type=Path, default=DEFAULT_DATA_ROOT / "train.json")
    train_parser.add_argument("--valid-json", type=Path, default=DEFAULT_DATA_ROOT / "valid.json")
    train_parser.add_argument("--epochs", type=int, default=80)
    train_parser.add_argument("--lr", type=float, default=1e-4)
    train_parser.add_argument("--lr-backbone", type=float, default=1e-5)
    train_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_parser.add_argument("--lr-drop", type=int, default=20)
    train_parser.add_argument("--lr-gamma", type=float, default=0.1)
    train_parser.add_argument("--clip-max-norm", type=float, default=0.1)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    train_parser.add_argument(
        "--pretrained-checkpoint",
        type=Path,
        default= PROJECT_ROOT / "checkpoints/pretrain_weight/detr-r50-e632da11.pth",
        help="Optional DETR checkpoint to warm-start matching backbone/transformer weights before fine-tuning.",
    )
    train_parser.add_argument("--resume", type=Path, default=None)
    train_parser.add_argument("--eval-every-epoch", action="store_true")
    train_parser.set_defaults(predict_after_train=True)
    train_parser.add_argument(
        "--predict-after-train",
        dest="predict_after_train",
        action="store_true",
        help="Run inference automatically after training finishes. This is enabled by default.",
    )
    train_parser.add_argument(
        "--no-predict-after-train",
        dest="predict_after_train",
        action="store_false",
        help="Disable the automatic post-training inference step.",
    )
    train_parser.add_argument("--test-dir", type=Path, default=DEFAULT_DATA_ROOT / "test")
    train_parser.add_argument("--output", type=Path, default=DEFAULT_PRED_PATH)
    train_parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=0,
        help="Stop training after this many consecutive epochs without valid_loss improvement; 0 disables early stopping.",
    )
    train_parser.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=0.0,
        help="Minimum valid_loss decrease required to reset the early-stopping counter.",
    )

    train_parser.add_argument("--cost-class", type=float, default=2.0)
    train_parser.add_argument("--cost-bbox", type=float, default=5.0)
    train_parser.add_argument("--cost-giou", type=float, default=2.0)
    train_parser.add_argument("--ce-loss-coef", type=float, default=2.0)
    train_parser.add_argument("--bbox-loss-coef", type=float, default=5.0)
    train_parser.add_argument("--giou-loss-coef", type=float, default=2.0)
    train_parser.add_argument("--eos-coef", type=float, default=0.05)
    train_parser.add_argument("--label-smoothing", type=float, default=0.05)
    train_parser.add_argument("--no-class-balance", action="store_true")
    train_parser.add_argument("--class-balance-power", type=float, default=0.5)
    train_parser.add_argument("--class-balance-min", type=float, default=0.5)
    train_parser.add_argument("--class-balance-max", type=float, default=3.0)
    train_parser.add_argument(
        "--no-augmentation",
        action="store_true",
        help="Disable train-time color jitter plus scale/translation augmentation.",
    )

    predict_parser = subparsers.add_parser("predict", help="Run inference and export pred.json.")
    add_shared_arguments(predict_parser)
    predict_parser.add_argument("--checkpoint", type=Path, required=True)
    predict_parser.add_argument("--test-dir", type=Path, default=DEFAULT_DATA_ROOT / "test")
    predict_parser.add_argument("--output", type=Path, default=DEFAULT_PRED_PATH)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed = getattr(args, "seed", 42)
    set_seed(seed)

    if args.command == "train":
        run_train(args)
    elif args.command == "predict":
        run_predict(args)
    else:
        raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
