#!/usr/bin/env python3
"""Compare baseline DETR and DN-DETR on the validation split.

This script keeps diagnosis separate from the main train/predict entry points.
It evaluates both checkpoints on the same validation annotations, reports a
small set of metrics, and saves side-by-side qualitative visualizations.

Usage example:
python src/diagnose_validation.py \
  --baseline-checkpoint checkpoints/dn_deformable_freeze_transformer/best.pth \
  --dn-checkpoint checkpoints/dn_deformable_digits/best.pth \
  --output-dir diagnostics/run3
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader

import main as baseline
import main_dn_deformable as dn_main


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "diagnostics" / "validation_compare"

if not hasattr(np, "float"):
    np.float = float  # type: ignore[attr-defined]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare baseline DETR and DN-DETR on the validation split.")
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--dn-checkpoint", type=Path, required=True)
    parser.add_argument("--valid-dir", type=Path, default=baseline.DEFAULT_DATA_ROOT / "valid")
    parser.add_argument("--valid-json", type=Path, default=baseline.DEFAULT_DATA_ROOT / "valid.json")
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--baseline-score-threshold", type=float, default=None)
    parser.add_argument("--baseline-top-k", type=int, default=None)
    parser.add_argument("--baseline-nms-iou-threshold", type=float, default=None)
    parser.add_argument("--dn-score-threshold", type=float, default=None)
    parser.add_argument("--dn-top-k", type=int, default=None)
    parser.add_argument("--dn-nms-iou-threshold", type=float, default=None)
    return parser.parse_args()


def load_checkpoint_payload(checkpoint_path: Path, device: torch.device) -> dict[str, Any]:
    return torch.load(checkpoint_path, map_location=device, weights_only=False)


def namespace_from_checkpoint_args(checkpoint_args: dict[str, Any] | None) -> argparse.Namespace:
    return argparse.Namespace(**(checkpoint_args or {}))


def value_or_default(value: Any, default: Any) -> Any:
    return default if value is None else value


def get_arg(args: argparse.Namespace, name: str, default: Any) -> Any:
    return getattr(args, name, default)


def build_baseline_eval_args(checkpoint_args: dict[str, Any], cli_args: argparse.Namespace) -> argparse.Namespace:
    args = namespace_from_checkpoint_args(checkpoint_args)
    args.image_size = int(get_arg(args, "image_size", 256))
    args.batch_size = cli_args.batch_size
    args.num_workers = cli_args.num_workers
    args.device = cli_args.device
    args.valid_dir = cli_args.valid_dir
    args.valid_json = cli_args.valid_json
    args.no_pretrained_backbone = True
    args.score_threshold = value_or_default(cli_args.baseline_score_threshold, get_arg(args, "score_threshold", 0.6))
    args.top_k = value_or_default(cli_args.baseline_top_k, get_arg(args, "top_k", 4))
    args.nms_iou_threshold = value_or_default(
        cli_args.baseline_nms_iou_threshold,
        get_arg(args, "nms_iou_threshold", 0.3),
    )
    return args


def build_dn_eval_args(checkpoint_args: dict[str, Any], cli_args: argparse.Namespace) -> argparse.Namespace:
    args = namespace_from_checkpoint_args(checkpoint_args)
    args.image_size = int(get_arg(args, "image_size", 256))
    args.batch_size = cli_args.batch_size
    args.num_workers = cli_args.num_workers
    args.device = cli_args.device
    args.valid_dir = cli_args.valid_dir
    args.valid_json = cli_args.valid_json
    args.no_pretrained_backbone = True
    args.pretrained_backbone = False
    args.score_threshold = value_or_default(cli_args.dn_score_threshold, get_arg(args, "score_threshold", 0.6))
    args.top_k = value_or_default(cli_args.dn_top_k, get_arg(args, "top_k", 4))
    args.nms_iou_threshold = value_or_default(
        cli_args.dn_nms_iou_threshold,
        get_arg(args, "nms_iou_threshold", 0.3),
    )
    return args


def normalize_device_arg(args: argparse.Namespace, device: torch.device) -> None:
    args.device = str(device)


def build_baseline_valid_loader(args: argparse.Namespace) -> tuple[baseline.CocoDigitDataset, DataLoader]:
    dataset = baseline.CocoDigitDataset(
        image_dir=args.valid_dir,
        annotation_path=args.valid_json,
        transform=baseline.FixedSizeDetrTransform(image_size=args.image_size, augment=False),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=baseline.detr_collate_fn,
        pin_memory=True,
    )
    return dataset, loader


def build_dn_valid_loader(args: argparse.Namespace) -> tuple[baseline.CocoDigitDataset, DataLoader]:
    dataset = baseline.CocoDigitDataset(
        image_dir=args.valid_dir,
        annotation_path=args.valid_json,
        transform=baseline.FixedSizeDetrTransform(image_size=args.image_size, augment=False),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=dn_main.dn_collate_fn,
        pin_memory=True,
    )
    return dataset, loader


def build_baseline_criterion(args: argparse.Namespace, device: torch.device) -> baseline.SetCriterion:
    matcher = baseline.HungarianMatcher(
        cost_class=float(get_arg(args, "cost_class", 1.0)),
        cost_bbox=float(get_arg(args, "cost_bbox", 5.0)),
        cost_giou=float(get_arg(args, "cost_giou", 2.0)),
    )
    return baseline.SetCriterion(
        num_classes=10,
        matcher=matcher,
        eos_coef=float(get_arg(args, "eos_coef", 0.1)),
        bbox_loss_coef=float(get_arg(args, "bbox_loss_coef", 5.0)),
        giou_loss_coef=float(get_arg(args, "giou_loss_coef", 2.0)),
    ).to(device)


def predictions_by_image_id(predictions: list[dict[str, float]]) -> dict[int, list[dict[str, float]]]:
    grouped: dict[int, list[dict[str, float]]] = defaultdict(list)
    for prediction in predictions:
        grouped[int(prediction["image_id"])].append(prediction)
    for image_predictions in grouped.values():
        image_predictions.sort(key=lambda item: item["score"], reverse=True)
    return dict(grouped)


def draw_boxes(
    image: Image.Image,
    boxes_xyxy: list[tuple[float, float, float, float]],
    labels: list[str],
    color: str,
) -> Image.Image:
    result = image.convert("RGB").copy()
    draw = ImageDraw.Draw(result)
    font = ImageFont.load_default()
    for box, label in zip(boxes_xyxy, labels):
        x0, y0, x1, y1 = box
        draw.rectangle((x0, y0, x1, y1), outline=color, width=3)
        text_bbox = draw.textbbox((x0, y0), label, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        text_y = max(0, y0 - text_h - 4)
        draw.rectangle((x0, text_y, x0 + text_w + 6, text_y + text_h + 4), fill=color)
        draw.text((x0 + 3, text_y + 2), label, fill="white", font=font)
    return result


def build_comparison_canvas(
    image_info: dict[str, Any],
    image_dir: Path,
    gt_annotations: list[dict[str, Any]],
    baseline_predictions: list[dict[str, float]],
    dn_predictions: list[dict[str, float]],
    category_name_by_id: dict[int, str],
) -> Image.Image:
    image_path = image_dir / image_info["file_name"]
    with Image.open(image_path) as img:
        original = img.convert("RGB")

    gt_boxes = []
    gt_labels = []
    for ann in gt_annotations:
        x, y, w, h = ann["bbox"]
        gt_boxes.append((float(x), float(y), float(x + w), float(y + h)))
        category_id = int(ann["category_id"])
        digit = category_name_by_id.get(category_id, str(category_id))
        gt_labels.append(f"cid={category_id} digit={digit}")

    baseline_boxes = []
    baseline_labels = []
    for pred in baseline_predictions:
        x, y, w, h = pred["bbox"]
        baseline_boxes.append((float(x), float(y), float(x + w), float(y + h)))
        category_id = int(pred["category_id"])
        digit = category_name_by_id.get(category_id, str(category_id))
        baseline_labels.append(f"cid={category_id} digit={digit} {pred['score']:.2f}")

    dn_boxes = []
    dn_labels = []
    for pred in dn_predictions:
        x, y, w, h = pred["bbox"]
        dn_boxes.append((float(x), float(y), float(x + w), float(y + h)))
        category_id = int(pred["category_id"])
        digit = category_name_by_id.get(category_id, str(category_id))
        dn_labels.append(f"cid={category_id} digit={digit} {pred['score']:.2f}")

    gt_panel = draw_boxes(original, gt_boxes, gt_labels, color="#1f9d55")
    baseline_panel = draw_boxes(original, baseline_boxes, baseline_labels, color="#1d4ed8")
    dn_panel = draw_boxes(original, dn_boxes, dn_labels, color="#dc2626")

    panel_width, panel_height = original.size
    title_height = 30
    canvas = Image.new("RGB", (panel_width * 3, panel_height + title_height), "white")
    canvas.paste(gt_panel, (0, title_height))
    canvas.paste(baseline_panel, (panel_width, title_height))
    canvas.paste(dn_panel, (panel_width * 2, title_height))

    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    titles = [
        f"GT | img_id={image_info['id']} | size={image_info['width']}x{image_info['height']} | gt={len(gt_annotations)}",
        f"Baseline | pred={len(baseline_predictions)}",
        f"DN-DETR | pred={len(dn_predictions)}",
    ]
    for idx, title in enumerate(titles):
        draw.text((panel_width * idx + 8, 8), title, fill="black", font=font)
    return canvas


def sample_image_infos(images: list[dict[str, Any]], num_samples: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    if num_samples >= len(images):
        return list(images)
    return rng.sample(images, num_samples)


def summarize_metrics(
    *,
    model_name: str,
    losses: dict[str, float],
    eval_stats: dict[str, float] | None,
    predictions: list[dict[str, float]],
    num_images: int,
    threshold: float,
    top_k: int,
    nms_iou_threshold: float,
) -> dict[str, float | int | None]:
    return {
        "score_threshold": threshold,
        "top_k": top_k,
        "nms_iou_threshold": nms_iou_threshold,
        "num_predictions": len(predictions),
        "avg_predictions_per_image": len(predictions) / max(num_images, 1),
        "loss_total": float(losses.get("loss_total", losses.get("loss", 0.0))),
        "class_error": float(losses.get("class_error", 0.0)),
        "loss_ce": float(losses.get("loss_ce", 0.0)),
        "loss_bbox": float(losses.get("loss_bbox", 0.0)),
        "loss_giou": float(losses.get("loss_giou", 0.0)),
        "map": None if eval_stats is None else float(eval_stats["map"]),
        "ap50": None if eval_stats is None else float(eval_stats["ap50"]),
        "ap75": None if eval_stats is None else float(eval_stats["ap75"]),
        "ar100": None if eval_stats is None else float(eval_stats["ar100"]),
        "model": model_name,
    }


def print_metric_summary(title: str, metrics: dict[str, float | int | None]) -> None:
    parts = [
        title,
        f"thr={metrics['score_threshold']}",
        f"top_k={metrics['top_k']}",
        f"nms={metrics['nms_iou_threshold']}",
        f"mAP={metrics['map'] if metrics['map'] is not None else 'n/a'}",
        f"AP50={metrics['ap50'] if metrics['ap50'] is not None else 'n/a'}",
        f"avg_pred={metrics['avg_predictions_per_image']:.3f}",
        f"class_error={metrics['class_error']:.4f}",
        f"loss_ce={metrics['loss_ce']:.4f}",
        f"loss_bbox={metrics['loss_bbox']:.4f}",
        f"loss_giou={metrics['loss_giou']:.4f}",
    ]
    print(" | ".join(parts))


def main() -> None:
    cli_args = parse_args()
    cli_args.output_dir.mkdir(parents=True, exist_ok=True)
    visuals_dir = cli_args.output_dir / "samples"
    visuals_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(cli_args.device if cli_args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    baseline.set_seed(cli_args.seed)

    baseline_payload = load_checkpoint_payload(cli_args.baseline_checkpoint, device)
    dn_payload = load_checkpoint_payload(cli_args.dn_checkpoint, device)

    baseline_args = build_baseline_eval_args(baseline_payload.get("args", {}), cli_args)
    dn_args = build_dn_eval_args(dn_payload.get("args", {}), cli_args)
    normalize_device_arg(baseline_args, device)
    normalize_device_arg(dn_args, device)

    baseline_dataset, baseline_loader = build_baseline_valid_loader(baseline_args)
    _, dn_loader = build_dn_valid_loader(dn_args)

    baseline_model = baseline.build_model(baseline_args).to(device)
    baseline.load_checkpoint(cli_args.baseline_checkpoint, baseline_model, device=device)
    baseline_criterion = build_baseline_criterion(baseline_args, device)

    dn_model, dn_criterion, _ = dn_main.build_model(dn_args)
    dn_model = dn_model.to(device)
    dn_criterion = dn_criterion.to(device)
    dn_main.load_checkpoint(cli_args.dn_checkpoint, dn_model, device=device)

    print("Running baseline validation diagnostics...")
    baseline_losses = baseline.evaluate_loss(baseline_model, baseline_criterion, baseline_loader, device)
    baseline_predictions = baseline.generate_predictions(
        model=baseline_model,
        dataloader=baseline_loader,
        device=device,
        score_threshold=baseline_args.score_threshold,
        top_k=baseline_args.top_k,
        image_size=baseline_args.image_size,
        nms_iou_threshold=baseline_args.nms_iou_threshold,
    )
    baseline_eval_stats = baseline.maybe_run_coco_eval(cli_args.valid_json, baseline_predictions)

    print("Running DN-DETR validation diagnostics...")
    dn_losses = dn_main.evaluate_loss(dn_model, dn_criterion, dn_loader, device, dn_args)
    dn_predictions = dn_main.generate_predictions(
        model=dn_model,
        dataloader=dn_loader,
        device=device,
        args=dn_args,
    )
    dn_eval_stats = baseline.maybe_run_coco_eval(cli_args.valid_json, dn_predictions)

    baseline_metrics = summarize_metrics(
        model_name="baseline",
        losses=baseline_losses,
        eval_stats=baseline_eval_stats,
        predictions=baseline_predictions,
        num_images=len(baseline_dataset),
        threshold=float(baseline_args.score_threshold),
        top_k=int(baseline_args.top_k),
        nms_iou_threshold=float(baseline_args.nms_iou_threshold),
    )
    dn_metrics = summarize_metrics(
        model_name="dn_detr",
        losses=dn_losses,
        eval_stats=dn_eval_stats,
        predictions=dn_predictions,
        num_images=len(baseline_dataset),
        threshold=float(dn_args.score_threshold),
        top_k=int(dn_args.top_k),
        nms_iou_threshold=float(dn_args.nms_iou_threshold),
    )

    print_metric_summary("baseline", baseline_metrics)
    print_metric_summary("dn_detr", dn_metrics)

    baseline_preds_by_id = predictions_by_image_id(baseline_predictions)
    dn_preds_by_id = predictions_by_image_id(dn_predictions)
    sampled_infos = sample_image_infos(baseline_dataset.images, cli_args.num_samples, cli_args.seed)
    category_name_by_id = {
        int(category["id"]): str(category["name"])
        for category in baseline_dataset.categories
    }

    for sample_idx, image_info in enumerate(sampled_infos, start=1):
        image_id = int(image_info["id"])
        comparison = build_comparison_canvas(
            image_info=image_info,
            image_dir=cli_args.valid_dir,
            gt_annotations=baseline_dataset.annotations_by_image.get(image_id, []),
            baseline_predictions=baseline_preds_by_id.get(image_id, []),
            dn_predictions=dn_preds_by_id.get(image_id, []),
            category_name_by_id=category_name_by_id,
        )
        output_path = visuals_dir / f"{sample_idx:02d}_image_{image_id}.png"
        comparison.save(output_path)

    summary = {
        "baseline": baseline_metrics,
        "dn_detr": dn_metrics,
        "valid_dir": str(cli_args.valid_dir),
        "valid_json": str(cli_args.valid_json),
        "baseline_checkpoint": str(cli_args.baseline_checkpoint),
        "dn_checkpoint": str(cli_args.dn_checkpoint),
        "num_visualized_samples": len(sampled_infos),
        "sample_seed": cli_args.seed,
    }
    summary_path = cli_args.output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved summary to {summary_path}")
    print(f"Saved qualitative samples to {visuals_dir}")


if __name__ == "__main__":
    main()
