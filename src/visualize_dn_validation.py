#!/usr/bin/env python3
"""Visualize GT against one or two DN-DETR checkpoints on the validation split.

Examples:
./docker/run_dn_detr.sh python src/visualize_dn_validation.py \
  --dn-checkpoint checkpoints/dn_deformable_digits/best.pth \
  --dn-score-threshold 0.05 \
    --dn-top-k 0 \
    --dn-nms-iou-threshold 0.1 \
    --num-samples 10 \
  --output-dir diagnostics/dn_gt_run3

./docker/run_dn_detr.sh python src/visualize_dn_validation.py \
  --dn-checkpoint checkpoints/dn_deformable_digits/best.pth \
  --compare-dn-checkpoint checkpoints/dn_deformable_aug/best_map.pth \
  --output-dir diagnostics/dn_compare_run3
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

import main as baseline
import main_dn_deformable as dn_main


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "diagnostics" / "dn_validation_visuals"

if not hasattr(np, "float"):
    np.float = float  # type: ignore[attr-defined]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize GT and DN-DETR predictions on the validation split.")
    parser.add_argument("--dn-checkpoint", type=Path, required=True, help="Primary DN-DETR checkpoint to visualize.")
    parser.add_argument(
        "--compare-dn-checkpoint",
        type=Path,
        default=None,
        help="Optional second DN-DETR checkpoint for side-by-side comparison.",
    )
    parser.add_argument("--valid-dir", type=Path, default=baseline.DEFAULT_DATA_ROOT / "valid")
    parser.add_argument("--valid-json", type=Path, default=baseline.DEFAULT_DATA_ROOT / "valid.json")
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dn-score-threshold", type=float, default=None)
    parser.add_argument("--dn-top-k", type=int, default=None)
    parser.add_argument("--dn-nms-iou-threshold", type=float, default=None)
    parser.add_argument("--compare-score-threshold", type=float, default=None)
    parser.add_argument("--compare-top-k", type=int, default=None)
    parser.add_argument("--compare-nms-iou-threshold", type=float, default=None)
    return parser.parse_args()


def load_checkpoint_payload(checkpoint_path: Path, device: torch.device) -> dict[str, Any]:
    return torch.load(checkpoint_path, map_location=device, weights_only=False)


def namespace_from_checkpoint_args(checkpoint_args: dict[str, Any] | None) -> argparse.Namespace:
    return argparse.Namespace(**(checkpoint_args or {}))


def value_or_default(value: Any, default: Any) -> Any:
    return default if value is None else value


def get_arg(args: argparse.Namespace, name: str, default: Any) -> Any:
    return getattr(args, name, default)


def build_dn_eval_args(
    checkpoint_args: dict[str, Any],
    cli_args: argparse.Namespace,
    *,
    score_threshold: float | None,
    top_k: int | None,
    nms_iou_threshold: float | None,
) -> argparse.Namespace:
    args = namespace_from_checkpoint_args(checkpoint_args)
    args.image_size = int(get_arg(args, "image_size", 720))
    args.batch_size = cli_args.batch_size
    args.num_workers = cli_args.num_workers
    args.device = cli_args.device
    args.valid_dir = cli_args.valid_dir
    args.valid_json = cli_args.valid_json
    args.no_pretrained_backbone = True
    args.pretrained_backbone = False
    args.score_threshold = value_or_default(score_threshold, get_arg(args, "score_threshold", 0.2))
    args.top_k = value_or_default(top_k, get_arg(args, "top_k", 4))
    args.nms_iou_threshold = value_or_default(nms_iou_threshold, get_arg(args, "nms_iou_threshold", 0.4))
    return args


def normalize_device_arg(args: argparse.Namespace, device: torch.device) -> None:
    args.device = str(device)


def build_dn_valid_loader(args: argparse.Namespace):
    dataset = baseline.CocoDigitDataset(
        image_dir=args.valid_dir,
        annotation_path=args.valid_json,
        transform=baseline.FixedSizeDetrTransform(image_size=args.image_size, augment=False),
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=dn_main.dn_collate_fn,
        pin_memory=True,
    )
    return dataset, loader


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


def format_gt_labels(annotations: list[dict[str, Any]], category_name_by_id: dict[int, str]) -> tuple[list[tuple[float, float, float, float]], list[str]]:
    boxes = []
    labels = []
    for ann in annotations:
        x, y, w, h = ann["bbox"]
        boxes.append((float(x), float(y), float(x + w), float(y + h)))
        category_id = int(ann["category_id"])
        digit = category_name_by_id.get(category_id, str(category_id))
        labels.append(f"cid={category_id} digit={digit}")
    return boxes, labels


def format_prediction_labels(
    predictions: list[dict[str, float]],
    category_name_by_id: dict[int, str],
) -> tuple[list[tuple[float, float, float, float]], list[str]]:
    boxes = []
    labels = []
    for pred in predictions:
        x, y, w, h = pred["bbox"]
        boxes.append((float(x), float(y), float(x + w), float(y + h)))
        category_id = int(pred["category_id"])
        digit = category_name_by_id.get(category_id, str(category_id))
        labels.append(f"cid={category_id} digit={digit} {pred['score']:.2f}")
    return boxes, labels


def build_canvas(
    *,
    image_info: dict[str, Any],
    image_dir: Path,
    gt_annotations: list[dict[str, Any]],
    primary_predictions: list[dict[str, float]],
    compare_predictions: list[dict[str, float]] | None,
    category_name_by_id: dict[int, str],
    primary_title: str,
    compare_title: str | None,
) -> Image.Image:
    image_path = image_dir / image_info["file_name"]
    with Image.open(image_path) as img:
        original = img.convert("RGB")

    gt_boxes, gt_labels = format_gt_labels(gt_annotations, category_name_by_id)
    primary_boxes, primary_labels = format_prediction_labels(primary_predictions, category_name_by_id)

    panels = [
        draw_boxes(original, gt_boxes, gt_labels, color="#1f9d55"),
        draw_boxes(original, primary_boxes, primary_labels, color="#dc2626"),
    ]
    titles = [
        f"GT | img_id={image_info['id']} | gt={len(gt_annotations)}",
        f"{primary_title} | pred={len(primary_predictions)}",
    ]

    if compare_predictions is not None:
        compare_boxes, compare_labels = format_prediction_labels(compare_predictions, category_name_by_id)
        panels.append(draw_boxes(original, compare_boxes, compare_labels, color="#1d4ed8"))
        titles.append(f"{compare_title} | pred={len(compare_predictions)}")

    panel_width, panel_height = original.size
    title_height = 30
    canvas = Image.new("RGB", (panel_width * len(panels), panel_height + title_height), "white")
    for idx, panel in enumerate(panels):
        canvas.paste(panel, (panel_width * idx, title_height))

    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
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
    checkpoint_name: str,
    losses: dict[str, float],
    eval_stats: dict[str, float] | None,
    predictions: list[dict[str, float]],
    num_images: int,
    threshold: float,
    top_k: int,
    nms_iou_threshold: float,
) -> dict[str, float | int | None | str]:
    return {
        "checkpoint": checkpoint_name,
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
    }


def print_metric_summary(title: str, metrics: dict[str, float | int | None | str]) -> None:
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


def run_dn_checkpoint(
    checkpoint_path: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[argparse.Namespace, baseline.CocoDigitDataset, list[dict[str, float]], dict[str, float], dict[str, float] | None]:
    payload = load_checkpoint_payload(checkpoint_path, device)
    eval_args = build_dn_eval_args(
        payload.get("args", {}),
        args,
        score_threshold=args.dn_score_threshold,
        top_k=args.dn_top_k,
        nms_iou_threshold=args.dn_nms_iou_threshold,
    )
    normalize_device_arg(eval_args, device)

    dataset, loader = build_dn_valid_loader(eval_args)
    model, criterion, _ = dn_main.build_model(eval_args)
    model = model.to(device)
    criterion = criterion.to(device)
    dn_main.load_checkpoint(checkpoint_path, model, device=device)

    losses = dn_main.evaluate_loss(model, criterion, loader, device, eval_args)
    predictions, _ = dn_main.generate_predictions(model, loader, device, eval_args)
    eval_stats = baseline.maybe_run_coco_eval(args.valid_json, predictions)
    return eval_args, dataset, predictions, losses, eval_stats


def run_compare_dn_checkpoint(
    checkpoint_path: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[argparse.Namespace, list[dict[str, float]], dict[str, float], dict[str, float] | None]:
    payload = load_checkpoint_payload(checkpoint_path, device)
    eval_args = build_dn_eval_args(
        payload.get("args", {}),
        args,
        score_threshold=args.compare_score_threshold,
        top_k=args.compare_top_k,
        nms_iou_threshold=args.compare_nms_iou_threshold,
    )
    normalize_device_arg(eval_args, device)

    _, loader = build_dn_valid_loader(eval_args)
    model, criterion, _ = dn_main.build_model(eval_args)
    model = model.to(device)
    criterion = criterion.to(device)
    dn_main.load_checkpoint(checkpoint_path, model, device=device)

    losses = dn_main.evaluate_loss(model, criterion, loader, device, eval_args)
    predictions, _ = dn_main.generate_predictions(model, loader, device, eval_args)
    eval_stats = baseline.maybe_run_coco_eval(args.valid_json, predictions)
    return eval_args, predictions, losses, eval_stats


def main() -> None:
    cli_args = parse_args()
    cli_args.output_dir.mkdir(parents=True, exist_ok=True)
    visuals_dir = cli_args.output_dir / "samples"
    visuals_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(cli_args.device if cli_args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    baseline.set_seed(cli_args.seed)

    primary_args, dataset, primary_predictions, primary_losses, primary_eval_stats = run_dn_checkpoint(
        cli_args.dn_checkpoint,
        cli_args,
        device,
    )
    primary_metrics = summarize_metrics(
        checkpoint_name=str(cli_args.dn_checkpoint),
        losses=primary_losses,
        eval_stats=primary_eval_stats,
        predictions=primary_predictions,
        num_images=len(dataset),
        threshold=float(primary_args.score_threshold),
        top_k=int(primary_args.top_k),
        nms_iou_threshold=float(primary_args.nms_iou_threshold),
    )
    print_metric_summary("dn_primary", primary_metrics)

    compare_predictions_by_id: dict[int, list[dict[str, float]]] | None = None
    compare_metrics: dict[str, float | int | None | str] | None = None
    compare_title: str | None = None

    if cli_args.compare_dn_checkpoint is not None:
        compare_args, compare_predictions, compare_losses, compare_eval_stats = run_compare_dn_checkpoint(
            cli_args.compare_dn_checkpoint,
            cli_args,
            device,
        )
        compare_metrics = summarize_metrics(
            checkpoint_name=str(cli_args.compare_dn_checkpoint),
            losses=compare_losses,
            eval_stats=compare_eval_stats,
            predictions=compare_predictions,
            num_images=len(dataset),
            threshold=float(compare_args.score_threshold),
            top_k=int(compare_args.top_k),
            nms_iou_threshold=float(compare_args.nms_iou_threshold),
        )
        print_metric_summary("dn_compare", compare_metrics)
        compare_predictions_by_id = predictions_by_image_id(compare_predictions)
        compare_title = f"DN-B {cli_args.compare_dn_checkpoint.parent.name}"

    primary_predictions_by_id = predictions_by_image_id(primary_predictions)
    sampled_infos = sample_image_infos(dataset.images, cli_args.num_samples, cli_args.seed)
    category_name_by_id = {
        int(category["id"]): str(category["name"])
        for category in dataset.categories
    }
    primary_title = f"DN-A {cli_args.dn_checkpoint.parent.name}"

    for sample_idx, image_info in enumerate(sampled_infos, start=1):
        image_id = int(image_info["id"])
        comparison = build_canvas(
            image_info=image_info,
            image_dir=cli_args.valid_dir,
            gt_annotations=dataset.annotations_by_image.get(image_id, []),
            primary_predictions=primary_predictions_by_id.get(image_id, []),
            compare_predictions=None if compare_predictions_by_id is None else compare_predictions_by_id.get(image_id, []),
            category_name_by_id=category_name_by_id,
            primary_title=primary_title,
            compare_title=compare_title,
        )
        output_path = visuals_dir / f"{sample_idx:02d}_image_{image_id}.png"
        comparison.save(output_path)

    summary = {
        "valid_dir": str(cli_args.valid_dir),
        "valid_json": str(cli_args.valid_json),
        "dn_primary": primary_metrics,
        "dn_compare": compare_metrics,
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
