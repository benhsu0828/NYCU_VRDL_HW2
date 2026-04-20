#!/usr/bin/env python3
"""Offline canvas augmentation for COCO detection training data.

This script builds a new train image folder plus a matching COCO JSON by
copying each source image once (orig) and generating extra canvas-augmented
copies with synchronized bbox offsets.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

from PIL import Image, ImageFile
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="COCO detection offline canvas augmentation.")
    parser.add_argument("--input-image-dir", type=Path, required=True, help="Source train image directory.")
    parser.add_argument("--input-json", type=Path, required=True, help="Source COCO train annotation json.")
    parser.add_argument("--output-image-dir", type=Path, required=True, help="Output image directory.")
    parser.add_argument("--output-json", type=Path, required=True, help="Output COCO annotation json.")
    parser.add_argument(
        "--copies-per-image",
        type=int,
        default=2,
        help="Total copies per source image. First copy is orig; remaining copies are canvas-aug (default: 2).",
    )
    parser.add_argument(
        "--canvas-scale-factor",
        type=float,
        default=1.2,
        help="Canvas size multiplier for augmented copies (default: 1.2).",
    )
    parser.add_argument(
        "--random-expand-noise-std",
        type=float,
        default=0.08,
        help="Noise std in normalized scale [0,1]. Pixel std uses std*255 (default: 0.08).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--overwrite", action="store_true", help="Remove output paths before generation.")
    return parser.parse_args()


def load_coco(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        coco = json.load(f)
    required = {"images", "annotations", "categories"}
    missing = required.difference(coco.keys())
    if missing:
        raise ValueError(f"Input json missing keys: {sorted(missing)}")
    return coco


def prepare_output_paths(output_image_dir: Path, output_json: Path, overwrite: bool) -> None:
    if overwrite and output_image_dir.exists():
        shutil.rmtree(output_image_dir)
    output_image_dir.mkdir(parents=True, exist_ok=True)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    if overwrite and output_json.exists():
        output_json.unlink()


def build_annotations_by_image(annotations: list[dict]) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = {}
    for ann in annotations:
        image_id = int(ann["image_id"])
        grouped.setdefault(image_id, []).append(ann)
    return grouped


def load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def make_noise_canvas(height: int, width: int, noise_std: float) -> Image.Image:
    pixel_std = max(0.0, float(noise_std)) * 255.0
    if pixel_std <= 0.0:
        return Image.new("RGB", (width, height), color=(127, 127, 127))

    # effect_noise returns grayscale Gaussian-like noise centered around mid-gray.
    noise_gray = Image.effect_noise((width, height), sigma=pixel_std)
    return Image.merge("RGB", (noise_gray, noise_gray, noise_gray))


def clip_bbox_xywh(bbox: list[float], width: int, height: int) -> list[float] | None:
    x, y, w, h = [float(v) for v in bbox]
    x0 = max(0.0, min(x, float(width)))
    y0 = max(0.0, min(y, float(height)))
    x1 = max(0.0, min(x + w, float(width)))
    y1 = max(0.0, min(y + h, float(height)))
    new_w = x1 - x0
    new_h = y1 - y0
    if new_w <= 0.0 or new_h <= 0.0:
        return None
    return [x0, y0, new_w, new_h]


def remap_annotations(
    source_annotations: list[dict],
    new_image_id: int,
    ann_id_start: int,
    dx: int,
    dy: int,
    out_width: int,
    out_height: int,
) -> tuple[list[dict], int]:
    new_annotations: list[dict] = []
    next_ann_id = ann_id_start

    for ann in source_annotations:
        bbox = ann.get("bbox", None)
        if bbox is None or len(bbox) != 4:
            continue

        shifted_bbox = [float(bbox[0]) + dx, float(bbox[1]) + dy, float(bbox[2]), float(bbox[3])]
        clipped = clip_bbox_xywh(shifted_bbox, out_width, out_height)
        if clipped is None:
            continue

        next_ann_id += 1
        new_annotations.append(
            {
                "id": next_ann_id,
                "image_id": new_image_id,
                "bbox": [round(clipped[0], 4), round(clipped[1], 4), round(clipped[2], 4), round(clipped[3], 4)],
                "category_id": int(ann["category_id"]),
                "area": round(clipped[2] * clipped[3], 4),
                "iscrowd": int(ann.get("iscrowd", 0)),
            }
        )

    return new_annotations, next_ann_id


def run_augmentation(args: argparse.Namespace) -> dict:
    if args.copies_per_image < 1:
        raise ValueError("--copies-per-image must be >= 1")
    if args.canvas_scale_factor < 1.0:
        raise ValueError("--canvas-scale-factor must be >= 1.0")

    random.seed(args.seed)

    coco = load_coco(args.input_json)
    images = coco["images"]
    annotations_by_image = build_annotations_by_image(coco["annotations"])

    prepare_output_paths(args.output_image_dir, args.output_json, args.overwrite)

    new_images: list[dict] = []
    new_annotations: list[dict] = []

    next_image_id = 0
    next_ann_id = 0

    for image_info in tqdm(images, desc="Generating offline canvas aug"):
        src_image_id = int(image_info["id"])
        src_file_name = str(image_info["file_name"])
        src_path = args.input_image_dir / src_file_name
        if not src_path.exists():
            raise FileNotFoundError(f"Image not found: {src_path}")

        src = load_rgb(src_path)
        src_w, src_h = src.size
        src_anns = annotations_by_image.get(src_image_id, [])
        stem = Path(src_file_name).stem

        for copy_idx in range(args.copies_per_image):
            is_orig = copy_idx == 0

            if is_orig:
                out = src.copy()
                out_w, out_h = src_w, src_h
                dx, dy = 0, 0
                suffix = "orig"
            else:
                out_w = max(src_w, int(round(src_w * args.canvas_scale_factor)))
                out_h = max(src_h, int(round(src_h * args.canvas_scale_factor)))
                max_left = out_w - src_w
                max_top = out_h - src_h
                dx = random.randint(0, max_left) if max_left > 0 else 0
                dy = random.randint(0, max_top) if max_top > 0 else 0

                out = make_noise_canvas(out_h, out_w, args.random_expand_noise_std)
                out.paste(src, (dx, dy))
                suffix = f"canvas{copy_idx}"

            next_image_id += 1
            out_name = f"{next_image_id:06d}_{stem}_{suffix}.png"
            out_path = args.output_image_dir / out_name
            out.save(out_path)

            new_images.append(
                {
                    "id": next_image_id,
                    "file_name": out_name,
                    "height": int(out_h),
                    "width": int(out_w),
                }
            )

            remapped, next_ann_id = remap_annotations(
                source_annotations=src_anns,
                new_image_id=next_image_id,
                ann_id_start=next_ann_id,
                dx=dx,
                dy=dy,
                out_width=out_w,
                out_height=out_h,
            )
            new_annotations.extend(remapped)

    out_coco = {
        "images": new_images,
        "annotations": new_annotations,
        "categories": coco["categories"],
    }

    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(out_coco, f)

    return out_coco


def validate_coco(coco: dict, sample_count: int = 20) -> None:
    image_meta = {int(img["id"]): img for img in coco["images"]}
    anns = coco["annotations"]

    bad = 0
    if anns:
        step = max(1, len(anns) // sample_count)
        sampled = anns[::step][:sample_count]
    else:
        sampled = []

    for ann in sampled:
        image_id = int(ann["image_id"])
        img = image_meta.get(image_id)
        if img is None:
            bad += 1
            continue
        x, y, w, h = [float(v) for v in ann["bbox"]]
        iw, ih = float(img["width"]), float(img["height"])
        valid = w > 0 and h > 0 and x >= 0 and y >= 0 and x + w <= iw + 1e-5 and y + h <= ih + 1e-5
        if not valid:
            bad += 1

    print(f"Output images: {len(coco['images'])}")
    print(f"Output annotations: {len(coco['annotations'])}")
    print(f"Validation sample size: {len(sampled)}")
    print(f"Invalid sampled bboxes: {bad}")


def main() -> None:
    args = parse_args()
    out_coco = run_augmentation(args)
    validate_coco(out_coco, sample_count=20)
    print(f"Saved images to: {args.output_image_dir}")
    print(f"Saved json to: {args.output_json}")


if __name__ == "__main__":
    main()
