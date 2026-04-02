#!/usr/bin/env python3
"""Offline data augmentation for image classification.

This script expands each training image into a fixed number of images.
Default behavior is 1 -> 3 (one original copy + two augmented copies).
"""

import argparse
import math
import random
import shutil
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFile
from torchvision import transforms
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline augmentation: expand dataset size per image.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Input train directory, e.g. ./data/train")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory, e.g. ./data_aug/train")
    parser.add_argument(
        "--copies-per-image",
        type=int,
        default=3,
        help="Total output images per source image (default: 3).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete output directory first if it already exists.",
    )
    parser.add_argument(
        "--mixup-copies-per-image",
        type=int,
        default=0,
        help="Additional same-class mixup-blend images per source image (default: 0).",
    )
    parser.add_argument(
        "--mixup-alpha",
        type=float,
        default=0.2,
        help="Beta(alpha, alpha) used by offline same-class mixup blend.",
    )
    parser.add_argument(
        "--resize-short-side",
        type=int,
        default=256,
        help="Resize short side to this value before optional center crop. Set <= 0 to disable.",
    )
    parser.add_argument(
        "--center-crop-size",
        type=int,
        default=224,
        help="Center crop size after resizing. Set <= 0 to disable.",
    )
    parser.add_argument(
        "--deterministic-only",
        action="store_true",
        help="Disable random offline augmentation and only apply deterministic preprocess.",
    )
    parser.add_argument(
        "--balance-to-max2x",
        action="store_true",
        help="Balance every class to 2x of the largest source class count.",
    )
    parser.add_argument(
        "--v4-fixed-1p5x",
        action="store_true",
        help=(
            "Generate a fixed 1.5x train set per class for V4 policy: "
            "0.5x plain + 0.5x same-class mixup + 0.5x random erasing."
        ),
    )
    return parser.parse_args()


def build_augmentation_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.08)],
                p=0.8,
            ),
            transforms.RandomApply(
                [
                    transforms.RandomAffine(
                        degrees=12,
                        translate=(0.05, 0.05),
                        scale=(0.9, 1.1),
                        shear=5,
                    )
                ],
                p=0.7,
            ),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2))], p=0.3),
        ]
    )


def build_deterministic_transform(resize_short_side: int, center_crop_size: int) -> transforms.Compose | None:
    steps = []
    if resize_short_side > 0:
        steps.append(transforms.Resize(resize_short_side))
    if center_crop_size > 0:
        steps.append(transforms.CenterCrop(center_crop_size))
    if not steps:
        return None
    return transforms.Compose(steps)


def collect_images(class_dir: Path) -> list[Path]:
    return sorted(
        [p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    )


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def save_rgb_jpg(image: Image.Image, path: Path) -> None:
    image.convert("RGB").save(path, format="JPEG", quality=95)


def sample_mix_ratio(alpha: float) -> float:
    if alpha <= 0:
        return 1.0
    return torch.distributions.Beta(alpha, alpha).sample().item()


def mix_images_same_class(image_a: Image.Image, image_b: Image.Image, alpha: float) -> Image.Image:
    lam = sample_mix_ratio(alpha)

    # Use the anchor image size as the canonical size for blending.
    if image_b.size != image_a.size:
        if hasattr(Image, "Resampling"):
            image_b = image_b.resize(image_a.size, resample=Image.Resampling.BILINEAR)
        else:
            image_b = image_b.resize(image_a.size, resample=Image.BILINEAR)

    tensor_a = transforms.ToTensor()(image_a)
    tensor_b = transforms.ToTensor()(image_b)
    mixed = lam * tensor_a + (1.0 - lam) * tensor_b
    return transforms.ToPILImage()(mixed.clamp(0.0, 1.0))


def load_rgb_image(path: Path) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB")


def apply_deterministic(image: Image.Image, deterministic_transform: transforms.Compose | None) -> Image.Image:
    if deterministic_transform is None:
        return image
    return deterministic_transform(image)


def apply_random_erasing(
    image: Image.Image,
    min_area_ratio: float = 0.03,
    max_area_ratio: float = 0.12,
    min_aspect_ratio: float = 0.3,
    max_aspect_ratio: float = 3.3,
) -> Image.Image:
    width, height = image.size
    area = width * height
    if area <= 0:
        return image

    out = image.copy()
    draw = ImageDraw.Draw(out)

    for _ in range(10):
        target_area = random.uniform(min_area_ratio, max_area_ratio) * area
        aspect_ratio = random.uniform(min_aspect_ratio, max_aspect_ratio)
        erase_w = int(round(math.sqrt(target_area * aspect_ratio)))
        erase_h = int(round(math.sqrt(target_area / aspect_ratio)))

        if erase_w < width and erase_h < height and erase_w > 0 and erase_h > 0:
            x0 = random.randint(0, width - erase_w)
            y0 = random.randint(0, height - erase_h)
            fill = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255),
            )
            draw.rectangle([x0, y0, x0 + erase_w, y0 + erase_h], fill=fill)
            return out

    return out


def save_class_v4_fixed_1p5x(
    class_dir: Path,
    output_class_dir: Path,
    image_paths: list[Path],
    deterministic_transform: transforms.Compose | None,
    mixup_alpha: float,
) -> int:
    generated = 0
    seq = 0

    target_total = max(1, int(round(1.5 * len(image_paths))))
    base = target_total // 3
    rem = target_total % 3
    plain_count = base + (1 if rem > 0 else 0)
    mix_count = base + (1 if rem > 1 else 0)
    erase_count = base

    def load_base_image(path: Path) -> Image.Image:
        img = load_rgb_image(path)
        return apply_deterministic(img, deterministic_transform)

    for _ in range(plain_count):
        image_path = random.choice(image_paths)
        seq += 1
        img = load_base_image(image_path)
        output_plain = output_class_dir / f"{seq:06d}_{image_path.stem}_plain.jpg"
        save_rgb_jpg(img, output_plain)
        generated += 1

    for _ in range(mix_count):
        image_a_path = random.choice(image_paths)
        image_b_path = random.choice(image_paths)
        seq += 1
        image_a = load_base_image(image_a_path)
        image_b = load_base_image(image_b_path)
        if len(image_paths) > 1 and mixup_alpha > 0:
            out_img = mix_images_same_class(image_a, image_b, mixup_alpha)
            suffix = "mix"
        else:
            out_img = image_a
            suffix = "plain"
        output_mix = output_class_dir / f"{seq:06d}_{image_a_path.stem}_{suffix}.jpg"
        save_rgb_jpg(out_img, output_mix)
        generated += 1

    for _ in range(erase_count):
        image_path = random.choice(image_paths)
        seq += 1
        img = load_base_image(image_path)
        out_img = apply_random_erasing(img)
        output_erase = output_class_dir / f"{seq:06d}_{image_path.stem}_erase.jpg"
        save_rgb_jpg(out_img, output_erase)
        generated += 1

    print(
        f"V4 class {class_dir.name}: target={target_total}, "
        f"plain={plain_count}, mix={mix_count}, erase={erase_count}"
    )
    return generated


def save_class_fixed_copies(
    class_dir: Path,
    output_class_dir: Path,
    image_paths: list[Path],
    copies_per_image: int,
    mixup_copies_per_image: int,
    mixup_alpha: float,
    augment_transform: transforms.Compose,
    deterministic_transform: transforms.Compose | None,
    deterministic_only: bool,
) -> int:
    generated = 0

    for idx, image_path in enumerate(tqdm(image_paths, desc=f"Augment class {class_dir.name}"), start=1):
        sample_prefix = f"{idx:06d}_{image_path.stem}"
        img = load_rgb_image(image_path)
        img = apply_deterministic(img, deterministic_transform)

        output_original = output_class_dir / f"{sample_prefix}_orig.jpg"
        save_rgb_jpg(img, output_original)
        generated += 1

        if deterministic_only:
            continue

        for aug_idx in range(1, copies_per_image):
            aug_img = augment_transform(img.copy())
            output_aug = output_class_dir / f"{sample_prefix}_aug{aug_idx}.jpg"
            save_rgb_jpg(aug_img, output_aug)
            generated += 1

        if mixup_copies_per_image > 0 and len(image_paths) > 1:
            for mix_idx in range(1, mixup_copies_per_image + 1):
                partner_path = random.choice(image_paths)
                partner_img = load_rgb_image(partner_path)
                partner_img = apply_deterministic(partner_img, deterministic_transform)
                mixed_img = mix_images_same_class(img, partner_img, mixup_alpha)
                output_mix = output_class_dir / f"{sample_prefix}_mix{mix_idx}.jpg"
                save_rgb_jpg(mixed_img, output_mix)
                generated += 1

    return generated


def save_class_balanced(
    class_dir: Path,
    output_class_dir: Path,
    image_paths: list[Path],
    target_count: int,
    augment_transform: transforms.Compose,
    deterministic_transform: transforms.Compose | None,
    deterministic_only: bool,
    mixup_alpha: float,
) -> int:
    generated = 0
    seq = 0

    # Keep one deterministic base copy for each source first.
    for image_path in image_paths:
        seq += 1
        img = load_rgb_image(image_path)
        img = apply_deterministic(img, deterministic_transform)
        output_original = output_class_dir / f"{seq:06d}_{image_path.stem}_orig.jpg"
        save_rgb_jpg(img, output_original)
        generated += 1
        if generated >= target_count:
            return generated

    if deterministic_only:
        # Fallback for deterministic-only mode: repeat deterministic copies to hit target.
        while generated < target_count:
            image_path = random.choice(image_paths)
            seq += 1
            img = load_rgb_image(image_path)
            img = apply_deterministic(img, deterministic_transform)
            output_copy = output_class_dir / f"{seq:06d}_{image_path.stem}_copy.jpg"
            save_rgb_jpg(img, output_copy)
            generated += 1
        return generated

    while generated < target_count:
        image_path = random.choice(image_paths)
        seq += 1
        base_img = load_rgb_image(image_path)
        base_img = apply_deterministic(base_img, deterministic_transform)

        can_mix = len(image_paths) > 1 and mixup_alpha > 0
        if can_mix and random.random() < 0.35:
            partner_path = random.choice(image_paths)
            partner_img = load_rgb_image(partner_path)
            partner_img = apply_deterministic(partner_img, deterministic_transform)
            out_img = mix_images_same_class(base_img, partner_img, mixup_alpha)
            suffix = "mix"
        else:
            out_img = augment_transform(base_img.copy())
            suffix = "aug"

        output_path = output_class_dir / f"{seq:06d}_{image_path.stem}_{suffix}.jpg"
        save_rgb_jpg(out_img, output_path)
        generated += 1

    return generated


def save_flat_deterministic(
    input_dir: Path,
    output_dir: Path,
    deterministic_transform: transforms.Compose | None,
) -> tuple[int, int]:
    image_paths = collect_images(input_dir)
    if not image_paths:
        raise RuntimeError(f"No images found in flat directory: {input_dir}")

    generated = 0
    for image_path in tqdm(image_paths, desc="Preprocess flat dataset"):
        img = load_rgb_image(image_path)
        img = apply_deterministic(img, deterministic_transform)
        output_path = output_dir / f"{image_path.stem}.jpg"
        save_rgb_jpg(img, output_path)
        generated += 1

    return len(image_paths), generated


def main() -> None:
    args = parse_args()

    if args.copies_per_image < 1:
        raise ValueError("--copies-per-image must be >= 1")
    if args.mixup_copies_per_image < 0:
        raise ValueError("--mixup-copies-per-image must be >= 0")
    if args.v4_fixed_1p5x and args.balance_to_max2x:
        raise ValueError("--v4-fixed-1p5x cannot be used with --balance-to-max2x")
    if args.balance_to_max2x and args.deterministic_only:
        print("[WARN] --balance-to-max2x with --deterministic-only will duplicate deterministic images.")
    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")

    random.seed(args.seed)

    prepare_output_dir(args.output_dir, args.overwrite)
    augment_transform = build_augmentation_transform()
    deterministic_transform = build_deterministic_transform(args.resize_short_side, args.center_crop_size)

    class_dirs = sorted([p for p in args.input_dir.iterdir() if p.is_dir()])
    if not class_dirs:
        if args.balance_to_max2x:
            raise RuntimeError("--balance-to-max2x requires class subfolders under --input-dir")
        if not args.deterministic_only:
            raise RuntimeError(
                "Flat input directory is supported only with --deterministic-only. "
                "Use class subfolders for train augmentation."
            )

        total_source_images, total_generated_images = save_flat_deterministic(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            deterministic_transform=deterministic_transform,
        )
        print("Preprocess finished.")
        print(f"Input images:  {total_source_images}")
        print(f"Output images: {total_generated_images}")
        print(f"Expansion ratio: {total_generated_images / total_source_images:.2f}x")
        return

    class_to_images: dict[str, list[Path]] = {}
    max_class_count = 0
    for class_dir in class_dirs:
        image_paths = collect_images(class_dir)
        class_to_images[class_dir.name] = image_paths
        max_class_count = max(max_class_count, len(image_paths))

    if max_class_count == 0:
        raise RuntimeError(f"No images found under class folders in {args.input_dir}")

    total_source_images = 0
    total_generated_images = 0

    target_per_class = max_class_count * 2 if args.balance_to_max2x else None
    if target_per_class is not None:
        print(f"Class balancing enabled. Target per class: {target_per_class}")

    for class_dir in class_dirs:
        output_class_dir = args.output_dir / class_dir.name
        output_class_dir.mkdir(parents=True, exist_ok=True)

        image_paths = class_to_images[class_dir.name]
        if not image_paths:
            print(f"[WARN] Skip empty class folder: {class_dir}")
            continue

        total_source_images += len(image_paths)

        if args.v4_fixed_1p5x:
            generated = save_class_v4_fixed_1p5x(
                class_dir=class_dir,
                output_class_dir=output_class_dir,
                image_paths=image_paths,
                deterministic_transform=deterministic_transform,
                mixup_alpha=args.mixup_alpha,
            )
        elif target_per_class is None:
            generated = save_class_fixed_copies(
                class_dir=class_dir,
                output_class_dir=output_class_dir,
                image_paths=image_paths,
                copies_per_image=args.copies_per_image,
                mixup_copies_per_image=args.mixup_copies_per_image,
                mixup_alpha=args.mixup_alpha,
                augment_transform=augment_transform,
                deterministic_transform=deterministic_transform,
                deterministic_only=args.deterministic_only,
            )
        else:
            generated = save_class_balanced(
                class_dir=class_dir,
                output_class_dir=output_class_dir,
                image_paths=image_paths,
                target_count=target_per_class,
                augment_transform=augment_transform,
                deterministic_transform=deterministic_transform,
                deterministic_only=args.deterministic_only,
                mixup_alpha=args.mixup_alpha,
            )

        total_generated_images += generated

    print("Augmentation finished.")
    print(f"Input images:  {total_source_images}")
    print(f"Output images: {total_generated_images}")
    if total_source_images > 0:
        print(f"Expansion ratio: {total_generated_images / total_source_images:.2f}x")


if __name__ == "__main__":
    main()
