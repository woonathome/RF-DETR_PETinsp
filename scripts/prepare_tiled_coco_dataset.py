#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from PIL import Image
from tqdm import tqdm


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
SPLITS = ("train", "valid", "test")


@dataclass(frozen=True)
class SourceSample:
    image_path: Path
    label_path: Path
    relative_image_path: Path
    relative_key: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert YOLO-labeled full images to tiled COCO dataset for RF-DETR training."
        )
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset"))
    parser.add_argument("--image-subdir", type=Path, default=Path("image"))
    parser.add_argument("--label-subdir", type=Path, default=Path("label"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data") / "rfdetr_tiled_coco",
    )
    parser.add_argument("--resize-size", type=int, default=2048)
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-box-area", type=float, default=1.0)
    parser.add_argument("--class-names-file", type=Path, default=None)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument(
        "--keep-empty-tiles",
        action="store_true",
        help="Keep tiles with no defect boxes. Default keeps defect-only tiles.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove existing output directory if it exists.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.grid_size <= 0:
        raise ValueError("--grid-size must be a positive integer.")
    if args.tile_size <= 0:
        raise ValueError("--tile-size must be a positive integer.")
    if args.resize_size != args.grid_size * args.tile_size:
        raise ValueError(
            "--resize-size must match grid_size * tile_size "
            f"(current: {args.resize_size} vs {args.grid_size * args.tile_size})."
        )
    if not (0.0 <= args.val_ratio < 1.0 and 0.0 <= args.test_ratio < 1.0):
        raise ValueError("--val-ratio and --test-ratio must be in [0, 1).")
    if args.val_ratio + args.test_ratio >= 1.0:
        raise ValueError("--val-ratio + --test-ratio must be < 1.0.")
    if args.min_box_area < 0:
        raise ValueError("--min-box-area must be >= 0.")


def prepare_output_dirs(output_root: Path, overwrite: bool) -> Dict[str, Path]:
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output path already exists: {output_root}. "
                "Use --overwrite to regenerate."
            )
        shutil.rmtree(output_root)

    split_dirs = {}
    for split in SPLITS:
        split_dir = output_root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        split_dirs[split] = split_dir
    (output_root / "metadata").mkdir(parents=True, exist_ok=True)
    return split_dirs


def collect_samples(
    image_root: Path,
    label_root: Path,
    max_images: int | None = None,
) -> Tuple[List[SourceSample], List[Path]]:
    image_paths = sorted(
        p for p in image_root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if max_images is not None:
        image_paths = image_paths[: max_images]

    samples: List[SourceSample] = []
    missing_labels: List[Path] = []
    for image_path in image_paths:
        relative_image_path = image_path.relative_to(image_root)
        label_path = label_root / relative_image_path.with_suffix(".txt")
        if not label_path.exists():
            missing_labels.append(relative_image_path)
            continue
        relative_key = relative_image_path.with_suffix("").as_posix()
        samples.append(
            SourceSample(
                image_path=image_path,
                label_path=label_path,
                relative_image_path=relative_image_path,
                relative_key=relative_key,
            )
        )
    return samples, missing_labels


def parse_yolo_file(label_path: Path) -> List[Tuple[int, float, float, float, float]]:
    rows: List[Tuple[int, float, float, float, float]] = []
    for raw in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        cx, cy, w, h = map(float, parts[1:5])
        rows.append((cls, cx, cy, w, h))
    return rows


def discover_yolo_classes(samples: Sequence[SourceSample]) -> List[int]:
    class_ids = set()
    for sample in samples:
        for cls, *_ in parse_yolo_file(sample.label_path):
            class_ids.add(cls)
    return sorted(class_ids)


def load_class_names(
    class_names_file: Path | None, yolo_ids: Sequence[int]
) -> Dict[int, str]:
    if class_names_file is None:
        return {cid: f"class_{cid}" for cid in yolo_ids}

    raw_names = [
        x.strip()
        for x in class_names_file.read_text(encoding="utf-8", errors="ignore").splitlines()
        if x.strip()
    ]
    name_map: Dict[int, str] = {}
    for cid in yolo_ids:
        if cid < len(raw_names):
            name_map[cid] = raw_names[cid]
        else:
            name_map[cid] = f"class_{cid}"
    return name_map


def split_samples(
    samples: Sequence[SourceSample],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Dict[str, str]:
    shuffled = list(samples)
    random.Random(seed).shuffle(shuffled)

    total = len(shuffled)
    n_test = int(round(total * test_ratio))
    n_valid = int(round(total * val_ratio))

    if total >= 3 and test_ratio > 0 and n_test == 0:
        n_test = 1
    if total >= 3 and val_ratio > 0 and n_valid == 0:
        n_valid = 1
    if n_test + n_valid >= total:
        overflow = n_test + n_valid - total + 1
        n_valid = max(0, n_valid - overflow)

    n_train = total - n_valid - n_test
    split_map: Dict[str, str] = {}

    for sample in shuffled[:n_train]:
        split_map[sample.relative_key] = "train"
    for sample in shuffled[n_train : n_train + n_valid]:
        split_map[sample.relative_key] = "valid"
    for sample in shuffled[n_train + n_valid :]:
        split_map[sample.relative_key] = "test"
    return split_map


def yolo_to_xyxy_resized(
    yolo_rows: Sequence[Tuple[int, float, float, float, float]],
    resized_w: int,
    resized_h: int,
) -> List[Tuple[int, float, float, float, float]]:
    xyxy_boxes = []
    for cls, cx, cy, bw, bh in yolo_rows:
        x1 = (cx - bw / 2.0) * resized_w
        y1 = (cy - bh / 2.0) * resized_h
        x2 = (cx + bw / 2.0) * resized_w
        y2 = (cy + bh / 2.0) * resized_h

        x1 = max(0.0, min(float(resized_w), x1))
        y1 = max(0.0, min(float(resized_h), y1))
        x2 = max(0.0, min(float(resized_w), x2))
        y2 = max(0.0, min(float(resized_h), y2))

        if x2 <= x1 or y2 <= y1:
            continue
        xyxy_boxes.append((cls, x1, y1, x2, y2))
    return xyxy_boxes


def intersect_with_tile(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    tile_x1: int,
    tile_y1: int,
    tile_x2: int,
    tile_y2: int,
) -> Tuple[float, float, float, float] | None:
    ix1 = max(x1, tile_x1)
    iy1 = max(y1, tile_y1)
    ix2 = min(x2, tile_x2)
    iy2 = min(y2, tile_y2)
    if ix2 <= ix1 or iy2 <= iy1:
        return None
    return ix1 - tile_x1, iy1 - tile_y1, ix2 - ix1, iy2 - iy1


def default_coco_template(categories: List[Dict]) -> Dict:
    return {
        "info": {
            "description": "RF-DETR tiled defect dataset",
            "date_created": datetime.now(timezone.utc).isoformat(),
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": categories,
    }


def safe_stem_from_relative_key(relative_key: str) -> str:
    return relative_key.replace("/", "__").replace("\\", "__").replace(" ", "_")


def main() -> None:
    args = parse_args()
    validate_args(args)

    dataset_root = args.dataset_root.resolve()
    image_root = (dataset_root / args.image_subdir).resolve()
    label_root = (dataset_root / args.label_subdir).resolve()
    output_root = args.output_root.resolve()

    if not image_root.exists():
        raise FileNotFoundError(f"Image root not found: {image_root}")
    if not label_root.exists():
        raise FileNotFoundError(f"Label root not found: {label_root}")

    split_dirs = prepare_output_dirs(output_root, overwrite=args.overwrite)

    samples, missing_labels = collect_samples(
        image_root=image_root, label_root=label_root, max_images=args.max_images
    )
    if not samples:
        raise RuntimeError("No valid image/label pairs found.")

    yolo_ids = discover_yolo_classes(samples)
    if not yolo_ids:
        raise RuntimeError("No classes discovered from labels.")

    class_name_map = load_class_names(args.class_names_file, yolo_ids)
    yolo_to_coco = {yolo_id: idx + 1 for idx, yolo_id in enumerate(yolo_ids)}
    categories = [
        {
            "id": yolo_to_coco[yolo_id],
            "name": class_name_map[yolo_id],
            "supercategory": "defect",
        }
        for yolo_id in yolo_ids
    ]

    split_map = split_samples(
        samples=samples,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    coco_by_split = {
        split: default_coco_template(categories=categories) for split in SPLITS
    }
    image_id_counter = {split: 1 for split in SPLITS}
    ann_id_counter = {split: 1 for split in SPLITS}

    split_stats = {
        split: {
            "source_images": 0,
            "saved_tiles": 0,
            "saved_annotations": 0,
            "skipped_empty_tiles": 0,
        }
        for split in SPLITS
    }
    manifest_rows: List[Dict[str, str | int]] = []

    for sample in tqdm(samples, desc="Processing source images"):
        split = split_map[sample.relative_key]
        split_stats[split]["source_images"] += 1

        yolo_rows = parse_yolo_file(sample.label_path)
        if not yolo_rows and not args.keep_empty_tiles:
            continue

        with Image.open(sample.image_path) as img:
            image = img.convert("RGB")
        if image.size != (args.resize_size, args.resize_size):
            image = image.resize((args.resize_size, args.resize_size), Image.Resampling.BILINEAR)

        resized_boxes = yolo_to_xyxy_resized(
            yolo_rows=yolo_rows,
            resized_w=args.resize_size,
            resized_h=args.resize_size,
        )

        sample_stem = safe_stem_from_relative_key(sample.relative_key)

        for row in range(args.grid_size):
            for col in range(args.grid_size):
                tile_x1 = col * args.tile_size
                tile_y1 = row * args.tile_size
                tile_x2 = tile_x1 + args.tile_size
                tile_y2 = tile_y1 + args.tile_size

                tile_boxes = []
                for yolo_cls, x1, y1, x2, y2 in resized_boxes:
                    intersected = intersect_with_tile(
                        x1,
                        y1,
                        x2,
                        y2,
                        tile_x1,
                        tile_y1,
                        tile_x2,
                        tile_y2,
                    )
                    if intersected is None:
                        continue
                    bx, by, bw, bh = intersected
                    if bw * bh < args.min_box_area:
                        continue
                    tile_boxes.append((yolo_cls, bx, by, bw, bh))

                if not tile_boxes and not args.keep_empty_tiles:
                    split_stats[split]["skipped_empty_tiles"] += 1
                    continue

                tile = image.crop((tile_x1, tile_y1, tile_x2, tile_y2))
                tile_file_name = f"{sample_stem}__r{row:02d}_c{col:02d}.jpg"
                tile_output_path = split_dirs[split] / tile_file_name
                tile.save(tile_output_path, format="JPEG", quality=95)

                image_id = image_id_counter[split]
                image_id_counter[split] += 1

                coco_by_split[split]["images"].append(
                    {
                        "id": image_id,
                        "file_name": tile_file_name,
                        "width": args.tile_size,
                        "height": args.tile_size,
                    }
                )

                for yolo_cls, bx, by, bw, bh in tile_boxes:
                    ann_id = ann_id_counter[split]
                    ann_id_counter[split] += 1
                    coco_by_split[split]["annotations"].append(
                        {
                            "id": ann_id,
                            "image_id": image_id,
                            "category_id": yolo_to_coco[yolo_cls],
                            "bbox": [bx, by, bw, bh],
                            "area": bw * bh,
                            "iscrowd": 0,
                        }
                    )

                split_stats[split]["saved_tiles"] += 1
                split_stats[split]["saved_annotations"] += len(tile_boxes)
                manifest_rows.append(
                    {
                        "split": split,
                        "tile_file_name": tile_file_name,
                        "source_image": sample.relative_image_path.as_posix(),
                        "tile_row": row,
                        "tile_col": col,
                        "num_boxes": len(tile_boxes),
                    }
                )

    for split in SPLITS:
        ann_path = split_dirs[split] / "_annotations.coco.json"
        with ann_path.open("w", encoding="utf-8") as f:
            json.dump(coco_by_split[split], f, ensure_ascii=False, indent=2)

    metadata_dir = output_root / "metadata"
    summary = {
        "dataset_root": str(dataset_root),
        "image_root": str(image_root),
        "label_root": str(label_root),
        "output_root": str(output_root),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "settings": {
            "resize_size": args.resize_size,
            "grid_size": args.grid_size,
            "tile_size": args.tile_size,
            "val_ratio": args.val_ratio,
            "test_ratio": args.test_ratio,
            "seed": args.seed,
            "min_box_area": args.min_box_area,
            "keep_empty_tiles": args.keep_empty_tiles,
        },
        "totals": {
            "source_image_files_scanned": len(samples) + len(missing_labels),
            "source_images_with_labels": len(samples),
            "missing_label_files": len(missing_labels),
            "split_source_images": {
                split: split_stats[split]["source_images"] for split in SPLITS
            },
            "split_saved_tiles": {
                split: split_stats[split]["saved_tiles"] for split in SPLITS
            },
            "split_saved_annotations": {
                split: split_stats[split]["saved_annotations"] for split in SPLITS
            },
            "split_skipped_empty_tiles": {
                split: split_stats[split]["skipped_empty_tiles"] for split in SPLITS
            },
        },
    }
    with (metadata_dir / "preprocess_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    class_mapping_payload = {
        "yolo_to_coco": {str(k): v for k, v in yolo_to_coco.items()},
        "class_names": {str(k): class_name_map[k] for k in yolo_ids},
    }
    with (metadata_dir / "class_mapping.json").open("w", encoding="utf-8") as f:
        json.dump(class_mapping_payload, f, ensure_ascii=False, indent=2)

    with (metadata_dir / "missing_labels.txt").open("w", encoding="utf-8") as f:
        for rel in missing_labels:
            f.write(rel.as_posix() + "\n")

    manifest_path = metadata_dir / "tile_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "split",
                "tile_file_name",
                "source_image",
                "tile_row",
                "tile_col",
                "num_boxes",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    print("Preprocessing finished.")
    print(f"Output directory: {output_root}")
    print(f"Source images with labels: {len(samples)}")
    print(f"Missing labels skipped: {len(missing_labels)}")
    for split in SPLITS:
        print(
            f"[{split}] source_images={split_stats[split]['source_images']} "
            f"tiles={split_stats[split]['saved_tiles']} "
            f"annotations={split_stats[split]['saved_annotations']}"
        )


if __name__ == "__main__":
    main()
