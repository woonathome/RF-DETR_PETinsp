#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
SPLITS = ("train", "valid", "test")


@dataclass(frozen=True)
class SourceSample:
    index: int
    image_path: Path
    label_path: Path
    image_stem: str
    image_name: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Rebuild preprocessing pipeline:\n"
            "1) filename-rule relabel to unknown,\n"
            "2) pockmark top-contrast keep (top 20%), others -> unknown,\n"
            "3) save secondary YOLO dataset,\n"
            "4) resize 2048 + 8x8 tiling + train/valid/test split."
        )
    )
    p.add_argument("--source-root", type=Path, default=Path("Dataset-v3.v1i.yolov5pytorch"))
    p.add_argument("--images-subdir", type=Path, default=Path("train/images"))
    p.add_argument("--labels-subdir", type=Path, default=Path("train/labels"))
    p.add_argument("--secondary-root", type=Path, default=Path("data") / "dataset_stage2_refined")
    p.add_argument("--output-root", type=Path, default=Path("data") / "rfdetr_tiled_coco")
    p.add_argument("--resize-size", type=int, default=2048)
    p.add_argument("--grid-size", type=int, default=8)
    p.add_argument("--tile-size", type=int, default=256)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--test-ratio", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-box-area", type=float, default=1.0)
    p.add_argument("--split-strategy", choices=["dominant_class", "random"], default="dominant_class")
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument("--keep-empty-tiles", action="store_true")
    p.add_argument("--pockmark-top-percent", type=float, default=0.20)
    p.add_argument("--pockmark-border-px", type=int, default=2)
    p.add_argument("--color-keyword", type=str, default="colordistribution")
    p.add_argument("--gas-keyword", type=str, default="gas")
    p.add_argument("--air-keyword", type=str, default="air")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.resize_size != args.grid_size * args.tile_size:
        raise ValueError("--resize-size must equal grid_size * tile_size.")
    if not (0 <= args.val_ratio < 1 and 0 <= args.test_ratio < 1):
        raise ValueError("--val-ratio and --test-ratio must be in [0, 1).")
    if args.val_ratio + args.test_ratio >= 1:
        raise ValueError("--val-ratio + --test-ratio must be < 1.")
    if not (0.0 < args.pockmark_top_percent <= 1.0):
        raise ValueError("--pockmark-top-percent must be in (0, 1].")
    if args.pockmark_border_px < 1:
        raise ValueError("--pockmark-border-px must be >= 1.")


def parse_yolo_file(path: Path) -> List[Tuple[int, float, float, float, float]]:
    rows = []
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
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


def write_yolo_file(path: Path, rows: Sequence[Tuple[int, float, float, float, float]]) -> None:
    lines = [f"{int(c)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for c, cx, cy, w, h in rows]
    path.write_text("\n".join(lines), encoding="utf-8")


def load_names_from_data_yaml(source_root: Path) -> Dict[int, str]:
    y = source_root / "data.yaml"
    if not y.exists():
        return {}
    try:
        import yaml  # type: ignore
        payload = yaml.safe_load(y.read_text(encoding="utf-8"))
    except Exception:
        return {}
    names = payload.get("names", [])
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, list):
        return {i: str(v) for i, v in enumerate(names)}
    return {}


def normalize_token(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalnum())


def resolve_class_id(names: Dict[int, str], candidates: Sequence[str], fallback: int) -> int:
    candidate_set = {c.lower() for c in candidates}
    for cls_id, name in names.items():
        if str(name).strip().lower() in candidate_set:
            return int(cls_id)
    return fallback


def collect_samples(
    images_dir: Path,
    labels_dir: Path,
    max_images: int | None,
) -> Tuple[List[SourceSample], List[str], Dict[int, List[Tuple[int, float, float, float, float]]]]:
    image_paths = sorted(
        p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if max_images is not None:
        image_paths = image_paths[:max_images]
    samples = []
    missing = []
    rows_by_index: Dict[int, List[Tuple[int, float, float, float, float]]] = {}
    for i, img in enumerate(image_paths):
        lbl = labels_dir / f"{img.stem}.txt"
        if not lbl.exists():
            missing.append(img.name)
            continue
        samples.append(SourceSample(i, img, lbl, img.stem, img.name))
        rows_by_index[i] = parse_yolo_file(lbl)
    return samples, missing, rows_by_index


def yolo_to_xyxy_resized(
    rows: Sequence[Tuple[int, float, float, float, float]],
    resized_w: int,
    resized_h: int,
) -> List[Tuple[int, int, float, float, float, float]]:
    out = []
    for row_idx, (cls, cx, cy, bw, bh) in enumerate(rows):
        x1 = max(0.0, min(float(resized_w), (cx - bw / 2.0) * resized_w))
        y1 = max(0.0, min(float(resized_h), (cy - bh / 2.0) * resized_h))
        x2 = max(0.0, min(float(resized_w), (cx + bw / 2.0) * resized_w))
        y2 = max(0.0, min(float(resized_h), (cy + bh / 2.0) * resized_h))
        if x2 <= x1 or y2 <= y1:
            continue
        out.append((row_idx, cls, x1, y1, x2, y2))
    return out


def compute_box_contrast(gray: np.ndarray, x1: float, y1: float, x2: float, y2: float, border_px: int) -> float:
    h, w = gray.shape
    ix1, iy1 = max(0, int(math.floor(x1))), max(0, int(math.floor(y1)))
    ix2, iy2 = min(w, int(math.ceil(x2))), min(h, int(math.ceil(y2)))
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inner = gray[iy1:iy2, ix1:ix2]
    if inner.size == 0:
        return 0.0
    ox1, oy1 = max(0, ix1 - border_px), max(0, iy1 - border_px)
    ox2, oy2 = min(w, ix2 + border_px), min(h, iy2 + border_px)
    outer = gray[oy1:oy2, ox1:ox2]
    if outer.size == 0:
        return 0.0
    mask = np.ones(outer.shape, dtype=bool)
    mask[iy1 - oy1 : iy2 - oy1, ix1 - ox1 : ix2 - ox1] = False
    ring = outer[mask]
    if ring.size == 0:
        return 0.0
    return abs(float(inner.mean()) - float(ring.mean()))


def allocate_counts(n: int, val_ratio: float, test_ratio: float) -> Tuple[int, int, int]:
    n_test = int(round(n * test_ratio))
    n_valid = int(round(n * val_ratio))
    if n >= 3 and test_ratio > 0 and n_test == 0:
        n_test = 1
    if n >= 3 and val_ratio > 0 and n_valid == 0:
        n_valid = 1
    if n_test + n_valid >= n:
        n_valid = max(0, n_valid - (n_test + n_valid - n + 1))
    return n - n_valid - n_test, n_valid, n_test


def split_samples(
    samples: Sequence[SourceSample],
    rows_by_index: Dict[int, List[Tuple[int, float, float, float, float]]],
    split_strategy: str,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[Dict[int, str], Dict[str, Dict[str, int]]]:
    split_map: Dict[int, str] = {}
    stats: Dict[str, Dict[str, int]] = {}
    if split_strategy == "random":
        shuffled = list(samples)
        random.Random(seed).shuffle(shuffled)
        n_train, n_valid, _ = allocate_counts(len(shuffled), val_ratio, test_ratio)
        for s in shuffled[:n_train]:
            split_map[s.index] = "train"
        for s in shuffled[n_train : n_train + n_valid]:
            split_map[s.index] = "valid"
        for s in shuffled[n_train + n_valid :]:
            split_map[s.index] = "test"
        stats["all"] = {"total": len(shuffled), "train": n_train, "valid": n_valid, "test": len(shuffled) - n_train - n_valid}
        return split_map, stats

    by_stratum: Dict[int, List[SourceSample]] = defaultdict(list)
    for s in samples:
        hist = Counter(int(r[0]) for r in rows_by_index[s.index])
        dominant = sorted(hist.items(), key=lambda x: (-x[1], x[0]))[0][0] if hist else -1
        by_stratum[dominant].append(s)

    for stratum in sorted(by_stratum.keys()):
        group = by_stratum[stratum]
        rnd = random.Random(seed + (stratum + 101) * 97)
        rnd.shuffle(group)
        n_train, n_valid, _ = allocate_counts(len(group), val_ratio, test_ratio)
        for s in group[:n_train]:
            split_map[s.index] = "train"
        for s in group[n_train : n_train + n_valid]:
            split_map[s.index] = "valid"
        for s in group[n_train + n_valid :]:
            split_map[s.index] = "test"
        stats[str(stratum)] = {"total": len(group), "train": n_train, "valid": n_valid, "test": len(group) - n_train - n_valid}
    return split_map, stats


def intersect_with_tile(x1: float, y1: float, x2: float, y2: float, tx1: int, ty1: int, tx2: int, ty2: int):
    ix1, iy1 = max(x1, tx1), max(y1, ty1)
    ix2, iy2 = min(x2, tx2), min(y2, ty2)
    if ix2 <= ix1 or iy2 <= iy1:
        return None
    return ix1 - tx1, iy1 - ty1, ix2 - ix1, iy2 - iy1


def main() -> None:
    args = parse_args()
    validate_args(args)

    source_root = args.source_root.resolve()
    images_dir = (source_root / args.images_subdir).resolve()
    labels_dir = (source_root / args.labels_subdir).resolve()
    secondary_root = args.secondary_root.resolve()
    output_root = args.output_root.resolve()
    if not images_dir.exists() or not labels_dir.exists():
        raise FileNotFoundError("source images/labels path not found.")

    for p in [secondary_root, output_root]:
        if p.exists():
            if not args.overwrite:
                raise FileExistsError(f"Path already exists: {p}. Use --overwrite.")
            shutil.rmtree(p)

    secondary_images_dir = secondary_root / "train" / "images"
    secondary_labels_dir = secondary_root / "train" / "labels"
    secondary_images_dir.mkdir(parents=True, exist_ok=True)
    secondary_labels_dir.mkdir(parents=True, exist_ok=True)
    (secondary_root / "metadata").mkdir(parents=True, exist_ok=True)
    split_dirs = {s: output_root / s for s in SPLITS}
    for d in split_dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    (output_root / "metadata").mkdir(parents=True, exist_ok=True)

    data_yaml = source_root / "data.yaml"
    if data_yaml.exists():
        shutil.copy2(data_yaml, secondary_root / "data.yaml")

    samples, missing_labels, original_rows_by_index = collect_samples(images_dir, labels_dir, args.max_images)
    if not samples:
        raise RuntimeError("No image/label pairs found.")

    names = load_names_from_data_yaml(source_root)
    unknown_id = resolve_class_id(names, ["unknown"], 7)
    pockmark_id = resolve_class_id(names, ["pockmark"], 5)
    air_id = resolve_class_id(names, ["airbubble", "air_bubble", "air"], 0)
    gas_id = resolve_class_id(names, ["gasbubble", "gas_bubble", "gas"], 4)
    color_id = resolve_class_id(names, ["color-distribution", "color_distribution", "colordistribution"], 2)

    # Step-2: filename-rule relabeling to unknown.
    stage2_rows_by_index: Dict[int, List[Tuple[int, float, float, float, float]]] = {}
    step2_counter = Counter()
    for s in samples:
        fname = normalize_token(s.image_stem)
        keep_air = normalize_token(args.air_keyword) in fname
        keep_gas = normalize_token(args.gas_keyword) in fname
        keep_color = normalize_token(args.color_keyword) in fname
        out = []
        for cls, cx, cy, w, h in original_rows_by_index[s.index]:
            new_cls = cls
            if cls == air_id and not keep_air:
                new_cls = unknown_id
                step2_counter["airbubble_to_unknown"] += 1
            elif cls == gas_id and not keep_gas:
                new_cls = unknown_id
                step2_counter["gasbubble_to_unknown"] += 1
            elif cls == color_id and not keep_color:
                new_cls = unknown_id
                step2_counter["color_distribution_to_unknown"] += 1
            out.append((new_cls, cx, cy, w, h))
        stage2_rows_by_index[s.index] = out

    # Step-3: pockmark top 20% keep, others to unknown.
    scored = []
    for s in tqdm(samples, desc="Scoring pockmark contrast"):
        rows = stage2_rows_by_index[s.index]
        boxes = yolo_to_xyxy_resized(rows, args.resize_size, args.resize_size)
        pock_boxes = [b for b in boxes if b[1] == pockmark_id]
        if not pock_boxes:
            continue
        with Image.open(s.image_path) as img:
            im = img.convert("RGB")
        if im.size != (args.resize_size, args.resize_size):
            im = im.resize((args.resize_size, args.resize_size), Image.Resampling.BILINEAR)
        gray = np.asarray(im, dtype=np.float32).mean(axis=2)
        for row_idx, _, x1, y1, x2, y2 in pock_boxes:
            scored.append(((s.index, row_idx), compute_box_contrast(gray, x1, y1, x2, y2, args.pockmark_border_px)))

    keep_keys = set()
    pock_stats = {"total_pockmark_boxes": 0, "keep_count": 0, "to_unknown_count": 0, "contrast_threshold": 0.0}
    if scored:
        scored.sort(key=lambda x: x[1], reverse=True)
        k = max(1, int(math.ceil(len(scored) * args.pockmark_top_percent)))
        keep_keys = {key for key, _ in scored[:k]}
        pock_stats = {
            "total_pockmark_boxes": len(scored),
            "keep_count": k,
            "to_unknown_count": len(scored) - k,
            "contrast_threshold": float(scored[k - 1][1]),
        }

    final_rows_by_index: Dict[int, List[Tuple[int, float, float, float, float]]] = {}
    for s in samples:
        rows = []
        for row_idx, (cls, cx, cy, w, h) in enumerate(stage2_rows_by_index[s.index]):
            if cls == pockmark_id and (s.index, row_idx) not in keep_keys:
                rows.append((unknown_id, cx, cy, w, h))
            else:
                rows.append((cls, cx, cy, w, h))
        final_rows_by_index[s.index] = rows

    # Step-4: write secondary refined YOLO dataset.
    stage2_manifest = []
    for s in tqdm(samples, desc="Writing secondary refined dataset"):
        shutil.copy2(s.image_path, secondary_images_dir / s.image_name)
        write_yolo_file(secondary_labels_dir / f"{s.image_stem}.txt", final_rows_by_index[s.index])
        stage2_manifest.append({"image_name": s.image_name, "num_boxes": len(final_rows_by_index[s.index])})
    with (secondary_root / "metadata" / "stage2_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_name", "num_boxes"])
        writer.writeheader()
        writer.writerows(stage2_manifest)

    # Step-5: split + tiling.
    split_map, strata_stats = split_samples(samples, final_rows_by_index, args.split_strategy, args.val_ratio, args.test_ratio, args.seed)
    yolo_ids = sorted(set(names.keys())) if names else sorted({c for rows in final_rows_by_index.values() for c, *_ in rows})
    if unknown_id not in yolo_ids:
        yolo_ids.append(unknown_id)
        yolo_ids = sorted(yolo_ids)
    yolo_to_coco = {yid: yid + 1 for yid in yolo_ids}
    class_name_map = {yid: names.get(yid, f"class_{yid}") for yid in yolo_ids}
    categories = [{"id": yolo_to_coco[y], "name": class_name_map[y], "supercategory": "defect"} for y in yolo_ids]

    coco = {s: {"info": {"description": "RF-DETR tiled dataset (rebuilt preprocessing)", "date_created": datetime.now(timezone.utc).isoformat()}, "licenses": [], "images": [], "annotations": [], "categories": categories} for s in SPLITS}
    iid = {s: 1 for s in SPLITS}
    aid = {s: 1 for s in SPLITS}
    split_stats = {s: {"source_images": 0, "saved_tiles": 0, "saved_annotations": 0, "skipped_empty_tiles": 0} for s in SPLITS}
    tile_manifest = []

    for s in tqdm(samples, desc="Tiling refined dataset"):
        split = split_map[s.index]
        split_stats[split]["source_images"] += 1
        rows = final_rows_by_index[s.index]
        if not rows and not args.keep_empty_tiles:
            continue
        with Image.open(secondary_images_dir / s.image_name) as im:
            image = im.convert("RGB")
        if image.size != (args.resize_size, args.resize_size):
            image = image.resize((args.resize_size, args.resize_size), Image.Resampling.BILINEAR)
        boxes = yolo_to_xyxy_resized(rows, args.resize_size, args.resize_size)
        for r in range(args.grid_size):
            for c in range(args.grid_size):
                tx1, ty1 = c * args.tile_size, r * args.tile_size
                tx2, ty2 = tx1 + args.tile_size, ty1 + args.tile_size
                tile_boxes = []
                for _, cls, x1, y1, x2, y2 in boxes:
                    inter = intersect_with_tile(x1, y1, x2, y2, tx1, ty1, tx2, ty2)
                    if inter is None:
                        continue
                    bx, by, bw, bh = inter
                    if bw * bh < args.min_box_area:
                        continue
                    tile_boxes.append((cls, bx, by, bw, bh))
                if not tile_boxes and not args.keep_empty_tiles:
                    split_stats[split]["skipped_empty_tiles"] += 1
                    continue
                tile = image.crop((tx1, ty1, tx2, ty2))
                tile_name = f"img{s.index:05d}_r{r:02d}_c{c:02d}.jpg"
                tile.save(split_dirs[split] / tile_name, format="JPEG", quality=95)
                image_id = iid[split]
                iid[split] += 1
                coco[split]["images"].append({"id": image_id, "file_name": tile_name, "width": args.tile_size, "height": args.tile_size})
                for cls, bx, by, bw, bh in tile_boxes:
                    if cls not in yolo_to_coco:
                        continue
                    ann_id = aid[split]
                    aid[split] += 1
                    coco[split]["annotations"].append({"id": ann_id, "image_id": image_id, "category_id": yolo_to_coco[cls], "bbox": [bx, by, bw, bh], "area": bw * bh, "iscrowd": 0})
                split_stats[split]["saved_tiles"] += 1
                split_stats[split]["saved_annotations"] += len(tile_boxes)
                tile_manifest.append({"split": split, "tile_file_name": tile_name, "source_image": s.image_name, "tile_row": r, "tile_col": c, "num_boxes": len(tile_boxes)})

    for split in SPLITS:
        (split_dirs[split] / "_annotations.coco.json").write_text(json.dumps(coco[split], ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "source_root": str(source_root),
        "secondary_root": str(secondary_root),
        "output_root": str(output_root),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "class_ids": {"airbubble": air_id, "gasbubble": gas_id, "color_distribution": color_id, "pockmark": pockmark_id, "unknown": unknown_id},
        "settings": {"pockmark_top_percent": args.pockmark_top_percent, "pockmark_border_px": args.pockmark_border_px, "air_keyword": args.air_keyword, "gas_keyword": args.gas_keyword, "color_keyword": args.color_keyword, "val_ratio": args.val_ratio, "test_ratio": args.test_ratio, "split_strategy": args.split_strategy},
        "totals": {
            "source_images_with_labels": len(samples),
            "source_images_missing_labels": len(missing_labels),
            "step2_filename_rule_counts": dict(step2_counter),
            "step3_pockmark_filter": pock_stats,
            "split_source_images": {s: split_stats[s]["source_images"] for s in SPLITS},
            "split_saved_tiles": {s: split_stats[s]["saved_tiles"] for s in SPLITS},
            "split_saved_annotations": {s: split_stats[s]["saved_annotations"] for s in SPLITS},
            "split_skipped_empty_tiles": {s: split_stats[s]["skipped_empty_tiles"] for s in SPLITS},
            "strata_split_counts": strata_stats,
        },
    }
    (secondary_root / "metadata" / "preprocess_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_root / "metadata" / "preprocess_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_root / "metadata" / "class_mapping.json").write_text(json.dumps({"yolo_to_coco": {str(k): v for k, v in yolo_to_coco.items()}, "class_names": {str(k): class_name_map[k] for k in yolo_ids}}, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_root / "metadata" / "missing_source_labels.txt").write_text("\n".join(missing_labels), encoding="utf-8")
    with (output_root / "metadata" / "tile_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "tile_file_name", "source_image", "tile_row", "tile_col", "num_boxes"])
        writer.writeheader()
        writer.writerows(tile_manifest)

    print("Preprocessing complete.")
    print(f"Secondary refined YOLO dataset: {secondary_root}")
    print(f"Final tiled COCO dataset: {output_root}")
    print(f"Source images with labels: {len(samples)}")
    print(f"Source images missing labels: {len(missing_labels)}")
    print(
        "Filename-rule relabel counts: "
        f"air={step2_counter['airbubble_to_unknown']}, "
        f"gas={step2_counter['gasbubble_to_unknown']}, "
        f"color={step2_counter['color_distribution_to_unknown']}"
    )
    print(
        "Pockmark filter: "
        f"total={pock_stats['total_pockmark_boxes']}, "
        f"keep={pock_stats['keep_count']}, "
        f"to_unknown={pock_stats['to_unknown_count']}"
    )
    for split in SPLITS:
        print(
            f"[{split}] source_images={split_stats[split]['source_images']} "
            f"tiles={split_stats[split]['saved_tiles']} "
            f"annotations={split_stats[split]['saved_annotations']}"
        )


if __name__ == "__main__":
    main()
