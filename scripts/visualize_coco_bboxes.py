#!/usr/bin/env python
from __future__ import annotations

import argparse
import inspect
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from PIL import Image, ImageDraw, ImageFont


MODEL_CLASS_BY_SIZE = {
    "nano": "RFDETRNano",
    "small": "RFDETRSmall",
    "medium": "RFDETRMedium",
    "large": "RFDETRLarge",
}

PALETTE = [
    (230, 57, 70),
    (29, 53, 87),
    (69, 123, 157),
    (42, 157, 143),
    (233, 196, 106),
    (244, 162, 97),
    (231, 111, 81),
    (102, 45, 145),
    (0, 128, 255),
    (255, 99, 71),
    (46, 204, 113),
    (241, 196, 15),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Draw COCO GT bboxes and/or RF-DETR prediction bboxes for a split."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data") / "rfdetr_tiled_coco",
        help="COCO dataset root that contains train/valid/test folders.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "valid", "test"],
        default="test",
        help="Which split to visualize.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs") / "vis" / "test_vis",
        help="Root directory to save visualizations.",
    )
    parser.add_argument(
        "--mode",
        choices=["gt", "pred", "both"],
        default="gt",
        help="Visualization mode.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=200,
        help="Maximum number of images to visualize (use <=0 for all).",
    )
    parser.add_argument("--line-width", type=int, default=2)
    parser.add_argument(
        "--skip-empty",
        action="store_true",
        help="Skip images that have no GT annotations (applies when GT is drawn).",
    )
    parser.add_argument(
        "--model-size",
        choices=sorted(MODEL_CLASS_BY_SIZE.keys()),
        default="medium",
        help="RF-DETR model size for prediction mode.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint path (.pth/.ckpt). If omitted, uses --run-dir/checkpoint_best_total.pth.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("runs") / "rfdetr-medium-7cls",
        help="Training run directory that contains checkpoint_best_total.pth.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Prediction confidence threshold.",
    )
    return parser.parse_args()


def resolve_image_path(split_dir: Path, file_name: str) -> Path:
    p = Path(file_name)
    if p.is_absolute():
        return p
    return split_dir / p


def color_for_category(category_id: int) -> Tuple[int, int, int]:
    return PALETTE[(int(category_id) - 1) % len(PALETTE)]


def draw_box_with_label(
    draw: ImageDraw.ImageDraw,
    box: Tuple[float, float, float, float],
    label: str,
    color: Tuple[int, int, int],
    line_width: int,
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont | None,
) -> None:
    x1, y1, x2, y2 = box
    draw.rectangle((x1, y1, x2, y2), outline=color, width=max(1, line_width))
    text_pos = (x1 + 2, max(0.0, y1 - 12))
    draw.text(text_pos, label, fill=color, font=font)


def resolve_model_checkpoint(args: argparse.Namespace) -> Path:
    if args.checkpoint is not None:
        ckpt = args.checkpoint.resolve()
    else:
        ckpt = (args.run_dir.resolve() / "checkpoint_best_total.pth")
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    return ckpt


def load_predictor(args: argparse.Namespace):
    try:
        import rfdetr
    except ImportError as exc:
        raise ImportError(
            "rfdetr package is not installed. Install dependencies first:\n"
            "  pip install -r requirements.txt"
        ) from exc

    class_name = MODEL_CLASS_BY_SIZE[args.model_size]
    model_cls = getattr(rfdetr, class_name, None)
    if model_cls is None:
        raise AttributeError(
            f"Cannot find {class_name} in rfdetr package. "
            "Please update rfdetr to a newer version."
        )

    ckpt = resolve_model_checkpoint(args)
    init_sig = inspect.signature(model_cls.__init__)
    init_has_var_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in init_sig.parameters.values()
    )
    init_kwargs = {}
    resolved_ckpt = str(ckpt)
    if "pretrain_weights" in init_sig.parameters or init_has_var_kwargs:
        init_kwargs["pretrain_weights"] = resolved_ckpt
    elif "weights" in init_sig.parameters:
        init_kwargs["weights"] = resolved_ckpt
    elif "checkpoint_path" in init_sig.parameters:
        init_kwargs["checkpoint_path"] = resolved_ckpt
    else:
        raise TypeError(
            "This rfdetr version does not expose a known checkpoint init argument. "
            "Please update the script for your rfdetr version."
        )

    print(f"Loading prediction model from: {ckpt}")
    model = model_cls(**init_kwargs)
    return model


def extract_predictions(detections: Any) -> List[Tuple[int, float, float, float, float, float]]:
    rows: List[Tuple[int, float, float, float, float, float]] = []
    if detections is None:
        return rows

    xyxy = getattr(detections, "xyxy", None)
    class_id = getattr(detections, "class_id", None)
    confidence = getattr(detections, "confidence", None)
    if xyxy is not None:
        total = len(xyxy)
        for i in range(total):
            box = xyxy[i]
            x1, y1, x2, y2 = [float(v) for v in box]
            cid = int(class_id[i]) if class_id is not None else -1
            conf = float(confidence[i]) if confidence is not None else -1.0
            rows.append((cid, conf, x1, y1, x2, y2))
        return rows

    if isinstance(detections, list):
        for d in detections:
            if not isinstance(d, dict):
                continue
            box = d.get("xyxy", d.get("bbox"))
            if box is None or len(box) < 4:
                continue
            x1, y1, x2, y2 = [float(v) for v in box[:4]]
            cid = int(d.get("class_id", d.get("class", -1)))
            conf = float(d.get("confidence", d.get("score", -1.0)))
            rows.append((cid, conf, x1, y1, x2, y2))
    return rows


def map_pred_class_name(pred_class_id: int, id_to_name: Dict[int, str]) -> str:
    # Handle both 0-based and 1-based prediction class ids.
    if pred_class_id in id_to_name:
        return id_to_name[pred_class_id]
    if (pred_class_id + 1) in id_to_name:
        return id_to_name[pred_class_id + 1]
    return f"class_{pred_class_id}"


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    split_dir = dataset_dir / args.split
    ann_path = split_dir / "_annotations.coco.json"
    output_dir = args.output_dir.resolve()

    if not ann_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {ann_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    mode_gt = args.mode in {"gt", "both"}
    mode_pred = args.mode in {"pred", "both"}
    gt_dir = output_dir / "gt" if mode_gt else None
    pred_dir = output_dir / "pred" if mode_pred else None
    if gt_dir is not None:
        gt_dir.mkdir(parents=True, exist_ok=True)
    if pred_dir is not None:
        pred_dir.mkdir(parents=True, exist_ok=True)

    payload = json.loads(ann_path.read_text(encoding="utf-8"))
    images = payload.get("images", [])
    annotations = payload.get("annotations", [])
    categories = payload.get("categories", [])

    id_to_name: Dict[int, str] = {int(c["id"]): str(c["name"]) for c in categories}
    ann_by_image: Dict[int, List[Dict]] = defaultdict(list)
    for ann in annotations:
        ann_by_image[int(ann["image_id"])].append(ann)

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    predictor = load_predictor(args) if mode_pred else None

    max_images = args.max_images
    limit = len(images) if max_images <= 0 else min(len(images), max_images)

    saved_gt = 0
    saved_pred = 0
    skipped = 0
    for img in images[:limit]:
        image_id = int(img["id"])
        file_name = str(img["file_name"])
        image_path = resolve_image_path(split_dir, file_name)
        anns = ann_by_image.get(image_id, [])
        if args.skip_empty and not anns:
            skipped += 1
            continue
        if not image_path.exists():
            print(f"[WARN] missing image: {image_path}")
            skipped += 1
            continue

        with Image.open(image_path) as im:
            base_image = im.convert("RGB")

        stem = Path(file_name).stem
        if mode_gt:
            canvas_gt = base_image.copy()
            draw_gt = ImageDraw.Draw(canvas_gt)
            for ann in anns:
                cat_id = int(ann["category_id"])
                cat_name = id_to_name.get(cat_id, f"class_{cat_id}")
                x, y, w, h = [float(v) for v in ann["bbox"]]
                x1 = max(0.0, x)
                y1 = max(0.0, y)
                x2 = max(x1 + 1.0, x + w)
                y2 = max(y1 + 1.0, y + h)
                draw_box_with_label(
                    draw=draw_gt,
                    box=(x1, y1, x2, y2),
                    label=cat_name,
                    color=color_for_category(cat_id),
                    line_width=args.line_width,
                    font=font,
                )
            out_gt = gt_dir / f"{stem}_gt.jpg"  # type: ignore[arg-type]
            canvas_gt.save(out_gt, quality=95)
            saved_gt += 1

        if mode_pred and predictor is not None:
            canvas_pred = base_image.copy()
            draw_pred = ImageDraw.Draw(canvas_pred)
            detections = predictor.predict(base_image, threshold=args.threshold)
            for cid, conf, x1, y1, x2, y2 in extract_predictions(detections):
                name = map_pred_class_name(cid, id_to_name)
                label = f"{name} {conf:.2f}" if conf >= 0 else name
                draw_box_with_label(
                    draw=draw_pred,
                    box=(x1, y1, x2, y2),
                    label=label,
                    color=color_for_category(cid + 1),
                    line_width=args.line_width,
                    font=font,
                )
            out_pred = pred_dir / f"{stem}_pred.jpg"  # type: ignore[arg-type]
            canvas_pred.save(out_pred, quality=95)
            saved_pred += 1

    print(
        f"Done. split={args.split} mode={args.mode} "
        f"saved_gt={saved_gt} saved_pred={saved_pred} skipped={skipped} output_dir={output_dir}"
    )


if __name__ == "__main__":
    main()
