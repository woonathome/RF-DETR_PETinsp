#!/usr/bin/env python
from __future__ import annotations

import argparse
import inspect
from pathlib import Path

from PIL import Image


MODEL_CLASS_BY_SIZE = {
    "nano": "RFDETRNano",
    "small": "RFDETRSmall",
    "medium": "RFDETRMedium",
    "large": "RFDETRLarge",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RF-DETR inference on one tile image.")
    parser.add_argument("--image-path", type=Path, required=True)
    parser.add_argument("--model-size", choices=sorted(MODEL_CLASS_BY_SIZE.keys()), default="medium")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional custom weights. If omitted, pretrained COCO weights are used.",
    )
    parser.add_argument("--threshold", type=float, default=0.3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path = args.image_path.resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

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

    init_sig = inspect.signature(model_cls.__init__)
    init_has_var_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in init_sig.parameters.values()
    )
    init_kwargs = {}
    if args.checkpoint is not None:
        resolved_ckpt = str(args.checkpoint.resolve())
        if "pretrain_weights" in init_sig.parameters or init_has_var_kwargs:
            init_kwargs["pretrain_weights"] = resolved_ckpt
        elif "weights" in init_sig.parameters:
            init_kwargs["weights"] = resolved_ckpt
        elif "checkpoint_path" in init_sig.parameters:
            init_kwargs["checkpoint_path"] = resolved_ckpt
        else:
            raise TypeError(
                "This rfdetr version does not expose a known checkpoint init argument. "
                "Please remove --checkpoint or update script for your version."
            )
    model = model_cls(**init_kwargs)

    image = Image.open(image_path).convert("RGB")
    detections = model.predict(image, threshold=args.threshold)

    print(f"Detections: {len(detections)}")
    if hasattr(detections, "xyxy"):
        for idx, box in enumerate(detections.xyxy):
            cls = int(detections.class_id[idx]) if hasattr(detections, "class_id") else -1
            conf = float(detections.confidence[idx]) if hasattr(detections, "confidence") else -1.0
            x1, y1, x2, y2 = [float(v) for v in box]
            print(
                f"{idx:03d} class={cls} conf={conf:.4f} "
                f"xyxy=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f})"
            )


if __name__ == "__main__":
    main()
