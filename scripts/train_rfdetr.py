#!/usr/bin/env python
from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path


MODEL_CLASS_BY_SIZE = {
    "nano": "RFDETRNano",
    "small": "RFDETRSmall",
    "medium": "RFDETRMedium",
    "large": "RFDETRLarge",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune RF-DETR on custom COCO dataset.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data") / "rfdetr_tiled_coco",
        help="COCO dataset root with train/valid/test split folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs") / "rfdetr-medium",
        help="Directory to store checkpoints and logs.",
    )
    parser.add_argument(
        "--model-size",
        choices=sorted(MODEL_CLASS_BY_SIZE.keys()),
        default="medium",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum-steps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the latest checkpoint in output dir.",
    )
    parser.add_argument(
        "--pretrain-weights",
        type=Path,
        default=None,
        help="Optional local checkpoint path to initialize from.",
    )
    return parser.parse_args()


def ensure_dataset_layout(dataset_dir: Path) -> None:
    expected = [
        dataset_dir / "train" / "_annotations.coco.json",
        dataset_dir / "valid" / "_annotations.coco.json",
    ]
    missing = [p for p in expected if not p.exists()]
    if missing:
        joined = "\n".join(str(p) for p in missing)
        raise FileNotFoundError(
            "Dataset layout is incomplete. Missing files:\n"
            f"{joined}\n"
            "Run scripts/prepare_tiled_coco_dataset.py first."
        )


def summarize_split(dataset_dir: Path, split: str) -> str:
    ann_path = dataset_dir / split / "_annotations.coco.json"
    if not ann_path.exists():
        return f"{split}: not found"
    payload = json.loads(ann_path.read_text(encoding="utf-8"))
    num_images = len(payload.get("images", []))
    num_annotations = len(payload.get("annotations", []))
    num_categories = len(payload.get("categories", []))
    return (
        f"{split}: images={num_images}, annotations={num_annotations}, "
        f"categories={num_categories}"
    )


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    output_dir = args.output_dir.resolve()
    ensure_dataset_layout(dataset_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Dataset summary")
    print(summarize_split(dataset_dir, "train"))
    print(summarize_split(dataset_dir, "valid"))
    print(summarize_split(dataset_dir, "test"))

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
    if args.pretrain_weights is not None:
        resolved_ckpt = str(args.pretrain_weights.resolve())
        if "pretrain_weights" in init_sig.parameters or init_has_var_kwargs:
            init_kwargs["pretrain_weights"] = resolved_ckpt
        elif "weights" in init_sig.parameters:
            init_kwargs["weights"] = resolved_ckpt
        elif "checkpoint_path" in init_sig.parameters:
            init_kwargs["checkpoint_path"] = resolved_ckpt
        else:
            raise TypeError(
                "This rfdetr version does not expose a known checkpoint init argument. "
                "Please initialize with pretrained defaults or update script for your version."
            )
    model = model_cls(**init_kwargs)

    requested_train_kwargs = {
        "dataset_dir": str(dataset_dir),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "lr": args.lr,
        "output_dir": str(output_dir),
    }
    if args.resume:
        requested_train_kwargs["resume"] = True

    train_sig = inspect.signature(model.train)
    train_has_var_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in train_sig.parameters.values()
    )
    if train_has_var_kwargs:
        train_kwargs = dict(requested_train_kwargs)
    else:
        supported_names = set(train_sig.parameters.keys())
        train_kwargs = {
            key: value
            for key, value in requested_train_kwargs.items()
            if key in supported_names
        }

    dropped = sorted(set(requested_train_kwargs.keys()) - set(train_kwargs.keys()))
    if dropped:
        print(
            "Warning: current rfdetr version does not support these train args and they were skipped: "
            + ", ".join(dropped)
        )

    print("Starting RF-DETR training with arguments:")
    for k, v in train_kwargs.items():
        print(f"  {k}: {v}")

    model.train(**train_kwargs)


if __name__ == "__main__":
    main()
