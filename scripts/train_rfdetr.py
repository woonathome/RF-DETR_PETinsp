#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import importlib
import inspect
import json
import random
import shutil
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

MODEL_CLASS_BY_SIZE = {
    "nano": "RFDETRNano",
    "small": "RFDETRSmall",
    "medium": "RFDETRMedium",
    "large": "RFDETRLarge",
}

MODEL_CONFIG_CLASS_BY_SIZE = {
    "nano": "RFDETRNanoConfig",
    "small": "RFDETRSmallConfig",
    "medium": "RFDETRMediumConfig",
    "large": "RFDETRLargeConfig",
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
        "--num-workers",
        type=int,
        default=8,
        help="DataLoader workers. RTX3090에서는 8~12 권장.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the latest checkpoint in output_dir.",
    )
    parser.add_argument(
        "--resume-best",
        action="store_true",
        help="Resume from output_dir/checkpoint_best_total.pth if it exists.",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Resume from a specific checkpoint path (.ckpt/.pth).",
    )
    parser.add_argument(
        "--pretrain-weights",
        type=Path,
        default=None,
        help="Optional local checkpoint path to initialize from.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base seed for reproducibility and epoch-wise augmentation randomness.",
    )
    parser.add_argument(
        "--disable-augment",
        action="store_true",
        help="Disable custom Albumentations training augmentations.",
    )
    parser.add_argument(
        "--tensorboard",
        action="store_true",
        help="Enable TensorBoard logger (epoch metrics are always printed to console).",
    )
    parser.add_argument(
        "--no-progress-bar",
        action="store_true",
        help="Disable training progress bar.",
    )
    parser.add_argument(
        "--force-high-level-api",
        action="store_true",
        help="Force model.train(...) path instead of PTL custom API.",
    )
    parser.add_argument(
        "--include-classes",
        nargs="+",
        default=None,
        help=(
            "Class names to train on. "
            "Supports space-separated or comma-separated values."
        ),
    )
    parser.add_argument(
        "--exclude-classes",
        nargs="+",
        default=None,
        help=(
            "Class names to exclude from training. "
            "Supports space-separated or comma-separated values."
        ),
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


def load_coco_category_info(dataset_dir: Path) -> tuple[int, List[str]]:
    ann_path = dataset_dir / "train" / "_annotations.coco.json"
    payload = json.loads(ann_path.read_text(encoding="utf-8"))
    categories = payload.get("categories", [])
    if not categories:
        raise ValueError("No categories found in train/_annotations.coco.json")
    sorted_cats = sorted(categories, key=lambda c: int(c["id"]))
    class_names = [str(cat["name"]) for cat in sorted_cats]
    return len(sorted_cats), class_names


def parse_class_tokens(raw_values: List[str] | None) -> List[str]:
    if not raw_values:
        return []
    out: List[str] = []
    seen = set()
    for raw in raw_values:
        for token in str(raw).split(","):
            name = token.strip()
            if not name:
                continue
            key = name.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(name)
    return out


def resolve_selected_class_names(
    all_class_names: List[str],
    include_tokens: List[str],
    exclude_tokens: List[str],
) -> List[str]:
    lowered_to_name = {name.lower(): name for name in all_class_names}

    missing_include = [n for n in include_tokens if n.lower() not in lowered_to_name]
    if missing_include:
        raise ValueError(
            "Unknown class name(s) in --include-classes: " + ", ".join(missing_include)
        )
    missing_exclude = [n for n in exclude_tokens if n.lower() not in lowered_to_name]
    if missing_exclude:
        print(
            "Warning: class name(s) in --exclude-classes were not found and will be ignored: "
            + ", ".join(missing_exclude)
        )

    if include_tokens:
        selected = [lowered_to_name[n.lower()] for n in include_tokens]
    else:
        selected = list(all_class_names)

    exclude_set = {n.lower() for n in exclude_tokens}
    selected = [n for n in selected if n.lower() not in exclude_set]
    if not selected:
        raise ValueError("No classes left after include/exclude filtering.")
    return selected


def build_filtered_dataset(
    source_dataset_dir: Path,
    filtered_dataset_dir: Path,
    selected_class_names: List[str],
) -> Path:
    if filtered_dataset_dir.exists():
        shutil.rmtree(filtered_dataset_dir)
    filtered_dataset_dir.mkdir(parents=True, exist_ok=True)

    selected_set = set(selected_class_names)
    global_name_to_new_id = {name: idx + 1 for idx, name in enumerate(selected_class_names)}
    global_categories = [
        {"id": idx + 1, "name": name, "supercategory": "defect"}
        for idx, name in enumerate(selected_class_names)
    ]

    for split in ("train", "valid", "test"):
        src_ann = source_dataset_dir / split / "_annotations.coco.json"
        if not src_ann.exists():
            continue
        payload = json.loads(src_ann.read_text(encoding="utf-8"))

        old_id_to_name = {
            int(cat["id"]): str(cat["name"])
            for cat in payload.get("categories", [])
        }
        old_id_to_new_id = {
            old_id: global_name_to_new_id[name]
            for old_id, name in old_id_to_name.items()
            if name in selected_set
        }

        dst_split_dir = filtered_dataset_dir / split
        dst_split_dir.mkdir(parents=True, exist_ok=True)
        src_split_dir = (source_dataset_dir / split).resolve()

        filtered_images = []
        for img in payload.get("images", []):
            new_img = copy.deepcopy(img)
            file_name = str(new_img.get("file_name", ""))
            if not file_name:
                continue
            abs_image_path = (src_split_dir / file_name).resolve()
            new_img["file_name"] = str(abs_image_path)
            filtered_images.append(new_img)

        filtered_annotations = []
        for ann in payload.get("annotations", []):
            old_cat_id = int(ann.get("category_id", -1))
            if old_cat_id not in old_id_to_new_id:
                continue
            new_ann = copy.deepcopy(ann)
            new_ann["category_id"] = old_id_to_new_id[old_cat_id]
            filtered_annotations.append(new_ann)

        filtered_payload = copy.deepcopy(payload)
        filtered_payload["images"] = filtered_images
        filtered_payload["annotations"] = filtered_annotations
        filtered_payload["categories"] = global_categories

        dst_ann = dst_split_dir / "_annotations.coco.json"
        dst_ann.write_text(
            json.dumps(filtered_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return filtered_dataset_dir


def prepare_dataset_for_class_selection(
    args: argparse.Namespace,
    dataset_dir: Path,
    output_dir: Path,
) -> tuple[Path, List[str]]:
    class_count, all_class_names = load_coco_category_info(dataset_dir)
    _ = class_count

    include_tokens = parse_class_tokens(args.include_classes)
    exclude_tokens = parse_class_tokens(args.exclude_classes)
    selected_class_names = resolve_selected_class_names(
        all_class_names=all_class_names,
        include_tokens=include_tokens,
        exclude_tokens=exclude_tokens,
    )

    if selected_class_names == all_class_names:
        return dataset_dir, selected_class_names

    filtered_dataset_dir = output_dir / "_filtered_dataset"
    build_filtered_dataset(
        source_dataset_dir=dataset_dir,
        filtered_dataset_dir=filtered_dataset_dir,
        selected_class_names=selected_class_names,
    )

    selection_meta = {
        "source_dataset_dir": str(dataset_dir),
        "filtered_dataset_dir": str(filtered_dataset_dir),
        "all_class_names": all_class_names,
        "selected_class_names": selected_class_names,
        "include_classes": include_tokens,
        "exclude_classes": exclude_tokens,
    }
    (output_dir / "class_selection.json").write_text(
        json.dumps(selection_meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Class-filtered dataset created: {filtered_dataset_dir}")
    return filtered_dataset_dir, selected_class_names


def build_requested_aug_config() -> Dict[str, Any]:
    """
    Requested policy:
    - HorizontalFlip p=0.2
    - RGBShift p=0.2 OR HSV(HueSaturationValue) p=0.2 OR ChannelShuffle p=0.2

    Implementation detail:
    - RF-DETR docs state OneOf container always fires, and child `p` is used as
      selection weight.
    - We add NoOp with weight 0.4, so each color transform is selected with
      exact relative probability 0.2, and 0.4 means no color transform.
    """
    return {
        "HorizontalFlip": {"p": 0.2},
        "OneOf": {
            "transforms": [
                {
                    "RGBShift": {
                        "r_shift_limit": 20,
                        "g_shift_limit": 20,
                        "b_shift_limit": 20,
                        "p": 0.2,
                    }
                },
                {
                    "HueSaturationValue": {
                        "hue_shift_limit": 10,
                        "sat_shift_limit": 20,
                        "val_shift_limit": 20,
                        "p": 0.2,
                    }
                },
                {"ChannelShuffle": {"p": 0.2}},
                {"NoOp": {"p": 0.4}},
            ],
        },
    }


def resolve_resume_path(
    output_dir: Path,
    do_resume: bool,
    resume_from: Path | None = None,
    resume_best: bool = False,
) -> str | None:
    if resume_from is not None:
        resolved = resume_from.resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"--resume-from file not found: {resolved}")
        print(f"Resume checkpoint selected (--resume-from): {resolved}")
        return str(resolved)

    if resume_best:
        best_total = output_dir / "checkpoint_best_total.pth"
        if best_total.exists():
            chosen = best_total.resolve()
            print(f"Resume checkpoint selected (--resume-best): {chosen}")
            return str(chosen)

        best_candidates = sorted(
            [
                p
                for p in output_dir.glob("checkpoint_best*.pth")
                if p.is_file()
            ],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if best_candidates:
            chosen = best_candidates[0].resolve()
            print(f"Resume checkpoint selected (--resume-best fallback): {chosen}")
            return str(chosen)

        print(
            f"Warning: --resume-best was set but no best checkpoint was found in {output_dir}. "
            "Falling back to --resume behavior if enabled."
        )

    if not do_resume and not resume_best:
        return None

    for preferred in ("checkpoint_last.ckpt", "checkpoint_last.pth"):
        p = output_dir / preferred
        if p.exists():
            chosen = p.resolve()
            print(f"Resume checkpoint selected (preferred last): {chosen}")
            return str(chosen)

    candidates = sorted(
        [
            p
            for p in output_dir.glob("*")
            if p.is_file() and p.suffix.lower() in {".ckpt", ".pth"} and p.name.startswith("checkpoint")
        ],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        chosen = candidates[0].resolve()
        print(f"Resume checkpoint selected (--resume): {chosen}")
        return str(chosen)

    print(
        f"Warning: resume was requested but no checkpoint file was found in {output_dir}. "
        "Starting from scratch."
    )
    return None


def run_high_level_train(
    rfdetr: Any,
    args: argparse.Namespace,
    dataset_dir: Path,
    output_dir: Path,
    aug_config: Dict[str, Any] | None,
    class_count: int,
) -> None:
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
    if "num_classes" in init_sig.parameters or init_has_var_kwargs:
        init_kwargs["num_classes"] = class_count
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
        "eval_interval": 1,
        "log_per_class_metrics": True,
        "progress_bar": None if args.no_progress_bar else "tqdm",
        "num_workers": args.num_workers,
        "seed": args.seed,
        "tensorboard": args.tensorboard,
        "aug_config": {} if args.disable_augment else aug_config,
    }
    resume_path = resolve_resume_path(
        output_dir=output_dir,
        do_resume=args.resume,
        resume_from=args.resume_from,
        resume_best=args.resume_best,
    )
    if resume_path:
        requested_train_kwargs["resume"] = resume_path

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

    print("Starting RF-DETR training (high-level API) with arguments:")
    for k, v in train_kwargs.items():
        print(f"  {k}: {v}")
    print(
        "Note: per-epoch metrics are produced by RF-DETR COCOEvalCallback "
        "(val/mAP_50_95, val/F1, etc.) when eval_interval=1."
    )
    try:
        model.train(**train_kwargs)
    except ModuleNotFoundError as exc:
        if exc.name == "pytorch_lightning":
            raise ModuleNotFoundError(
                "Missing dependency: pytorch_lightning\n"
                "Please install and retry:\n"
                "  python -m pip install pytorch-lightning\n"
                "or reinstall all deps:\n"
                "  python -m pip install -r requirements.txt"
            ) from exc
        raise


def main() -> None:
    args = parse_args()
    source_dataset_dir = args.dataset_dir.resolve()
    output_dir = args.output_dir.resolve()
    ensure_dataset_layout(source_dataset_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_dir, class_names = prepare_dataset_for_class_selection(
        args=args,
        dataset_dir=source_dataset_dir,
        output_dir=output_dir,
    )
    class_count = len(class_names)
    aug_config = build_requested_aug_config()
    if args.disable_augment:
        print("Augmentation is disabled by --disable-augment")
    else:
        aug_path = output_dir / "augmentation_config.json"
        aug_path.write_text(
            json.dumps(aug_config, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Saved augmentation config: {aug_path}")

    print("Dataset summary")
    print(summarize_split(dataset_dir, "train"))
    print(summarize_split(dataset_dir, "valid"))
    print(summarize_split(dataset_dir, "test"))
    print(f"Detected class count: {class_count}")
    print(f"Selected classes: {class_names}")

    try:
        import rfdetr
    except ImportError as exc:
        raise ImportError(
            "rfdetr package is not installed. Install dependencies first:\n"
            "  pip install -r requirements.txt"
        ) from exc

    # rfdetr training path requires pytorch_lightning internally.
    try:
        import pytorch_lightning  # noqa: F401
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency: pytorch_lightning\n"
            "Please install and retry:\n"
            "  python -m pip install pytorch-lightning\n"
            "or reinstall all deps:\n"
            "  python -m pip install -r requirements.txt"
        ) from exc

    if args.force_high_level_api:
        run_high_level_train(
            rfdetr=rfdetr,
            args=args,
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            aug_config=aug_config,
            class_count=class_count,
        )
        return

    # Prefer PTL custom API because it allows explicit epoch metric printing callbacks.
    try:
        from rfdetr.config import TrainConfig
        from rfdetr.training import RFDETRDataModule, RFDETRModelModule, build_trainer
    except Exception as exc:
        print(
            "Warning: custom training API import failed. Falling back to high-level model.train()."
        )
        print(f"Reason: {exc}")
        run_high_level_train(
            rfdetr=rfdetr,
            args=args,
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            aug_config=aug_config,
            class_count=class_count,
        )
        return

    model_config_name = MODEL_CONFIG_CLASS_BY_SIZE[args.model_size]
    config_module = importlib.import_module("rfdetr.config")
    model_config_cls = getattr(config_module, model_config_name, None)
    if model_config_cls is None:
        print(
            f"Warning: {model_config_name} not found in rfdetr.config. "
            "Falling back to high-level model.train()."
        )
        run_high_level_train(
            rfdetr=rfdetr,
            args=args,
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            aug_config=aug_config,
            class_count=class_count,
        )
        return

    model_config_kwargs = {"num_classes": class_count}
    if args.pretrain_weights is not None:
        model_config_kwargs["pretrain_weights"] = str(args.pretrain_weights.resolve())

    model_config_sig = inspect.signature(model_config_cls.__init__)
    model_config_has_var_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD
        for p in model_config_sig.parameters.values()
    )
    if not model_config_has_var_kwargs:
        model_config_kwargs = {
            k: v
            for k, v in model_config_kwargs.items()
            if k in model_config_sig.parameters
        }
    model_config = model_config_cls(**model_config_kwargs)

    resume_for_config = resolve_resume_path(
        output_dir=output_dir,
        do_resume=args.resume,
        resume_from=args.resume_from,
        resume_best=args.resume_best,
    )

    train_config_kwargs = {
        "dataset_dir": str(dataset_dir),
        "output_dir": str(output_dir),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "lr": args.lr,
        "num_workers": args.num_workers,
        "eval_interval": 1,
        "log_per_class_metrics": True,
        "tensorboard": args.tensorboard,
        "progress_bar": None if args.no_progress_bar else "tqdm",
        "seed": args.seed,
        "class_names": class_names,
        "aug_config": {} if args.disable_augment else aug_config,
        "resume": resume_for_config,
    }
    train_config_sig = inspect.signature(TrainConfig.__init__)
    train_config_has_var_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD
        for p in train_config_sig.parameters.values()
    )
    if not train_config_has_var_kwargs:
        train_config_kwargs = {
            k: v
            for k, v in train_config_kwargs.items()
            if k in train_config_sig.parameters
        }
    train_config = TrainConfig(**train_config_kwargs)

    # Callback imports: support both old/new lightning package names.
    try:
        from pytorch_lightning import Callback
    except Exception:
        from lightning.pytorch.callbacks import Callback  # type: ignore

    class LastCheckpointSaver(Callback):
        """
        Save a rolling 'last' checkpoint every validation epoch.
        """

        def __init__(self, output_dir: Path) -> None:
            super().__init__()
            self.output_dir = Path(output_dir)
            self.ckpt_path = self.output_dir / "checkpoint_last.ckpt"

        def _save(self, trainer) -> None:
            if getattr(trainer, "is_global_zero", True) is False:
                return
            self.output_dir.mkdir(parents=True, exist_ok=True)
            trainer.save_checkpoint(str(self.ckpt_path), weights_only=False)
            print(f"[Checkpoint] saved last: {self.ckpt_path}")

        def on_validation_epoch_end(self, trainer, pl_module) -> None:
            if getattr(trainer, "sanity_checking", False):
                return
            self._save(trainer)

        def on_exception(self, trainer, pl_module, exception) -> None:
            self._save(trainer)

    class EpochMetricsPrinter(Callback):
        def __init__(self) -> None:
            super().__init__()
            self._preferred = [
                "val/mAP_50_95",
                "val/mAP_50",
                "val/F1",
                "val/precision",
                "val/recall",
                "val/loss",
                "train/loss",
            ]

        @staticmethod
        def _to_float(v: Any) -> float | None:
            if v is None:
                return None
            if isinstance(v, torch.Tensor):
                if v.numel() != 1:
                    return None
                return float(v.detach().cpu().item())
            if isinstance(v, (float, int)):
                return float(v)
            return None

        def on_validation_epoch_end(self, trainer, pl_module) -> None:
            if getattr(trainer, "sanity_checking", False):
                return
            metrics = trainer.callback_metrics
            epoch = int(trainer.current_epoch) + 1
            chunks = [f"epoch={epoch:03d}"]
            for key in self._preferred:
                if key in metrics:
                    fv = self._to_float(metrics.get(key))
                    if fv is not None:
                        chunks.append(f"{key}={fv:.4f}")
            if len(chunks) > 1:
                print("[EpochMetrics] " + " | ".join(chunks))
            else:
                print(f"[EpochMetrics] epoch={epoch:03d} | no validation metrics found")

    class EpochAugmentationSeedCallback(Callback):
        """
        Ensures a different RNG seed per epoch so Albumentations random outcomes
        change across epochs.
        """

        def __init__(self, base_seed: int) -> None:
            super().__init__()
            self.base_seed = int(base_seed)

        def on_train_epoch_start(self, trainer, pl_module) -> None:
            epoch = int(trainer.current_epoch)
            epoch_seed = self.base_seed + epoch
            random.seed(epoch_seed)
            np.random.seed(epoch_seed % (2**32 - 1))
            torch.manual_seed(epoch_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(epoch_seed)
            print(f"[AugmentSeed] epoch={epoch + 1:03d} seed={epoch_seed}")

    module = RFDETRModelModule(model_config, train_config)
    datamodule = RFDETRDataModule(model_config, train_config)
    trainer = build_trainer(train_config, model_config)
    trainer.callbacks.extend(
        [
            LastCheckpointSaver(output_dir=output_dir),
            EpochAugmentationSeedCallback(base_seed=args.seed),
            EpochMetricsPrinter(),
        ]
    )

    print("Starting RF-DETR training (custom PTL API) with arguments:")
    print(f"  source_dataset_dir: {source_dataset_dir}")
    print(f"  dataset_dir: {dataset_dir}")
    print(f"  output_dir: {output_dir}")
    print(f"  model_size: {args.model_size}")
    print(f"  epochs: {args.epochs}")
    print(f"  batch_size: {args.batch_size}")
    print(f"  grad_accum_steps: {args.grad_accum_steps}")
    print(f"  lr: {args.lr}")
    print(f"  num_workers: {args.num_workers}")
    print(f"  eval_interval: 1")
    print(f"  log_per_class_metrics: True")
    print(f"  tensorboard: {args.tensorboard}")
    print(f"  progress_bar: {None if args.no_progress_bar else 'tqdm'}")
    print(f"  seed(base): {args.seed}")
    print(f"  augmentation: {'disabled' if args.disable_augment else 'enabled'}")
    print(f"  resume: {args.resume}")
    print(f"  resume_best: {args.resume_best}")

    trainer.fit(module, datamodule, ckpt_path=getattr(train_config, "resume", None) or None)


if __name__ == "__main__":
    main()
