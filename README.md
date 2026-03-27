# RF-DETR Pipeline for PET inspection Pipeline

This project builds a training dataset from:
- `Dataset-v3.v1i.yolov5pytorch/train/images`
- `Dataset-v3.v1i.yolov5pytorch/train/labels`

Workflow:
1. Resize full images from `2046x2046` to `2048x2048`
2. Filename rule relabel to `unknown`:
   - `color-distribution`, `gasbubble`, `airbubble` labels become `unknown` when filename does not contain `colordistribution`, `gas`, `air`
3. For `pockmark`, compute contrast = `|mean(inner bbox) - mean(outer 2px ring)|`
4. Keep top 20% pockmark boxes as `pockmark`, relabel remaining as `unknown`
5. Save refined secondary YOLO dataset
6. Tile to `8x8` (`256x256`)
7. Create COCO splits (`train/valid/test`)
8. Train RF-DETR

## 1) Install

```bash
python -m pip install -r requirements.txt
```

## 2) Preprocess and Split (same ratio policy)

```bash
python scripts/prepare_tiled_coco_dataset.py ^
  --source-root ./Dataset-v3.v1i.yolov5pytorch ^
  --images-subdir train/images ^
  --labels-subdir train/labels ^
  --secondary-root ./data/dataset_stage2_refined ^
  --output-root ./data/rfdetr_tiled_coco ^
  --val-ratio 0.15 ^
  --test-ratio 0.10 ^
  --split-strategy dominant_class ^
  --pockmark-top-percent 0.20 ^
  --pockmark-border-px 2 ^
  --seed 42 ^
  --overwrite
```

Notes:
- `split-strategy=dominant_class` keeps split ratios per dominant-class stratum.
- This is used to keep train/valid/test ratio behavior consistent across subgroups.
- Secondary refined YOLO dataset is saved at `--secondary-root` (step-4 artifact).

## 3) Train

```bash
python scripts/train_rfdetr.py ^
  --dataset-dir ./data/rfdetr_tiled_coco ^
  --output-dir ./runs/rfdetr-medium ^
  --model-size medium ^
  --epochs 100 ^
  --batch-size 8 ^
  --grad-accum-steps 2 ^
  --num-workers 8 ^
  --lr 1e-4 ^
  --tensorboard
```

Train only 7 classes (exclude `unknown`):

```bash
python scripts/train_rfdetr.py ^
  --dataset-dir ./data/rfdetr_tiled_coco ^
  --output-dir ./runs/rfdetr-medium-7cls ^
  --model-size medium ^
  --epochs 100 ^
  --batch-size 8 ^
  --grad-accum-steps 2 ^
  --num-workers 8 ^
  --lr 1e-4 ^
  --exclude-classes unknown ^
  --tensorboard
```

You can also choose explicit class names:

```bash
python scripts/train_rfdetr.py ... --include-classes airbubble blackspot color-distribution dust gasbubble pockmark scratch
```

## 4) Resume

`last` checkpoint is now saved every epoch as:
- `checkpoint_last.ckpt`

Auto resume (prefers `checkpoint_last.ckpt`):

```bash
python scripts/train_rfdetr.py ^
  --dataset-dir ./data/rfdetr_tiled_coco ^
  --output-dir ./runs/rfdetr-medium-7cls ^
  --model-size medium ^
  --epochs 100 ^
  --batch-size 8 ^
  --grad-accum-steps 2 ^
  --num-workers 8 ^
  --lr 1e-4 ^
  --exclude-classes unknown ^
  --resume ^
  --tensorboard
```

Resume from best total checkpoint:

```bash
python scripts/train_rfdetr.py ^
  --dataset-dir ./data/rfdetr_tiled_coco ^
  --output-dir ./runs/rfdetr-medium-7cls ^
  --model-size medium ^
  --epochs 100 ^
  --batch-size 8 ^
  --grad-accum-steps 2 ^
  --num-workers 8 ^
  --lr 1e-4 ^
  --exclude-classes unknown ^
  --resume-best ^
  --tensorboard
```

From specific checkpoint:

```bash
python scripts/train_rfdetr.py ^
  --dataset-dir ./data/rfdetr_tiled_coco ^
  --output-dir ./runs/rfdetr-medium-7cls ^
  --model-size medium ^
  --epochs 100 ^
  --batch-size 8 ^
  --grad-accum-steps 2 ^
  --num-workers 8 ^
  --lr 1e-4 ^
  --exclude-classes unknown ^
  --resume-from ./runs/rfdetr-medium-7cls/checkpoint_best_total.pth ^
  --tensorboard
```

## 5) Check broken images if DataLoader fails

```bash
python scripts/check_coco_images.py --dataset-dir ./data/rfdetr_tiled_coco
python scripts/check_coco_images.py --dataset-dir ./data/rfdetr_tiled_coco --clean
```

## 6) Class count notebook

Use `data_preprocess.ipynb` to inspect:
- source YOLO class bbox counts
- processed COCO class bbox counts
- per-split class distribution
