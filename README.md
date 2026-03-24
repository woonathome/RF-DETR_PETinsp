# RF-DETR Fine-tuning Repo (Torch)

`2046x2046` 원본 이미지를 `2048x2048`로 리사이즈 후 `8x8` 타일링(`256x256`)하여, 결함이 있는 타일만 남겨 RF-DETR을 학습하기 위한 레포 구성입니다.

## 1) 디렉토리 구조

```text
2603 Tester Model/
├─ dataset/
│  ├─ image/
│  └─ label/
├─ scripts/
│  ├─ prepare_tiled_coco_dataset.py
│  ├─ train_rfdetr.py
│  └─ predict_tile.py
├─ requirements.txt
└─ README.md
```

## 2) 환경 구성 (RTX3090 권장)

Python 3.10+ 권장.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
pip install -r requirements.txt
```

## 3) 데이터 전처리 (타일링 + 결함 타일만 추출 + train/valid/test 분할)

프로젝트 루트(`2603 Tester Model`)에서 실행:

```bash
python scripts/prepare_tiled_coco_dataset.py ^
  --dataset-root ./dataset ^
  --output-root ./data/rfdetr_tiled_coco ^
  --resize-size 2048 ^
  --grid-size 8 ^
  --tile-size 256 ^
  --val-ratio 0.15 ^
  --test-ratio 0.10 ^
  --seed 42 ^
  --overwrite
```

기본 동작:
- YOLO 라벨(`class x_center y_center width height`)을 읽어 COCO bbox로 변환
- 타일별 박스 클리핑
- 박스가 없는 타일은 기본적으로 제거 (결함 타일만 유지)
- 원본 이미지 단위로 split 적용 (같은 원본의 타일이 다른 split으로 섞이지 않음)
- 라벨 없는 이미지는 자동 스킵하며 `metadata/missing_labels.txt`로 기록

전처리 결과:

```text
data/rfdetr_tiled_coco/
├─ train/
│  ├─ *.jpg
│  └─ _annotations.coco.json
├─ valid/
│  ├─ *.jpg
│  └─ _annotations.coco.json
├─ test/
│  ├─ *.jpg
│  └─ _annotations.coco.json
└─ metadata/
   ├─ preprocess_summary.json
   ├─ class_mapping.json
   ├─ missing_labels.txt
   └─ tile_manifest.csv
```

## 4) RF-DETR 학습

```bash
python scripts/train_rfdetr.py ^
  --dataset-dir ./data/rfdetr_tiled_coco ^
  --output-dir ./runs/rfdetr-medium ^
  --model-size medium ^
  --epochs 100 ^
  --batch-size 8 ^
  --grad-accum-steps 2 ^
  --lr 1e-4
```

RTX3090 권장 시작점:
- `model-size=medium`
- `batch-size=8`
- `grad-accum-steps=2` (effective batch 16)

OOM 발생 시:
1. `--batch-size 4`로 감소
2. `--model-size small` 사용

## 5) 단일 이미지 추론 테스트

```bash
python scripts/predict_tile.py ^
  --image-path ./data/rfdetr_tiled_coco/test/some_tile.jpg ^
  --model-size medium ^
  --checkpoint ./runs/rfdetr-medium/best.pth ^
  --threshold 0.3
```

## 6) 클래스 이름 커스터마이징 (선택)

기본 클래스명은 `class_0 ... class_7`로 생성됩니다.  
원하면 한 줄에 하나씩 클래스명을 적은 txt 파일을 만들고:

```bash
python scripts/prepare_tiled_coco_dataset.py --class-names-file ./class_names.txt ...
```

`class_names.txt` 예시:

```text
blackspot
air
gas
pockmark
...
```

