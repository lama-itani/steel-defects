# Computer vision to detect steel defects

A PyTorch object-detection pipeline that classifies and localizes surface defects on steel. Built for Cloudera as an
automotive/manufacturing quality-control demo, it trains a RetinaNet detector to draw bounding boxes around six defect
types on 200×200 images of steel surfaces.

## Dataset

- **Source:** [NEU Steel Surface Defect dataset](https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database) (Kaggle).
- **Size:** 1,800 images total, 200×200 px, Pascal VOC XML annotations.
- **Current split:** 1,440 train / 360 val (stratified 80/20, produced by `src/utils/split_dataset.py`).
- **Classes (6 + background):** `crazing`, `inclusion`, `patches`, `pitted_surface`, `rolled-in_scale`, `scratches`.

## Model

- **Architecture:** `torchvision.models.detection.retinanet_resnet50_fpn_v2` with a ResNet-50 FPN v2 backbone,
  ImageNet-pretrained. Classification head is replaced for 6 defect classes + background.
- **Training:** SGD + momentum, `StepLR` scheduler, gradient clipping, early stopping, best-checkpoint saving.
- **Augmentation:** Albumentations pipeline (see `src/utils/transforms_pipeline.py`) with `min_area` / `min_visibility`
  guards on bboxes.
- **Metric:** mAP and mAP@50 via `torchmetrics.MeanAveragePrecision`. mAP@50 is the model-selection signal.

## Project structure

```
project-root/
├── README.md
├── requirements.txt
├── correction-plan.md
├── 2026-08-11-code-review-and-p0-fixes-summary.md
│
├── data/
│   ├── raw/
│   │   ├── train_images/            # 1440 .jpg training images
│   │   ├── train_annotations/       # matching Pascal VOC XML
│   │   ├── valid_images/            # 360 .jpg validation images
│   │   ├── valid_annotations/       # matching Pascal VOC XML
│   │   └── backup_original_split/   # backup written by split_dataset.py
│   └── processed/
│
├── jobs/
│   └── model_training_job.py        # headless training CLI (entry point)
│
├── models/                          # runtime checkpoints + metrics (retinanet_best.pth, training_metrics.json, …)
│
├── notebooks/
│   ├── EDA_cv_steel.ipynb           # exploratory data analysis (class dist., bbox viz)
│   ├── EDA_functions.py             # standalone EDA helpers
│   ├── train_cv_steel.ipynb         # interactive training notebook
│   ├── models/                      # checkpoints written from the notebook
│   └── visualizations/              # training-curve PNGs
│
├── outputs/
│   └── anchor_explained_viz.html    # RetinaNet anchor-box explainer
│
└── src/
    └── utils/
        ├── config.py                # Config dataclass, CLASS_MAP, pick_device(), set_seed(), seed_worker()
        ├── dataset.py               # SteelDefectDataset + collate_func
        ├── parse_xml.py             # Pascal VOC XML → box dicts; raises AnnotationParseError
        ├── split_dataset.py         # one-shot stratified 80/20 resplit (backs up originals)
        ├── transforms_pipeline.py   # Albumentations train/val augmentation pipelines
        ├── trainEval_pipeline.py    # train_one_epoch() + evaluate() (BN-safe, mAP-based)
        └── test_pipeline.py         # 7-test smoke script
```

## Setup

```bash
python -m venv venv-CVsteel
source venv-CVsteel/bin/activate
pip install -r requirements.txt

# split_dataset.py uses sklearn.model_selection.train_test_split but scikit-learn
# is not currently in requirements.txt — install it if you plan to resplit:
pip install scikit-learn
```

Python 3.14 is what the existing `venv-CVsteel/` was built with; any recent 3.x with the pinned versions in
`requirements.txt` should work. There is no `pyproject.toml` yet, so scripts manage `sys.path` themselves — always run
from the project root.

## Usage

### 1. (Optional) Resplit the dataset

Run once if you want a fresh stratified 80/20 split. The script backs up the current split into
`data/raw/backup_original_split/` before overwriting.

```bash
python src/utils/split_dataset.py
```

### 2a. Interactive training — notebook

Open `notebooks/train_cv_steel.ipynb` in Jupyter. It mirrors the headless job: builds the dataset + dataloaders,
constructs the model, trains, and writes the best checkpoint to `notebooks/models/retinanet_best.pth` along with
training-curve PNGs under `notebooks/visualizations/`.

### 2b. Headless training — CLI job

```bash
# From project root:
python jobs/model_training_job.py                             # defaults from Config
python jobs/model_training_job.py --epochs 24 --lr 0.0005
python jobs/model_training_job.py --resume models/retinanet_best.pth
```

Flags: `--epochs`, `--batch-size`, `--lr`, `--seed`, `--num-workers`, `--resume`. Outputs `models/retinanet_best.pth`,
`models/retinanet_final.pth`, and `models/training_metrics.json`.

### 3. Smoke tests

```bash
python src/utils/test_pipeline.py
```

Runs 7 sequential checks (dataset loading, normalization, dataloader, model creation, inference, loss/backward,
checkpoint I/O). Exits non-zero on the first failure.

### 4. EDA

Open `notebooks/EDA_cv_steel.ipynb` for class distributions and bounding-box visualizations.

## Status

A code-review pass on **2026-08-11** landed a batch of P0 correctness fixes. Full detail lives in
[`correction-plan.md`](./correction-plan.md) and
[`2026-08-11-code-review-and-p0-fixes-summary.md`](./2026-08-11-code-review-and-p0-fixes-summary.md); the highlights:

**Fixed (P0):**
- `evaluate()` was flipping the model back to `.train()` mid-validation, corrupting BatchNorm running stats.
  Now stays in `.eval()` throughout; `val_loss` is reported as `NaN` and mAP@50 is the sole model-selection signal.
- `SteelDefectDataset.__getitem__` crashed on defect-free (empty-annotation) images — now returns correctly shaped
  zero tensors.
- `split_dataset.py` was stratifying on only the first `<object>` tag (wrong for multi-defect images) and had a
  broken `__main__` path. Both fixed.
- `parse_xml.py` now raises `AnnotationParseError` with `None` checks and coordinate validation instead of
  cryptic `AttributeError`s.
- Added `iscrowd` field to detection targets (required by torchvision losses and COCO eval).
- Albumentations `BboxParams` gained `min_area` / `min_visibility` guards.
- Notebook: best-model checkpoint was never saved (dead conditional). Fixed. Nested-quote f-string (Python <3.12
  incompatible) fixed. Broken confusion-matrix cell gated behind `RUN_CONFUSION_MATRIX = False`.

**New files:**
- `src/utils/config.py` — central `Config` dataclass, `CLASS_MAP`, `pick_device()`, `set_seed()`, `seed_worker()`.
- `jobs/model_training_job.py` — reproducible headless training CLI with resume support.

**Pending (P1/P2):**
- `src/utils/logging_utils.py` (replace `print` with `logging`).
- Convert `test_pipeline.py` to proper `pytest` under `tests/`.
- `pyproject.toml` to eliminate `sys.path` manipulation.
- AMP, `tqdm`, and LR-scheduler hook inside `train_one_epoch`.
- IoU-matched confusion matrix.
- Manifest-based split (P3).
