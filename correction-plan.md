# Correction Plan — Steel Defect Detection

## Context

PyTorch RetinaNet pipeline for detecting 6 classes of steel-surface defects (Cloudera / automotive-manufacturing QC). All working code lives in `src/utils/`. This document lists every weakness found during the review and the recommended fix, prioritized.

## Files reviewed

- `src/utils/dataset.py` — `SteelDefectDataset` (torch Dataset) + `collate_func`
- `src/utils/parse_xml.py` — Pascal VOC XML → dict list
- `src/utils/split_dataset.py` — stratified 80/20 resplit utility
- `src/utils/transforms_pipeline.py` — Albumentations train/val transforms
- `src/utils/trainEval_pipeline.py` — `train_one_epoch` + `evaluate` (mAP)
- `src/utils/test_pipeline.py` — sequential smoke tests

## Weaknesses found

### Correctness / robustness bugs
1. **`evaluate()` flips model between `.train()` and `.eval()` inside the batch loop** (`trainEval_pipeline.py:43–48`). Toggling `.train()` re-enables BatchNorm running-mean updates on validation data — statistics drift toward the val set. Real training bug.
2. **`split_dataset.py` stratifies on only the first `<object>`** (`get_defect_class`, lines 19–24). Multi-defect images are mislabeled for stratification → skewed class balance.
3. **`split_dataset.py`'s `__main__` block resolves the data root incorrectly** (lines 206–207: `Path(__file__).parent` from `src/utils/` yields `src/utils/data/raw`).
4. **`parse_xml.py` has no error handling.** Missing `<name>`, `<bndbox>`, or malformed XML raises cryptic `AttributeError` on `.text`. No coordinate validation.
5. **Empty-annotation images crash** in `dataset.py:55` — `(boxes_tensor[:, 2] - boxes_tensor[:, 0])` throws `IndexError` when there are 0 boxes.
6. **Missing `iscrowd` field** in the target dict — some torchvision detection loss paths and COCO-style eval expect it.
7. **No bbox sanity for augmentations** (`transforms_pipeline.py`): `BboxParams` has no `min_area` / `min_visibility`, so rotations/flips can produce degenerate boxes.

### Design / duplication
8. **`test_pipeline.py` re-implements `SteelDefectDataset` inline** (lines 48–96) instead of importing from `src/utils/dataset.py`. The two definitions have already drifted (test version drops `area`).
9. **`dataset.py` duplicates target-dict construction** across the `if transforms` / `else` branches (lines 51–64).
10. **`class_map` is hardcoded in two places** (`dataset.py`, `test_pipeline.py`) — should be a shared constant/config.
11. **Duplicated print block** in `split_dataset.py:114–117` (`"Copying files to new split..."` appears twice).
12. **Unused imports** in `trainEval_pipeline.py`: `confusion_matrix`, `seaborn`.

### MLOps / robustness gaps
13. **`print` everywhere** instead of `logging` — no log levels, no file sink, no timestamps.
14. **No central config** (paths, class map, hyperparameters, seed, device) — hardcoded values scattered.
15. **`test_pipeline.py` is not `pytest`-compatible** — `sys.exit` + `try/except`, cannot be CI-integrated.
16. **`test_pipeline.py` writes a real checkpoint** to `models/test_checkpoint.pth` without cleanup.
17. **`sys.path` manipulation** in `test_pipeline.py:16–19` is fragile — a proper installable package would fix it.
18. **Device selection ignores CUDA** in `test_pipeline.py:33` (only MPS ↔ CPU).
19. **`train_one_epoch` lacks** AMP, LR scheduler hook, tqdm progress, checkpoint hook, per-component loss logging.
20. **No reproducibility harness** — seeds only set in the test script; the real training loop does not seed dataloader workers or cuDNN.
21. **`split_dataset.py` copies files instead of writing a manifest** — doubles disk use and locks the split into on-disk layout.

## Improvement plan (prioritized)

### P0 — Correctness bugs
- **Fix `evaluate()` train/eval toggling** (`trainEval_pipeline.py`): keep model in `.eval()`; drop `val_loss` computed via train-mode hack; rely on mAP/mAP@50.
- **Fix stratification in `split_dataset.py`** — use multi-label-aware strategy (`MultilabelStratifiedShuffleSplit`) or fall back to most-frequent class per image.
- **Fix `__main__` path** in `split_dataset.py`: `project_root = Path(__file__).resolve().parents[2]`.
- **Handle empty-annotation images** in `dataset.py`: build zero-shaped tensors and compute `area` conditionally.
- **Harden `parse_xml.py`** with `try/except ET.ParseError`, `None` checks on `find(...)`, and `xmin<xmax`/`ymin<ymax` validation.

### P1 — De-duplication and structure
- **Delete inline `SteelDefectDataset` in `test_pipeline.py`**; import from `src.utils.dataset`. Same for `collate_func` and transforms.
- **Refactor `dataset.py:__getitem__`** to build the target once.
- **Extract shared constants** (`CLASS_MAP`, `IMG_MEAN`, `IMG_STD`, num_classes) into `src/utils/config.py`.
- **Remove unused imports** in `trainEval_pipeline.py`.
- **Remove duplicated print block** at `split_dataset.py:114–117`.

### P2 — MLOps hardening
- Add `src/utils/logging_utils.py` with `get_logger()`; replace `print` calls in pipeline modules.
- Convert `test_pipeline.py` to `pytest` (`tests/test_pipeline.py`), use `tmp_path` for checkpoint I/O.
- Add `pyproject.toml` so `pip install -e .` replaces `sys.path` hacks.
- Extend device selection: `cuda` → `mps` → `cpu`.
- Add `bbox_params` safeguards (`min_area=1`, `min_visibility=0.1`).
- Enhance `train_one_epoch`: optional AMP, LR scheduler hook, tqdm progress bar, per-component loss dict.
- Add reproducibility helper `set_seed(seed)` (random, numpy, torch, cuDNN, dataloader `worker_init_fn`).

### P3 — Nice-to-haves
- Rewrite `split_dataset.py` to emit `train.txt`/`val.txt` manifests instead of copying files.
- Add `models.py` (as promised in README) with configurable `num_classes`.
- Add type hints and docstrings across the public API in `src/utils/`.

## Critical files to modify

- `src/utils/dataset.py` — items 4, 5, 6, 9, 10
- `src/utils/parse_xml.py` — item 4
- `src/utils/split_dataset.py` — items 2, 3, 11, P3
- `src/utils/transforms_pipeline.py` — item 7
- `src/utils/trainEval_pipeline.py` — items 1, 12, P2 training loop
- `src/utils/test_pipeline.py` — items 8, 15, 16, 17, 18 (candidate for full rewrite as `tests/test_pipeline.py`)
- **New:** `src/utils/config.py`, `src/utils/logging_utils.py`, `src/utils/models.py`, `pyproject.toml`

## Verification

1. **Unit smoke:** `pytest tests/` — rewritten tests run green with no side-effects.
2. **Dataset:** load a synthetic image with 0 boxes and confirm `dataset[i]` returns without crashing.
3. **Parse:** feed `parse_xml` a malformed XML fixture and confirm it raises a clear, custom error.
4. **Split:** run `split_dataset.py` on a small copy of `data/raw/`; per-class ratio in train ≈ overall ratio (±2%).
5. **Train step:** one `train_one_epoch` + `evaluate` cycle on a 4-image subset; confirm BN running stats unchanged after eval (compare `state_dict()` snapshots).
6. **Reproducibility:** two training epochs with the same seed → identical loss trajectory.
