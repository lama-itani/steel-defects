"""
Test pipeline to ensure dataset, dataloader, model, and training loop work correctly.
It tests sequentially:
1. Dataset loading: XML parsing, class mapping, image loading
2. DataLoader: Batching, collate function, variable-sized targets
3. Model creation: Architecture modification, device transfer
4. Inference: Forward pass outputs correct format
5. Training mode: Loss computation
6. Optimizer step: Backward pass and parameter update
7. Checkpoint saving: Save/load functionality
Exits on the first failure.
"""
import math
import sys
import tempfile
from pathlib import Path

# The actual paths/params come from src/utils/config.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import torch
from torchvision.models.detection import retinanet_resnet50_fpn_v2
from torchvision.models.detection.retinanet import RetinaNetClassificationHead

from src.utils.config import Config, set_seed, pick_device
from src.utils.dataset import SteelDefectDataset, collate_func
from src.utils.transforms_pipeline import get_val_transforms, get_train_transforms
from src.utils.trainEval_pipeline import train_one_epoch, evaluate

config = Config(batch_size=2, num_workers=0)

# --- Fixture -----------------------------------------------------------------
# Default: a 3-image synthetic fixture generated into a temp dir, so this runs
# offline, on CPU, with no dataset on disk and no pretrained-weight download.
# Pass --real to run the identical checks against the on-disk NEU dataset.
USE_SYNTHETIC = "--real" not in sys.argv

FIXTURE_SIZE = 200          # NEU images are 200x200
_fixture_tmp = None         # module-level: temp dir must outlive every test


def _write_voc_xml(path, filename, boxes, size=FIXTURE_SIZE):
    """Hand-write a minimal Pascal VOC annotation. `boxes` may be empty."""
    objects = "".join(
        f"  <object>\n"
        f"    <name>{label}</name>\n"
        f"    <bndbox>\n"
        f"      <xmin>{xmin}</xmin>\n"
        f"      <ymin>{ymin}</ymin>\n"
        f"      <xmax>{xmax}</xmax>\n"
        f"      <ymax>{ymax}</ymax>\n"
        f"    </bndbox>\n"
        f"  </object>\n"
        for label, xmin, ymin, xmax, ymax in boxes
    )
    path.write_text(
        f"<annotation>\n"
        f"  <filename>{filename}</filename>\n"
        f"  <size><width>{size}</width><height>{size}</height><depth>3</depth></size>\n"
        f"{objects}</annotation>\n"
    )


def build_synthetic_fixture():
    """
    Write 3 textured 200x200 JPEGs + matching VOC XML into a temp dir:
    000_defect_free : zero <object> tags (exercises the empty-target branch)
    001_crazing     : one box
    002_scratches   : two boxes, two classes (exercises multi-object collate)
    Sorted glob order puts the defect-free sample at index 0 on purpose.
    Returns (img_dir, ann_dir).
    """
    global _fixture_tmp
    _fixture_tmp = tempfile.TemporaryDirectory(prefix="steel_fixture_")
    root = Path(_fixture_tmp.name)
    img_dir, ann_dir = root / "images", root / "annotations"
    img_dir.mkdir()
    ann_dir.mkdir()

    rng = np.random.default_rng(config.seed)
    samples = [
        ("000_defect_free", []),
        ("001_crazing", [("crazing", 20, 30, 120, 140)]),
        ("002_scratches", [("scratches", 10, 10, 90, 60),
                            ("inclusion", 100, 110, 180, 190)]),
    ]
    for stem, boxes in samples:
        # Textured grey, not flat: keeps the normalization check meaningful and
        # gives the augmentation pipeline something real to act on.
        img = rng.integers(90, 165, size=(FIXTURE_SIZE, FIXTURE_SIZE, 3), dtype=np.uint8)
        for _, xmin, ymin, xmax, ymax in boxes:
            patch = img[ymin:ymax, xmin:xmax].astype(np.int16) + 60
            img[ymin:ymax, xmin:xmax] = np.clip(patch, 0, 255).astype(np.uint8)
        cv2.imwrite(str(img_dir / f"{stem}.jpg"), img)
        _write_voc_xml(ann_dir / f"{stem}.xml", f"{stem}.jpg", boxes)

    return img_dir, ann_dir


if USE_SYNTHETIC:
    device = torch.device("cpu")        # forced: determinism, no GPU needed
    img_dir, ann_dir = build_synthetic_fixture()
else:
    device = pick_device()
    img_dir, ann_dir = config.train_img, config.train_ann

set_seed(config.seed)

print("=" * 70)
print("STEEL DEFECT DETECTION - PIPELINE TEST")
print("=" * 70)

# TEST 1: Dataset loading
print("\n[TEST 1] Dataset Loading")
print("-" * 70)
try:
    dataset = SteelDefectDataset(img_dir, ann_dir, transforms=get_val_transforms())
    print(f"Dataset created. Total images: {len(dataset)}")

    if len(dataset) > 0:
        image, target = dataset[0]
        print(f"Image shape: {image.shape}")
        print(f"Boxes shape: {target['boxes'].shape}")
        print(f"Labels: {target['labels'].tolist()}")
        if target['labels'].numel() > 0:
            assert target['labels'].min().item() >= 1 and target['labels'].max().item() <= 6, "Labels out of range!"
            print("Labels in valid range [1-6]")
        else:
            print("Sample has zero boxes (defect-free image) — pipeline handled it cleanly.")

        # Cover every sample, not just index 0. Capped on the real dataset.
        n_check = len(dataset) if USE_SYNTHETIC else min(len(dataset), 5)
        for i in range(n_check):
            img_i, tgt_i = dataset[i]
            assert img_i.ndim == 3 and img_i.shape[0] == 3, \
                f"Sample {i}: expected CHW with 3 channels, got {tuple(img_i.shape)}"
            lengths = {tgt_i[k].shape[0] for k in ("boxes", "labels", "area", "iscrowd")}
            assert len(lengths) == 1, f"Sample {i}: target field lengths disagree ({lengths})"
            if tgt_i["labels"].numel():
                assert tgt_i["labels"].min().item() >= 1 and tgt_i["labels"].max().item() <= 6, \
                    f"Sample {i}: label out of range [1-6]"
        print(f"Checked {n_check} sample(s): consistent target fields, labels in range")
    else:
        print("No images found in dataset!")
        sys.exit(1)
except Exception as e:
    print(f"Dataset loading failed: {e}")
    sys.exit(1)

# Test 1.1: Verify normalization
print("\n[TEST 1.5] Verify Normalization")
print("-" * 70)
try:
    image, target = dataset[0]
    img_min, img_max = image.min().item(), image.max().item()
    img_mean = image.mean().item()

    print(f"Image range: [{img_min:.3f}, {img_max:.3f}]  mean: {img_mean:.3f}")

    unnormalized = (0 <= img_min < 0.1) and (0.9 < img_max <= 1.0)
    assert not unnormalized, (
        f"No normalization applied — raw [0,1] range detected "
        f"[{img_min:.3f}, {img_max:.3f}]"
    )
    print("Normalization present (not raw [0,1])")
except Exception as e:
    print(f"Normalization check failed: {e}")
    sys.exit(1)

# Test 1.2: Train-transform path (augementation + zero-box branch)
print("\n[TEST 1.2] Train Transform (Augmentation + Zero-Box)")
print("-" * 70)
try:
    train_tf_dataset = SteelDefectDataset(img_dir, ann_dir, transforms=get_train_transforms())

    img0, tgt0 = train_tf_dataset[0]
    assert img0.ndim == 3 and img0.shape[0] == 3, f"Bad CHW shape: {tuple(img0.shape)}"
    if USE_SYNTHETIC:
        assert tgt0["boxes"].shape == (0, 4), \
            f"Fixture sample 0 must be defect-free, got boxes {tuple(tgt0['boxes'].shape)}"
        assert tgt0["labels"].shape == (0,) and tgt0["area"].shape == (0,) \
            and tgt0["iscrowd"].shape == (0,), "Zero-box target has wrong field shapes"
        assert tgt0["boxes"].dtype == torch.float32 and tgt0["labels"].dtype == torch.int64
        print("Zero-box branch OK under get_train_transforms()")

    idx_boxed, tgt_boxed = None, None
    for i in range(len(train_tf_dataset)):
        _, t = train_tf_dataset[i]
        if t["labels"].numel() > 0:
            idx_boxed, tgt_boxed = i, t
            break
    assert tgt_boxed is not None, "No annotated sample survived augmentation — path untested"

    b = tgt_boxed["boxes"]
    assert b.shape[1] == 4 and b.shape[0] == tgt_boxed["labels"].shape[0], \
        "Augmented boxes/labels disagree in length"
    assert torch.all(b[:, 2] > b[:, 0]) and torch.all(b[:, 3] > b[:, 1]), \
        "Augmentation produced a degenerate box (xmax<=xmin or ymax<=ymin)"
    print(f"Augmented sample {idx_boxed}: {b.shape[0]} box(es), all non-degenerate")
except Exception as e:
    print(f"Train-transform check failed: {e}")
    sys.exit(1)

# TEST 2: DataLoader
print("\n[TEST 2] DataLoader")
print("-" * 70)
try:
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_func,
    )
    batch_images, batch_targets = next(iter(loader))
    print(f"Batch size: {len(batch_images)}  shapes: {[img.shape for img in batch_images]}")
except Exception as e:
    print(f"DataLoader failed: {e}")
    sys.exit(1)

# TEST 3: Model creation
print("\n[TEST 3] Model Creation")
print("-" * 70)
try:
    # weights=None on the synthetic path: no network, no torch-hub cache needed.
    model = retinanet_resnet50_fpn_v2(weights="DEFAULT" if not USE_SYNTHETIC else None)
    num_anchors = model.head.classification_head.num_anchors
    model.head.classification_head = RetinaNetClassificationHead(
        in_channels=256,
        num_anchors=num_anchors,
        num_classes=config.num_classes,
    )
    if USE_SYNTHETIC:
        # Skip the 800px upscale so the CPU run is seconds, not minutes.
        model.transform.min_size = (FIXTURE_SIZE,)
        model.transform.max_size = FIXTURE_SIZE
    model = model.to(device)
    print(f"Model created and moved to {device}. num_anchors={num_anchors} num_classes={config.num_classes}")
except Exception as e:
    print(f"Model creation failed: {e}")
    sys.exit(1)

# TEST 4: Model inference
print("\n[TEST 4] Model Inference (eval mode)")
print("-" * 70)
try:
    model.eval()
    with torch.no_grad():
        test_images = [img.to(device) for img in batch_images[:2]]
        predictions = model(test_images)
        assert set(predictions[0].keys()) == {"boxes", "labels", "scores"}, \
            f"Unexpected prediction keys: {sorted(predictions[0].keys())}"
        print(f"Predictions: {len(predictions)}  keys: {list(predictions[0].keys())}")
        print(f"Pred boxes: {predictions[0]['boxes'].shape}  scores: {predictions[0]['scores'].shape}")
except Exception as e:
    print(f"Model inference failed: {e}")
    sys.exit(1)

# TEST 5: Training mode (loss computation)
print("\n[TEST 5] Training Mode (loss computation)")
print("-" * 70)
try:
    model.train()
    test_images = [img.to(device) for img in batch_images[:2]]
    test_targets = [{k: v.to(device) for k, v in t.items()} for t in batch_targets[:2]]

    loss_dict = model(test_images, test_targets)
    for loss_name, loss_value in loss_dict.items():
        print(f"  {loss_name}: {loss_value.item():.4f}")

    total_loss = sum(loss for loss in loss_dict.values())
    print(f"Total loss: {total_loss.item():.4f}")
    total_loss.backward()
    print("Backward pass successful")
except Exception as e:
    print(f"Training mode failed: {e}")
    sys.exit(1)

# TEST 6: Optimizer step
print("\n[TEST 6] Optimizer Step")
print("-" * 70)
try:
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer.zero_grad()

    tracked = [p for p in model.head.classification_head.parameters() if p.requires_grad]
    assert tracked, "No trainable parameters in the classification head"
    before = [p.detach().clone() for p in tracked]

    loss_dict = model(test_images, test_targets)
    total_loss = sum(loss for loss in loss_dict.values())
    total_loss.backward()
    optimizer.step()

    assert any(not torch.equal(b, p) for b, p in zip(before, tracked)), \
        "optimizer.step() did not change any classification-head parameter"
    print(f"Optimizer step OK. Loss before step: {total_loss.item():.4f}")
except Exception as e:
    print(f"Optimizer step failed: {e}")
    sys.exit(1)

# TEST 7: Checkpoint saving (uses a temp file so we don't litter models/)
print("\n[TEST 7] Checkpoint Saving")
print("-" * 70)
try:
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = Path(tmp) / "test_checkpoint.pth"
        saved = {k: v.detach().clone() for k, v in model.state_dict().items()}
        torch.save(model.state_dict(), ckpt)
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        for k, v in model.state_dict().items():
            assert torch.equal(v, saved[k]), f"Checkpoint mismatch on parameter {k}"
        print(f"Checkpoint saved & reloaded from {ckpt}")
except Exception as e:
    print(f"Checkpoint saving/loading failed: {e}")
    sys.exit(1)

# TEST 8: The repo's own training/eval code (trainEval_pipeline)
print("\n[TEST 8] trainEval_pipeline Smoke Cycle")
print("-" * 70)
try:
    smoke_optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)

    train_loss = train_one_epoch(model, loader, smoke_optimizer, device)
    assert isinstance(train_loss, float) and math.isfinite(train_loss), \
        f"train_one_epoch returned {train_loss!r} (expected a finite float)"
    assert model.training, "train_one_epoch should leave the model in train mode"
    print(f"train_one_epoch OK. Mean loss: {train_loss:.4f}")

    results = evaluate(model, loader, device)

    # The P0 BatchNorm fix: evaluate() must never flip back to .train().
    assert not model.training, \
        "evaluate() left the model in train mode — BatchNorm running stats get corrupted"

    # These exact keys are indexed by the notebook and jobs/model_training_job.py.
    required = {"map_50", "map", "val_loss", "precision", "recall", "map_per_class"}
    missing = required - set(results)
    assert not missing, f"evaluate() lost keys downstream code indexes: {sorted(missing)}"

    # notebook does results["map_50"].item() unconditionally -> must stay a tensor
    assert torch.is_tensor(results["map_50"]), \
        f"map_50 must be a tensor, got {type(results['map_50']).__name__}"
    # notebook's _val_loss_is_meaningful() checks isinstance(v, float) and isnan(v)
    assert isinstance(results["val_loss"], float) and math.isnan(results["val_loss"]), \
        f"val_loss should be float('nan'), got {results['val_loss']!r}"

    print(f"evaluate OK. keys: {sorted(results)}")
    print(f"mAP@50: {results['map_50'].item():.4f} (meaningless on a random model — shape check only)")
except Exception as e:
    print(f"trainEval_pipeline smoke cycle failed: {e}")
    sys.exit(1)

print("\n" + "=" * 70)
print("ALL TESTS PASSED SUCCESSFULLY!")
print("=" * 70)