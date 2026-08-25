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
import sys
import tempfile
from pathlib import Path

# The actual paths/params come from src/utils/config.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torchvision.models.detection import retinanet_resnet50_fpn_v2
from torchvision.models.detection.retinanet import RetinaNetClassificationHead

from src.utils.config import Config, set_seed, pick_device
from src.utils.dataset import SteelDefectDataset, collate_func
from src.utils.transforms_pipeline import get_val_transforms


config = Config(batch_size=2, num_workers=0)

device = pick_device()
set_seed(config.seed)

print("=" * 70)
print("STEEL DEFECT DETECTION - PIPELINE TEST")
print("=" * 70)

# TEST 1: Dataset loading
print("\n[TEST 1] Dataset Loading")
print("-" * 70)
try:
    dataset = SteelDefectDataset(config.train_img, config.train_ann, transforms=get_val_transforms())
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
    else:
        print("No images found in dataset!")
        sys.exit(1)
except Exception as e:
    print(f"Dataset loading failed: {e}")
    sys.exit(1)

# Test 1.5: Verify normalization
print("\n[TEST 1.5] Verify Normalization")
print("-" * 70)
try:
    image, target = dataset[1]
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
    model = retinanet_resnet50_fpn_v2(weights="DEFAULT")
    num_anchors = model.head.classification_head.num_anchors
    model.head.classification_head = RetinaNetClassificationHead(
        in_channels=256,
        num_anchors=num_anchors,
        num_classes=config.num_classes,
    )
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

print("\n" + "=" * 70)
print("ALL TESTS PASSED SUCCESSFULLY!")
print("=" * 70)
