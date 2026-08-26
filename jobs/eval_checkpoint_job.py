"""
Evaluate an existing checkpoint against the validation set and record the mAP.

Answers the one open question blocking the demo: what is this model's actual
mAP@50? Does NOT train anything.

Usage:
    # smoke test first — 32 images, confirms it works end to end
    python jobs/eval_checkpoint_job.py --limit 32

    # the real run
    python jobs/eval_checkpoint_job.py

    # other checkpoint
    python jobs/eval_checkpoint_job.py --ckpt models/retinanet_final.pth
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torchvision.models.detection import retinanet_resnet50_fpn_v2
from torchvision.models.detection.retinanet import RetinaNetClassificationHead

from src.utils.config import Config, CLASS_NAMES, set_seed, pick_device
from src.utils.dataset import SteelDefectDataset, collate_func
from src.utils.transforms_pipeline import get_val_transforms
from src.utils.trainEval_pipeline import evaluate


def build_model(num_classes: int) -> torch.nn.Module:
    """Same architecture as jobs/model_training_job.py, but weights=None.

    The checkpoint supplies every weight, so downloading DEFAULT weights first
    would be pure wasted bandwidth and disk.
    """
    model = retinanet_resnet50_fpn_v2(weights=None)
    num_anchors = model.head.classification_head.num_anchors
    model.head.classification_head = RetinaNetClassificationHead(
        in_channels=256,
        num_anchors=num_anchors,
        num_classes=num_classes,
    )
    return model


def load_checkpoint(model: torch.nn.Module, path: Path, device) -> dict:
    """Handle both checkpoint layouts in this repo.

    jobs/model_training_job.py saves a dict with 'model_state_dict' + metadata.
    The notebook's final save is a bare state_dict. Accept either.
    """
    try:
        ck = torch.load(path, map_location=device, weights_only=True)
    except Exception:
        # metadata-bearing checkpoints contain non-tensor values
        ck = torch.load(path, map_location=device, weights_only=False)

    meta = {}
    if isinstance(ck, dict) and "model_state_dict" in ck:
        state = ck["model_state_dict"]
        meta = {k: ck.get(k) for k in ("epoch", "best_map")}
        print(f"Checkpoint metadata: epoch={meta.get('epoch')} best_map={meta.get('best_map')}")
    else:
        state = ck
        print("Bare state_dict checkpoint (no recorded metrics).")

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        # strict=False so we can SEE the mismatch instead of dying on it, but a
        # non-empty list here means the number below is not trustworthy.
        print(f"WARNING missing keys:    {len(missing)}  {missing[:5]}")
        print(f"WARNING unexpected keys: {len(unexpected)}  {unexpected[:5]}")
        print("WARNING: architecture does not match the checkpoint. mAP is meaningless.")
    else:
        print("All weights loaded cleanly.")
    return meta


class ProgressLoader:
    """Thin iterable wrapper that prints progress.

    evaluate() has no logging and never calls len(), so wrapping the loader is
    enough to stop a 20-minute pass from looking hung — and it avoids editing
    trainEval_pipeline.py, which is out of scope.
    """

    def __init__(self, loader, every: int = 5):
        self.loader = loader
        self.every = every
        self.total = len(loader)

    def __iter__(self):
        start = time.time()
        for i, batch in enumerate(self.loader, 1):
            yield batch
            if i % self.every == 0 or i == self.total:
                el = time.time() - start
                eta = el / i * (self.total - i)
                print(f"  batch {i}/{self.total}  elapsed {el:.0f}s  eta {eta:.0f}s", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate a checkpoint. No training.")
    p.add_argument("--ckpt", type=Path, default=None)
    p.add_argument("--batch-size", type=int, default=4, help="Small: CPU box, 800px upscale.")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--limit", type=int, default=None, help="Evaluate only N images (smoke test).")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    cfg = Config()
    ckpt = args.ckpt or (cfg.models_dir / "retinanet_best.pth")
    out = args.out or (cfg.models_dir / "eval_report.json")

    if not ckpt.exists():
        sys.exit(f"Checkpoint not found: {ckpt}")

    device = pick_device()
    set_seed(cfg.seed)
    print(f"Device: {device} | checkpoint: {ckpt}")

    val_ds = SteelDefectDataset(cfg.val_img, cfg.val_ann, transforms=get_val_transforms())
    if len(val_ds) == 0:
        sys.exit(f"No images found under {cfg.val_img}")
    if args.limit:
        val_ds = torch.utils.data.Subset(val_ds, range(min(args.limit, len(val_ds))))
    print(f"Val images: {len(val_ds)}")

    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_func,
    )

    model = build_model(cfg.num_classes)
    meta = load_checkpoint(model, ckpt, device)
    model = model.to(device)

    print("\nEvaluating (MeanAveragePrecision is slow on CPU — progress below)...")
    t0 = time.time()
    results = evaluate(model, ProgressLoader(val_loader), device)
    took = time.time() - t0

    map50 = float(results["map_50"])
    map_all = float(results["map"])
    print("\n" + "=" * 60)
    print(f"mAP@50 : {map50:.4f}")
    print(f"mAP    : {map_all:.4f}")
    print(f"took   : {took:.0f}s on {len(val_ds)} images")

    per_class = results.get("map_per_class")
    per_class_out = None
    if torch.is_tensor(per_class) and per_class.ndim > 0:
        per_class_out = {}
        print("\nPer-class AP:")
        for i, name in enumerate(CLASS_NAMES):
            if i < len(per_class):
                v = float(per_class[i])
                per_class_out[name] = v
                print(f"  {name:<20}: {v:.4f}")
    print("=" * 60)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "checkpoint": str(ckpt),
        "checkpoint_epoch": meta.get("epoch"),
        "checkpoint_best_map": meta.get("best_map"),
        "num_val_images": len(val_ds),
        "eval_seconds": round(took, 1),
        "map_50": map50,
        "map": map_all,
        "per_class_ap": per_class_out,
    }, indent=2))
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()