"""
Headless training entry point for the steel-defect RetinaNet detector.

Mirrors what the training cell in notebooks/train_cv_steel.ipynb does, but
driven from src.utils.config.Config so it is reproducible from a shell and
suitable for a job scheduler (CML, Slurm, cron, GitHub Actions...).

Usage:
    python jobs/model_training_job.py                      # default config
    python jobs/model_training_job.py --epochs 24 --lr .0005
    python jobs/model_training_job.py --resume models/retinanet_best.pth
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

# Make `src.utils.*` importable when this file is run as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torchvision.models.detection import retinanet_resnet50_fpn_v2
from torchvision.models.detection.retinanet import RetinaNetClassificationHead

from src.utils.config import Config, CLASS_NAMES, set_seed, seed_worker
from src.utils.dataset import SteelDefectDataset, collate_func
from src.utils.transforms_pipeline import get_train_transforms, get_val_transforms
from src.utils.trainEval_pipeline import train_one_epoch, evaluate


def build_model(num_classes: int, pretrained: bool = True) -> torch.nn.Module:
    """Build a RetinaNet with a replaced classification head."""
    model = retinanet_resnet50_fpn_v2(weights="DEFAULT" if pretrained else None)

    num_anchors = model.head.classification_head.num_anchors
    model.head.classification_head = RetinaNetClassificationHead(
        in_channels=256,
        num_anchors=num_anchors,
        num_classes=num_classes,
    )

    # RetinaNet paper: bias init to counter class imbalance at start of training.
    prior_prob = 0.01
    bias_value = -math.log((1 - prior_prob) / prior_prob)
    torch.nn.init.constant_(model.head.classification_head.cls_logits.bias, bias_value)
    return model


def build_dataloaders(cfg: Config):
    train_ds = SteelDefectDataset(cfg.train_img, cfg.train_ann, transforms=get_train_transforms())
    val_ds = SteelDefectDataset(cfg.val_img, cfg.val_ann, transforms=get_val_transforms())

    generator = torch.Generator()
    generator.manual_seed(cfg.seed)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        collate_fn=collate_func,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        collate_fn=collate_func,
        worker_init_fn=seed_worker,
    )
    return train_loader, val_loader


def save_checkpoint(path: Path, model, optimizer, scheduler, cfg: Config, epoch: int, best_map: float, history: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_map": best_map,
            "config": cfg.to_serializable_dict(),
            "history": dict(history),
        },
        path,
    )


def save_metrics_json(path: Path, history: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = {
        k: [float(v) if torch.is_tensor(v) else v for v in vals]
        for k, vals in history.items()
    }
    with open(path, "w") as f:
        json.dump(clean, f, indent=2)


def train(cfg: Config, resume: Path | None = None) -> dict:
    set_seed(cfg.seed)

    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    cfg.visualizations_dir.mkdir(parents=True, exist_ok=True)

    device = cfg.device
    print("=" * 70)
    print(f"Steel Defect Detection — headless training job")
    print(f"Device: {device} | workers: {cfg.num_workers}")
    print(f"Epochs: {cfg.num_epochs} | batch: {cfg.batch_size} | lr: {cfg.learning_rate}")
    print(f"Data root: {cfg.data_root}")
    print("=" * 70)

    train_loader, val_loader = build_dataloaders(cfg)
    model = build_model(cfg.num_classes, pretrained=cfg.backbone_pretrained).to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg.learning_rate,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=cfg.step_size,
        gamma=cfg.gamma,
    )

    start_epoch = 0
    history: dict = defaultdict(list)
    best_map = 0.0
    patience_counter = 0

    if resume and resume.exists():
        print(f"Resuming from {resume}")
        ckpt = torch.load(resume, map_location=device)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
            if "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "scheduler_state_dict" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            start_epoch = int(ckpt.get("epoch", 0))
            best_map = float(ckpt.get("best_map", 0.0))
            history = defaultdict(list, ckpt.get("history", {}))
        else:
            model.load_state_dict(ckpt)
        print(f"Resumed at epoch {start_epoch}, best mAP so far {best_map:.4f}")

    bar = "=" * 70
    for epoch in range(start_epoch, cfg.num_epochs):
        print(f"\n{bar}\nEpoch {epoch + 1}/{cfg.num_epochs}\n{bar}")

        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        results = evaluate(model, val_loader, device)

        val_loss = results["val_loss"]
        val_map50 = results["map_50"].item() if torch.is_tensor(results["map_50"]) else float(results["map_50"])
        precision = results.get("precision", torch.tensor(0.0))
        precision = precision.item() if torch.is_tensor(precision) else float(precision)
        recall = results.get("recall", torch.tensor(0.0))
        recall = recall.item() if torch.is_tensor(recall) else float(recall)
        per_class_ap = results.get("map_per_class")
        if torch.is_tensor(per_class_ap):
            per_class_ap = per_class_ap.cpu().numpy()

        current_lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_map50"].append(val_map50)
        history["learning_rate"].append(current_lr)
        history["precision"].append(precision)
        history["recall"].append(recall)
        if per_class_ap is not None:
            history["per_class_ap"].append(per_class_ap.tolist())

        scheduler.step()

        improved = val_map50 > best_map
        if improved:
            best_map = val_map50
            patience_counter = 0
            save_checkpoint(
                cfg.models_dir / "retinanet_best.pth",
                model, optimizer, scheduler, cfg,
                epoch=epoch + 1, best_map=best_map, history=history,
            )
            print(f">>> Best model saved (mAP: {best_map:.4f}) <<<")
        else:
            patience_counter += 1
            print(f"mAP@.5 not improved. Patience: {patience_counter}/{cfg.early_stopping_patience}")

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss} | Val mAP@.5: {val_map50:.4f}")
        print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | LR: {current_lr:.6f}")

        if per_class_ap is not None:
            if getattr(per_class_ap, "ndim", 1) == 0:
                print(f"Overall AP: {float(per_class_ap):.4f}")
            else:
                for idx, ap in enumerate(per_class_ap):
                    if idx < len(CLASS_NAMES):
                        print(f"  {CLASS_NAMES[idx]:<20}: {float(ap):.4f}")

        if patience_counter >= cfg.early_stopping_patience:
            print(f"\nEarly stopping at epoch {epoch + 1} (mAP@.5 not improved for {cfg.early_stopping_patience} epochs)")
            break

    save_metrics_json(cfg.models_dir / "training_metrics.json", history)
    torch.save(model.state_dict(), cfg.models_dir / "retinanet_final.pth")

    print("\n" + bar)
    print(f"Training complete. Best val mAP@.5: {best_map:.4f}")
    print(bar)
    return history


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Headless RetinaNet trainer for steel defects.")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--resume", type=Path, default=None, help="Path to checkpoint to resume from.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config()
    if args.epochs is not None:
        cfg.num_epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.lr is not None:
        cfg.learning_rate = args.lr
    if args.seed is not None:
        cfg.seed = args.seed
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers

    train(cfg, resume=args.resume)


if __name__ == "__main__":
    main()
