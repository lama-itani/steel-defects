from typing import Dict

import torch
from torch.utils.data import DataLoader
from torchmetrics.detection import MeanAveragePrecision


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0

    for batch_idx, (images, targets) in enumerate(loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)  # gradient clipping
        optimizer.step()

        total_loss += losses.item()

        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch [{batch_idx + 1}/{len(loader)}] Loss: {losses.item():.4f}")

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """
    Validation pass. Stays in .eval() the whole time — do NOT flip to .train()
    to read losses, since that updates BatchNorm running statistics on val data.

    Torchvision detection models only expose the loss dict in training mode, so
    we deliberately drop val_loss here and rely on mAP / mAP@50 as the primary
    signal. `val_loss` is kept in the return dict as NaN for callers that log it.
    """
    model.eval()
    metric = MeanAveragePrecision(
        box_format='xyxy',
        iou_type='bbox',
        class_metrics=True,
    )

    for images, targets in loader:
        images = [img.to(device) for img in images]

        predictions = model(images)

        preds = [
            {
                "boxes": pred["boxes"].cpu(),
                "scores": pred["scores"].cpu(),
                "labels": pred["labels"].cpu(),
            }
            for pred in predictions
        ]

        targs = [
            {
                "boxes": t["boxes"].cpu(),
                "labels": t["labels"].cpu(),
            }
            for t in targets
        ]

        metric.update(preds, targs)

    raw_results = metric.compute()
    results = {
        'map_50': raw_results['map_50'],
        'map': raw_results['map'],
        # val_loss cannot be computed in .eval() for torchvision detection models
        # without corrupting BN stats. Kept as NaN so downstream code (history logs,
        # plots) still has a stable key. Rely on map/map_50 for model selection.
        'val_loss': float('nan'),
        'precision': raw_results.get('precision', torch.tensor(0.0)),
        'recall': raw_results.get('recall', torch.tensor(0.0)),
    }
    # Ensure 'map_per_class' is a tensor and not a scalar
    if 'map_per_class' in raw_results:
        results['map_per_class'] = raw_results['map_per_class']

    return results
