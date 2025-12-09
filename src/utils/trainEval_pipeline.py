import torch
from torchmetrics.detection import MeanAveragePrecision

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    
    for batch_idx, (images, targets) in enumerate(loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k,v in t.items()} for t in targets]
        
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        total_loss += losses.item()
        
        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch [{batch_idx + 1}/{len(loader)}] Loss: {losses.item():.4f}")
    
    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    metric = MeanAveragePrecision(iou_thresholds=[.5])

    for images, targets in loader:
        images = [img.to(device) for img in images]
        predictions = model(images)

        preds = [
            {
                "boxes": pred["boxes"],
                "scores": pred["scores"],
                "labels": pred["labels"]
            }
            for pred in predictions
        ]

        targs = [
            {
                "boxes": t["boxes"].to(device),
                "labels": t["labels"].to(device)
            }
            for t in targets
        ]
        
        metric.update(preds, targs)

    return metric.compute()