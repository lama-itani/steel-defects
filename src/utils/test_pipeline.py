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
Code will exit on the first failure.
"""
import sys  
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
print(f"Project root added to sys.path: {project_root}")

from pathlib import Path
import torch
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from torchvision.models.detection import retinanet_resnet50_fpn_v2
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
from src.utils.parse_xml import parse_xml

# Setup
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)

class Config:
    DATA_ROOT = Path(project_root) / "data"/ "raw"
    TRAIN_IMG = DATA_ROOT / "train_images"
    TRAIN_ANN = DATA_ROOT / "train_annotations"
    NUM_CLASSES = 7
    BATCH_SIZE = 2
    NUM_WORKERS = 0

config = Config()

# Import dataset class (simplified version for testing)
class SteelDefectDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, ann_dir, transforms=None):
        self.img_dir = Path(img_dir)
        self.ann_dir = Path(ann_dir)
        self.transforms = transforms
        self.images = sorted(list(self.img_dir.glob("*.jpg")))
        self.class_map = {
            "crazing": 1, "inclusion": 2, "patches": 3,
            "pitted_surface": 4, "rolled-in_scale": 5, "scratches": 6
        }
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        xml_path = self.ann_dir / img_path.name.replace(".jpg", ".xml")
        boxes_data = parse_xml(str(xml_path))
        
        boxes = []
        labels = []
        for box in boxes_data:
            boxes.append([box["xmin"], box["ymin"], box["xmax"], box["ymax"]])
            labels.append(self.class_map[box["label"]])
        
        if self.transforms:
            transformed = self.transforms(
                image = image, 
                bboxes = boxes, 
                labels = labels
                )
            image = transformed["image"]
            target = {
                "boxes": torch.as_tensor(transformed["bboxes"], dtype = torch.float32),
                "labels": torch.as_tensor(transformed["labels"], dtype = torch.int64),
                "image_id": torch.as_tensor([idx])
            }
        else:
                        
            target = {
            "boxes": torch.as_tensor(boxes, dtype = torch.float32),
            "labels": torch.as_tensor(labels, dtype = torch.int64),
            "image_id": torch.as_tensor([idx])
            }

        return image, target

def get_transforms():
    return A.Compose([
        A.Normalize(mean = (.485, .456, .406), std = (.229, .224, .225)),
        ToTensorV2()
    ], bbox_params = A.BboxParams(format = "pascal_voc", label_fields = ["labels"]))

def collate_func(batch):
    return tuple(zip(*batch))

print("=" * 70)
print("STEEL DEFECT DETECTION - PIPELINE TEST")
print("=" * 70)

# TEST 1: Dataset loading
print("\n[TEST 1] Dataset Loading")
print("-" * 70)
try:
    dataset = SteelDefectDataset(config.TRAIN_IMG, config.TRAIN_ANN, transforms = get_transforms())
    print(f"Dataset created successfully!")
    print(f"Total images: {len(dataset)}")
    
    if len(dataset) > 0:
        image, target = dataset[0]
        print(f"Sample loaded successfully!")
        print(f"Image shape: {image.shape}")
        print(f"Boxes shape: {target['boxes'].shape}")
        print(f"Labels: {target['labels'].tolist()}")
        print(f"Label range: {target['labels'].min().item()}-{target['labels'].max().item()}")
        
        # Verify labels are in correct range
        assert target['labels'].min().item() >= 1 and target['labels'].max().item() <= 6, "Labels out of range!"
        print(f"Labels in valid range [1-6]")
    else:
        print(f"No images found in dataset!")
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
    
    print(f"Image range: [{img_min:.3f}, {img_max:.3f}]")
    print(f"Image mean: {img_mean:.3f}")
    
    # Used ImageNet normalization, expect roughly [-2, 2] range and mean near 0
    if -3 < img_min < -0.5 and 0.5 < img_max < 3:
        print("ImageNet normalization detected")
    elif 0 <= img_min < 0.1 and 0.9 < img_max <= 1.0:
        print("WARNING: No normalization applied (values in [0,1])")
    else:
        print(f"Unexpected range, check normalization")
        
except Exception as e:
    print(f"Normalization check failed: {e}")
    sys.exit(1)

# TEST 2: DataLoader
print("\n[TEST 2] DataLoader")
print("-" * 70)
try:
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size = config.BATCH_SIZE,
        shuffle = False,
        num_workers = config.NUM_WORKERS,
        collate_fn = collate_func
    )
    print(f"DataLoader created successfully")
    
    batch_images, batch_targets = next(iter(loader))
    print(f"Batch loaded successfully!")
    print(f"Batch size: {len(batch_images)}")
    print(f"First image shape: {batch_images[0].shape}")
    print(f"Images in batch have shapes: {[img.shape for img in batch_images]}")
except Exception as e:
    print(f"DataLoader failed: {e}")
    sys.exit(1)

# TEST 3: Model creation
print("\n[TEST 3] Model Creation")
print("-" * 70)
try:
    model = retinanet_resnet50_fpn_v2(weights = "DEFAULT")
    print(f"Base model loaded")
    
    num_anchors = model.head.classification_head.num_anchors
    model.head.classification_head = RetinaNetClassificationHead(
        in_channels = 256,
        num_anchors = num_anchors,
        num_classes = config.NUM_CLASSES
    )
    print(f"Classification head replaced")
    print(f"Num anchors: {num_anchors}")
    print(f"Num classes: {config.NUM_CLASSES}")
    
    model = model.to(device)
    print(f"Model moved to device: {device}")
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
        print(f"Inference successful")
        print(f"Number of predictions: {len(predictions)}")
        print(f"Prediction keys: {list(predictions[0].keys())}")
        print(f"Pred boxes shape: {predictions[0]['boxes'].shape}")
        print(f"Pred scores shape: {predictions[0]['scores'].shape}")
        print(f"Pred labels range: {predictions[0]['labels'].min().item()}-{predictions[0]['labels'].max().item()}")
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
    print(f"Forward pass in training mode successful")
    print(f"Loss components: {list(loss_dict.keys())}")
    
    for loss_name, loss_value in loss_dict.items():
        print(f"  {loss_name}: {loss_value.item():.4f}")
    
    total_loss = sum(loss for loss in loss_dict.values())
    print(f"Total loss: {total_loss.item():.4f}")
    
    # Test backward pass
    total_loss.backward()
    print(f"Backward pass successful")
except Exception as e:
    print(f"Training mode failed: {e}")
    sys.exit(1)

# TEST 6: Optimizer step
print("\n[TEST 6] Optimizer Step")
print("-" * 70)
try:
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.001, momentum = 0.9)
    optimizer.zero_grad()
    
    # Fresh forward pass
    loss_dict = model(test_images, test_targets)
    total_loss = sum(loss for loss in loss_dict.values())
    
    total_loss.backward()
    optimizer.step()
    print(f"Optimizer step successful")
    print(f"Loss before step: {total_loss.item():.4f}")
except Exception as e:
    print(f"Optimizer step failed: {e}")
    sys.exit(1)

# TEST 7: Checkpoint saving
print("\n[TEST 7] Checkpoint Saving")
print("-" * 70)
try:
    Path("models").mkdir(exist_ok = True)
    torch.save(model.state_dict(), "models/test_checkpoint.pth")
    print(f"Checkpoint saved successfully")
    print(f"Location: models/test_checkpoint.pth")
    
    # Verify can load
    model.load_state_dict(torch.load("models/test_checkpoint.pth"))
    print(f"Checkpoint loaded successfully")
except Exception as e:
    print(f"Checkpoint saving/loading failed: {e}")
    sys.exit(1)

print("\n" + "=" * 70)
print("ALL TESTS PASSED SUCCESSFULLY!")
print("=" * 70)
print("\nYour pipeline is ready to be fully trained.")
