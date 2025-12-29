import torch
from pathlib import Path
import cv2
from src.utils.parse_xml import parse_xml

class SteelDefectDataset(torch.utils.data.Dataset): # inherits from PyTorch Dataset class, making it compatible w/ DataLoader for batch and // processing
    def __init__(self, img_dir, ann_dir, transforms = None): # takes 3 inputs: image directory, annotation directory, and optional transforms (default no augmentation)
        self.img_dir = Path(img_dir)
        self.ann_dir = Path(ann_dir)
        self.transforms = transforms

        # Logic of loading a dataset
        self.images = sorted(list(self.img_dir.glob("*.jpg"))) # find all .jpg files, sort them in order
        self.class_map = { 
            "crazing" : 1,
            "inclusion" : 2,
            "patches" : 3,
            "pitted_surface" : 4,
            "rolled-in_scale" : 5,
            "scratches" : 6,
        }  # mapping defect names (categorical) to integers (numerical)
    def __len__(self): # runs total number of images
        return len(self.images) 
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.images[idx]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # convert BGR, i.e. OpenCV format, to RGB, i.e. PyTorch/RetinaNet format

        # Parse annotations using parse_xml function
        xml_path = self.ann_dir / img_path.name.replace(".jpg", ".xml")
        boxes_data = parse_xml(str(xml_path))

        # Convert XML to lists: [x1, y1, x2, y2], labels = integers
        boxes = []
        labels = []
        for box in boxes_data:
            boxes.append([box["xmin"], box["ymin"], box["xmax"], box["ymax"]])
            labels.append(self.class_map[box["label"]])

        # Apply transforms before converting to tensors
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
    

# Create data loaders
def collate_func(batch):
    return tuple(zip(*batch)) # transpose batch to a list of tensors instead of stack of images into a single tensor 
                                #[(img1, target1), (img2, target2)] >> ([img1, img2],[target1, target2])