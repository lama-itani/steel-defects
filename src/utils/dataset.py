from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from src.utils.parse_xml import parse_xml

# Shared class map — kept here for backward compat; the canonical source is
# src.utils.config.CLASS_MAP if callers want to import from a single place.
CLASS_MAP = {
    "crazing": 1,
    "inclusion": 2,
    "patches": 3,
    "pitted_surface": 4,
    "rolled-in_scale": 5,
    "scratches": 6,
}


class SteelDefectDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for steel-defect images with Pascal VOC XML annotations.
    Compatible with torchvision detection models (RetinaNet, Faster R-CNN, ...).
    """

    def __init__(
        self,
        img_dir: Union[str, Path],
        ann_dir: Union[str, Path],
        transforms: Optional[Callable] = None,
    ) -> None:
        self.img_dir = Path(img_dir)
        self.ann_dir = Path(ann_dir)
        self.transforms = transforms
        self.images = sorted(list(self.img_dir.glob("*.jpg")))
        self.class_map = CLASS_MAP

    def __len__(self) -> int:
        return len(self.images)

    def _build_target(
        self,
        boxes: List[List[float]],
        labels: List[int],
        idx: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Assemble the torchvision-detection target dict from raw box/label lists.
        Handles the zero-box case (defect-free images) cleanly.
        """
        if len(boxes) == 0:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes_tensor = torch.as_tensor(np.array(boxes, dtype=np.float32), dtype=torch.float32)
            labels_tensor = torch.as_tensor(labels, dtype=torch.int64)
            area = (boxes_tensor[:, 2] - boxes_tensor[:, 0]) * (boxes_tensor[:, 3] - boxes_tensor[:, 1])
            iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        return {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.as_tensor([idx]),
            "area": area,
            "iscrowd": iscrowd,
        }

    def __getitem__(self, idx: int) -> Tuple[Union[np.ndarray, torch.Tensor], Dict[str, torch.Tensor]]:
        img_path = self.images[idx]
        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # OpenCV BGR -> RGB (torch/RetinaNet)

        xml_path = self.ann_dir / img_path.name.replace(".jpg", ".xml")
        boxes_data = parse_xml(str(xml_path)) if xml_path.exists() else []

        boxes = []
        labels = []
        for box in boxes_data:
            boxes.append([box["xmin"], box["ymin"], box["xmax"], box["ymax"]])
            labels.append(self.class_map[box["label"]])

        if self.transforms:
            # Albumentations does not accept an empty bbox list under some
            # versions; guard by only passing bboxes/labels when present.
            if boxes:
                transformed = self.transforms(image=image, bboxes=boxes, labels=labels)
                image = transformed["image"]
                boxes = list(transformed["bboxes"])
                labels = list(transformed["labels"])
            else:
                transformed = self.transforms(image=image, bboxes=[], labels=[])
                image = transformed["image"]
                boxes, labels = [], []

        target = self._build_target(boxes, labels, idx)
        return image, target


def collate_func(batch: List[Tuple[Any, Dict[str, torch.Tensor]]]) -> Tuple[Tuple[Any, ...], ...]:
    """
    Transpose a batch of (image, target) pairs into (images, targets) tuples.
    Necessary because detection targets have variable-length bbox tensors.
    """
    return tuple(zip(*batch))
