"""
Albumentations augmentation pipelines for the steel-defect detector.

bbox_params includes min_area / min_visibility so that geometric augmentations
(flips, rotations) that shrink or clip a box off-frame drop it cleanly instead
of leaving a degenerate zero-area annotation in the target dict.
"""
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ImageNet stats (RetinaNet backbone was pretrained on ImageNet)
IMG_MEAN = (0.485, 0.456, 0.406)
IMG_STD = (0.229, 0.224, 0.225)

_BBOX_PARAMS = A.BboxParams(
    format="pascal_voc",
    label_fields=["labels"],
    min_area=1.0,          # drop boxes smaller than 1 px^2 after augment
    min_visibility=0.1,    # drop boxes with <10% of their area still visible
)


def get_train_transforms():
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.GaussNoise(p=0.2),
            A.Normalize(mean=IMG_MEAN, std=IMG_STD),
            ToTensorV2(),
        ],
        bbox_params=_BBOX_PARAMS,
    )


def get_val_transforms():
    """Validation: clean images, only format conversion + normalization."""
    return A.Compose(
        [
            A.Normalize(mean=IMG_MEAN, std=IMG_STD),
            ToTensorV2(),
        ],
        bbox_params=_BBOX_PARAMS,
    )
