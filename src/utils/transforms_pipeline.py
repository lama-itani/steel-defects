import albumentations as A # note that albumentationsx was installed and but you still albumentations
from albumentations.pytorch import ToTensorV2

def get_train_transforms():
    return A.Compose([
        A.HorizontalFlip(p = .5), # probability of 50% to flip the image left-right >> defects visible from any orientation
        A.VerticalFlip(p = .5), # flip top-bottom >> surfaces viewed from any angle
        A.RandomRotate90(p = .5), # rotate 90, 180 or 270 degrees >> increase orientation variety
        A.RandomBrightnessContrast(p = .3), # adjust brightness/contrast >> image quality
        A.GaussNoise(p = .2), # add random noise
        A.Normalize(mean = (.485, .456, .406), std = (.229, .224, .225)), # normalizing based on ImageNet stats (imp. for transfer learning)
        ToTensorV2() # converts NumPy array to PyTorch tensor, divides pix by 255 ([0,1] range), permutes format from [H,W,C] to [C,H,W].
    ], bbox_params = A.BboxParams(format = "pascal_voc", label_fields = ["labels"])) # XML annotation has a Pascal Voc format

def get_val_transforms(): # for validation on clean images, converting only to tensor format, no augmentation.
    return A.Compose([
        A.Normalize(mean = (.485, .456, .406), std = (.229, .224, .225)),
        ToTensorV2()
    ], bbox_params = A.BboxParams(format = "pascal_voc", label_fields = ["labels"]))