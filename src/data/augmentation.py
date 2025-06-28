"""
BioViT3R-Beta Training Augmentation Pipelines
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_training_augmentation():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomRotate90(p=0.3),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.5),
        A.OneOf([
            A.MotionBlur(blur_limit=3, p=0.2),
            A.GaussianBlur(blur_limit=3, p=0.2)
        ], p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
