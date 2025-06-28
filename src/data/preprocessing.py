"""
BioViT3R-Beta Data Preprocessing Module
Handles image and video preprocessing for plant analysis pipeline.
"""

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageEnhance
import albumentations as A
from albumentations.pytorch import ToTensorV2
import logging
from typing import Union, Tuple, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class PlantImagePreprocessor:
    """Preprocessing pipeline for plant images with specialized augmentations."""

    def __init__(self, target_size: Tuple[int, int] = (512, 512), 
                 normalize: bool = True,
                 augment: bool = False):
        """
        Initialize preprocessor with configuration.

        Args:
            target_size: Target image dimensions (height, width)
            normalize: Whether to apply ImageNet normalization
            augment: Whether to apply data augmentation
        """
        self.target_size = target_size
        self.normalize = normalize
        self.augment = augment

        # ImageNet statistics for normalization
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        self._build_transforms()

    def _build_transforms(self):
        """Build albumentations transform pipeline."""
        transforms_list = [
            A.Resize(height=self.target_size[0], width=self.target_size[1])
        ]

        if self.augment:
            transforms_list.extend([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.RandomRotate90(p=0.3),
                A.ShiftScaleRotate(
                    shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.5
                ),
                A.OneOf([
                    A.HueSaturationValue(
                        hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.5
                    ),
                    A.RandomBrightnessContrast(
                        brightness_limit=0.2, contrast_limit=0.2, p=0.5
                    ),
                    A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3)
                ], p=0.7),
                A.OneOf([
                    A.MotionBlur(blur_limit=3, p=0.2),
                    A.GaussianBlur(blur_limit=3, p=0.2),
                    A.MedianBlur(blur_limit=3, p=0.1)
                ], p=0.3),
                A.GaussNoise(var_limit=(0, 0.02), p=0.2)
            ])

        if self.normalize:
            transforms_list.extend([
                A.Normalize(mean=self.mean, std=self.std),
                ToTensorV2()
            ])
        else:
            transforms_list.append(ToTensorV2())

        self.transform = A.Compose(transforms_list)

    def preprocess_image(self, image: Union[str, np.ndarray, Image.Image]) -> torch.Tensor:
        """
        Preprocess a single image.

        Args:
            image: Input image (file path, numpy array, or PIL Image)

        Returns:
            Preprocessed image tensor
        """
        # Load image if path provided
        if isinstance(image, (str, Path)):
            image = cv2.imread(str(image))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image, Image.Image):
            image = np.array(image)

        # Ensure image is RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            pass  # Already RGB
        elif len(image.shape) == 3 and image.shape[2] == 4:
            image = image[:, :, :3]  # Remove alpha channel
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")

        # Apply transforms
        transformed = self.transform(image=image)
        return transformed['image']

    def preprocess_batch(self, images: List[Union[str, np.ndarray, Image.Image]]) -> torch.Tensor:
        """
        Preprocess a batch of images.

        Args:
            images: List of input images

        Returns:
            Batch tensor of preprocessed images
        """
        processed_images = []
        for img in images:
            processed_img = self.preprocess_image(img)
            processed_images.append(processed_img.unsqueeze(0))

        return torch.cat(processed_images, dim=0)


class VGGTPreprocessor:
    """Specialized preprocessing for VGGT 3D reconstruction."""

    def __init__(self, target_size: Tuple[int, int] = (384, 384)):
        """
        Initialize VGGT preprocessor.

        Args:
            target_size: Target size for VGGT model input
        """
        self.target_size = target_size
        self.transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def preprocess_for_vggt(self, image: Union[str, np.ndarray, Image.Image]) -> torch.Tensor:
        """
        Preprocess image specifically for VGGT model.

        Args:
            image: Input image

        Returns:
            VGGT-ready image tensor
        """
        # Convert to PIL Image if needed
        if isinstance(image, (str, Path)):
            image = Image.open(str(image)).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        return self.transform(image).unsqueeze(0)


class FruitDetectionPreprocessor:
    """Preprocessing for fruit detection models (ACFR + MinneApple)."""

    def __init__(self, target_size: int = 800, max_size: int = 1333):
        """
        Initialize fruit detection preprocessor.

        Args:
            target_size: Minimum target size
            max_size: Maximum allowed size
        """
        self.target_size = target_size
        self.max_size = max_size

        self.transform = A.Compose([
            A.LongestMaxSize(max_size=max_size),
            A.PadIfNeeded(
                min_height=target_size, min_width=target_size,
                border_mode=cv2.BORDER_CONSTANT, value=0
            ),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

    def preprocess_for_detection(self, image: Union[str, np.ndarray, Image.Image]) -> torch.Tensor:
        """
        Preprocess image for fruit detection.

        Args:
            image: Input image

        Returns:
            Detection-ready image tensor
        """
        # Load and convert image
        if isinstance(image, (str, Path)):
            image = cv2.imread(str(image))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image, Image.Image):
            image = np.array(image)

        # Apply transforms
        transformed = self.transform(image=image)
        return transformed['image'].unsqueeze(0)


def enhance_image_quality(image: np.ndarray, 
                         brightness: float = 1.0,
                         contrast: float = 1.0,
                         saturation: float = 1.0) -> np.ndarray:
    """
    Enhance image quality for better analysis.

    Args:
        image: Input image array
        brightness: Brightness factor (1.0 = no change)
        contrast: Contrast factor (1.0 = no change)
        saturation: Saturation factor (1.0 = no change)

    Returns:
        Enhanced image array
    """
    # Convert to PIL for enhancement
    pil_image = Image.fromarray(image)

    # Apply enhancements
    if brightness != 1.0:
        enhancer = ImageEnhance.Brightness(pil_image)
        pil_image = enhancer.enhance(brightness)

    if contrast != 1.0:
        enhancer = ImageEnhance.Contrast(pil_image)
        pil_image = enhancer.enhance(contrast)

    if saturation != 1.0:
        enhancer = ImageEnhance.Color(pil_image)
        pil_image = enhancer.enhance(saturation)

    return np.array(pil_image)


def remove_background(image: np.ndarray, 
                     method: str = 'grabcut',
                     mask_threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove background from plant images.

    Args:
        image: Input image
        method: Background removal method ('grabcut', 'watershed', 'threshold')
        mask_threshold: Threshold for binary mask

    Returns:
        Tuple of (foreground_image, mask)
    """
    if method == 'grabcut':
        return _grabcut_segmentation(image)
    elif method == 'watershed':
        return _watershed_segmentation(image)
    elif method == 'threshold':
        return _threshold_segmentation(image, mask_threshold)
    else:
        raise ValueError(f"Unknown method: {method}")


def _grabcut_segmentation(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """GrabCut-based background removal."""
    height, width = image.shape[:2]

    # Initialize mask
    mask = np.zeros((height, width), np.uint8)

    # Define rectangle (rough foreground region)
    rect = (width//10, height//10, width-width//5, height-height//5)

    # GrabCut algorithm
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

    # Generate final mask
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    # Apply mask
    result = image * mask2[:, :, np.newaxis]

    return result, mask2


def _watershed_segmentation(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Watershed-based segmentation."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Threshold
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

    # Sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    # Unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # Apply watershed
    markers = cv2.watershed(image, markers)
    mask = (markers > 1).astype(np.uint8)

    result = image * mask[:, :, np.newaxis]

    return result, mask


def _threshold_segmentation(image: np.ndarray, threshold: float) -> Tuple[np.ndarray, np.ndarray]:
    """Simple threshold-based segmentation."""
    # Convert to HSV for better plant segmentation
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Define range for green color (plants)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])

    # Create mask
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Morphological operations
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Convert to float and normalize
    mask = mask.astype(np.float32) / 255.0
    mask = (mask > threshold).astype(np.uint8)

    result = image * mask[:, :, np.newaxis]

    return result, mask
