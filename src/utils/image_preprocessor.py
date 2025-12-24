#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image Preprocessor Module

Provides grayscale conversion with CLAHE enhancement and sharpening
for improved YOLO detection performance.

Usage:
    from src.utils.image_preprocessor import ImagePreprocessor

    preprocessor = ImagePreprocessor()
    processed_frame = preprocessor.process(frame)
"""

import cv2
import numpy as np


class ImagePreprocessor:
    """
    Image preprocessor for grayscale + CLAHE + sharpening pipeline

    This preprocessing improves detection accuracy for models trained
    on grayscale images with contrast enhancement.
    """

    def __init__(self, clip_limit: float = 2.0, grid_size: tuple = (8, 8)):
        """
        Initialize preprocessor

        Args:
            clip_limit: CLAHE clip limit (default: 2.0)
            grid_size: CLAHE tile grid size (default: (8, 8))
        """
        self.clip_limit = clip_limit
        self.grid_size = grid_size

        # Create CLAHE object
        self.clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=grid_size
        )

        # Sharpening kernel
        self.sharpen_kernel = np.array([
            [0, -1,  0],
            [-1, 5, -1],
            [0, -1,  0]
        ], dtype=np.float32)

        self.enabled = True

    def process(self, img: np.ndarray) -> np.ndarray:
        """
        Process image: BGR → Gray + CLAHE + Sharpen → 3ch BGR

        Args:
            img: Input BGR image (numpy array)

        Returns:
            Processed image as 3-channel BGR (for YOLO compatibility)
        """
        if not self.enabled:
            return img

        if img is None:
            return img

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        enhanced = self.clahe.apply(gray)

        # Apply sharpening
        sharpened = cv2.filter2D(enhanced, -1, self.sharpen_kernel)

        # Convert back to 3-channel BGR (YOLO expects 3 channels)
        return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)

    def process_batch(self, frames: list) -> list:
        """
        Process a batch of frames

        Args:
            frames: List of BGR images

        Returns:
            List of processed images
        """
        return [self.process(frame) for frame in frames]

    def enable(self):
        """Enable preprocessing"""
        self.enabled = True

    def disable(self):
        """Disable preprocessing (passthrough mode)"""
        self.enabled = False

    def is_enabled(self) -> bool:
        """Check if preprocessing is enabled"""
        return self.enabled


# Singleton instance for convenience
_default_preprocessor = None

def get_preprocessor(clip_limit: float = 2.0, grid_size: tuple = (8, 8)) -> ImagePreprocessor:
    """
    Get or create default preprocessor instance

    Args:
        clip_limit: CLAHE clip limit
        grid_size: CLAHE tile grid size

    Returns:
        ImagePreprocessor instance
    """
    global _default_preprocessor
    if _default_preprocessor is None:
        _default_preprocessor = ImagePreprocessor(clip_limit, grid_size)
    return _default_preprocessor


def gray_sharpen_transform(img: np.ndarray) -> np.ndarray:
    """
    Convenience function for single image processing

    Args:
        img: Input BGR image

    Returns:
        Processed 3-channel BGR image
    """
    return get_preprocessor().process(img)
