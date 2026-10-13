"""
BoneGuard AI - Preprocessing & Training Augmentation Pipeline
============================================================
Project Title: Advanced Bone Fracture Assessment and Clinical Decision Support
               Through Deep Learning-Based X-Ray Image Analysis

Model Architecture Target: DenseNet121 (+ DG-CBAM)
Target Resolution: 224 x 224 x 3

This module provides standard, reproducible, and medically conservative
preprocessing and training-time data augmentation transforms.
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import random

# ImageNet standard normalization parameters expected by pretrained DenseNet121
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
TARGET_SIZE = (224, 224)


def load_and_preprocess_image(image_path, target_size=TARGET_SIZE, normalize=True):
    """
    Standard deterministic preprocessing pipeline for inference, validation, and testing.
    Steps:
    1. Read image safely (supporting various bit depths and formats).
    2. Convert to single-channel grayscale if needed.
    3. Replicate across 3 channels (RGB) for pretrained DenseNet121 backbone compatibility.
    4. Resize to 224 x 224 with high-quality Lanczos/Bicubic interpolation.
    5. Convert pixel intensities to float [0.0, 1.0] and apply ImageNet normalization.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    # Read image using PIL for format consistency
    with Image.open(image_path) as img:
        img = img.convert("L")  # Convert to standard 8-bit grayscale radiograph
        img_resized = img.resize(target_size, Image.Resampling.LANCZOS)
        
        # Convert to 3-channel RGB array
        img_rgb = Image.merge("RGB", (img_resized, img_resized, img_resized))
        arr = np.array(img_rgb, dtype=np.float32) / 255.0

    if normalize:
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD

    # Return shape (H, W, C)
    return arr


def conservative_train_augmentation(pil_image):
    """
    Conservative training-time augmentation:
    - Slight rotation (+/- 10 degrees)
    - Mild horizontal flip (p=0.5)
    - Slight zoom / scale (0.95 - 1.05)
    - Mild brightness adjustment (0.9 - 1.1)
    - Mild contrast adjustment (0.9 - 1.1)
    
    Strictly avoids:
    - Extreme distortion
    - Vertical flipping
    - Aggressive cropping / shear
    """
    img = pil_image.copy()

    # 1. Horizontal Flip
    if random.random() > 0.5:
        img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

    # 2. Slight Rotation (-10 to +10 degrees)
    angle = random.uniform(-10.0, 10.0)
    img = img.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False, fillcolor=0)

    # 3. Slight Zoom / Crop (0.95 - 1.05)
    zoom_factor = random.uniform(0.95, 1.05)
    w, h = img.size
    if zoom_factor > 1.0:
        # Slight zoom in: crop center and resize back
        crop_w = int(w / zoom_factor)
        crop_h = int(h / zoom_factor)
        left = (w - crop_w) // 2
        top = (h - crop_h) // 2
        img = img.crop((left, top, left + crop_w, top + crop_h)).resize((w, h), Image.Resampling.BICUBIC)
    elif zoom_factor < 1.0:
        # Slight zoom out: resize and pad center
        new_w = int(w * zoom_factor)
        new_h = int(h * zoom_factor)
        img_scaled = img.resize((new_w, new_h), Image.Resampling.BICUBIC)
        new_canvas = Image.new("L", (w, h), 0)
        paste_x = (w - new_w) // 2
        paste_y = (h - new_h) // 2
        new_canvas.paste(img_scaled, (paste_x, paste_y))
        img = new_canvas

    # 4. Mild Brightness Variation (0.9 - 1.1)
    b_factor = random.uniform(0.90, 1.10)
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(b_factor)

    # 5. Mild Contrast Variation (0.9 - 1.1)
    c_factor = random.uniform(0.90, 1.10)
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(c_factor)

    return img


def get_preprocessed_tensor(image_path, is_training=False, target_size=TARGET_SIZE, normalize=True):
    """
    Complete single-image loader returning preprocessed (and optionally augmented) array.
    """
    with Image.open(image_path) as img:
        img_gray = img.convert("L")
        
        if is_training:
            img_gray = conservative_train_augmentation(img_gray)
            
        img_resized = img_gray.resize(target_size, Image.Resampling.LANCZOS)
        img_rgb = Image.merge("RGB", (img_resized, img_resized, img_resized))
        arr = np.array(img_rgb, dtype=np.float32) / 255.0

    if normalize:
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD

    # Transpose to channel-first format (C, H, W) standard in PyTorch
    arr = np.transpose(arr, (2, 0, 1))
    return arr
