from __future__ import annotations

from pathlib import Path
from typing import Union

import cv2
import numpy as np
from PIL import Image

ImageInput = Union[Image.Image, np.ndarray, str, Path]


def load_image(image: ImageInput) -> np.ndarray:
    """Load an image as contiguous uint8 BGR, the format used internally."""
    if image is None:
        raise ValueError("No image was provided.")

    if isinstance(image, (str, Path)):
        path = Path(image)
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            raise ValueError("Only JPG, JPEG and PNG images are supported.")
        data = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if data is None:
            raise ValueError(f"Unable to decode image: {path}")
    elif isinstance(image, Image.Image):
        rgb = np.asarray(image.convert("RGB"))
        data = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    elif isinstance(image, np.ndarray):
        data = image
    else:
        raise TypeError(f"Unsupported image type: {type(image).__name__}")

    if data.ndim == 2:
        data = cv2.cvtColor(data, cv2.COLOR_GRAY2BGR)
    elif data.ndim == 3 and data.shape[2] == 4:
        data = cv2.cvtColor(data, cv2.COLOR_BGRA2BGR)
    elif data.ndim != 3 or data.shape[2] != 3:
        raise ValueError(f"Expected an HxWx3 image, got shape {data.shape}.")

    if data.dtype != np.uint8:
        if np.issubdtype(data.dtype, np.floating) and data.max(initial=0) <= 1.0:
            data = data * 255.0
        data = np.clip(data, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(data)


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def ensure_max_pixels(image: np.ndarray, max_pixels: int) -> None:
    pixels = int(image.shape[0] * image.shape[1])
    if pixels > max_pixels:
        raise ValueError(
            f"Input contains {pixels:,} pixels; configured limit is {max_pixels:,}. "
            "Resize the image or increase runtime.max_input_pixels."
        )


def safe_output_name(prefix: str = "restored") -> str:
    from datetime import datetime

    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png"
