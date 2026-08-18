from __future__ import annotations

import cv2
import numpy as np


def blend_face_crop(original: np.ndarray, restored: np.ndarray, strength: float) -> np.ndarray:
    """Softly blend an aligned restored face with its original crop."""
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength == 0.0:
        return original.copy()
    if restored.shape[:2] != original.shape[:2]:
        restored = cv2.resize(restored, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_LANCZOS4)

    h, w = original.shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)
    margin_x, margin_y = max(1, w // 20), max(1, h // 20)
    cv2.ellipse(mask, (w // 2, h // 2), (w // 2 - margin_x, h // 2 - margin_y), 0, 0, 360, 1, -1)
    sigma = max(3, min(h, w) // 25) | 1
    mask = cv2.GaussianBlur(mask, (sigma, sigma), 0)[..., None] * strength
    mixed = original.astype(np.float32) * (1.0 - mask) + restored.astype(np.float32) * mask
    return np.clip(mixed, 0, 255).astype(np.uint8)
