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


def blend_hair_crop(
    base: np.ndarray,
    original: np.ndarray,
    restored: np.ndarray,
    parsing_labels: np.ndarray,
    strength: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Enhance parsed hair while retaining the source hair color.

    CelebAMask-HQ/facexlib label 17 represents hair. CodeFormer supplies the
    reconstructed luminance/detail; source chroma is retained to limit color
    drift and a soft semantic mask prevents a hard halo at the hair boundary.
    """
    strength = float(np.clip(strength, 0.0, 1.0))
    h, w = original.shape[:2]
    labels = cv2.resize(parsing_labels.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    hard_mask = (labels == 17).astype(np.float32)
    if strength == 0.0 or not np.any(hard_mask):
        return base, np.zeros((h, w), dtype=np.float32)

    restored = cv2.resize(restored, (w, h), interpolation=cv2.INTER_LANCZOS4)
    # Preserve original a/b chroma and take reconstructed luminance detail.
    original_lab = cv2.cvtColor(original, cv2.COLOR_BGR2LAB)
    restored_lab = cv2.cvtColor(restored, cv2.COLOR_BGR2LAB)
    hair_lab = restored_lab.copy()
    hair_lab[:, :, 1:] = original_lab[:, :, 1:]
    color_stable = cv2.cvtColor(hair_lab, cv2.COLOR_LAB2BGR)

    # A restrained unsharp mask adds local strand contrast without changing size.
    blurred = cv2.GaussianBlur(color_stable, (0, 0), 1.1)
    sharpened = cv2.addWeighted(color_stable, 1.35, blurred, -0.35, 0)
    soft_mask = cv2.GaussianBlur(hard_mask, (31, 31), 7)
    soft_mask = np.clip(soft_mask * strength, 0.0, 1.0)[..., None]
    mixed = base.astype(np.float32) * (1.0 - soft_mask) + sharpened.astype(np.float32) * soft_mask
    return np.clip(mixed, 0, 255).astype(np.uint8), hard_mask


def paste_faces_with_hair(
    helper, upsample_img: np.ndarray, parsing_maps: list[np.ndarray], hair_strength: float
) -> np.ndarray:
    """Inverse-warp restored faces with facexlib face and hair-aware masks."""
    h, w = helper.input_img.shape[:2]
    h_up, w_up = int(h * helper.upscale_factor), int(w * helper.upscale_factor)
    output = cv2.resize(upsample_img, (w_up, h_up), interpolation=cv2.INTER_LANCZOS4).astype(np.float32)
    if len(helper.restored_faces) != len(helper.inverse_affine_matrices):
        raise RuntimeError("Restored faces and inverse affine matrices have different lengths.")

    # facexlib's standard parsing mask includes facial features but excludes hair.
    face_labels = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15}
    for index, (restored_face, inverse_affine) in enumerate(
        zip(helper.restored_faces, helper.inverse_affine_matrices)
    ):
        matrix = inverse_affine.copy()
        if helper.upscale_factor > 1:
            matrix[:, 2] += 0.5 * helper.upscale_factor

        if index >= len(parsing_maps):
            raise RuntimeError("Missing face parsing map for restored face.")
        labels = parsing_maps[index]
        labels = cv2.resize(labels.astype(np.uint8), restored_face.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
        face_mask = np.isin(labels, list(face_labels)).astype(np.float32)
        face_mask = cv2.GaussianBlur(face_mask, (101, 101), 11)
        face_mask = cv2.GaussianBlur(face_mask, (101, 101), 11)
        hair_mask = (labels == 17).astype(np.float32) if hair_strength > 0 else np.zeros_like(face_mask)
        # The aligned crop is already blended according to hair_strength;
        # paste its entire soft hair region to avoid applying strength twice.
        hair_soft = cv2.GaussianBlur(hair_mask.astype(np.float32), (31, 31), 7)
        mask = np.maximum(face_mask, hair_soft)
        mask[:10, :] = mask[-10:, :] = 0
        mask[:, :10] = mask[:, -10:] = 0

        warped_face = cv2.warpAffine(restored_face, matrix, (w_up, h_up), flags=cv2.INTER_LINEAR)
        warped_mask = cv2.warpAffine(mask, matrix, (w_up, h_up), flags=cv2.INTER_CUBIC)
        warped_mask = np.clip(warped_mask, 0.0, 1.0)[..., None]
        output = warped_mask * warped_face.astype(np.float32) + (1.0 - warped_mask) * output
    return np.clip(output, 0, 255).astype(np.uint8)
