"""Lightweight, weight-free image deblurring utilities.

The project normally passes BGR images to :class:`DeblurEngine`, but the
algorithms below operate on every colour channel independently.  Consequently
RGB and BGR inputs are both safe, and an alpha channel is preserved verbatim.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from numbers import Integral, Real

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)

_SUPPORTED_METHODS = frozenset({"auto", "unsharp", "wiener"})


def _validate_image(image: np.ndarray) -> np.ndarray:
    """Validate the public uint8 image contract without changing its layout."""
    if not isinstance(image, np.ndarray):
        raise TypeError(f"image must be a numpy.ndarray, got {type(image).__name__}.")
    if image.dtype != np.uint8:
        raise TypeError(f"image must have dtype uint8, got {image.dtype}.")
    if image.size == 0 or any(dimension == 0 for dimension in image.shape):
        raise ValueError("image must not be empty.")
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] in (1, 3, 4):
        return image
    raise ValueError(
        "image must have shape HxW, HxWx1, HxWx3, or HxWx4; "
        f"got {image.shape}."
    )


def _validate_real(name: str, value: object, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number.")
    result = float(value)
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise ValueError(f"{name} must be between {minimum:g} and {maximum:g}.")
    return result


def _validate_positive_real(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number.")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be greater than zero.")
    return result


def _validate_tile_size(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("tile_size must be an integer.")
    result = int(value)
    if not 128 <= result <= 8192:
        raise ValueError("tile_size must be between 128 and 8192 pixels.")
    return result


def _colour_plane(image: np.ndarray) -> tuple[np.ndarray, np.ndarray | None, bool]:
    """Return processable channels, optional alpha, and a single-channel flag."""
    if image.ndim == 2:
        return image, None, True
    if image.shape[2] == 1:
        return image[:, :, 0], None, True
    if image.shape[2] == 4:
        return image[:, :, :3], image[:, :, 3].copy(), False
    return image, None, False


def _luminance(image: np.ndarray) -> np.ndarray:
    """Make a channel-order-neutral luminance proxy for blur measurement."""
    if image.ndim == 2:
        return image
    # A mean avoids assuming whether callers supplied RGB or OpenCV-style BGR.
    return np.mean(image, axis=2, dtype=np.float32)


def measure_blur(image: np.ndarray) -> float:
    """Return a Laplacian-variance sharpness score (larger means sharper).

    The longest side is capped at 1024 pixels so scores are inexpensive and
    reasonably comparable for large inputs.  Alpha is deliberately ignored.
    A completely flat or single-pixel image returns ``0.0``.
    """
    checked = _validate_image(image)
    colour, _, _ = _colour_plane(checked)
    gray = _luminance(colour)

    height, width = gray.shape
    longest_side = max(height, width)
    if longest_side > 1024:
        ratio = 1024.0 / longest_side
        resized_width = max(1, round(width * ratio))
        resized_height = max(1, round(height * ratio))
        gray = cv2.resize(
            gray,
            (resized_width, resized_height),
            interpolation=cv2.INTER_AREA,
        )
    if min(gray.shape) < 2:
        return 0.0

    # ksize=1 is the conventional variance-of-Laplacian focus measure.  Its
    # scale also keeps the default auto thresholds intuitive (roughly 35--180).
    laplacian = cv2.Laplacian(
        gray,
        cv2.CV_32F,
        ksize=1,
        borderType=cv2.BORDER_REFLECT101,
    )
    score = float(np.var(laplacian, dtype=np.float64))
    return score if math.isfinite(score) else 0.0


class DeblurEngine:
    """Traditional deblurring and sharpening that needs no model weights.

    Parameters can be supplied through a configuration mapping.  Unknown keys
    are ignored so this class can safely receive a larger application config.

    Supported methods:

    * ``auto``: measure sharpness and attenuate/skip enhancement on clear images.
    * ``unsharp``: multi-scale, noise-gated unsharp masking.
    * ``wiener``: regularised Gaussian Wiener deconvolution followed by a mild
      unsharp pass.  It is more aggressive and is intended for visibly blurred
      inputs.
    """

    def __init__(self, config: Mapping[str, object] | None = None):
        if config is None:
            config = {}
        if not isinstance(config, Mapping):
            raise TypeError("config must be a mapping or None.")

        # ``tile`` mirrors Real-ESRGAN's existing configuration name, while
        # ``tile_size`` is the clearer standalone spelling.
        self.tile_size = _validate_tile_size(
            config.get("tile_size", config.get("tile", 1024))
        )
        self.auto_blur_threshold = _validate_positive_real(
            "auto_blur_threshold", config.get("auto_blur_threshold", 35.0)
        )
        self.auto_sharp_threshold = _validate_positive_real(
            "auto_sharp_threshold", config.get("auto_sharp_threshold", 180.0)
        )
        if self.auto_blur_threshold >= self.auto_sharp_threshold:
            raise ValueError("auto_blur_threshold must be less than auto_sharp_threshold.")

    @staticmethod
    def measure_blur(image: np.ndarray) -> float:
        """Expose :func:`measure_blur` from an engine instance as well."""
        return measure_blur(image)

    def enhance(
        self,
        image_bgr: np.ndarray,
        strength: float = 0.5,
        method: str = "auto",
    ) -> np.ndarray:
        """Enhance a uint8 gray/RGB/BGR/RGBA/BGRA image.

        ``strength`` is in ``[0, 1]``.  Zero returns a contiguous copy with
        pixel values unchanged.  The result always has the same shape as the
        input and dtype ``uint8``.
        """
        image = _validate_image(image_bgr)
        amount = _validate_real("strength", strength, 0.0, 1.0)
        if not isinstance(method, str):
            raise TypeError("method must be a string.")
        method_key = method.strip().lower()
        if method_key not in _SUPPORTED_METHODS:
            choices = ", ".join(sorted(_SUPPORTED_METHODS))
            raise ValueError(f"Unknown deblur method {method!r}; choose one of: {choices}.")
        if amount == 0.0:
            return np.ascontiguousarray(image.copy())

        colour, alpha, single_channel = _colour_plane(image)
        effective_strength = amount
        selected_method = method_key
        score: float | None = None
        if method_key == "auto":
            score = measure_blur(image)
            effective_strength *= self._auto_attenuation(score)
            selected_method = "unsharp"
            if effective_strength <= 1e-6:
                LOGGER.debug("Skipping deblur: sharpness score %.2f is already high", score)
                return np.ascontiguousarray(image.copy())

        if min(colour.shape[:2]) < 2:
            return np.ascontiguousarray(image.copy())
        if selected_method == "wiener":
            processed = self._process_tiled(
                colour,
                effective_strength,
                self._wiener_patch,
                overlap=self._wiener_overlap(effective_strength),
            )
        else:
            processed = self._process_tiled(
                colour,
                effective_strength,
                self._unsharp_patch,
                overlap=self._unsharp_overlap(effective_strength),
            )

        if score is not None:
            LOGGER.debug(
                "Auto deblur: sharpness=%.2f, requested=%.2f, effective=%.2f",
                score,
                amount,
                effective_strength,
            )
        if single_channel:
            if image.ndim == 3:
                processed = processed[:, :, None]
            return np.ascontiguousarray(processed)
        if alpha is not None:
            processed = np.dstack((processed, alpha))
        return np.ascontiguousarray(processed)

    # ``deblur`` reads naturally to external callers and keeps ``enhance``
    # aligned with the existing SuperResolutionEngine API.
    def deblur(
        self,
        image: np.ndarray,
        strength: float = 0.5,
        method: str = "auto",
    ) -> np.ndarray:
        return self.enhance(image, strength=strength, method=method)

    def _auto_attenuation(self, score: float) -> float:
        if score <= self.auto_blur_threshold:
            return 1.0
        if score >= self.auto_sharp_threshold:
            return 0.0
        span = self.auto_sharp_threshold - self.auto_blur_threshold
        # Smoothstep prevents an abrupt visual transition around thresholds.
        position = (self.auto_sharp_threshold - score) / span
        return position * position * (3.0 - 2.0 * position)

    @staticmethod
    def _unsharp_overlap(strength: float) -> int:
        largest_sigma = 1.8 + 1.8 * strength
        return max(8, math.ceil(4.0 * largest_sigma) + 2)

    @staticmethod
    def _wiener_overlap(strength: float) -> int:
        # FFT deconvolution has a wider effective support than Gaussian blur.
        return max(24, math.ceil(20.0 + 20.0 * strength))

    def _process_tiled(
        self,
        image: np.ndarray,
        strength: float,
        processor,
        overlap: int,
    ) -> np.ndarray:
        height, width = image.shape[:2]
        output = np.empty_like(image)
        for top in range(0, height, self.tile_size):
            bottom = min(top + self.tile_size, height)
            source_top = max(0, top - overlap)
            source_bottom = min(height, bottom + overlap)
            for left in range(0, width, self.tile_size):
                right = min(left + self.tile_size, width)
                source_left = max(0, left - overlap)
                source_right = min(width, right + overlap)
                patch = image[source_top:source_bottom, source_left:source_right]
                enhanced = processor(patch, strength)
                crop_top = top - source_top
                crop_left = left - source_left
                output[top:bottom, left:right] = enhanced[
                    crop_top:crop_top + (bottom - top),
                    crop_left:crop_left + (right - left),
                ]
        return output

    @staticmethod
    def _unsharp_patch(patch: np.ndarray, strength: float) -> np.ndarray:
        source = patch.astype(np.float32) / 255.0
        fine_sigma = 0.7 + 0.8 * strength
        coarse_sigma = 1.8 + 1.8 * strength
        fine_blur = cv2.GaussianBlur(
            source, (0, 0), fine_sigma, borderType=cv2.BORDER_REFLECT101
        )
        coarse_blur = cv2.GaussianBlur(
            source, (0, 0), coarse_sigma, borderType=cv2.BORDER_REFLECT101
        )
        fine_detail = source - fine_blur
        coarse_detail = source - coarse_blur

        # Soft-threshold only fine detail: it is where sensor/JPEG noise tends
        # to live.  Broader contours remain available for genuinely soft edges.
        noise_floor = (1.5 - 0.8 * strength) / 255.0
        fine_detail = np.sign(fine_detail) * np.maximum(
            np.abs(fine_detail) - noise_floor, 0.0
        )
        detail = 0.72 * fine_detail + 0.28 * coarse_detail
        gain = 0.6 * strength + 1.8 * strength * strength
        delta = detail * gain
        maximum_delta = strength * (12.0 + 36.0 * strength) / 255.0
        np.clip(delta, -maximum_delta, maximum_delta, out=delta)
        result = np.clip(source + delta, 0.0, 1.0)
        return np.rint(result * 255.0).astype(np.uint8)

    @staticmethod
    def _wiener_patch(patch: np.ndarray, strength: float) -> np.ndarray:
        source = patch.astype(np.float32) / 255.0
        was_gray = source.ndim == 2
        if was_gray:
            source = source[:, :, None]

        height, width, channels = source.shape
        sigma = 0.65 + 1.15 * strength
        regularisation = 0.035 - 0.022 * strength
        frequency_y = np.fft.fftfreq(height).astype(np.float32)[:, None]
        frequency_x = np.fft.rfftfreq(width).astype(np.float32)[None, :]
        transfer = np.exp(
            -2.0 * (math.pi ** 2) * (sigma ** 2)
            * (frequency_x * frequency_x + frequency_y * frequency_y)
        ).astype(np.float32)
        inverse_filter = transfer / (transfer * transfer + regularisation)

        restored = np.empty_like(source)
        for channel in range(channels):
            spectrum = np.fft.rfft2(source[:, :, channel])
            restored[:, :, channel] = np.fft.irfft2(
                spectrum * inverse_filter, s=(height, width)
            ).astype(np.float32)

        # Limit ringing/overshoot, then blend by strength.  A final mild
        # multi-scale pass restores local contrast lost to regularisation.
        maximum_delta = (18.0 + 42.0 * strength) / 255.0
        delta = np.clip(restored - source, -maximum_delta, maximum_delta)
        blended = np.clip(source + delta * (0.35 + 0.55 * strength), 0.0, 1.0)
        blended_u8 = np.rint(blended * 255.0).astype(np.uint8)
        if was_gray:
            blended_u8 = blended_u8[:, :, 0]
        return DeblurEngine._unsharp_patch(blended_u8, min(0.35, strength * 0.35))


__all__ = ["DeblurEngine", "measure_blur"]
