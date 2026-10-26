"""Lazy neural deblurring backends with a traditional CPU fallback.

The upstream EVSSM and Restormer repositories both vendor their own BasicSR
packages.  Loading only their standalone architecture files keeps those copies
out of ``sys.path`` and avoids colliding with Real-ESRGAN and CodeFormer.
"""

from __future__ import annotations

import importlib.util
import logging
import math
import runpy
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from .deblur import DeblurEngine

LOGGER = logging.getLogger(__name__)

_AI_METHODS = {"ai", "ai_auto", "evssm", "restormer"}
_TRADITIONAL_METHODS = {"auto", "unsharp", "wiener"}


def _resolve_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else project_root / path


def _checkpoint_state(checkpoint: Any) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise RuntimeError("The model checkpoint is not a state-dict mapping.")
    state: Any = checkpoint
    for key in ("params_ema", "params", "state_dict"):
        candidate = checkpoint.get(key)
        if isinstance(candidate, dict):
            state = candidate
            break
    if not isinstance(state, dict) or not state:
        raise RuntimeError("No model parameters were found in the checkpoint.")
    cleaned: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if torch.is_tensor(value):
            cleaned[str(key).removeprefix("module.")] = value
    if not cleaned:
        raise RuntimeError("The checkpoint does not contain tensor parameters.")
    return cleaned


def _safe_torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - compatibility with older supported torch
        return torch.load(path, map_location="cpu")


class HybridDeblurEngine:
    """Route light blur to OpenCV and real blur to EVSSM/Restormer.

    Public ``enhance`` and ``measure_blur`` signatures deliberately match the
    original :class:`DeblurEngine`, so the pipeline and third-party extensions
    remain backwards compatible.
    """

    LABELS = {
        "traditional": "Traditional",
        "evssm": "EVSSM (RealBlur-J)",
        "restormer": "Restormer (defocus)",
        "unchanged": "Skipped (already sharp)",
    }

    def __init__(
        self,
        config: dict[str, Any] | None,
        device: torch.device,
        project_root: str | Path,
    ) -> None:
        self.config = dict(config or {})
        self.device = device
        self.project_root = Path(project_root).resolve()
        self.traditional = DeblurEngine(self.config)
        self._models: dict[str, torch.nn.Module] = {}
        self._dtypes: dict[str, torch.dtype] = {}
        self._load_errors: dict[str, str] = {}
        self.last_backend = "traditional"
        self.last_backend_label = self.LABELS[self.last_backend]

    @property
    def ready(self) -> bool:
        return bool(self._models)

    def measure_blur(self, image: np.ndarray) -> float:
        return self.traditional.measure_blur(image)

    def backend_available(self, backend: str) -> tuple[bool, str]:
        key = backend.strip().lower()
        if key not in {"evssm", "restormer"}:
            return False, f"Unknown AI deblur backend: {backend}"
        section = self.config.get(key, {})
        arch = _resolve_path(self.project_root, section.get("architecture", ""))
        weights = _resolve_path(self.project_root, section.get("weights", ""))
        missing = [str(path) for path in (arch, weights) if not path.is_file()]
        if missing:
            return False, "missing " + ", ".join(missing)
        if importlib.util.find_spec("einops") is None:
            return False, "Python package 'einops' is not installed"
        if key == "evssm":
            if self.device.type != "cuda":
                return False, "EVSSM requires a CUDA device"
            if importlib.util.find_spec("mamba_ssm") is None:
                return False, "Python package 'mamba_ssm' is not installed"
        return True, "ready" if key in self._models else "standby"

    def status(self) -> str:
        parts = []
        for backend in ("evssm", "restormer"):
            available, detail = self.backend_available(backend)
            if backend in self._models:
                detail = "loaded"
            elif not available and backend in self._load_errors:
                detail = "error"
            parts.append(f"{backend.upper()}={detail}")
        return ", ".join(parts)

    def load(self, backend: str | None = None) -> None:
        selected = (backend or self.config.get("default_ai_backend", "evssm")).lower()
        if selected not in {"evssm", "restormer"}:
            raise ValueError(f"Unknown AI deblur backend: {selected}")
        if selected in self._models:
            return
        available, detail = self.backend_available(selected)
        if not available:
            raise RuntimeError(
                f"{selected.upper()} is not available: {detail}. "
                "Run `python scripts/prepare_ai_models.py --models deblur --install-deps`."
            )
        try:
            if selected == "evssm":
                model, dtype = self._load_evssm()
            else:
                model, dtype = self._load_restormer()
        except Exception as exc:
            self._load_errors[selected] = str(exc)
            raise RuntimeError(f"Could not load {selected.upper()}: {exc}") from exc
        self._models[selected] = model
        self._dtypes[selected] = dtype
        self._load_errors.pop(selected, None)
        LOGGER.info("AI deblur backend %s loaded on %s (%s)", selected, self.device, dtype)

    def _load_evssm(self) -> tuple[torch.nn.Module, torch.dtype]:
        section = self.config["evssm"]
        namespace = runpy.run_path(
            str(_resolve_path(self.project_root, section["architecture"]))
        )
        model_class = namespace.get("EVSSM")
        if model_class is None:
            raise RuntimeError("EVSSM architecture file does not define EVSSM.")
        model = model_class()
        weights = _resolve_path(self.project_root, section["weights"])
        model.load_state_dict(_checkpoint_state(_safe_torch_load(weights)), strict=True)
        # EVSSM's selective-scan and FFT path intentionally contains FP32
        # operations.  FP32 is the safe default even when the rest of the app
        # uses half precision.
        dtype = torch.float16 if bool(section.get("half", False)) else torch.float32
        model = model.eval().requires_grad_(False).to(self.device, dtype=dtype)
        return model, dtype

    def _load_restormer(self) -> tuple[torch.nn.Module, torch.dtype]:
        section = self.config["restormer"]
        namespace = runpy.run_path(
            str(_resolve_path(self.project_root, section["architecture"]))
        )
        model_class = namespace.get("Restormer")
        if model_class is None:
            raise RuntimeError("Restormer architecture file does not define Restormer.")
        model = model_class(
            inp_channels=3,
            out_channels=3,
            dim=48,
            num_blocks=[4, 6, 6, 8],
            num_refinement_blocks=4,
            heads=[1, 2, 4, 8],
            ffn_expansion_factor=2.66,
            bias=False,
            LayerNorm_type="WithBias",
            dual_pixel_task=False,
        )
        weights = _resolve_path(self.project_root, section["weights"])
        model.load_state_dict(_checkpoint_state(_safe_torch_load(weights)), strict=True)
        use_half = self.device.type == "cuda" and bool(section.get("half", True))
        dtype = torch.float16 if use_half else torch.float32
        model = model.eval().requires_grad_(False).to(self.device, dtype=dtype)
        return model, dtype

    def enhance(
        self,
        image_bgr: np.ndarray,
        strength: float = 0.5,
        method: str = "auto",
    ) -> np.ndarray:
        if not isinstance(method, str):
            raise TypeError("method must be a string.")
        key = method.strip().lower().replace("-", "_")
        if key in _TRADITIONAL_METHODS:
            self._set_last_backend("traditional")
            return self.traditional.enhance(image_bgr, strength=strength, method=key)
        if key not in _AI_METHODS:
            choices = ", ".join(sorted(_AI_METHODS | _TRADITIONAL_METHODS))
            raise ValueError(f"Unknown deblur method {method!r}; choose one of: {choices}.")
        if not isinstance(image_bgr, np.ndarray) or image_bgr.dtype != np.uint8:
            raise TypeError("AI deblur input must be a uint8 numpy.ndarray.")
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3 or image_bgr.size == 0:
            raise ValueError(f"AI deblur expects a non-empty HxWx3 image, got {image_bgr.shape}.")
        if isinstance(strength, bool) or not isinstance(strength, (int, float)):
            raise TypeError("strength must be a real number.")
        amount = float(strength)
        if not math.isfinite(amount) or not 0.0 <= amount <= 1.0:
            raise ValueError("strength must be between 0 and 1.")
        if amount == 0.0:
            self._set_last_backend("unchanged")
            return np.ascontiguousarray(image_bgr.copy())

        automatic = key in {"ai", "ai_auto"}
        selected = (
            str(self.config.get("default_ai_backend", "evssm")).lower()
            if automatic
            else key
        )
        if automatic and bool(self.config.get("skip_ai_when_sharp", True)):
            score = self.measure_blur(image_bgr)
            if score >= self.traditional.auto_sharp_threshold:
                self._set_last_backend("unchanged")
                return np.ascontiguousarray(image_bgr.copy())

        try:
            restored = self._enhance_ai(image_bgr, selected)
            self._set_last_backend(selected)
        except Exception as exc:
            if not automatic or not bool(self.config.get("fallback_to_traditional", True)):
                raise
            LOGGER.warning("AI deblur failed; using traditional Auto fallback: %s", exc)
            restored = self.traditional.enhance(image_bgr, strength=amount, method="auto")
            self._set_last_backend("traditional")
            return restored

        if amount < 1.0:
            restored = cv2.addWeighted(
                image_bgr, 1.0 - amount, restored, amount, 0.0
            )
        return np.ascontiguousarray(restored)

    def _set_last_backend(self, backend: str) -> None:
        self.last_backend = backend
        self.last_backend_label = self.LABELS.get(backend, backend)

    def _enhance_ai(self, image_bgr: np.ndarray, backend: str) -> np.ndarray:
        self.load(backend)
        section = self.config[backend]
        configured_tile = int(section.get("tile", 384))
        if configured_tile < 64:
            raise ValueError(f"{backend}.tile must be at least 64.")
        candidates = []
        for tile in (configured_tile, 384, 256, 192, 128):
            if tile >= 64 and tile not in candidates:
                candidates.append(tile)
        last_oom: BaseException | None = None
        for tile in candidates:
            try:
                return self._run_tiled(image_bgr, backend, tile)
            except torch.cuda.OutOfMemoryError as exc:
                last_oom = exc
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                LOGGER.warning("%s OOM at tile=%d; retrying smaller tiles", backend, tile)
        raise RuntimeError(
            f"Insufficient GPU memory for {backend.upper()} even with a 128px tile."
        ) from last_oom

    @torch.inference_mode()
    def _run_tiled(self, image_bgr: np.ndarray, backend: str, tile: int) -> np.ndarray:
        model = self._models[backend]
        dtype = self._dtypes[backend]
        section = self.config[backend]
        context = max(0, int(section.get("tile_overlap", 64)))
        height, width = image_bgr.shape[:2]
        output = np.empty_like(image_bgr)

        for top in range(0, height, tile):
            bottom = min(top + tile, height)
            source_top = max(0, top - context)
            source_bottom = min(height, bottom + context)
            for left in range(0, width, tile):
                right = min(left + tile, width)
                source_left = max(0, left - context)
                source_right = min(width, right + context)
                patch = image_bgr[source_top:source_bottom, source_left:source_right]
                prediction = self._predict_patch(patch, model, dtype)
                crop_top = top - source_top
                crop_left = left - source_left
                output[top:bottom, left:right] = prediction[
                    crop_top : crop_top + (bottom - top),
                    crop_left : crop_left + (right - left),
                ]
        return np.ascontiguousarray(output)

    def _predict_patch(
        self,
        patch_bgr: np.ndarray,
        model: torch.nn.Module,
        dtype: torch.dtype,
    ) -> np.ndarray:
        height, width = patch_bgr.shape[:2]
        rgb = patch_bgr[:, :, ::-1].copy()
        tensor = (
            torch.from_numpy(rgb)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(self.device, dtype=dtype)
            .div_(255.0)
        )
        multiple = 8
        pad_bottom = (multiple - height % multiple) % multiple
        pad_right = (multiple - width % multiple) % multiple
        if pad_bottom or pad_right:
            mode = "reflect" if min(height, width) > max(pad_bottom, pad_right) else "replicate"
            tensor = torch.nn.functional.pad(
                tensor, (0, pad_right, 0, pad_bottom), mode=mode
            )
        prediction = model(tensor)
        if isinstance(prediction, (tuple, list)):
            prediction = prediction[0]
        prediction = prediction[:, :, :height, :width].float().clamp_(0.0, 1.0)
        rgb_out = prediction[0].permute(1, 2, 0).cpu().numpy()
        bgr_out = np.rint(rgb_out[:, :, ::-1] * 255.0).clip(0, 255).astype(np.uint8)
        return np.ascontiguousarray(bgr_out)
