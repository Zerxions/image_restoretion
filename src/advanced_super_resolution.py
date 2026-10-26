"""Super-resolution routing for faithful DRCT and stable Real-ESRGAN."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from .super_resolution import SuperResolutionEngine

LOGGER = logging.getLogger(__name__)


def _resolve_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else project_root / path


def _safe_torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover
        return torch.load(path, map_location="cpu")


def _checkpoint_state(checkpoint: Any) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise RuntimeError("The DRCT checkpoint is not a mapping.")
    state: Any = checkpoint
    for key in ("params_ema", "params", "state_dict"):
        if isinstance(checkpoint.get(key), dict):
            state = checkpoint[key]
            break
    cleaned = {
        str(key).removeprefix("module."): value
        for key, value in state.items()
        if torch.is_tensor(value)
    }
    if not cleaned:
        raise RuntimeError("No tensor parameters were found in the DRCT checkpoint.")
    return cleaned


class DRCTSuperResolutionEngine:
    """Real-DRCT MSE x4 with CPU-output context tiling for 8GB GPUs."""

    EXPECTED_PARAMETERS = 14_139_579

    def __init__(
        self,
        config: dict[str, Any],
        device: torch.device,
        project_root: str | Path,
    ) -> None:
        self.config = dict(config)
        self.device = device
        self.project_root = Path(project_root).resolve()
        self.model: torch.nn.Module | None = None
        self.dtype = torch.float32

    @property
    def ready(self) -> bool:
        return self.model is not None

    def availability(self) -> tuple[bool, str]:
        architecture = _resolve_path(
            self.project_root, self.config.get("architecture", "")
        )
        weights = _resolve_path(self.project_root, self.config.get("weights", ""))
        missing = [str(path) for path in (architecture, weights) if not path.is_file()]
        return (False, "missing " + ", ".join(missing)) if missing else (True, "standby")

    def _load_architecture(self, path: Path):
        # Keep the downloaded upstream file pristine.  These narrow runtime
        # substitutions remove unused BasicSR/Einops imports and fix DRCT's
        # dynamic FP16 attention-mask dtype on non-training image sizes.
        source = path.read_text(encoding="utf-8")
        replacements = {
            "import torch.utils.checkpoint as checkpoint\n": "",
            "from basicsr.utils.registry import ARCH_REGISTRY\n": "",
            "from basicsr.archs.arch_util import to_2tuple, trunc_normal_\n": "",
            "from einops import rearrange\n": "",
            "@ARCH_REGISTRY.register()\n": "",
            "self.calculate_mask(x_size).to(x.device)": (
                "self.calculate_mask(x_size).to(device=x.device, dtype=x.dtype)"
            ),
            "torch.meshgrid([coords_h, coords_w])": (
                "torch.meshgrid(coords_h, coords_w, indexing='ij')"
            ),
        }
        for old, new in replacements.items():
            if old not in source:
                raise RuntimeError(
                    f"Unsupported DRCT architecture revision; marker not found: {old!r}"
                )
            source = source.replace(old, new)
        helper = (
            "\ndef to_2tuple(value):\n"
            "    return value if isinstance(value, tuple) else (value, value)\n\n"
            "trunc_normal_ = torch.nn.init.trunc_normal_\n\n"
        )
        insertion = source.index("\ndef drop_path")
        source = source[:insertion] + helper + source[insertion:]
        namespace: dict[str, Any] = {
            "__file__": str(path),
            "__name__": "drct_arch_local",
        }
        exec(compile(source, str(path), "exec"), namespace)
        model_class = namespace.get("DRCT")
        if model_class is None:
            raise RuntimeError("DRCT architecture file does not define DRCT.")
        return model_class

    def load(self) -> None:
        if self.ready:
            return
        available, detail = self.availability()
        if not available:
            raise RuntimeError(
                f"DRCT is not available: {detail}. Run "
                "`python scripts/prepare_ai_models.py --models sr`."
            )
        architecture = _resolve_path(self.project_root, self.config["architecture"])
        weights = _resolve_path(self.project_root, self.config["weights"])
        model_class = self._load_architecture(architecture)
        model = model_class(
            upscale=4,
            in_chans=3,
            img_size=64,
            window_size=16,
            compress_ratio=3,
            squeeze_factor=30,
            conv_scale=0.01,
            overlap_ratio=0.5,
            img_range=1.0,
            depths=[6] * 6,
            embed_dim=180,
            num_heads=[6] * 6,
            gc=32,
            mlp_ratio=2,
            upsampler="pixelshuffle",
            resi_connection="1conv",
        )
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        if parameter_count != self.EXPECTED_PARAMETERS:
            raise RuntimeError(
                f"Unexpected DRCT architecture ({parameter_count:,} parameters; "
                f"expected {self.EXPECTED_PARAMETERS:,})."
            )
        model.load_state_dict(_checkpoint_state(_safe_torch_load(weights)), strict=True)
        use_half = self.device.type == "cuda" and bool(self.config.get("half", True))
        self.dtype = torch.float16 if use_half else torch.float32
        self.model = model.eval().requires_grad_(False).to(self.device, dtype=self.dtype)
        LOGGER.info("DRCT MSE x4 loaded on %s (%s)", self.device, self.dtype)

    def enhance(self, image_bgr: np.ndarray, scale: int) -> np.ndarray:
        if scale not in (2, 4):
            raise ValueError("Scale must be 2 or 4.")
        if not isinstance(image_bgr, np.ndarray) or image_bgr.dtype != np.uint8:
            raise TypeError("DRCT input must be a uint8 numpy.ndarray.")
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3 or image_bgr.size == 0:
            raise ValueError(f"DRCT expects a non-empty HxWx3 image, got {image_bgr.shape}.")
        self.load()
        configured = int(self.config.get("tile", 256))
        if configured < 64:
            raise ValueError("drct.tile must be at least 64.")
        candidates = []
        for tile in (configured, 256, 192, 128, 96, 64):
            tile = max(64, tile - tile % 16)
            if tile not in candidates:
                candidates.append(tile)
        last_oom: BaseException | None = None
        for tile in candidates:
            try:
                return self._run_tiled(image_bgr, scale, tile)
            except torch.cuda.OutOfMemoryError as exc:
                last_oom = exc
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                LOGGER.warning("DRCT OOM at tile=%d; retrying smaller tiles", tile)
        raise RuntimeError("Insufficient GPU memory for DRCT at the smallest tile.") from last_oom

    @torch.inference_mode()
    def _run_tiled(self, image_bgr: np.ndarray, scale: int, tile: int) -> np.ndarray:
        assert self.model is not None
        height, width = image_bgr.shape[:2]
        context = max(0, int(self.config.get("tile_pad", 32)))
        output = np.empty((height * scale, width * scale, 3), dtype=np.uint8)

        for top in range(0, height, tile):
            bottom = min(top + tile, height)
            source_top = max(0, top - context)
            source_bottom = min(height, bottom + context)
            for left in range(0, width, tile):
                right = min(left + tile, width)
                source_left = max(0, left - context)
                source_right = min(width, right + context)
                patch = image_bgr[source_top:source_bottom, source_left:source_right]
                predicted_x4 = self._predict_x4(patch)
                if scale == 2:
                    predicted = cv2.resize(
                        predicted_x4,
                        (patch.shape[1] * 2, patch.shape[0] * 2),
                        interpolation=cv2.INTER_LANCZOS4,
                    )
                else:
                    predicted = predicted_x4
                crop_top = (top - source_top) * scale
                crop_left = (left - source_left) * scale
                output[
                    top * scale : bottom * scale,
                    left * scale : right * scale,
                ] = predicted[
                    crop_top : crop_top + (bottom - top) * scale,
                    crop_left : crop_left + (right - left) * scale,
                ]
        return np.ascontiguousarray(output)

    def _predict_x4(self, patch_bgr: np.ndarray) -> np.ndarray:
        assert self.model is not None
        height, width = patch_bgr.shape[:2]
        target_height = max(16, math.ceil(height / 16) * 16)
        target_width = max(16, math.ceil(width / 16) * 16)
        tensor = (
            torch.from_numpy(patch_bgr[:, :, ::-1].copy())
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(self.device, dtype=self.dtype)
            .div_(255.0)
        )
        pad_bottom = target_height - height
        pad_right = target_width - width
        if pad_bottom or pad_right:
            mode = "reflect" if min(height, width) > max(pad_bottom, pad_right) else "replicate"
            tensor = torch.nn.functional.pad(
                tensor, (0, pad_right, 0, pad_bottom), mode=mode
            )
        prediction = self.model(tensor).float().clamp_(0.0, 1.0)
        prediction = prediction[:, :, : height * 4, : width * 4]
        rgb = prediction[0].permute(1, 2, 0).cpu().numpy()
        bgr = np.rint(rgb[:, :, ::-1] * 255.0).clip(0, 255).astype(np.uint8)
        return np.ascontiguousarray(bgr)


class SuperResolutionRouter:
    """Choose DRCT for fidelity, with the existing Real-ESRGAN as fallback."""

    LABELS = {"drct": "DRCT-MSE", "realesrgan": "Real-ESRGAN"}

    def __init__(
        self,
        realesrgan_config: dict[str, Any],
        advanced_config: dict[str, Any] | None,
        device: torch.device,
        project_root: str | Path,
    ) -> None:
        self.config = dict(advanced_config or {})
        self.device = device
        self.project_root = Path(project_root).resolve()
        stable_config = dict(realesrgan_config)
        stable_config["weights"] = str(
            _resolve_path(self.project_root, stable_config["weights"])
        )
        self.realesrgan = SuperResolutionEngine(stable_config, device)
        self.drct = DRCTSuperResolutionEngine(
            dict(self.config.get("drct", {})), device, self.project_root
        )
        self.last_backend = "realesrgan"
        self.last_backend_label = self.LABELS[self.last_backend]

    @property
    def ready(self) -> bool:
        return self.realesrgan.ready or self.drct.ready

    def status(self) -> str:
        available, detail = self.drct.availability()
        drct = "loaded" if self.drct.ready else detail
        realesrgan = "loaded" if self.realesrgan.ready else "standby"
        if not available and not self.drct.ready:
            drct = "missing"
        return f"DRCT={drct}, Real-ESRGAN={realesrgan}"

    def load(self) -> None:
        backend = str(self.config.get("default_backend", "drct")).lower()
        if backend == "drct":
            self.drct.load()
        elif backend == "realesrgan":
            self.realesrgan.load()
        else:
            raise ValueError(f"Unknown super-resolution backend: {backend}")

    def enhance(self, image_bgr: np.ndarray, scale: int) -> np.ndarray:
        return self.enhance_with_backend(image_bgr, scale, "auto")

    def enhance_with_backend(
        self, image_bgr: np.ndarray, scale: int, backend: str = "auto"
    ) -> np.ndarray:
        key = backend.strip().lower().replace("-", "")
        aliases = {
            "auto": "auto",
            "drct": "drct",
            "faithful": "drct",
            "realesrgan": "realesrgan",
            "real_esrgan": "realesrgan",
            "fast": "realesrgan",
        }
        selected = aliases.get(key)
        if selected is None:
            raise ValueError(f"Unknown super-resolution backend: {backend}")
        if selected == "auto":
            selected = str(self.config.get("default_backend", "drct")).lower()
            try:
                result = self._enhance_selected(image_bgr, scale, selected)
            except Exception as exc:
                if selected == "realesrgan" or not bool(
                    self.config.get("fallback_to_realesrgan", True)
                ):
                    raise
                LOGGER.warning("DRCT failed; using Real-ESRGAN fallback: %s", exc)
                selected = "realesrgan"
                result = self._enhance_selected(image_bgr, scale, selected)
        else:
            result = self._enhance_selected(image_bgr, scale, selected)
        self.last_backend = selected
        self.last_backend_label = self.LABELS[selected]
        return result

    def _enhance_selected(
        self, image_bgr: np.ndarray, scale: int, backend: str
    ) -> np.ndarray:
        if backend == "drct":
            return self.drct.enhance(image_bgr, scale)
        if backend == "realesrgan":
            return self.realesrgan.enhance(image_bgr, scale)
        raise ValueError(f"Unknown super-resolution backend: {backend}")
