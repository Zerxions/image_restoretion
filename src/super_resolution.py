from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from .compat import ensure_basicsr_torchvision_compat

LOGGER = logging.getLogger(__name__)


class SuperResolutionEngine:
    """Real-ESRGAN wrapper with lazy, single-instance model loading."""

    def __init__(self, config: dict, device: torch.device):
        self.config = config
        self.device = device
        self.upsampler = None

    @property
    def ready(self) -> bool:
        return self.upsampler is not None

    def load(self) -> None:
        if self.ready:
            return
        model_path = Path(self.config["weights"])
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Real-ESRGAN weights not found: {model_path}. See weights/README.md."
            )
        try:
            ensure_basicsr_torchvision_compat()
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
        except Exception as exc:
            raise RuntimeError(
                "Could not import Real-ESRGAN. Install requirements and check the "
                "basicsr/torchvision compatibility notes in README.md."
            ) from exc

        model = RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64,
            num_block=23, num_grow_ch=32, scale=4,
        )
        self.upsampler = RealESRGANer(
            scale=4,
            model_path=str(model_path),
            model=model,
            tile=int(self.config.get("tile", 0)),
            tile_pad=int(self.config.get("tile_pad", 10)),
            pre_pad=int(self.config.get("pre_pad", 0)),
            half=self.device.type == "cuda" and bool(self.config.get("half", True)),
            device=self.device,
        )
        LOGGER.info("Real-ESRGAN loaded successfully on %s", self.device)

    def enhance(self, image_bgr: np.ndarray, scale: int) -> np.ndarray:
        if scale not in (2, 4):
            raise ValueError("Scale must be 2 or 4.")
        self.load()
        try:
            output, _ = self.upsampler.enhance(image_bgr, outscale=scale)
            return output
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            raise RuntimeError(
                "Insufficient GPU memory during super-resolution. Enable tiling, "
                "reduce tile size, choose 2x, or use a smaller image."
            ) from exc
