from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Callable

import numpy as np
import torch
import yaml

from .face_detector import FaceDetector
from .face_restore import FaceRestorationEngine
from .image_utils import ensure_max_pixels, load_image
from .super_resolution import SuperResolutionEngine

LOGGER = logging.getLogger(__name__)


@dataclass
class RestorationResult:
    image_bgr: np.ndarray
    message: str
    face_count: int
    pipeline: str


class ImageRestorationPipeline:
    """Extensible orchestration point for image (and future video) restorers."""

    MODES = {"Auto": "auto", "Portrait": "portrait", "Landscape": "landscape"}

    def __init__(self, config_path: str | Path = "configs/config.yaml"):
        config_path = Path(config_path)
        with config_path.open("r", encoding="utf-8") as stream:
            self.config = yaml.safe_load(stream)
        preference = self.config.get("runtime", {}).get("device", "auto")
        use_cuda = torch.cuda.is_available() and preference != "cpu"
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.detector = FaceDetector(self.config["face_detector"], self.device)
        self.super_resolution = SuperResolutionEngine(self.config["realesrgan"], self.device)
        self.face_restoration = FaceRestorationEngine(
            self.config["face_restoration"], self.config["face_detector"], self.device
        )
        self._lock = Lock()
        LOGGER.info("Pipeline initialized; inference device=%s", self.device)

    def status(self) -> str:
        gpu = torch.cuda.get_device_name(0) if self.device.type == "cuda" else "CPU fallback"
        states = (
            f"Real-ESRGAN={'loaded' if self.super_resolution.ready else 'standby'}, "
            f"RetinaFace={'loaded' if self.detector.model is not None else 'standby'}, "
            f"Face restore={'loaded' if self.face_restoration.model is not None else 'standby'}"
        )
        return f"Device: {gpu} · {states} · Models load once"

    def warmup(self) -> str:
        """Optionally preload all models; useful for deployment health checks."""
        self.super_resolution.load()
        self.detector.load()
        self.face_restoration.load()
        return f"All models loaded successfully on {self.device}"

    def restore(
        self,
        image,
        mode: str = "Auto",
        scale: int = 2,
        face_strength: float = 0.7,
        hair_strength: float = 0.35,
        progress: Callable[[float, str], None] | None = None,
    ) -> RestorationResult:
        mode_key = self.MODES.get(mode, mode.lower())
        if mode_key not in self.MODES.values():
            raise ValueError(f"Unknown restoration mode: {mode}")
        image_bgr = load_image(image)
        ensure_max_pixels(image_bgr, int(self.config["runtime"].get("max_input_pixels", 24_000_000)))
        LOGGER.info("Input image size: %dx%d", image_bgr.shape[1], image_bgr.shape[0])

        def report(value: float, text: str) -> None:
            LOGGER.info(text)
            if progress:
                progress(value, text)

        with self._lock:
            report(0.1, "Detecting faces")
            faces = [] if mode_key == "landscape" else self.detector.detect(image_bgr)
            LOGGER.info("Detected faces: %d", len(faces))
            use_faces = mode_key == "portrait" or (mode_key == "auto" and bool(faces))
            chosen = "Real-ESRGAN + face restoration" if use_faces else "Real-ESRGAN"
            report(0.35, f"Running {chosen}")
            enhanced = self.super_resolution.enhance(image_bgr, int(scale))
            restored_count = 0
            if use_faces:
                report(0.7, "Restoring and blending faces")
                enhanced, restored_count = self.face_restoration.restore(
                    enhanced, float(face_strength), float(hair_strength)
                )
            report(1.0, "Restoration complete")
        message = (
            f"Complete · {image_bgr.shape[1]}×{image_bgr.shape[0]} → "
            f"{enhanced.shape[1]}×{enhanced.shape[0]} · faces: {restored_count} · {chosen}"
        )
        return RestorationResult(enhanced, message, restored_count, chosen)

    def register_extension(self, name: str, engine: object) -> None:
        """Reserved hook for SUPIR, DiffBIR, inpainting, video and TensorRT engines."""
        if not hasattr(self, "extensions"):
            self.extensions: dict[str, object] = {}
        self.extensions[name] = engine
