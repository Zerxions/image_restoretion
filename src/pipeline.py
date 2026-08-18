from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import numpy as np
import torch
import yaml

from .advanced_super_resolution import SuperResolutionRouter
from .ai_deblur import HybridDeblurEngine
from .face_detector import FaceDetector
from .face_restore import FaceRestorationEngine
from .image_utils import ensure_max_pixels, load_image

LOGGER = logging.getLogger(__name__)


@dataclass
class RestorationResult:
    image_bgr: np.ndarray
    message: str
    face_count: int
    pipeline: str
    blur_score: float | None = None
    deblur_applied: bool = False


class ImageRestorationPipeline:
    """Extensible orchestration point for image (and future video) restorers."""

    MODES = {"Auto": "auto", "Portrait": "portrait", "Landscape": "landscape"}
    TASKS = {
        "Restore": "restore",
        "Smart": "smart",
        "Deblur": "deblur",
        "SuperResolution": "super_resolution",
        "Deblur+SR": "deblur_sr",
    }
    TASK_ALIASES = {
        "restore": "restore",
        "smart": "smart",
        "deblur": "deblur",
        "superresolution": "super_resolution",
        "super-resolution": "super_resolution",
        "super_resolution": "super_resolution",
        "deblur+sr": "deblur_sr",
        "deblur_sr": "deblur_sr",
    }

    def __init__(self, config_path: str | Path = "configs/config.yaml"):
        config_path = Path(config_path).expanduser().resolve()
        self.project_root = config_path.parent.parent
        with config_path.open("r", encoding="utf-8") as stream:
            self.config = yaml.safe_load(stream)
        preference = self.config.get("runtime", {}).get("device", "auto")
        use_cuda = torch.cuda.is_available() and preference != "cpu"
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.deblur = HybridDeblurEngine(
            self.config.get("deblur"), self.device, self.project_root
        )
        self.detector = FaceDetector(self.config["face_detector"], self.device)
        self.super_resolution = SuperResolutionRouter(
            self.config["realesrgan"],
            self.config.get("advanced_super_resolution"),
            self.device,
            self.project_root,
        )
        self.face_restoration = FaceRestorationEngine(
            self.config["face_restoration"],
            self.config["face_detector"],
            self.device,
            self.project_root,
        )
        self._lock = Lock()
        LOGGER.info("Pipeline initialized; inference device=%s", self.device)

    def status(self) -> str:
        gpu = torch.cuda.get_device_name(0) if self.device.type == "cuda" else "CPU fallback"
        states = (
            f"Deblur[{self.deblur.status()}], "
            f"SR[{self.super_resolution.status()}], "
            f"RetinaFace={'loaded' if self.detector.model is not None else 'standby'}, "
            f"Face restore={self.face_restoration.backend or 'standby'}"
        )
        return f"Device: {gpu} · {states} · Core models are cached"

    def warmup(self) -> str:
        """Preload the configured primary backends for deployment health checks."""
        self.deblur.load()
        self.super_resolution.load()
        self.detector.load()
        self.face_restoration.load()
        return f"Primary models loaded successfully on {self.device}"

    def restore(
        self,
        image,
        mode: str = "Auto",
        scale: int = 2,
        face_strength: float = 0.7,
        progress: Callable[[float, str], None] | None = None,
        *,
        task: str = "Restore",
        deblur_strength: float = 0.5,
        deblur_method: str = "auto",
        sr_backend: str = "auto",
    ) -> RestorationResult:
        if not isinstance(mode, str):
            raise TypeError("Restoration mode must be a string.")
        mode_key = self.MODES.get(mode, mode.lower())
        if mode_key not in self.MODES.values():
            raise ValueError(f"Unknown restoration mode: {mode}")
        if not isinstance(task, str):
            raise TypeError("Restoration task must be a string.")
        task_key = self.TASKS.get(task, self.TASK_ALIASES.get(task.strip().lower()))
        if task_key not in self.TASKS.values():
            raise ValueError(f"Unknown restoration task: {task}")

        uses_deblur = task_key in {"smart", "deblur", "deblur_sr"}
        uses_super_resolution = task_key in {
            "restore", "smart", "super_resolution", "deblur_sr",
        }
        uses_face_routing = task_key in {"restore", "smart"}
        if uses_super_resolution and int(scale) not in (2, 4):
            raise ValueError("Scale must be 2 or 4 when super-resolution is enabled.")

        image_bgr = load_image(image)
        ensure_max_pixels(
            image_bgr,
            int(self.config["runtime"].get("max_input_pixels", 24_000_000)),
        )
        LOGGER.info("Input image size: %dx%d", image_bgr.shape[1], image_bgr.shape[0])

        if uses_super_resolution:
            output_pixels = int(image_bgr.shape[0] * image_bgr.shape[1] * int(scale) ** 2)
            max_output_pixels = int(
                self.config["runtime"].get("max_output_pixels", 100_000_000)
            )
            if output_pixels > max_output_pixels:
                raise ValueError(
                    f"Requested output would contain {output_pixels:,} pixels; "
                    f"configured limit is {max_output_pixels:,}. Choose 2x, resize "
                    "the input, or increase runtime.max_output_pixels."
                )

        def report(value: float, text: str) -> None:
            LOGGER.info(text)
            if progress:
                progress(value, text)

        enhanced = image_bgr
        stages: list[str] = []
        blur_score: float | None = None
        deblur_applied = False
        restored_count = 0

        with self._lock:
            if uses_deblur:
                report(0.1, "Analyzing blur")
                blur_score = self.deblur.measure_blur(enhanced)
                report(0.22, f"Deblurring ({deblur_method})")
                deblurred = self.deblur.enhance(
                    enhanced,
                    strength=float(deblur_strength),
                    method=deblur_method,
                )
                deblur_applied = not np.array_equal(deblurred, enhanced)
                enhanced = deblurred
                if deblur_applied:
                    backend_label = getattr(
                        self.deblur, "last_backend_label", deblur_method
                    )
                    if deblur_method in {"auto", "unsharp", "wiener"}:
                        backend_label = deblur_method
                    stages.append(f"Deblur ({backend_label})")
                else:
                    LOGGER.info(
                        "Deblur produced no change (sharpness=%.2f, strength=%s)",
                        blur_score,
                        deblur_strength,
                    )

            faces = []
            if uses_face_routing and mode_key != "landscape":
                report(0.4, "Detecting faces")
                faces = self.detector.detect(enhanced)
                LOGGER.info("Detected faces: %d", len(faces))
            use_faces = uses_face_routing and (
                mode_key == "portrait" or (mode_key == "auto" and bool(faces))
            )

            if uses_super_resolution:
                report(0.6, f"Running {sr_backend} {scale}x super-resolution")
                enhance_with_backend = getattr(
                    self.super_resolution, "enhance_with_backend", None
                )
                if callable(enhance_with_backend):
                    enhanced = enhance_with_backend(
                        enhanced, int(scale), sr_backend
                    )
                else:
                    enhanced = self.super_resolution.enhance(enhanced, int(scale))
                sr_label = getattr(
                    self.super_resolution, "last_backend_label", "Real-ESRGAN"
                )
                stages.append(f"{sr_label} {scale}x")

            restored_count = 0
            if use_faces:
                report(0.82, "Restoring and blending faces")
                enhanced, restored_count = self.face_restoration.restore(
                    enhanced,
                    float(face_strength),
                )
                if restored_count:
                    backend = self.face_restoration.backend or "face restoration"
                    stage = (
                        str(backend).upper()
                        if backend == "gfpgan"
                        else "RestoreFormer++"
                        if backend == "restoreformer"
                        else str(backend).title()
                    )
                    stages.append(stage)
            report(1.0, "Restoration complete")

        chosen = " → ".join(stages) if stages else "Input unchanged"
        blur_summary = f" · sharpness: {blur_score:.1f}" if blur_score is not None else ""
        message = (
            f"Complete · {image_bgr.shape[1]}×{image_bgr.shape[0]} → "
            f"{enhanced.shape[1]}×{enhanced.shape[0]} · faces: {restored_count}"
            f"{blur_summary} · {chosen}"
        )
        return RestorationResult(
            enhanced,
            message,
            restored_count,
            chosen,
            blur_score=blur_score,
            deblur_applied=deblur_applied,
        )

    def register_extension(self, name: str, engine: object) -> None:
        """Reserved hook for SUPIR, DiffBIR, inpainting, video and TensorRT engines."""
        if not hasattr(self, "extensions"):
            self.extensions: dict[str, object] = {}
        self.extensions[name] = engine
