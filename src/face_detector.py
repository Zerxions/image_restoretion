from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

LOGGER = logging.getLogger(__name__)


class FaceDetector:
    """RetinaFace detector backed by facexlib."""

    def __init__(self, config: dict, device: torch.device):
        self.config = config
        self.device = device
        self.model = None

    def load(self) -> None:
        if self.model is not None:
            return
        try:
            from facexlib.detection import init_detection_model
        except Exception as exc:
            raise RuntimeError("facexlib is not installed correctly.") from exc

        model_name = self.config.get("model", "retinaface_resnet50")
        model_path = self.config.get("weights")
        if model_path and not Path(model_path).is_file():
            raise FileNotFoundError(f"Face detector weights not found: {model_path}")
        kwargs = {"device": self.device}
        if model_path:
            kwargs["model_path"] = model_path
        self.model = init_detection_model(model_name, **kwargs)
        LOGGER.info("Face detector %s loaded successfully", model_name)

    @torch.inference_mode()
    def detect(self, image_bgr: np.ndarray) -> list[list[float]]:
        self.load()
        detections = self.model.detect_faces(
            image_bgr, float(self.config.get("confidence_threshold", 0.8))
        )
        if detections is None:
            return []
        if torch.is_tensor(detections):
            detections = detections.detach().cpu().numpy()
        return [row[:5].astype(float).tolist() for row in np.asarray(detections)]
