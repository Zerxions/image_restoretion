from __future__ import annotations

import sys
import unittest
from pathlib import Path
from threading import Lock

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import ImageRestorationPipeline  # noqa: E402


class FakeDeblur:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def measure_blur(self, image: np.ndarray) -> float:
        self.calls.append("measure")
        return 12.5

    def enhance(self, image: np.ndarray, strength: float, method: str) -> np.ndarray:
        self.calls.append("deblur")
        return np.clip(image.astype(np.int16) + 1, 0, 255).astype(np.uint8)


class FakeDetector:
    def __init__(self, calls: list[str], has_faces: bool = True) -> None:
        self.calls = calls
        self.has_faces = has_faces
        self.model = object()

    def detect(self, image: np.ndarray) -> list[list[float]]:
        self.calls.append("detect")
        return [[0.0, 0.0, 5.0, 5.0, 0.99]] if self.has_faces else []


class FakeSuperResolution:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls
        self.ready = True

    def enhance(self, image: np.ndarray, scale: int) -> np.ndarray:
        self.calls.append("sr")
        return np.repeat(np.repeat(image, scale, axis=0), scale, axis=1)


class FakeFaceRestoration:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls
        self.model = object()
        self.backend = "codeformer"

    def restore(self, image: np.ndarray, strength: float) -> tuple[np.ndarray, int]:
        self.calls.append("face")
        return image.copy(), 1


def make_pipeline(
    *,
    has_faces: bool = True,
    max_input_pixels: int = 1_000_000,
    max_output_pixels: int = 1_000_000,
) -> tuple[ImageRestorationPipeline, list[str]]:
    """Build a pipeline shell whose engines never load model weights."""
    calls: list[str] = []
    pipeline = ImageRestorationPipeline.__new__(ImageRestorationPipeline)
    pipeline.config = {
        "runtime": {
            "max_input_pixels": max_input_pixels,
            "max_output_pixels": max_output_pixels,
        }
    }
    pipeline.device = torch.device("cpu")
    pipeline.deblur = FakeDeblur(calls)
    pipeline.detector = FakeDetector(calls, has_faces=has_faces)
    pipeline.super_resolution = FakeSuperResolution(calls)
    pipeline.face_restoration = FakeFaceRestoration(calls)
    pipeline._lock = Lock()
    return pipeline, calls


class PipelineRoutingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.image = np.full((6, 8, 3), 40, dtype=np.uint8)

    def test_restore_routes_detect_then_sr_then_face(self) -> None:
        pipeline, calls = make_pipeline()

        result = pipeline.restore(self.image, mode="Auto", scale=2, task="Restore")

        self.assertEqual(calls, ["detect", "sr", "face"])
        self.assertEqual(result.image_bgr.shape, (12, 16, 3))
        self.assertEqual(result.face_count, 1)
        self.assertIsNone(result.blur_score)
        self.assertFalse(result.deblur_applied)

    def test_smart_routes_deblur_detect_sr_then_face(self) -> None:
        pipeline, calls = make_pipeline()

        result = pipeline.restore(
            self.image,
            mode="Auto",
            scale=4,
            task="Smart",
            deblur_strength=0.6,
            deblur_method="unsharp",
        )

        self.assertEqual(calls, ["measure", "deblur", "detect", "sr", "face"])
        self.assertEqual(result.image_bgr.shape, (24, 32, 3))
        self.assertEqual(result.blur_score, 12.5)
        self.assertTrue(result.deblur_applied)
        self.assertIn("Deblur", result.pipeline)
        self.assertIn("Real-ESRGAN 4x", result.pipeline)

    def test_deblur_only_keeps_original_dimensions_and_skips_face_models(self) -> None:
        pipeline, calls = make_pipeline()

        result = pipeline.restore(
            self.image,
            mode="Portrait",
            scale=99,
            task="Deblur",
            deblur_strength=0.7,
        )

        self.assertEqual(calls, ["measure", "deblur"])
        self.assertEqual(result.image_bgr.shape, self.image.shape)
        self.assertEqual(result.face_count, 0)
        self.assertTrue(result.deblur_applied)

    def test_super_resolution_only_skips_deblur_and_face_models(self) -> None:
        pipeline, calls = make_pipeline()

        result = pipeline.restore(
            self.image,
            mode="Portrait",
            scale=2,
            task="SuperResolution",
        )

        self.assertEqual(calls, ["sr"])
        self.assertEqual(result.image_bgr.shape, (12, 16, 3))
        self.assertEqual(result.pipeline, "Real-ESRGAN 2x")

    def test_deblur_plus_sr_runs_in_order_and_skips_face_models(self) -> None:
        pipeline, calls = make_pipeline()

        result = pipeline.restore(
            self.image,
            mode="Auto",
            scale=2,
            task="Deblur+SR",
            deblur_method="wiener",
        )

        self.assertEqual(calls, ["measure", "deblur", "sr"])
        self.assertEqual(result.image_bgr.shape, (12, 16, 3))
        self.assertEqual(result.face_count, 0)
        self.assertIn("Deblur (wiener)", result.pipeline)
        self.assertTrue(result.pipeline.index("Deblur") < result.pipeline.index("Real-ESRGAN"))

    def test_output_pixel_limit_fails_before_any_engine_runs(self) -> None:
        pipeline, calls = make_pipeline(max_output_pixels=191)

        with self.assertRaisesRegex(ValueError, "Requested output would contain 192 pixels"):
            pipeline.restore(self.image, scale=2, task="SuperResolution")

        self.assertEqual(calls, [])

    def test_legacy_positional_call_keeps_original_restore_behavior(self) -> None:
        pipeline, calls = make_pipeline()

        result = pipeline.restore(self.image, "Landscape", 2, 0.7)

        self.assertEqual(calls, ["sr"])
        self.assertEqual(result.image_bgr.shape, (12, 16, 3))
        self.assertEqual(result.pipeline, "Real-ESRGAN 2x")

    def test_auto_without_faces_skips_face_restoration(self) -> None:
        pipeline, calls = make_pipeline(has_faces=False)

        result = pipeline.restore(self.image, mode="Auto", scale=2, task="Restore")

        self.assertEqual(calls, ["detect", "sr"])
        self.assertEqual(result.face_count, 0)


if __name__ == "__main__":
    unittest.main()
