from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.advanced_super_resolution import (  # noqa: E402
    DRCTSuperResolutionEngine,
    SuperResolutionRouter,
)
from src.ai_deblur import HybridDeblurEngine  # noqa: E402


class HybridDeblurTests(unittest.TestCase):
    def config(self) -> dict:
        return {
            "default_ai_backend": "evssm",
            "fallback_to_traditional": True,
            "skip_ai_when_sharp": False,
            "tile_size": 128,
            "auto_blur_threshold": 35.0,
            "auto_sharp_threshold": 180.0,
            "evssm": {
                "architecture": "missing/EVSSM.py",
                "weights": "missing/evssm.pth",
            },
            "restormer": {
                "architecture": "missing/restormer.py",
                "weights": "missing/restormer.pth",
            },
        }

    def test_ai_auto_falls_back_to_traditional_when_assets_are_missing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            engine = HybridDeblurEngine(self.config(), torch.device("cpu"), directory)
            image = np.zeros((48, 64, 3), dtype=np.uint8)
            image[:, 32:] = 255

            output = engine.enhance(image, strength=0.5, method="ai")

            self.assertEqual(output.shape, image.shape)
            self.assertEqual(output.dtype, np.uint8)
            self.assertEqual(engine.last_backend, "traditional")

    def test_explicit_ai_backend_reports_missing_assets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            engine = HybridDeblurEngine(self.config(), torch.device("cpu"), directory)
            image = np.zeros((16, 16, 3), dtype=np.uint8)

            with self.assertRaisesRegex(RuntimeError, "EVSSM is not available"):
                engine.enhance(image, strength=1.0, method="evssm")

    def test_ai_strength_blends_source_and_model_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            engine = HybridDeblurEngine(self.config(), torch.device("cpu"), directory)
            engine._enhance_ai = lambda image, backend: np.full_like(image, 200)  # type: ignore[method-assign]
            image = np.full((8, 10, 3), 40, dtype=np.uint8)

            output = engine.enhance(image, strength=0.25, method="restormer")

            np.testing.assert_array_equal(output, np.full_like(image, 80))
            self.assertEqual(engine.last_backend, "restormer")


class NearestX4(torch.nn.Module):
    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.repeat_interleave(4, dim=2).repeat_interleave(4, dim=3)


class DRCTTilingTests(unittest.TestCase):
    def make_engine(self) -> DRCTSuperResolutionEngine:
        engine = DRCTSuperResolutionEngine.__new__(DRCTSuperResolutionEngine)
        engine.config = {"tile_pad": 5}
        engine.device = torch.device("cpu")
        engine.project_root = PROJECT_ROOT
        engine.model = NearestX4().eval()
        engine.dtype = torch.float32
        return engine

    def test_odd_sized_image_has_exact_two_and_four_x_shapes(self) -> None:
        image = np.random.default_rng(7).integers(
            0, 256, size=(23, 31, 3), dtype=np.uint8
        )
        engine = self.make_engine()

        output_2x = engine._run_tiled(image, scale=2, tile=16)
        output_4x = engine._run_tiled(image, scale=4, tile=16)

        self.assertEqual(output_2x.shape, (46, 62, 3))
        self.assertEqual(output_4x.shape, (92, 124, 3))
        self.assertEqual(output_2x.dtype, np.uint8)
        self.assertTrue(output_4x.flags.c_contiguous)


class FakeSR:
    def __init__(self, value: int, error: Exception | None = None) -> None:
        self.value = value
        self.error = error
        self.ready = False

    def enhance(self, image: np.ndarray, scale: int) -> np.ndarray:
        if self.error:
            raise self.error
        return np.full((image.shape[0] * scale, image.shape[1] * scale, 3), self.value, np.uint8)


class SuperResolutionRouterTests(unittest.TestCase):
    def make_router(self) -> SuperResolutionRouter:
        router = SuperResolutionRouter.__new__(SuperResolutionRouter)
        router.config = {"default_backend": "drct", "fallback_to_realesrgan": True}
        router.drct = FakeSR(90)
        router.realesrgan = FakeSR(30)
        router.last_backend = "realesrgan"
        router.last_backend_label = "Real-ESRGAN"
        return router

    def test_explicit_backend_and_auto_fallback_are_reported(self) -> None:
        image = np.zeros((5, 7, 3), dtype=np.uint8)
        router = self.make_router()

        drct = router.enhance_with_backend(image, 2, "drct")
        self.assertEqual(int(drct[0, 0, 0]), 90)
        self.assertEqual(router.last_backend_label, "DRCT-MSE")

        router.drct = FakeSR(0, RuntimeError("not ready"))
        fallback = router.enhance_with_backend(image, 4, "auto")
        self.assertEqual(int(fallback[0, 0, 0]), 30)
        self.assertEqual(router.last_backend, "realesrgan")


if __name__ == "__main__":
    unittest.main()
