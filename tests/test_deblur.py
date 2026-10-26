from __future__ import annotations

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.deblur import DeblurEngine, measure_blur  # noqa: E402


class DeblurEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = DeblurEngine({"tile_size": 128})

    def test_supported_inputs_keep_shape_and_uint8_dtype(self) -> None:
        images = (
            np.arange(32 * 24, dtype=np.uint8).reshape(32, 24),
            np.arange(32 * 24, dtype=np.uint8).reshape(32, 24, 1),
            np.zeros((32, 24, 3), dtype=np.uint8),
            np.zeros((32, 24, 4), dtype=np.uint8),
        )

        for image in images:
            with self.subTest(shape=image.shape):
                output = self.engine.enhance(image, strength=0.25, method="unsharp")
                self.assertEqual(output.shape, image.shape)
                self.assertEqual(output.dtype, np.uint8)
                self.assertTrue(output.flags.c_contiguous)
                self.assertGreaterEqual(measure_blur(image), 0.0)

    def test_invalid_inputs_are_rejected(self) -> None:
        invalid_cases = (
            ([1, 2, 3], TypeError),
            (np.zeros((8, 8), dtype=np.float32), TypeError),
            (np.zeros((0, 8, 3), dtype=np.uint8), ValueError),
            (np.zeros((8, 8, 2), dtype=np.uint8), ValueError),
            (np.zeros((1, 2, 3, 4), dtype=np.uint8), ValueError),
        )

        for image, error_type in invalid_cases:
            with self.subTest(value_type=type(image).__name__, shape=getattr(image, "shape", None)):
                with self.assertRaises(error_type):
                    self.engine.enhance(image)

    def test_alpha_channel_is_preserved_exactly(self) -> None:
        rng = np.random.default_rng(20260818)
        image = rng.integers(0, 256, size=(64, 80, 4), dtype=np.uint8)
        image[:, :, 3] = np.arange(80, dtype=np.uint8)[None, :]

        output = self.engine.enhance(image, strength=0.8, method="unsharp")

        np.testing.assert_array_equal(output[:, :, 3], image[:, :, 3])
        self.assertTrue(np.any(output[:, :, :3] != image[:, :, :3]))

    def test_zero_strength_returns_exact_contiguous_copy(self) -> None:
        source = np.arange(48 * 64 * 3, dtype=np.uint8).reshape(48, 64, 3)
        non_contiguous = source[:, ::2]
        self.assertFalse(non_contiguous.flags.c_contiguous)

        output = self.engine.enhance(non_contiguous, strength=0.0, method="wiener")

        np.testing.assert_array_equal(output, non_contiguous)
        self.assertTrue(output.flags.c_contiguous)
        self.assertFalse(np.shares_memory(output, non_contiguous))

    def test_auto_leaves_an_already_clear_image_unchanged(self) -> None:
        checkerboard = ((np.indices((64, 64)).sum(axis=0) % 2) * 255).astype(np.uint8)
        clear = np.repeat(checkerboard[:, :, None], 3, axis=2)
        self.assertGreaterEqual(measure_blur(clear), self.engine.auto_sharp_threshold)

        output = self.engine.enhance(clear, strength=1.0, method="auto")

        np.testing.assert_array_equal(output, clear)
        self.assertIsNot(output, clear)

    def test_unsharp_improves_sharpness_of_synthetic_blur(self) -> None:
        sharp = np.zeros((128, 128, 3), dtype=np.uint8)
        sharp[:, 64:] = 255
        blurred = cv2.GaussianBlur(sharp, (0, 0), sigmaX=3.0, sigmaY=3.0)
        before = measure_blur(blurred)

        restored = self.engine.enhance(blurred, strength=1.0, method="unsharp")
        after = measure_blur(restored)

        self.assertEqual(restored.shape, blurred.shape)
        self.assertGreater(after, before * 1.5)


if __name__ == "__main__":
    unittest.main()
