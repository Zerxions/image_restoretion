#!/usr/bin/env python3
"""Run small CUDA forward passes through every prepared AI backend."""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.advanced_super_resolution import DRCTSuperResolutionEngine  # noqa: E402
from src.ai_deblur import HybridDeblurEngine  # noqa: E402
from src.face_restore import FaceRestorationEngine  # noqa: E402


def release() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_model(name: str, operation):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    output = operation()
    elapsed = time.perf_counter() - started
    peak = (
        torch.cuda.max_memory_allocated() / 2**30
        if torch.cuda.is_available()
        else 0.0
    )
    if isinstance(output, tuple):
        output = output[0]
    print(
        f"[ok] {name}: shape={output.shape}, dtype={output.dtype}, "
        f"time={elapsed:.2f}s, peak_allocated={peak:.2f}GiB"
    )
    return output


def synthetic_image(height: int, width: int) -> np.ndarray:
    y, x = np.indices((height, width))
    image = np.stack(
        (
            (x * 255 // max(1, width - 1)),
            (y * 255 // max(1, height - 1)),
            ((x + y) * 255 // max(1, height + width - 2)),
        ),
        axis=2,
    ).astype(np.uint8)
    return cv2.GaussianBlur(image, (0, 0), 1.5)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        default="evssm,restormer,drct,face",
        help="Comma-separated: evssm,restormer,drct,face",
    )
    args = parser.parse_args()
    requested = {item.strip().lower() for item in args.models.split(",") if item.strip()}
    unknown = requested - {"evssm", "restormer", "drct", "face"}
    if unknown:
        raise ValueError(f"Unknown model(s): {', '.join(sorted(unknown))}")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available to this interpreter. Activate image_restore and "
            "verify WSL exposes /dev/dxg."
        )
    device = torch.device("cuda")
    print(f"CUDA: {torch.cuda.get_device_name(0)} · torch={torch.__version__}")
    with (PROJECT_ROOT / "configs/config.yaml").open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    deblur_input = synthetic_image(64, 96)
    if "evssm" in requested:
        engine = HybridDeblurEngine(config["deblur"], device, PROJECT_ROOT)
        run_model(
            "EVSSM RealBlur-J",
            lambda: engine.enhance(deblur_input, strength=1.0, method="evssm"),
        )
        del engine
        release()
    if "restormer" in requested:
        engine = HybridDeblurEngine(config["deblur"], device, PROJECT_ROOT)
        run_model(
            "Restormer defocus",
            lambda: engine.enhance(deblur_input, strength=1.0, method="restormer"),
        )
        del engine
        release()
    if "drct" in requested:
        engine = DRCTSuperResolutionEngine(
            config["advanced_super_resolution"]["drct"], device, PROJECT_ROOT
        )
        sr_input = synthetic_image(32, 48)
        run_model("DRCT-MSE 2x", lambda: engine.enhance(sr_input, 2))
        del engine
        release()
    if "face" in requested:
        engine = FaceRestorationEngine(
            config["face_restoration"], config["face_detector"], device, PROJECT_ROOT
        )
        engine.load()
        face_input = synthetic_image(512, 512)
        run_model(
            f"{engine.backend} face crop",
            lambda: engine._restore_crop(face_input, strength=1.0),
        )
        del engine
        release()
    print("All requested CUDA smoke tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
