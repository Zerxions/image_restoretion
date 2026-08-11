from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.transforms.functional import normalize

from .compat import ensure_basicsr_torchvision_compat
from .fusion import blend_face_crop

LOGGER = logging.getLogger(__name__)


class FaceRestorationEngine:
    """Aligned-face restoration using CodeFormer, with GFPGAN fallback."""

    def __init__(self, config: dict, detector_config: dict, device: torch.device):
        self.config = config
        self.detector_config = detector_config
        self.device = device
        self.model = None
        self.backend = None

    def _load_codeformer(self) -> None:
        ensure_basicsr_torchvision_compat()
        repo = Path(self.config.get("codeformer_repo", "models/codeformer"))
        arch_file = repo / "basicsr" / "archs" / "codeformer_arch.py"
        weights = Path(self.config["codeformer_weights"])
        if not arch_file.is_file() or not weights.is_file():
            raise FileNotFoundError(
                f"CodeFormer repository/weights missing ({arch_file}, {weights})."
            )
        # Import the installed BasicSR before loading local CodeFormer files so
        # the repository's bundled, incomplete BasicSR copy cannot shadow it.
        import basicsr.archs  # noqa: F401
        # CodeFormer adds vqgan_arch to BasicSR, but installing its bundled
        # BasicSR fork alongside pip BasicSR would create a package collision.
        # Register only the required local module under the expected namespace.
        vqgan_name = "basicsr.archs.vqgan_arch"
        if vqgan_name not in sys.modules:
            vqgan_file = repo / "basicsr" / "archs" / "vqgan_arch.py"
            vqgan_spec = importlib.util.spec_from_file_location(vqgan_name, vqgan_file)
            vqgan_module = importlib.util.module_from_spec(vqgan_spec)
            assert vqgan_spec.loader is not None
            sys.modules[vqgan_name] = vqgan_module
            vqgan_spec.loader.exec_module(vqgan_module)
        spec = importlib.util.spec_from_file_location("codeformer_arch_local", arch_file)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        CodeFormer = module.CodeFormer
        model = CodeFormer(
            dim_embd=512, codebook_size=1024, n_head=8, n_layers=9,
            connect_list=["32", "64", "128", "256"],
        ).to(self.device)
        checkpoint = torch.load(weights, map_location="cpu")
        model.load_state_dict(checkpoint.get("params_ema", checkpoint.get("params", checkpoint)), strict=True)
        self.model = model.eval()
        self.backend = "codeformer"

    def _load_gfpgan(self) -> None:
        ensure_basicsr_torchvision_compat()
        from gfpgan.archs.gfpganv1_clean_arch import GFPGANv1Clean

        weights = Path(self.config["gfpgan_weights"])
        if not weights.is_file():
            raise FileNotFoundError(f"GFPGAN weights not found: {weights}")
        model = GFPGANv1Clean(
            out_size=512, num_style_feat=512, channel_multiplier=2,
            decoder_load_path=None, fix_decoder=False, num_mlp=8,
            input_is_latent=True, different_w=True, narrow=1, sft_half=True,
        ).to(self.device)
        checkpoint = torch.load(weights, map_location="cpu")
        model.load_state_dict(checkpoint.get("params_ema", checkpoint.get("params", checkpoint)), strict=True)
        self.model = model.eval()
        self.backend = "gfpgan"

    def load(self) -> None:
        if self.model is not None:
            return
        preferred = self.config.get("backend", "codeformer").lower()
        errors = []
        loaders = [self._load_codeformer, self._load_gfpgan] if preferred == "codeformer" else [self._load_gfpgan, self._load_codeformer]
        for loader in loaders:
            try:
                loader()
                LOGGER.info("Face restoration model %s loaded successfully", self.backend)
                return
            except Exception as exc:
                errors.append(f"{loader.__name__}: {exc}")
                LOGGER.warning("Face backend failed to load: %s", exc)
        raise RuntimeError("No face restoration backend could be loaded. " + " | ".join(errors))

    @torch.inference_mode()
    def _restore_crop(self, crop_bgr: np.ndarray, strength: float) -> np.ndarray:
        resized = cv2.resize(crop_bgr, (512, 512), interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(resized[:, :, ::-1].copy()).permute(2, 0, 1).float().div(255).unsqueeze(0).to(self.device)
        normalize(tensor, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        if self.backend == "codeformer":
            # CodeFormer w=0 favors restoration quality; w=1 favors source fidelity.
            output = self.model(tensor, w=1.0 - strength, adain=True)[0]
        else:
            output = self.model(tensor, return_rgb=False, weight=0.5)[0]
        output = output.squeeze(0).clamp(-1, 1).add(1).div(2).mul(255)
        output = output.permute(1, 2, 0).cpu().numpy().round().astype(np.uint8)
        # CodeFormer returns RGB tensors while the rest of this pipeline and
        # facexlib use OpenCV BGR images. GFPGAN with return_rgb=False already
        # returns BGR, so only swap channels for CodeFormer.
        if self.backend == "codeformer":
            output = output[:, :, ::-1]
        return np.ascontiguousarray(output)

    @torch.inference_mode()
    def restore(self, enhanced_bgr: np.ndarray, strength: float) -> tuple[np.ndarray, int]:
        if not 0.0 <= strength <= 1.0:
            raise ValueError("Face restore strength must be between 0 and 1.")
        if strength == 0.0:
            return enhanced_bgr, 0
        self.load()
        from facexlib.utils.face_restoration_helper import FaceRestoreHelper

        helper = FaceRestoreHelper(
            upscale_factor=1,
            face_size=512,
            crop_ratio=(1, 1),
            det_model=self.detector_config.get("model", "retinaface_resnet50"),
            save_ext="png",
            use_parse=True,
            device=self.device,
        )
        helper.clean_all()
        helper.read_image(enhanced_bgr)
        count = helper.get_face_landmarks_5(
            only_center_face=False,
            resize=640,
            eye_dist_threshold=5,
        )
        if count == 0:
            return enhanced_bgr, 0
        helper.align_warp_face()
        for crop in helper.cropped_faces:
            try:
                restored = self._restore_crop(crop, strength)
                # facexlib 0.3.0 accepts only the restored aligned face here;
                # the source crop is already incorporated by blend_face_crop.
                helper.add_restored_face(blend_face_crop(crop, restored, strength))
            except torch.cuda.OutOfMemoryError as exc:
                torch.cuda.empty_cache()
                raise RuntimeError("Insufficient GPU memory during face restoration.") from exc
        helper.get_inverse_affine(None)
        output = helper.paste_faces_to_input_image(upsample_img=enhanced_bgr)
        return output, int(count)
