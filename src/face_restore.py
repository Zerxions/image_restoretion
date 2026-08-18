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
    """Aligned-face restoration using RestoreFormer++, CodeFormer or GFPGAN."""

    def __init__(
        self,
        config: dict,
        detector_config: dict,
        device: torch.device,
        project_root: str | Path | None = None,
    ):
        self.config = config
        self.detector_config = detector_config
        self.device = device
        self.project_root = Path(project_root or Path.cwd()).resolve()
        self.model = None
        self.backend = None

    def _path(self, value: str | Path) -> Path:
        path = Path(value).expanduser()
        return path if path.is_absolute() else self.project_root / path

    @staticmethod
    def _load_checkpoint(path: Path):
        try:
            return torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:  # pragma: no cover - older supported torch
            return torch.load(path, map_location="cpu")

    def _load_codeformer(self) -> None:
        ensure_basicsr_torchvision_compat()
        repo = self._path(self.config.get("codeformer_repo", "models/codeformer"))
        arch_file = repo / "basicsr" / "archs" / "codeformer_arch.py"
        weights = self._path(self.config["codeformer_weights"])
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
        checkpoint = self._load_checkpoint(weights)
        model.load_state_dict(checkpoint.get("params_ema", checkpoint.get("params", checkpoint)), strict=True)
        self.model = model.eval().requires_grad_(False)
        self.backend = "codeformer"

    def _load_gfpgan(self) -> None:
        ensure_basicsr_torchvision_compat()
        from gfpgan.archs.gfpganv1_clean_arch import GFPGANv1Clean

        weights = self._path(self.config["gfpgan_weights"])
        if not weights.is_file():
            raise FileNotFoundError(f"GFPGAN weights not found: {weights}")
        model = GFPGANv1Clean(
            out_size=512, num_style_feat=512, channel_multiplier=2,
            decoder_load_path=None, fix_decoder=False, num_mlp=8,
            input_is_latent=True, different_w=True, narrow=1, sft_half=True,
        ).to(self.device)
        checkpoint = self._load_checkpoint(weights)
        model.load_state_dict(checkpoint.get("params_ema", checkpoint.get("params", checkpoint)), strict=True)
        self.model = model.eval().requires_grad_(False)
        self.backend = "gfpgan"

    def _load_restoreformer(self) -> None:
        arch_file = self._path(
            self.config.get(
                "restoreformer_architecture",
                "models/third_party/restoreformerplusplus/vqvae_arch.py",
            )
        )
        weights = self._path(
            self.config.get(
                "restoreformer_weights", "weights/RestoreFormer++.ckpt"
            )
        )
        if not arch_file.is_file() or not weights.is_file():
            raise FileNotFoundError(
                f"RestoreFormer++ architecture/weights missing ({arch_file}, {weights})."
            )
        spec = importlib.util.spec_from_file_location(
            "restoreformerplusplus_arch_local", arch_file
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        model = module.VQVAEGANMultiHeadTransformer(
            head_size=4, ex_multi_scale_num=1
        )
        checkpoint = self._load_checkpoint(weights)
        source = checkpoint.get("state_dict", checkpoint)
        if not isinstance(source, dict):
            raise RuntimeError("RestoreFormer++ checkpoint has no state_dict.")
        expected = model.state_dict()
        state = {}
        for key, value in source.items():
            normalized = str(key)
            for prefix in ("module.", "vqvae."):
                normalized = normalized.removeprefix(prefix)
            if normalized in expected and torch.is_tensor(value):
                state[normalized] = value
        missing, _ = model.load_state_dict(state, strict=False)
        if missing:
            preview = ", ".join(missing[:5])
            raise RuntimeError(
                f"RestoreFormer++ checkpoint is missing {len(missing)} parameters: {preview}"
            )
        use_half = self.device.type == "cuda" and bool(
            self.config.get("restoreformer_half", True)
        )
        dtype = torch.float16 if use_half else torch.float32
        self.model = (
            model.eval().requires_grad_(False).to(self.device, dtype=dtype)
        )
        self.backend = "restoreformer"

    def load(self) -> None:
        if self.model is not None:
            return
        preferred = self.config.get("backend", "restoreformer").lower()
        errors = []
        loader_map = {
            "restoreformer": self._load_restoreformer,
            "codeformer": self._load_codeformer,
            "gfpgan": self._load_gfpgan,
        }
        if preferred not in loader_map:
            raise ValueError(f"Unknown face restoration backend: {preferred}")
        order = [preferred] + [
            name for name in ("restoreformer", "codeformer", "gfpgan")
            if name != preferred
        ]
        loaders = [loader_map[name] for name in order]
        for loader in loaders:
            try:
                loader()
                LOGGER.info("Face restoration model %s loaded successfully", self.backend)
                return
            except Exception as exc:
                errors.append(f"{loader.__name__}: {exc}")
                if isinstance(exc, torch.cuda.OutOfMemoryError):
                    torch.cuda.empty_cache()
                LOGGER.warning("Face backend failed to load: %s", exc)
        raise RuntimeError("No face restoration backend could be loaded. " + " | ".join(errors))

    @torch.inference_mode()
    def _restore_crop(self, crop_bgr: np.ndarray, strength: float) -> np.ndarray:
        resized = cv2.resize(crop_bgr, (512, 512), interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(resized[:, :, ::-1].copy()).permute(2, 0, 1).float().div(255).unsqueeze(0).to(self.device)
        normalize(tensor, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        model_dtype = next(self.model.parameters()).dtype
        tensor = tensor.to(dtype=model_dtype)
        if self.backend == "codeformer":
            # CodeFormer w=0 favors restoration quality; w=1 favors source fidelity.
            output = self.model(tensor, w=1.0 - strength, adain=True)[0]
        elif self.backend == "restoreformer":
            output = self.model(tensor)[0]
        else:
            output = self.model(tensor, return_rgb=False, weight=0.5)[0]
        output = output.squeeze(0).clamp(-1, 1).add(1).div(2).mul(255)
        output = output.permute(1, 2, 0).cpu().numpy().round().astype(np.uint8)
        # CodeFormer and RestoreFormer++ return RGB tensors while the pipeline
        # and facexlib use BGR. GFPGAN with return_rgb=False already returns BGR.
        if self.backend in {"codeformer", "restoreformer"}:
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
