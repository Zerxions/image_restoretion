#!/usr/bin/env python3
"""Download pinned open-source restoration architectures and checkpoints.

Downloads are explicit rather than happening inside a Gradio request.  Files
are streamed to ``.part`` paths, validated, and atomically moved into place.
Run this script with the ``image_restore`` interpreter from the project root.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Asset:
    name: str
    url: str
    destination: str
    size: int
    sha256: str | None = None


ASSETS = {
    "evssm_arch": Asset(
        "EVSSM architecture",
        "https://raw.githubusercontent.com/kkkls/EVSSM/5098a5276640694a39a941119f9e8bcc3ece9fb0/models/EVSSM.py",
        "models/third_party/evssm/EVSSM.py",
        17_399,
        "d28b59d22be7b5a9d2e0247062b25142d62fc9fff16de66963f2e001620accd4",
    ),
    "evssm_license": Asset(
        "EVSSM MIT license",
        "https://raw.githubusercontent.com/kkkls/EVSSM/5098a5276640694a39a941119f9e8bcc3ece9fb0/license",
        "models/third_party/evssm/LICENSE",
        1_062,
        "ebbb97354efc45d568b5bd7100a3e70447256b7bbbe0c6d340cd382103751e30",
    ),
    "evssm_weights": Asset(
        "EVSSM RealBlur-J checkpoint",
        "https://drive.usercontent.google.com/download?id=1cItSJH6zGJjsWxGMM0jr1S0bOFZuP6gp&export=download&confirm=t",
        "weights/evssm_realblur_j.pth",
        68_885_494,
        "a78d1d4537a8966259dd2793ce6b91a3fd6983f66a6ffb3c04644822dd78c65e",
    ),
    "restormer_arch": Asset(
        "Restormer architecture",
        "https://raw.githubusercontent.com/swz30/Restormer/68dc6ac472db26f16361150cb7a96a1bc87da93f/basicsr/models/archs/restormer_arch.py",
        "models/third_party/restormer/restormer_arch.py",
        11_428,
        "3be243fa3c8e2cb2c9459eeac062b04d6084dbcc328292dae1c3e52ba6f7434a",
    ),
    "restormer_license": Asset(
        "Restormer MIT license",
        "https://raw.githubusercontent.com/swz30/Restormer/68dc6ac472db26f16361150cb7a96a1bc87da93f/LICENSE.md",
        "models/third_party/restormer/LICENSE",
        1_090,
        "2b93776512924bc095ec5d97a79d76cf1ab401ae641d1ae1f46e94f1c53a2e59",
    ),
    "restormer_weights": Asset(
        "Restormer single-image defocus checkpoint",
        "https://github.com/swz30/Restormer/releases/download/v1.0/single_image_defocus_deblurring.pth",
        "weights/restormer_defocus.pth",
        104_700_429,
        "7dce451f33f8f5e0faf7c4e3996e5dcc1bd425ecd1ada99b0f9750e490fd4c9e",
    ),
    "drct_arch": Asset(
        "DRCT architecture",
        "https://raw.githubusercontent.com/ming053l/DRCT/c0a8b51e69241cf7f8ae598053235ff498517935/drct/archs/DRCT_arch.py",
        "models/third_party/drct/DRCT_arch.py",
        31_758,
        "d70fee3763fab71fc8af44f0a476ebd847de1b10d4da110323df803904de18ca",
    ),
    "drct_license": Asset(
        "DRCT MIT license",
        "https://raw.githubusercontent.com/ming053l/DRCT/c0a8b51e69241cf7f8ae598053235ff498517935/LICENSE",
        "models/third_party/drct/LICENSE",
        1_070,
        "84c4b322533e9a8440d95e616047f9171c9b7adbbcbadf144e3418f71308d7ec",
    ),
    "drct_weights": Asset(
        "Real-DRCT MSE x4 checkpoint",
        "https://drive.usercontent.google.com/download?id=1y_ZRuR_1HPQO-azViqgynyEgfFhWqCAS&export=download&confirm=t",
        "weights/drct_mse_x4.pth",
        245_582_817,
        "fc745ffe4c806bef59e7f90f2f2bb4bccd36649ed16bd184bede5257ca5fb020",
    ),
    "restoreformer_arch": Asset(
        "RestoreFormer++ architecture",
        "https://raw.githubusercontent.com/wzhouxiff/RestoreFormerPlusPlus/59d250f45d6b11a3e620d62e6e067ca1288f93a9/RestoreFormer/modules/vqvae/vqvae_arch.py",
        "models/third_party/restoreformerplusplus/vqvae_arch.py",
        28_954,
        "ba4a1275d938a416ea5ce848d7c5418d34cc82ca8d6f1dfb1998a8ed225aa138",
    ),
    "restoreformer_license": Asset(
        "RestoreFormer++ Apache-2.0 license",
        "https://raw.githubusercontent.com/wzhouxiff/RestoreFormerPlusPlus/59d250f45d6b11a3e620d62e6e067ca1288f93a9/LICENSE",
        "models/third_party/restoreformerplusplus/LICENSE",
        11_357,
        "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4",
    ),
    "restoreformer_weights": Asset(
        "RestoreFormer++ v1.0.0 checkpoint",
        "https://github.com/wzhouxiff/RestoreFormerPlusPlus/releases/download/v1.0.0/RestoreFormer%2B%2B.ckpt",
        "weights/RestoreFormer++.ckpt",
        294_028_815,
        "613fe52805f86bf8c2bffff08ae9f7a0b99f408be1bf221767af6183038be3a2",
    ),
}


GROUPS = {
    "deblur": (
        "evssm_arch",
        "evssm_license",
        "evssm_weights",
        "restormer_arch",
        "restormer_license",
        "restormer_weights",
    ),
    "sr": ("drct_arch", "drct_license", "drct_weights"),
    "face": (
        "restoreformer_arch",
        "restoreformer_license",
        "restoreformer_weights",
    ),
}


MAMBA_WHEEL = (
    "https://github.com/state-spaces/mamba/releases/download/v2.2.4/"
    "mamba_ssm-2.2.4%2Bcu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def valid_existing(asset: Asset, destination: Path) -> bool:
    if not destination.is_file() or destination.stat().st_size != asset.size:
        return False
    return asset.sha256 is None or sha256(destination) == asset.sha256


def download(asset: Asset, *, force: bool) -> None:
    destination = PROJECT_ROOT / asset.destination
    if not force and valid_existing(asset, destination):
        print(f"[ready] {asset.name}: {destination.relative_to(PROJECT_ROOT)}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    if partial.exists():
        partial.unlink()
    print(f"[download] {asset.name} ({asset.size / 2**20:.1f} MiB)")
    try:
        with requests.get(asset.url, stream=True, timeout=(20, 120)) as response:
            response.raise_for_status()
            content_type = response.headers.get("content-type", "").lower()
            if "text/html" in content_type and asset.size > 1_000_000:
                raise RuntimeError(
                    "The host returned HTML instead of the checkpoint. Retry later or "
                    "download the official file manually to " + str(destination)
                )
            downloaded = 0
            next_report = 32 * 1024 * 1024
            with partial.open("wb") as stream:
                for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
                    if not chunk:
                        continue
                    stream.write(chunk)
                    downloaded += len(chunk)
                    if downloaded >= next_report:
                        print(f"  {downloaded / asset.size:6.1%}")
                        next_report += 32 * 1024 * 1024
        actual_size = partial.stat().st_size
        if actual_size != asset.size:
            raise RuntimeError(
                f"Size mismatch for {asset.name}: got {actual_size:,}, "
                f"expected {asset.size:,} bytes."
            )
        actual_hash = sha256(partial)
        if asset.sha256 and actual_hash != asset.sha256:
            raise RuntimeError(
                f"SHA-256 mismatch for {asset.name}: {actual_hash}"
            )
        os.replace(partial, destination)
        print(f"[saved] {destination.relative_to(PROJECT_ROOT)} sha256={actual_hash}")
    except Exception:
        if partial.exists():
            partial.unlink()
        raise


def install_dependencies(needs_evssm: bool) -> None:
    print(f"[python] {sys.executable}")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "einops==0.8.1"], check=True
    )
    if not needs_evssm:
        return
    import torch

    version_ok = torch.__version__.split("+", 1)[0].startswith("2.5.")
    python_ok = sys.version_info[:2] == (3, 10)
    cuda_ok = bool(torch.version.cuda and torch.version.cuda.startswith("12."))
    abi_ok = not bool(torch._C._GLIBCXX_USE_CXX11_ABI)
    if not all((version_ok, python_ok, cuda_ok, abi_ok)):
        raise RuntimeError(
            "The pinned EVSSM Mamba wheel requires Python 3.10, torch 2.5.x, "
            "CUDA 12.x and CXX11 ABI disabled. Restormer remains usable without it."
        )
    # Mamba 2.2.x imports generation symbols removed from Transformers 5.x.
    # Pin the last compatible 4.x line before installing the CUDA wheel.
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "transformers==4.44.2",
            MAMBA_WHEEL,
        ],
        check=True,
    )


def parse_groups(value: str) -> list[str]:
    requested = [item.strip().lower() for item in value.split(",") if item.strip()]
    if not requested or "all" in requested:
        return list(GROUPS)
    unknown = sorted(set(requested) - set(GROUPS))
    if unknown:
        raise argparse.ArgumentTypeError(
            f"Unknown model group(s): {', '.join(unknown)}; choose deblur,sr,face,all"
        )
    return requested


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        default="all",
        help="Comma-separated groups: deblur,sr,face,all (default: all)",
    )
    parser.add_argument(
        "--install-deps",
        action="store_true",
        help="Install einops and the exact torch-2.5 CUDA-12 Mamba wheel for EVSSM.",
    )
    parser.add_argument("--force", action="store_true", help="Redownload valid files.")
    args = parser.parse_args()
    groups = parse_groups(args.models)
    if args.install_deps:
        install_dependencies("deblur" in groups)
    keys: list[str] = []
    for group in groups:
        for key in GROUPS[group]:
            if key not in keys:
                keys.append(key)
    for key in keys:
        download(ASSETS[key], force=args.force)
    print("\nAI model preparation complete.")
    print("Keep the upstream LICENSE files when redistributing this project.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
