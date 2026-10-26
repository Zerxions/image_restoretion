# Model weights

Place model files here using these exact names:

```text
weights/
├── evssm_realblur_j.pth
├── restormer_defocus.pth
├── drct_mse_x4.pth
├── RestoreFormer++.ckpt
├── RealESRGAN_x4plus.pth
├── codeformer.pth
└── GFPGANv1.4.pth
```

Prepare the pinned AI architectures, licenses, and four new checkpoints with:

```bash
conda activate image_restore
python scripts/prepare_ai_models.py --models all --install-deps
```

The downloader streams to temporary files, validates official file sizes and pinned
source SHA-256 values, then moves each file into place atomically. Its sources are:

- EVSSM RealBlur-J: official EVSSM Google Drive checkpoint (motion/real camera blur).
- Restormer single-image defocus: official Restormer Google Drive checkpoint.
- Real-DRCT MSE x4: official DRCT Google Drive checkpoint (faithful SR).
- RestoreFormer++ v1.0.0: official GitHub release checkpoint.

- Real-ESRGAN: download `RealESRGAN_x4plus.pth` from the official
  [Real-ESRGAN releases](https://github.com/xinntao/Real-ESRGAN/releases).
- CodeFormer: download `codeformer.pth` from the official
  [CodeFormer releases](https://github.com/sczhou/CodeFormer/releases).
- GFPGAN fallback: download `GFPGANv1.4.pth` from the official
  [GFPGAN releases](https://github.com/TencentARC/GFPGAN/releases).

For CodeFormer, also clone its source into `models/codeformer`:

```bash
git clone --depth 1 https://github.com/sczhou/CodeFormer.git models/codeformer
```

The binary files are not committed because model licenses and large-file distribution
should be handled separately. Pinned upstream source licenses are saved beside the
downloaded architecture files under `models/third_party/`. Review checkpoint terms and
all transitive model licenses before commercial use.
