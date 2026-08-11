# Model weights

Place model files here using these exact names:

```text
weights/
├── RealESRGAN_x4plus.pth
├── codeformer.pth
└── GFPGANv1.4.pth
```

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

The files are not committed because model licenses and large-file distribution should
be handled separately. Review each upstream project's license before commercial use.
