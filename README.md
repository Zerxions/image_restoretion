# AI Image Restoration

一个面向 NVIDIA CUDA 的模块化图像修复系统。运动/真实相机模糊使用 EVSSM，散焦模糊
使用 Restormer；保真超分辨率使用 DRCT-MSE，Real-ESRGAN 作为快速回退；人脸由
RestoreFormer++ 修复，并保留 CodeFormer/GFPGAN 回退。所有大模型均延迟加载，CPU 仍可
使用传统 Unsharp/Wiener 去模糊和原有后端。

> 这是本地推理工程，首次运行前必须按“模型权重”章节准备代码和权重。
> 权重不会随本仓库分发。

## 功能特点

- Smart：AI 去模糊，再执行保真超分；根据 Auto/Portrait/Landscape 决定是否修复人脸
- Deblur：可选 EVSSM 运动去模糊、Restormer 散焦去模糊或无权重传统算法，保持原尺寸
- SuperResolution：DRCT-MSE 或 Real-ESRGAN 2x/4x 超分辨率
- Deblur+SR：先用所选模型去模糊再超分，避免直接放大模糊
- Restore：保留原有“超分 + 可选人脸恢复”流程，便于对比
- Auto/Portrait/Landscape：自动检测人脸、强制人像流程或完全跳过人脸模型
- AI Auto、EVSSM、Restormer、传统 Auto、Unsharp、Wiener 去模糊及可调混合强度
- 2x/4x 输出，以及 0～1 可调的人脸恢复强度
- CUDA 自动检测；CUDA 不可用时自动回退 CPU
- EVSSM、Restormer、DRCT、Real-ESRGAN 和人脸网络延迟加载并在服务进程内缓存
- RetinaFace 五点定位、512×512 人脸对齐、解析掩膜和逆仿射回贴
- 日志记录设备、图片尺寸、人脸数量、所选 pipeline 和模型加载状态
- 为 SUPIR、DiffBIR、Stable Diffusion Inpainting、视频修复及 TensorRT 留有扩展注册接口

## Demo 截图

启动 Web UI 后可将截图放到：

```text
examples/demo.png
```

建议截图同时包含 Original Image、Restored Image、模式和参数区域。

## 环境安装

### 方案 A：使用 environment.yml（推荐）

需要 NVIDIA 驱动支持 CUDA 12.x。Conda 安装的是 PyTorch 自带的 CUDA 12.1 runtime，
通常不要求系统安装完整 CUDA Toolkit。

```bash
cd AI_Image_Restoration
conda env create -f environment.yml
conda activate image_restore
```

### 方案 B：创建 Python 3.10 环境后使用 pip

```bash
cd AI_Image_Restoration
conda create -n image_restore python=3.10 -y
conda activate image_restore

# CUDA 12.1 示例；按 PyTorch 官网为当前驱动选择合适的 CUDA 12.x wheel。
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

验证 GPU：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
```

### basicsr 与较新 torchvision 的兼容性

部分 `basicsr==1.4.2` 发布包仍引用旧的
`torchvision.transforms.functional_tensor`。若导入时报该模块不存在，优先使用
`environment.yml` 中的版本范围；如果当前平台求解出了不兼容组合，可安装与 PyTorch
严格配套的 torchvision，或使用 Real-ESRGAN/BasicSR 官方仓库的最新源码版本。
不要混装 CPU 版 torch 和 CUDA 版 torchvision。

## AI 模型准备

在 `image_restore` 环境中运行一次显式准备脚本。它下载固定上游提交的架构、许可证及约
700 MiB 的 checkpoint，并安装与当前 Python 3.10 / Torch 2.5 / CUDA 12 ABI 匹配的
Mamba 预编译 wheel，不会替换 PyTorch：

```bash
conda activate image_restore
cd AI_Image_Restoration
python scripts/prepare_ai_models.py --models all --install-deps
```

可用 `--models deblur`、`--models sr` 或 `--models face` 只准备一个组。下载先写入 `.part`，
校验大小和固定源码 SHA-256 后再原子落盘。应用不会在 Web 请求中静默联网；AI Auto
不可用时只会明确记录并回退传统去模糊/Real-ESRGAN。

### AI 去模糊

- [EVSSM](https://github.com/kkkls/EVSSM) 使用官方 `RealBlur-J` 权重，适合普通
  JPEG/PNG 的真实相机运动模糊。
- [Restormer](https://github.com/swz30/Restormer) 使用官方 single-image DPDD 权重，
  适合单张 RGB 散焦/失焦图。
- [DRCT](https://github.com/ming053l/DRCT) 使用 Real-DRCT MSE x4 权重，默认选择
  低幻觉的保真结果；2x 在每个分块内部由原生 x4 结果缩放，避免整图 4x 中间缓冲。
- [RestoreFormer++](https://github.com/wzhouxiff/RestoreFormerPlusPlus) 对 512×512
  对齐人脸做 FP16 修复，再按 UI 强度软融合回原图。
- Laplacian 清晰度不能可靠区分运动与散焦，因此 UI 保留手动覆盖。
- 传统 Auto/Unsharp/Wiener 不需要权重，是 CPU 与缺依赖时的回退。

### 现有 Real-ESRGAN

从 [Real-ESRGAN 官方 Releases](https://github.com/xinntao/Real-ESRGAN/releases)
下载 `RealESRGAN_x4plus.pth`，放到：

```text
weights/RealESRGAN_x4plus.pth
```

2x 模式仍使用 x4plus 网络完整推理，再通过 Real-ESRGAN 的 `outscale` 重采样，因此只需
一份权重。它能减小输出文件和内存占用，但不会像原生 x2 网络那样明显降低模型推理显存。

### 现有 CodeFormer（回退）

克隆官方源码到预留目录：

```bash
git clone --depth 1 https://github.com/sczhou/CodeFormer.git models/codeformer
```

从 [CodeFormer 官方 Releases](https://github.com/sczhou/CodeFormer/releases)
下载 `codeformer.pth`，放到：

```text
weights/codeformer.pth
```

### 现有 GFPGAN（回退）

从 [GFPGAN 官方 Releases](https://github.com/TencentARC/GFPGAN/releases)
下载 `GFPGANv1.4.pth`，放到：

```text
weights/GFPGANv1.4.pth
```

CodeFormer 的源码或权重不可用时，系统会尝试 GFPGAN。要固定优先 GFPGAN，将
`configs/config.yaml` 中 `face_restoration.backend` 改为 `gfpgan`。

RetinaFace 和 face parsing 由 facexlib 管理。联网环境第一次使用时，facexlib 可能将
检测/解析权重下载到其包内 `weights` 目录；全新离线环境必须按 facexlib 的要求提前准备
这些文件。

模型目录最终应类似：

```text
models/codeformer/basicsr/archs/codeformer_arch.py
models/third_party/evssm/EVSSM.py
models/third_party/restormer/restormer_arch.py
models/third_party/drct/DRCT_arch.py
models/third_party/restoreformerplusplus/vqvae_arch.py
weights/evssm_realblur_j.pth
weights/restormer_defocus.pth
weights/drct_mse_x4.pth
weights/RestoreFormer++.ckpt
weights/RealESRGAN_x4plus.pth
weights/codeformer.pth
weights/GFPGANv1.4.pth
```

## 启动

```bash
conda activate image_restore
cd AI_Image_Restoration
python app.py
```

浏览器访问 `http://127.0.0.1:7860`。服务地址、端口及 Gradio share 开关在
`configs/config.yaml` 中配置。
如果从 IDE 启动，请将 Python Interpreter 选为 `image_restore` Conda 环境，不要使用
未安装 PyTorch/OpenCV 的系统 Python。

启动时 pipeline 只创建引擎；大型模型在第一次被相应任务使用时才加载。模型成功加载会
写入终端日志，状态中的 `standby` 表示尚未加载，不是错误。

## 测试

```bash
conda activate image_restore
cd AI_Image_Restoration
python -B -m unittest discover -s tests -v

# 权重准备后，用真实 CUDA 分别做一次小尺寸前向测试
python scripts/smoke_ai_models.py
```

## 参数说明

| 参数 | 可选值 | 说明 |
|---|---|---|
| Restoration Task | Smart / Deblur / SuperResolution / Deblur+SR / Restore | 选择独立或组合任务 |
| Content Mode | Auto / Portrait / Landscape | 仅 Smart/Restore 使用；控制人脸修复路由 |
| Deblur Method | AI / EVSSM / Restormer / Auto / Unsharp / Wiener | AI 运动/散焦模型或传统回退 |
| Deblur Strength | 0～1 | 0 保持输入不变；过高可能放大噪声或产生光晕、振铃 |
| Super-resolution Model | Auto / DRCT / Real-ESRGAN | 保真优先、固定 DRCT 或快速回退 |
| Scale | 2x / 4x | 含超分任务的最终长宽倍率；4x 输出占用更多内存 |
| Face Restore Strength | 0～1 | 0 保留超分后人脸，1 最大程度采用生成式修复结果 |
| `realesrgan.tile` | 0 或正整数 | 0 表示整图推理；显存不足可设 256、384、512 |
| `deblur.evssm/restormer.tile` | 正整数 | AI 去模糊核心分块；8GB 默认 256/384 |
| `advanced_super_resolution.drct.tile` | 16 的倍数 | DRCT 核心分块；8GB 默认 256 |
| `deblur.tile_size` | 128～8192 | 传统去模糊分块大小；较小值降低主机内存峰值 |
| `runtime.max_input_pixels` | 正整数 | 输入像素安全上限，避免意外超大图片耗尽内存 |
| `runtime.max_output_pixels` | 正整数 | 超分输出像素上限；长宽 4x 会产生 16 倍像素 |

`sharpness` 是 Laplacian 方差清晰度分数，通常越大越清晰，但它受图像尺寸、纹理和噪声
影响，不能跨内容当作绝对质量分数，也不能判断模糊类型。AI Auto 会跳过明显清晰的图片，
默认把其余图片交给 EVSSM；明显散焦时应手动选 Restormer。

CodeFormer 内部的 fidelity 权重与 UI 强度方向相反：系统会自动映射为
`fidelity = 1 - strength`，并额外将修复结果与原始对齐人脸做软融合，因此 UI 的语义始终是
“值越大，修复越强”。

## 项目结构

```text
AI_Image_Restoration/
├── app.py                       # Gradio 入口和进程级 pipeline 单例
├── requirements.txt
├── environment.yml
├── README.md
├── configs/
│   └── config.yaml              # 权重、推理和服务配置
├── models/
│   ├── realesrgan/              # 预留源码/扩展目录
│   ├── codeformer/              # CodeFormer 官方源码放置处
│   └── gfpgan/                  # 预留源码/扩展目录
├── src/
│   ├── pipeline.py              # 任务路由与组合流程调度
│   ├── deblur.py                # 自适应/Unsharp/Wiener 去模糊
│   ├── ai_deblur.py             # EVSSM/Restormer 懒加载、分块与回退
│   ├── advanced_super_resolution.py # DRCT 分块和 Real-ESRGAN 路由
│   ├── face_detector.py         # RetinaFace 检测
│   ├── face_restore.py          # RestoreFormer++/CodeFormer/GFPGAN 与回贴
│   ├── super_resolution.py      # Real-ESRGAN 封装
│   ├── image_utils.py           # 输入校验和颜色转换
│   └── fusion.py                # 对齐人脸软融合
├── tests/                        # 去模糊与 pipeline 路由测试
├── scripts/
│   ├── prepare_ai_models.py      # 固定版本下载、校验和可选依赖安装
│   └── smoke_ai_models.py        # 真实 CUDA checkpoint 冒烟测试
├── weights/
│   └── README.md                # 权重文件名和下载来源
└── examples/                    # 示例输入和 Demo 截图
```

## Pipeline 逻辑

```text
输入与格式校验
       │
       ├─ Deblur ── EVSSM / Restormer / Traditional ───── 输出（原尺寸）
       ├─ SuperResolution ──────────── DRCT / Real-ESRGAN ── 输出
       ├─ Deblur+SR ── 去模糊 ─────── DRCT / Real-ESRGAN ── 输出
       └─ Smart ── AI 去模糊 ── RetinaFace（Auto/Portrait）
                                  └─ DRCT（失败回退 Real-ESRGAN）
                                       └─ 可选 RestoreFormer++
                                            └─ CodeFormer / GFPGAN 回退
                                                └─ 软融合与逆仿射回贴 ── 输出
```

## 常见问题

### CUDA 不可用或版本不匹配

先运行 GPU 验证命令。如果 `torch.cuda.is_available()` 为 `False`，检查 NVIDIA 驱动、
torch wheel 是否为 CUDA 构建，以及 torch/torchvision 是否来自同一渠道。系统仍可在 CPU
运行，但 4x 超分和人脸修复会很慢。

### CUDA out of memory

程序会自动把 EVSSM/Restormer/DRCT 的 tile 逐级降低后重试。仍失败时，关闭其他 GPU
程序、缩小输入图，或手动降低相应 `tile`；Real-ESRGAN 可设为 256。
当前 2x 仍运行 x4plus 网络后再重采样，不能作为显存不足的主要解决方案。

### 去模糊后出现光晕、振铃或噪声

先降低 Deblur Strength。运动拖影选 EVSSM，散焦选 Restormer；Unsharp 适合轻度柔化，
Wiener 对不符合高斯假设的运动拖影容易产生振铃。

### 提示模型路径不存在

必须从项目根目录运行 `python app.py`。核对文件名大小写和配置相对路径。程序不会静默下载
Real-ESRGAN、CodeFormer 或 GFPGAN 主权重，以便离线部署可控。

### 第一次人脸检测停在下载

facexlib 可能在首次调用时下载 RetinaFace 或 face parsing 权重。离线服务器应先在联网机器
完成一次初始化并复制 facexlib 包内的 `weights` 目录，或根据 facexlib 文档预置权重。

### 人脸过度塑化

降低 Face Restore Strength（建议 0.4～0.7）。小脸、严重压缩脸和非真实人物可能出现生成式
细节；该参数同时控制 CodeFormer fidelity 与最终融合比例。

### 多用户请求与模型重复加载

`PIPELINE` 是进程级单例，主网络按需加载并缓存；内部锁避免多个请求同时争抢同一 GPU。
使用多个 worker 进程会让每个进程各加载一份模型，请按显存容量配置部署进程数。

## 扩展开发

`ImageRestorationPipeline.register_extension(name, engine)` 是通用扩展挂载点。新增 SUPIR、
DiffBIR、Inpainting 或 TensorRT 时建议实现独立 engine 并在 pipeline 中增加路由，不要修改
UI 的图像处理细节。视频修复可复用现有单帧 pipeline，再增加解码、跟踪、时序一致性和编码
模块。

## License 提示

工程代码与模型权重是两部分。EVSSM、Restormer、DRCT 的源码是 MIT，RestoreFormer++
源码是 Apache-2.0；下载目录中的权重未必单独声明与源码相同的条款。部署或商用前还须分别
检查 Real-ESRGAN、CodeFormer（非商业许可）、GFPGAN、facexlib 及所有 checkpoint 条款。
