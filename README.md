# AI Image Restoration

一个模块化的 AI 图像智能修复系统。系统先使用 Real-ESRGAN 对整张图片进行
2x/4x 超分、去噪和压缩伪影抑制；在自动或人像模式中，使用 RetinaFace 检测并
对齐人脸，再以 CodeFormer 修复（GFPGAN 自动回退），最后通过软掩膜和逆仿射变换
将人脸融合回高清图。

> 这是本地推理工程，首次运行前必须按“模型权重”章节准备代码和权重。
> 权重不会随本仓库分发。

## 功能特点

- Auto：检测到人脸时运行“全图增强 + 人脸恢复 + 融合”，否则仅运行全图增强
- Portrait：强制运行人像 pipeline，适合人物照片
- Landscape：仅运行 Real-ESRGAN，避免对非人像纹理做无意义的人脸处理
- 2x/4x 输出，以及 0～1 可调的人脸恢复强度
- 基于 facexlib 语义解析的独立头发增强，并保留原始头发色度
- CUDA 自动检测；CUDA 不可用时自动回退 CPU
- 模型延迟加载且每个服务进程只加载一次
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

## 模型权重

### 1. Real-ESRGAN

从 [Real-ESRGAN 官方 Releases](https://github.com/xinntao/Real-ESRGAN/releases)
下载 `RealESRGAN_x4plus.pth`，放到：

```text
weights/RealESRGAN_x4plus.pth
```

2x 模式仍使用 x4plus 网络，通过 Real-ESRGAN 的 `outscale` 做高质量重采样，因此只需
一份权重。

### 2. CodeFormer（优先）

克隆官方源码到预留目录：

```bash
git clone --depth 1 https://github.com/sczhou/CodeFormer.git models/codeformer
```

从 [CodeFormer 官方 Releases](https://github.com/sczhou/CodeFormer/releases)
下载 `codeformer.pth`，放到：

```text
weights/codeformer.pth
```

### 3. GFPGAN（备用）

从 [GFPGAN 官方 Releases](https://github.com/TencentARC/GFPGAN/releases)
下载 `GFPGANv1.4.pth`，放到：

```text
weights/GFPGANv1.4.pth
```

CodeFormer 的源码或权重不可用时，系统会尝试 GFPGAN。要固定优先 GFPGAN，将
`configs/config.yaml` 中 `face_restoration.backend` 改为 `gfpgan`。

RetinaFace 由 facexlib 管理。联网环境第一次使用时可能下载检测/解析模型到 torch
缓存；离线部署需提前准备 facexlib 所需权重缓存。配置中的检测权重字段可用于独立
检测器，但 facexlib 的人脸对齐 helper 仍按其自身缓存规则读取模型。

模型目录最终应类似：

```text
models/codeformer/basicsr/archs/codeformer_arch.py
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

启动时显示 `Initialized` 表示 pipeline 已创建；模型在第一次需要时加载，状态提示中的
“Models load once on first use” 属于正常状态。模型成功加载会写入终端日志。

## 参数说明

| 参数 | 可选值 | 说明 |
|---|---|---|
| Restoration Mode | Auto / Portrait / Landscape | 自动、人像或风景 pipeline |
| Scale | 2x / 4x | 最终长宽倍率；4x 使用更多显存和时间 |
| Face Restore Strength | 0～1 | 0 保留超分后人脸，1 最大程度采用生成式修复结果 |
| Hair Enhance Strength | 0～1 | 头发语义区域的细节融合强度；建议 0.25～0.5 |
| `realesrgan.tile` | 0 或正整数 | 0 表示整图推理；显存不足可设 256、384、512 |
| `runtime.max_input_pixels` | 正整数 | 输入像素安全上限，避免意外超大图片耗尽内存 |

CodeFormer 内部的 fidelity 权重与 UI 强度方向相反：系统会自动映射为
`fidelity = 1 - strength`，并额外将修复结果与原始对齐人脸做软融合，因此 UI 的语义始终是
“值越大，修复越强”。

头发增强使用 face parsing 的 hair 类别生成软掩膜，仅融合 CodeFormer 重建的亮度细节，
并保留输入头发的色度以降低染发、偏色和发型漂移风险。严重失焦时新增发丝仍属于模型推断，
强度过高可能产生不真实纹理。

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
│   ├── pipeline.py              # 自动模式判断与主流程调度
│   ├── face_detector.py         # RetinaFace 检测
│   ├── face_restore.py          # 对齐、CodeFormer/GFPGAN 推理与回贴
│   ├── super_resolution.py      # Real-ESRGAN 封装
│   ├── image_utils.py           # 输入校验和颜色转换
│   └── fusion.py                # 对齐人脸软融合
├── weights/
│   └── README.md                # 权重文件名和下载来源
└── examples/                    # 示例输入和 Demo 截图
```

## Pipeline 逻辑

```text
输入与格式校验
       │
       ├─ Landscape ─────────────── Real-ESRGAN ── 输出
       │
       └─ Auto / Portrait ── RetinaFace 检测
                                  │
                    无人脸（Auto）├─ Real-ESRGAN ── 输出
                                  │
                         有人脸   └─ Real-ESRGAN
                                         └─ 人脸对齐
                                            └─ CodeFormer / GFPGAN
                                               └─ 软融合与逆仿射回贴 ── 输出
```

## 常见问题

### CUDA 不可用或版本不匹配

先运行 GPU 验证命令。如果 `torch.cuda.is_available()` 为 `False`，检查 NVIDIA 驱动、
torch wheel 是否为 CUDA 构建，以及 torch/torchvision 是否来自同一渠道。系统仍可在 CPU
运行，但 4x 超分和人脸修复会很慢。

### CUDA out of memory

按顺序尝试：选择 2x；将 `configs/config.yaml` 中 `realesrgan.tile` 设为 512 或 256；
缩小输入图；关闭其他占用 GPU 的程序。tile 越小越省显存，但速度更慢且可能产生轻微接缝。

### 提示模型路径不存在

必须从项目根目录运行 `python app.py`。核对文件名大小写和配置相对路径。程序不会静默下载
Real-ESRGAN、CodeFormer 或 GFPGAN 主权重，以便离线部署可控。

### 第一次人脸检测停在下载

facexlib 可能在首次调用时下载 RetinaFace 或 face parsing 权重。离线服务器应先在联网机器
完成一次初始化并复制 torch hub/checkpoints 缓存，或根据 facexlib 文档预置权重。

### 人脸过度塑化

降低 Face Restore Strength（建议 0.4～0.7）。小脸、严重压缩脸和非真实人物可能出现生成式
细节；该参数同时控制 CodeFormer fidelity 与最终融合比例。

### 多用户请求与模型重复加载

`PIPELINE` 是进程级单例，模型按需加载一次；内部锁避免多个请求同时争抢同一 GPU 模型。
使用多个 worker 进程会让每个进程各加载一份模型，请按显存容量配置部署进程数。

## 扩展开发

`ImageRestorationPipeline.register_extension(name, engine)` 是通用扩展挂载点。新增 SUPIR、
DiffBIR、Inpainting 或 TensorRT 时建议实现独立 engine 并在 pipeline 中增加路由，不要修改
UI 的图像处理细节。视频修复可复用现有单帧 pipeline，再增加解码、跟踪、时序一致性和编码
模块。

## License 提示

工程代码与模型权重是两部分。部署或商用前请分别检查 Real-ESRGAN、CodeFormer、GFPGAN、
facexlib 及所用权重的上游许可证和适用条款。
