from __future__ import annotations

import logging
import os
from pathlib import Path

import gradio as gr

from src.image_utils import bgr_to_rgb
from src.pipeline import ImageRestorationPipeline

PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
LOGGER = logging.getLogger("app")

# This singleton is created once per server process; its main models are lazy-loaded and cached.
PIPELINE = ImageRestorationPipeline("configs/config.yaml")

TASK_KEYS = {
    "Smart 智能综合修复": "Smart",
    "Deblur 仅去模糊（保持原尺寸）": "Deblur",
    "SuperResolution 仅超分辨率": "SuperResolution",
    "Deblur+SR 去模糊后超分辨率": "Deblur+SR",
    "Restore 原有超分与人脸修复流程": "Restore",
}
MODE_KEYS = {
    "Auto 自动修复": "Auto",
    "Portrait 人像增强": "Portrait",
    "Landscape 风景增强": "Landscape",
}
DEBLUR_METHODS = {
    "AI 智能去模糊（EVSSM，推荐）": "ai",
    "EVSSM 运动/真实相机模糊": "evssm",
    "Restormer 散焦/失焦模糊": "restormer",
    "Auto 传统自适应（CPU 回退）": "auto",
    "Unsharp 多尺度清晰化": "unsharp",
    "Wiener 反卷积": "wiener",
}
SR_BACKENDS = {
    "Auto DRCT 保真优先（推荐）": "auto",
    "DRCT AI 保真超分": "drct",
    "Real-ESRGAN 快速超分": "realesrgan",
}


def run_restoration(
    image,
    task_label: str,
    mode_label: str,
    scale_label: str,
    sr_backend_label: str,
    deblur_method_label: str,
    deblur_strength: float,
    face_strength: float,
    progress=gr.Progress(),
):
    if image is None:
        raise gr.Error("Please upload a JPG, JPEG or PNG image first.")
    try:
        scale = int(scale_label.rstrip("x"))
        result = PIPELINE.restore(
            image=image,
            mode=MODE_KEYS[mode_label],
            scale=scale,
            face_strength=face_strength,
            progress=lambda value, text: progress(value, desc=text),
            task=TASK_KEYS[task_label],
            deblur_strength=deblur_strength,
            deblur_method=DEBLUR_METHODS[deblur_method_label],
            sr_backend=SR_BACKENDS[sr_backend_label],
        )
        return bgr_to_rgb(result.image_bgr), result.message, f"**Model status:** {PIPELINE.status()}"
    except FileNotFoundError as exc:
        LOGGER.exception("Required model is missing")
        raise gr.Error(f"Model file missing: {exc}") from exc
    except (ValueError, TypeError) as exc:
        LOGGER.warning("Invalid request: %s", exc)
        raise gr.Error(str(exc)) from exc
    except Exception as exc:
        LOGGER.exception("Restoration failed")
        raise gr.Error(f"Restoration failed: {exc}") from exc


with gr.Blocks(title="AI Image Restoration", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# AI Image Restoration\n"
        "EVSSM/Restormer AI 去模糊、DRCT/Real-ESRGAN 2x/4x 超分辨率，"
        "以及 RestoreFormer++ 人脸修复。"
    )
    model_status = gr.Markdown(f"**Model status:** {PIPELINE.status()}")

    with gr.Row():
        original = gr.Image(
            label="Original Image",
            type="pil",
            sources=["upload"],
            image_mode="RGB",
        )
        restored = gr.Image(label="Restored Image", type="numpy", interactive=False)

    with gr.Row():
        task = gr.Dropdown(
            list(TASK_KEYS),
            value="Smart 智能综合修复",
            label="Restoration Task",
            info="智能模式按去模糊 → 超分 → 可选人脸修复的顺序处理。",
        )
        mode = gr.Dropdown(
            list(MODE_KEYS),
            value="Auto 自动修复",
            label="Content Mode",
            info="仅 Smart/Restore 任务使用；Landscape 不运行人脸模型。",
        )
        scale = gr.Radio(
            ["2x", "4x"],
            value="2x",
            label="Super-resolution Scale",
            info="仅包含超分辨率的任务使用。",
        )
        sr_backend = gr.Dropdown(
            list(SR_BACKENDS),
            value="Auto DRCT 保真优先（推荐）",
            label="Super-resolution Model",
            info="DRCT 更偏保真；若高级模型不可用，Auto 回退 Real-ESRGAN。",
        )

    with gr.Row():
        deblur_method = gr.Dropdown(
            list(DEBLUR_METHODS),
            value="AI 智能去模糊（EVSSM，推荐）",
            label="Deblur Method",
            info="EVSSM 适合运动拖影；Restormer 适合散焦。AI 会跳过已经清晰的图。",
        )
        deblur_strength = gr.Slider(
            0.0,
            1.0,
            value=0.5,
            step=0.05,
            label="Deblur Strength",
            info="强度过高可能放大噪声并产生边缘光晕。",
        )
        strength = gr.Slider(
            0.0,
            1.0,
            value=0.7,
            step=0.05,
            label="Face Restore Strength",
            info="RestoreFormer++ 生成细节的混合比例；数值越高修复越强。",
        )

    gr.Markdown(
        "> **真实性提示：** AI 可以生成更清晰的合理细节，但严重模糊中已经丢失的信息无法被"
        "确定还原。证据、证件、医学等场景请降低强度，并保留原图。"
    )

    start = gr.Button("Start Restoration", variant="primary")
    run_status = gr.Textbox(label="Run Status", interactive=False)
    start.click(
        run_restoration,
        inputs=[
            original,
            task,
            mode,
            scale,
            sr_backend,
            deblur_method,
            deblur_strength,
            strength,
        ],
        outputs=[restored, run_status, model_status],
    )


if __name__ == "__main__":
    server = PIPELINE.config.get("server", {})
    demo.queue(default_concurrency_limit=int(server.get("concurrency_limit", 1))).launch(
        server_name=server.get("host", "127.0.0.1"),
        server_port=int(server.get("port", 7860)),
        share=bool(server.get("share", False)),
    )
