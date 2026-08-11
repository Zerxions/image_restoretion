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

# This singleton is created once per server process. Individual models are lazy-loaded once.
PIPELINE = ImageRestorationPipeline("configs/config.yaml")


def run_restoration(image, mode: str, scale_label: str, face_strength: float, progress=gr.Progress()):
    if image is None:
        raise gr.Error("Please upload a JPG, JPEG or PNG image first.")
    try:
        scale = int(scale_label.rstrip("x"))
        result = PIPELINE.restore(
            image=image,
            mode=mode.split(" ", 1)[0],
            scale=scale,
            face_strength=face_strength,
            progress=lambda value, text: progress(value, desc=text),
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
    gr.Markdown("# AI Image Restoration\nReal-ESRGAN full-image enhancement with CodeFormer/GFPGAN face restoration.")
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
        mode = gr.Dropdown(
            ["Auto 自动修复", "Portrait 人像增强", "Landscape 风景增强"],
            value="Auto 自动修复",
            label="Restoration Mode",
        )
        scale = gr.Radio(["2x", "4x"], value="2x", label="Scale")
        strength = gr.Slider(0.0, 1.0, value=0.7, step=0.05, label="Face Restore Strength")

    start = gr.Button("Start Restoration", variant="primary")
    run_status = gr.Textbox(label="Run Status", interactive=False)
    start.click(
        run_restoration,
        inputs=[original, mode, scale, strength],
        outputs=[restored, run_status, model_status],
    )


if __name__ == "__main__":
    server = PIPELINE.config.get("server", {})
    demo.queue(default_concurrency_limit=int(server.get("concurrency_limit", 1))).launch(
        server_name=server.get("host", "127.0.0.1"),
        server_port=int(server.get("port", 7860)),
        share=bool(server.get("share", False)),
    )
