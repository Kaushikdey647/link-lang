"""Admin UI — Indexing + Metrics only.

Mounted at /admin by FastAPI. Not intended for end users.

Standalone:
    python -m ui.admin_app
"""

from __future__ import annotations

import gradio as gr

from ui.indexing import build_indexing_tab
from ui.journey_tab import build_journey_tab
from ui.metrics_tab import build_metrics_tab

_CSS = """
.gradio-container { max-width: 1100px; margin: auto; }
footer { display: none !important; }
"""


def build_admin_ui() -> gr.Blocks:
    with gr.Blocks(
        title="Link-Lang Admin",
        theme=gr.themes.Soft(primary_hue="indigo"),
        css=_CSS,
    ) as demo:

        gr.Markdown(
            "# Link-Lang — Admin\n"
            "Indexing controls and live metrics. "
            "User-facing query UI is at [`/`](/)."
        )

        with gr.Tabs():
            with gr.Tab("⚙️ Indexing"):
                build_indexing_tab()

            with gr.Tab("🗺️ Pipeline Journey"):
                build_journey_tab()

            with gr.Tab("📊 Metrics"):
                build_metrics_tab()

    return demo


if __name__ == "__main__":
    build_admin_ui().launch(server_port=7861, share=False)
