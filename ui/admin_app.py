"""Admin UI — read-only observability: Serving + Ingestion.

Mounted at /admin by FastAPI. Not intended for end users.

Indexing is CLI-only (see pipeline/indexer.py) — this UI has no controls
that start, stop, or resume anything; it only observes live server metrics
and live Qdrant state.

Standalone:
    python -m ui.admin_app
"""

from __future__ import annotations

import gradio as gr

from ui.ingestion_tab import build_ingestion_tab
from ui.metrics_tab import build_serving_tab

_CSS = """
.gradio-container { max-width: 1100px; margin: auto; }
footer { display: none !important; }
"""


def build_admin_ui() -> gr.Blocks:
    with gr.Blocks(
        title="Bhasha Admin",
        theme=gr.themes.Soft(primary_hue="indigo"),
        css=_CSS,
    ) as demo:

        gr.Markdown(
            "# Bhasha — Admin\n"
            "Read-only observability. Indexing runs via CLI only. "
            "User-facing query UI is at [`/`](/)."
        )

        with gr.Tabs():
            with gr.Tab("📡 Serving"):
                build_serving_tab()

            with gr.Tab("📥 Ingestion"):
                build_ingestion_tab()

    return demo


if __name__ == "__main__":
    build_admin_ui().launch(server_port=7861, share=False)
