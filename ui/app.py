"""Gradio UI for the Link-Lang voice-enabled RAG pipeline.

Mounted on FastAPI at /ui. Can also run standalone:
    python -m ui.app
"""

from __future__ import annotations

import gradio as gr

from pipeline.rag import RAGChain
from stt import transcribe
from ui.indexing import build_indexing_tab
from ui.metrics_tab import build_metrics_tab

# ---------------------------------------------------------------------------
# Language options
# ---------------------------------------------------------------------------

LANGUAGES = {
    "Hindi (हिंदी)": "hi",
    "Bengali (বাংলা)": "bn",
    "Gujarati (ગુજરાતી)": "gu",
    "Kannada (ಕನ್ನಡ)": "kn",
    "Malayalam (മലയാളം)": "ml",
    "Marathi (मराठी)": "mr",
    "Nepali (नेपाली)": "ne",
    "Odia (ଓଡ଼ିଆ)": "or",
    "Punjabi (ਪੰਜਾਬੀ)": "pa",
    "Sanskrit (संस्कृतम्)": "sa",
    "Tamil (தமிழ்)": "ta",
    "Telugu (తెలుగు)": "te",
    "Urdu (اردو)": "ur",
    "Assamese (অসমীয়া)": "as",
}

SARVAM_LANG_CODES = {
    "hi": "hi-IN", "bn": "bn-IN", "gu": "gu-IN", "kn": "kn-IN",
    "ml": "ml-IN", "mr": "mr-IN", "ne": "ne-NP", "or": "or-IN",
    "pa": "pa-IN", "sa": "sa-IN", "ta": "ta-IN", "te": "te-IN",
    "ur": "ur-IN", "as": "as-IN",
}

# Cache RAGChain instances per language
_chains: dict[str, RAGChain] = {}


def _get_chain(lang: str) -> RAGChain:
    if lang not in _chains:
        _chains[lang] = RAGChain(lang=lang)
    return _chains[lang]


# ---------------------------------------------------------------------------
# Handler functions
# ---------------------------------------------------------------------------

def handle_voice(audio_path: str | None, lang_display: str) -> tuple:
    """STT → RAG → return (transcript, answer, passages_md, latency_md)."""
    if audio_path is None:
        return "", "Please record or upload an audio file.", "", ""

    lang = LANGUAGES[lang_display]
    sarvam_code = SARVAM_LANG_CODES.get(lang, "unknown")

    try:
        transcript = transcribe(audio_path, language_code=sarvam_code)
    except Exception as e:
        return "", f"⚠️ STT error: {e}", "", ""

    if not transcript.strip():
        return transcript, "⚠️ Could not transcribe audio — try speaking more clearly.", "", ""

    return _run_rag(transcript, lang)


def handle_text(query: str, lang_display: str) -> tuple:
    """Text query → RAG → return (answer, passages_md, latency_md)."""
    if not query.strip():
        return "Please enter a question.", "", ""

    lang = LANGUAGES[lang_display]
    answer, passages_md, latency_md = _run_rag(query, lang)[1:]
    return answer, passages_md, latency_md


def _run_rag(query: str, lang: str) -> tuple[str, str, str, str]:
    """Returns (transcript_or_query, answer, passages_md, latency_md)."""
    chain = _get_chain(lang)
    result = chain.invoke(query)

    # Guardrail badges
    input_badge = "✅ Passed" if result.input_guardrail.passed else f"🚫 Blocked — {result.input_guardrail.reason}"
    ground_badge = "✅ Grounded" if result.grounding_guardrail.passed else f"⚠️ Ungrounded — {result.grounding_guardrail.reason}"

    # Passages markdown
    if result.passages:
        parts = []
        for i, p in enumerate(result.passages, 1):
            selected = " ★" if p.get("is_selected") else ""
            ctype = p.get("chunk_type", "")
            parts.append(
                f"**[{i}]{selected} `{ctype}` — passage `{p.get('passage_id', '')}`**\n\n"
                f"{p.get('text', '')}"
            )
        passages_md = "\n\n---\n\n".join(parts)
    else:
        passages_md = "_No passages retrieved._"

    # Latency markdown
    lat = result.latency
    latency_md = (
        f"| Step | Latency |\n|---|---|\n"
        f"| Input guardrail | {lat.get('input_guardrail_ms', 0):.1f} ms |\n"
        f"| Retrieval (Qdrant ANN) | {lat.get('retrieval_ms', 0):.1f} ms |\n"
        f"| Generation (Sarvam-105B) | {lat.get('generation_ms', 0):.1f} ms |\n"
        f"| Grounding guardrail | {lat.get('grounding_guardrail_ms', 0):.1f} ms |\n"
        f"| **Total** | **{lat.get('total_ms', 0):.1f} ms** |\n\n"
        f"**Guardrails:** {input_badge} &nbsp;|&nbsp; {ground_badge}"
    )

    return query, result.answer, passages_md, latency_md


# ---------------------------------------------------------------------------
# Gradio layout
# ---------------------------------------------------------------------------

_CSS = """
#answer-box textarea { font-size: 1.1rem; line-height: 1.8; }
#transcript-box textarea { color: #555; font-style: italic; }
.gradio-container { max-width: 900px; margin: auto; }
footer { display: none !important; }
"""

def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title="Link-Lang — Multilingual Voice RAG",
        theme=gr.themes.Soft(primary_hue="indigo"),
        css=_CSS,
    ) as demo:

        gr.Markdown(
            "# Link-Lang\n"
            "**Voice-enabled multilingual RAG** over MS MARCO-XI &nbsp;·&nbsp; "
            "Sarvam STT + Qdrant + Sarvam-105B"
        )

        lang_selector = gr.Dropdown(
            choices=list(LANGUAGES.keys()),
            value="Hindi (हिंदी)",
            label="Language",
            scale=1,
        )

        with gr.Tabs():

            # ── Tab 1: Voice ──────────────────────────────────────────────
            with gr.Tab("🎤 Voice Query"):
                audio_in = gr.Audio(
                    sources=["microphone", "upload"],
                    type="filepath",
                    label="Speak your question",
                )
                voice_btn = gr.Button("Transcribe & Ask", variant="primary")

                transcript_box = gr.Textbox(
                    label="Transcript",
                    interactive=False,
                    elem_id="transcript-box",
                    placeholder="Transcript will appear here…",
                )
                voice_answer = gr.Textbox(
                    label="Answer",
                    interactive=False,
                    elem_id="answer-box",
                    lines=4,
                )

                with gr.Accordion("Retrieved Passages", open=False):
                    voice_passages = gr.Markdown()

                with gr.Accordion("Latency Breakdown", open=False):
                    voice_latency = gr.Markdown()

                voice_btn.click(
                    fn=handle_voice,
                    inputs=[audio_in, lang_selector],
                    outputs=[transcript_box, voice_answer, voice_passages, voice_latency],
                )

            # ── Tab 2: Text ───────────────────────────────────────────────
            with gr.Tab("⌨️ Text Query"):
                text_in = gr.Textbox(
                    label="Your question",
                    placeholder="मैनहट्टन परियोजना की सफलता का तत्काल प्रभाव क्या था?",
                    lines=2,
                )
                text_btn = gr.Button("Ask", variant="primary")

                text_answer = gr.Textbox(
                    label="Answer",
                    interactive=False,
                    elem_id="answer-box",
                    lines=4,
                )

                with gr.Accordion("Retrieved Passages", open=False):
                    text_passages = gr.Markdown()

                with gr.Accordion("Latency Breakdown", open=False):
                    text_latency = gr.Markdown()

                text_btn.click(
                    fn=handle_text,
                    inputs=[text_in, lang_selector],
                    outputs=[text_answer, text_passages, text_latency],
                )
                text_in.submit(
                    fn=handle_text,
                    inputs=[text_in, lang_selector],
                    outputs=[text_answer, text_passages, text_latency],
                )

            # ── Tab 3: Indexing ───────────────────────────────────────────
            with gr.Tab("⚙️ Indexing"):
                build_indexing_tab(lang_selector)

            # ── Tab 4: Metrics ────────────────────────────────────────────
            with gr.Tab("📊 Metrics"):
                build_metrics_tab()

    return demo


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    build_ui().launch(server_port=7860, share=False)
