"""Pipeline Journey tab — visualize every step from raw dataset record to Qdrant point.

Shows:
  1. A flowchart graph of the full pipeline
  2. JSON panels at each stage (what the data looks like)

Updates live when backend / chunker strategy is changed.
"""

from __future__ import annotations

import json
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.figure import Figure

import gradio as gr

from pipeline.embedder import AVAILABLE_BACKENDS, VECTOR_DIM_FOR
from pipeline.index_plan import IndexPlan, MODEL_NAME_FOR
from pipeline.chunking import REGISTRY
from ui.theme import BG as _BG

# ---------------------------------------------------------------------------
# Colours (dark theme) — background shared with the other admin tabs
# (ui/theme.py); the stage colors below are semantic (record/chunk/embed/...)
# and stay specific to this flowchart.
# ---------------------------------------------------------------------------

_RECORD  = ("#1e3a5f", "#93c5fd")   # bg, border
_CHUNK   = ("#1a3320", "#86efac")
_EMBED   = ("#2d1b4e", "#c4b5fd")
_DOC     = ("#1e3a5f", "#93c5fd")
_QDRANT  = ("#1e3220", "#6ee7b7")
_QUERY   = ("#3b1f1f", "#fca5a5")
_TEXT    = "#f1f5f9"
_ARROW   = "#64748b"
_LABEL   = "#94a3b8"

# ---------------------------------------------------------------------------
# Example record (realistic MSMARCO-XI Hindi entry)
# ---------------------------------------------------------------------------

_RECORD_EXAMPLE = {
    "query_id":    174,
    "passage_id":  "7527097__2",
    "lang":        "hi",
    "query":       "What is the capital of India?",          # English (original MSMARCO)
    "text":        "नई दिल्ली भारत की राजधानी और देश का सबसे बड़ा महानगर है।",   # Hindi (translated)
    "answer":      "नई दिल्ली",
    "is_selected": True,
    "query_type":  "description",
}

# ---------------------------------------------------------------------------
# Stage JSON builders
# ---------------------------------------------------------------------------

def _chunk_json(chunker: str) -> dict:
    rec = _RECORD_EXAMPLE
    base = dict(
        chunk_id    = f"{rec['passage_id']}__{chunker[:3]}",
        chunk_type  = chunker,
        lang        = rec["lang"],
        passage_id  = rec["passage_id"],
        query_id    = rec["query_id"],
        is_selected = rec["is_selected"],
        query       = rec["query"],
        answer      = rec["answer"],
        query_type  = rec["query_type"],
    )
    if chunker == "passage":
        base["text"]           = rec["text"]
        base["parent_passage"] = ""
        base["__embedded__"]   = "text  →  vernacular passage"
    elif chunker == "sentence":
        base["text"]           = "नई दिल्ली भारत की राजधानी है।"   # first sentence
        base["parent_passage"] = rec["text"]
        base["__embedded__"]   = "text  →  vernacular sentence"
    elif chunker == "qa_pair":
        base["text"]           = rec["query"] + " " + rec["text"]
        base["parent_passage"] = rec["text"]
        base["__embedded__"]   = "text  →  English query + vernacular passage"
    elif chunker == "english_query":
        base["text"]           = rec["query"]        # English question ← embedded
        base["parent_passage"] = rec["text"]         # Hindi passage   ← returned
        base["__embedded__"]   = "text  →  English question only"
    return base


def _embed_json(chunker: str, backend: str) -> dict:
    chunk  = _chunk_json(chunker)
    dim    = VECTOR_DIM_FOR.get(backend, 384)
    model  = MODEL_NAME_FOR.get(backend, backend)
    sample = [round(0.023 + i * 0.003, 4) for i in range(5)]
    needs_translate = chunker == "english_query"
    out = {
        "model":   model,
        "backend": backend,
        "dim":     dim,
    }
    if needs_translate:
        out["translate_step"] = {
            "user_input":    "भारत की राजधानी क्या है?",
            "engine":        "Sarvam sarvam-translate:v1",
            "→ english":     "What is the capital of India?",
        }
    out["input_text"] = chunk["text"]
    out["vector"]     = sample + [f"... ({dim} floats total)"]
    return out


def _document_json(chunker: str) -> dict:
    chunk = _chunk_json(chunker)
    meta  = {k: v for k, v in chunk.items() if k not in ("text", "__embedded__")}
    return {
        "page_content": chunk["text"],
        "metadata":     meta,
    }


def _qdrant_json(chunker: str, backend: str) -> dict:
    plan = IndexPlan(backend=backend, chunkers=[chunker])
    doc  = _document_json(chunker)
    dim  = VECTOR_DIM_FOR.get(backend, 384)
    sample = [round(0.023 + i * 0.003, 4) for i in range(5)]
    return {
        "collection": plan.collection_name,
        "id":         "a3f8d2b1-...",
        "vector":     sample + [f"... ({dim} floats)"],
        "payload":    doc,
    }


def _query_journey_json(chunker: str, backend: str) -> dict:
    needs_translate = chunker == "english_query"
    chunk  = _chunk_json(chunker)
    model  = MODEL_NAME_FOR.get(backend, backend)
    plan   = IndexPlan(backend=backend, chunkers=[chunker])
    ctx    = chunk.get("parent_passage") or chunk["text"]
    out: dict = {"user_voice": "भारत की राजधानी क्या है?  (Hindi speech)"}
    out["→ STT (Sarvam saaras:v3)"] = {"transcript": "भारत की राजधानी क्या है?", "detected_lang": "hi"}
    if needs_translate:
        out["→ translate (Sarvam)"] = {"input": "भारत की राजधानी क्या है?", "output": "What is the capital of India?"}
    out["→ embed"] = {"model": model, "input": chunk["text"] if not needs_translate else "What is the capital of India?"}
    out["→ qdrant search"] = {"collection": plan.collection_name, "filter": {"lang": "hi", "chunk_type": chunker}, "top_k": 5}
    out["→ context returned"] = ctx
    out["→ Sarvam LLM"]  = {"generates_in": "Hindi", "answer": "नई दिल्ली भारत की राजधानी है।"}
    return out


# ---------------------------------------------------------------------------
# Pipeline graph
# ---------------------------------------------------------------------------

def _draw_box(ax, x, y, w, h, label, sublabel, fill, border, fontsize=9):
    box = mpatches.FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.04",
        facecolor=fill, edgecolor=border, linewidth=1.5,
        zorder=3,
    )
    ax.add_patch(box)
    ax.text(x, y + 0.08, label, ha="center", va="center",
            fontsize=fontsize, color=_TEXT, fontweight="bold", zorder=4)
    if sublabel:
        ax.text(x, y - 0.22, sublabel, ha="center", va="center",
                fontsize=7, color=_LABEL, zorder=4)


def _draw_arrow(ax, x, y_top, y_bot, label=""):
    ax.annotate("", xy=(x, y_bot + 0.02),
                xytext=(x, y_top - 0.02),
                arrowprops=dict(arrowstyle="-|>", color=_ARROW,
                                lw=1.5, mutation_scale=14),
                zorder=2)
    if label:
        ax.text(x + 0.45, (y_top + y_bot) / 2, label,
                ha="left", va="center", fontsize=7, color=_LABEL,
                style="italic", zorder=4)


def _make_pipeline_graph(backend: str, chunker: str) -> Figure:
    model   = MODEL_NAME_FOR.get(backend, backend)
    dim     = VECTOR_DIM_FOR.get(backend, 384)
    chunk   = _chunk_json(chunker)
    is_en   = chunker == "english_query"

    # --- layout constants ---
    W, H    = 7.0, 0.75    # box width, box half-height
    X       = 5.0          # centre x
    GAP     = 0.35         # gap between box edge and arrow tip
    STEP    = 1.85         # vertical distance between box centres

    # Stage y-centres (top → bottom)
    y_record = 12.8
    y_chunk  = y_record  - STEP
    y_embed  = y_chunk   - STEP
    y_doc    = y_embed   - STEP
    y_qdrant = y_doc     - STEP
    # divider + query section
    y_div    = y_qdrant  - 1.1
    y_query  = y_div     - 1.1
    y_ctx    = y_query   - STEP

    fig_h = y_record + H + 0.5
    fig_w = 11.0

    fig = Figure(figsize=(fig_w, fig_h), dpi=100)
    fig.patch.set_facecolor(_BG)
    ax = fig.add_subplot(111)
    ax.set_facecolor(_BG)
    ax.set_xlim(0, fig_w)
    ax.set_ylim(y_ctx - H - 0.5, y_record + H + 0.3)
    ax.set_aspect("equal")
    ax.set_axis_off()

    # --- Stage 1: Dataset Record ---
    rec_sub = (
        f"query (EN): \"{_RECORD_EXAMPLE['query']}\"  |  "
        f"text (vernacular): [Hindi passage]  |  lang: hi"
    )
    _draw_box(ax, X, y_record, W, H, "① MSMARCO-XI Record  (PassageRecord)", rec_sub,
              *_RECORD)

    # --- Arrow 1 ---
    chunker_label = f"chunker: {chunker}"
    src_field  = "record.query  →  chunk.text" if is_en else "record.text  →  chunk.text"
    if chunker == "sentence":
        src_field = "sentence(record.text)  →  chunk.text"
    elif chunker == "qa_pair":
        src_field = "query+text  →  chunk.text"
    arrow1_lbl = f"{chunker_label}\n{src_field}"
    _draw_arrow(ax, X, y_record - H, y_chunk + H, "")
    ax.text(X + 0.45, (y_record - H + y_chunk + H) / 2,
            arrow1_lbl, ha="left", va="center", fontsize=7,
            color="#86efac", style="italic", zorder=4)
    if is_en:
        ax.text(X + 0.45, (y_record - H + y_chunk + H) / 2 - 0.2,
                "record.text  →  parent_passage  (returned at query time)",
                ha="left", va="center", fontsize=6.5,
                color="#fde68a", style="italic", zorder=4)

    # --- Stage 2: Chunk ---
    embedded_field = chunk.get("__embedded__", "")
    chunk_text_preview = chunk["text"] if chunk["text"].isascii() else "[vernacular text]"
    chunk_sub = f"chunk.text: \"{chunk_text_preview[:50]}\"  |  {embedded_field}"
    _draw_box(ax, X, y_chunk, W, H, "② Chunk Object", chunk_sub, *_CHUNK)

    # --- Arrow 2 ---
    _draw_arrow(ax, X, y_chunk - H, y_embed + H,
                f"embed_passages([chunk.text], backend=\"{backend}\")")

    # --- Stage 3: Embedding ---
    embed_sub = f"model: {model}  |  → vector[{dim}]"
    _draw_box(ax, X, y_embed, W, H, "③ Embedding", embed_sub, *_EMBED)

    # --- Arrow 3 ---
    _draw_arrow(ax, X, y_embed - H, y_doc + H, "LangChain Document wrapper")

    # --- Stage 4: LangChain Document ---
    doc_sub = ("page_content = chunk.text  |  "
               "metadata = {lang, chunk_type, passage_id, parent_passage, …}")
    _draw_box(ax, X, y_doc, W, H, "④ LangChain Document", doc_sub, *_DOC)

    # --- Arrow 4 ---
    plan = IndexPlan(backend=backend, chunkers=[chunker])
    _draw_arrow(ax, X, y_doc - H, y_qdrant + H,
                f"vectorstore.add_documents()  →  {plan.collection_name}")

    # --- Stage 5: Qdrant Point ---
    qdrant_sub = (f"vector[{dim}]  +  payload.page_content  +  "
                  "metadata.parent_passage  (vernacular context at retrieval)")
    _draw_box(ax, X, y_qdrant, W, H, "⑤ Qdrant Point", qdrant_sub, *_QDRANT)

    # --- Divider ---
    ax.axhline(y_div, xmin=0.05, xmax=0.95,
               color="#334155", linewidth=1.0, linestyle="--", zorder=1)
    ax.text(X, y_div + 0.1, "─── query time ───",
            ha="center", va="bottom", fontsize=8, color="#475569")

    # --- Arrow 5 ---
    steps = "user speech  →  Sarvam STT"
    if is_en:
        steps += "  →  Sarvam translate EN"
    steps += f"  →  embed (same {backend} model)  →  Qdrant filter(lang, chunk_type)"
    _draw_arrow(ax, X, y_div - 0.05, y_query + H, steps)

    # --- Stage 6: Query ---
    q_sub = (f"filter: {{lang: hi, chunk_type: {chunker}}}  →  "
             "retrieve top-k  →  extract parent_passage (or page_content)")
    _draw_box(ax, X, y_query, W, H, "⑥ Retrieval + Context", q_sub, *_QUERY)

    # --- Arrow 6 ---
    _draw_arrow(ax, X, y_query - H, y_ctx + H, "Sarvam LLM → answer in target language")

    # --- Stage 7: Output ---
    ctx_text = chunk.get("parent_passage") or chunk["text"]
    ctx_preview = ctx_text if ctx_text.isascii() else "[vernacular passage text]"
    out_sub  = f"context from parent_passage: \"{ctx_preview[:55]}\""
    _draw_box(ax, X, y_ctx, W, H, "⑦ Generated Answer (Hindi)", out_sub, *_QUERY)

    fig.tight_layout(pad=0.5)
    return fig


# ---------------------------------------------------------------------------
# Tab builder
# ---------------------------------------------------------------------------

def build_journey_tab() -> None:
    gr.Markdown(
        "Trace a single MSMARCO-XI record through the full indexing and retrieval pipeline. "
        "Change the **embedding model** and **chunking strategy** to see how the fields shift at each stage."
    )

    with gr.Group():
        with gr.Row():
            backend_dd = gr.Dropdown(
                choices=AVAILABLE_BACKENDS, value=AVAILABLE_BACKENDS[0],
                label="Embedding model (backend)", scale=1,
            )
            chunker_dd = gr.Dropdown(
                choices=list(REGISTRY.keys()), value="english_query",
                label="Chunking strategy", scale=1,
            )

    gr.Markdown("### Pipeline Flowchart")
    with gr.Group():
        graph_plot = gr.Plot(label="", show_label=False)

    gr.Markdown("### Stage-by-stage JSON")

    with gr.Accordion("① Raw Dataset Record  (same for all strategies)", open=True):
        gr.Code(
            value=json.dumps(_RECORD_EXAMPLE, ensure_ascii=False, indent=2),
            language="json", label="PassageRecord",
        )

    with gr.Accordion("② Chunk Object  (changes per strategy)", open=True):
        chunk_code = gr.Code(language="json", label="Chunk")

    with gr.Accordion("③ Embedding input/output", open=True):
        embed_code = gr.Code(language="json", label="embed_passages()")

    with gr.Accordion("④ LangChain Document", open=False):
        doc_code = gr.Code(language="json", label="Document")

    with gr.Accordion("⑤ Qdrant Point  (stored in collection)", open=True):
        qdrant_code = gr.Code(language="json", label="Qdrant payload")

    with gr.Accordion("⑥⑦ Query-time journey", open=True):
        query_code = gr.Code(language="json", label="Full query journey")

    # ── Update all components when selection changes ──────────────────────
    def _update(backend: str, chunker: str):
        return (
            _make_pipeline_graph(backend, chunker),
            json.dumps(_chunk_json(chunker),            ensure_ascii=False, indent=2),
            json.dumps(_embed_json(chunker, backend),   ensure_ascii=False, indent=2),
            json.dumps(_document_json(chunker),         ensure_ascii=False, indent=2),
            json.dumps(_qdrant_json(chunker, backend),  ensure_ascii=False, indent=2),
            json.dumps(_query_journey_json(chunker, backend), ensure_ascii=False, indent=2),
        )

    _OUTPUTS = [graph_plot, chunk_code, embed_code, doc_code, qdrant_code, query_code]

    backend_dd.change(fn=_update, inputs=[backend_dd, chunker_dd], outputs=_OUTPUTS)
    chunker_dd.change(fn=_update, inputs=[backend_dd, chunker_dd], outputs=_OUTPUTS)
