"""Prometheus metrics dashboard — Gradio tab.

Three independent timers:
  1s  → KPI markdown (serving + ingestion) + store.tick()
  5s  → time-series line charts (QPS, latency, indexing throughput)
  15s → distribution charts (bar/histograms, per-lang breakdowns)

All data comes from the in-process prometheus_client REGISTRY.
"""

from __future__ import annotations

from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import gradio as gr
from prometheus_client import REGISTRY

from ui.metrics_store import store, _sum_counter, _sum_gauge, _sum_histogram_field
from ui.theme import (
    BG as _BG, GRID as _GRID, TEXT as _TEXT, WHITE as _WHITE,
    INDIGO as _C1, CYAN as _C2, AMBER as _C3, RED as _C4, GREEN as _C5,
    MUTED as _MUTED, dark_figure,
)

# ---------------------------------------------------------------------------
# Chart style — shared with the other admin tabs (ui/theme.py)
# ---------------------------------------------------------------------------


def _fig(w: float = 6, h: float = 2.6) -> tuple[plt.Figure, plt.Axes]:
    return dark_figure(w, h, dpi=110)


def _no_data(ax: plt.Axes, msg: str = "No data yet") -> None:
    ax.text(0.5, 0.5, msg, ha="center", va="center",
            transform=ax.transAxes, color=_MUTED, fontsize=10)
    ax.set_axis_off()


def _tight(fig: plt.Figure) -> plt.Figure:
    fig.tight_layout(pad=1.0)
    return fig


# ---------------------------------------------------------------------------
# Low-level registry helpers (supplement metrics_store)
# ---------------------------------------------------------------------------

def _samples(name: str) -> list:
    for m in REGISTRY.collect():
        if m.name == name:
            return m.samples
    return []


def _counter_by_label(name: str, label: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for s in _samples(name):
        if s.name.endswith("_total"):
            k = s.labels.get(label, "?")
            out[k] = out.get(k, 0) + s.value
    return out


def _counter_by_two_labels(name: str, l1: str, l2: str) -> dict[tuple, float]:
    out: dict[tuple, float] = {}
    for s in _samples(name):
        if s.name.endswith("_total"):
            k = (s.labels.get(l1, "?"), s.labels.get(l2, "?"))
            out[k] = out.get(k, 0) + s.value
    return out


def _histogram_percentiles(
    name: str, label: str,
    percentiles: tuple[float, ...] = (0.5, 0.9, 0.99),
) -> dict[str, dict[str, float]]:
    buckets: dict[str, list[tuple[float, float]]] = {}
    counts:  dict[str, float] = {}
    for s in _samples(name):
        lv = s.labels.get(label, "all")
        if s.name.endswith("_bucket"):
            le = float(s.labels.get("le", "inf"))
            buckets.setdefault(lv, []).append((le, s.value))
        elif s.name.endswith("_count"):
            counts[lv] = s.value
    result: dict[str, dict[str, float]] = {}
    for lv, bkt in buckets.items():
        bkt.sort(key=lambda x: x[0])
        total = counts.get(lv, 0)
        if total == 0:
            continue
        pmap: dict[str, float] = {}
        for p in percentiles:
            target = p * total
            for le, cum in bkt:
                if cum >= target:
                    pmap[f"p{int(p*100)}"] = le
                    break
        result[lv] = pmap
    return result


def _histogram_per_bucket(name: str) -> tuple[list[float], list[float]]:
    bucket_map: dict[float, float] = {}
    for s in _samples(name):
        if s.name.endswith("_bucket"):
            le = float(s.labels.get("le", "inf"))
            bucket_map[le] = bucket_map.get(le, 0) + s.value
    if not bucket_map:
        return [], []
    items = sorted((le, v) for le, v in bucket_map.items() if le != float("inf"))
    les    = [x[0] for x in items]
    cumul  = [x[1] for x in items]
    per    = [cumul[0]] + [cumul[i] - cumul[i-1] for i in range(1, len(cumul))]
    return les, per


# ---------------------------------------------------------------------------
# KPI markdown (fast — 1s)
# ---------------------------------------------------------------------------

def _serving_kpi_md() -> str:
    total    = _sum_counter("rag_requests")
    qps      = store.qps(60)
    p50_pipe = store.mean_latency_ms("pipeline", 60)
    p50_gen  = store.mean_latency_ms("generation", 60)
    p50_ret  = store.mean_latency_ms("retrieval", 60)
    p50_stt  = store.mean_latency_ms("stt", 60)
    in_blk   = _sum_counter("rag_input_guardrail_blocked")
    gr_blk   = _sum_counter("rag_grounding_guardrail_blocked")
    blk_rate = ((in_blk + gr_blk) / total * 100) if total else 0
    ts       = datetime.now().strftime("%H:%M:%S")
    return (
        f"#### Serving &nbsp; <small>_updated {ts}_</small>\n\n"
        f"| Metric | Value |\n|:---|---:|\n"
        f"| Total Requests | **{int(total):,}** |\n"
        f"| QPS (60 s) | **{qps:.3f} req/s** |\n"
        f"| Pipeline mean (60 s) | **{p50_pipe:.0f} ms** |\n"
        f"| Generation mean (60 s) | **{p50_gen:.0f} ms** |\n"
        f"| Retrieval mean (60 s) | **{p50_ret:.0f} ms** |\n"
        f"| STT mean (60 s) | **{p50_stt:.0f} ms** |\n"
        f"| Guardrail block rate | **{blk_rate:.1f}%** |\n"
        f"| Input blocks | **{int(in_blk):,}** |\n"
        f"| Grounding blocks | **{int(gr_blk):,}** |\n"
    )


def _ingestion_kpi_md() -> str:
    running  = store.indexing_running()
    chunks   = _sum_gauge("indexing_chunks_done")
    target   = _sum_gauge("indexing_chunks_target")
    thru     = _sum_gauge("indexing_throughput_chunks_per_min")
    progress = (chunks / target * 100) if target else 0
    status   = "🟢 Running" if running else "⚫ Idle"
    eta_str  = ""
    if running and thru > 0 and target > chunks:
        eta_min = (target - chunks) / thru
        eta_str = f" &nbsp;·&nbsp; ETA ~{eta_min:.0f} min"
    ts = datetime.now().strftime("%H:%M:%S")
    return (
        f"#### Ingestion &nbsp; <small>_updated {ts}_</small>\n\n"
        f"| Metric | Value |\n|:---|---:|\n"
        f"| Status | **{status}{eta_str}** |\n"
        f"| Chunks Done | **{int(chunks):,}** / {int(target):,} |\n"
        f"| Progress | **{progress:.1f}%** |\n"
        f"| Throughput | **{thru:,.0f} chunks/min** |\n"
    )


def tick_kpis() -> tuple[str, str]:
    """1-second timer: sample the store AND return updated KPI markdown."""
    store.tick()
    return _serving_kpi_md(), _ingestion_kpi_md()


# ---------------------------------------------------------------------------
# Time-series charts (5s)
# ---------------------------------------------------------------------------

def _fmt_elapsed(ax: plt.Axes, xs: list[float]) -> None:
    """Format x-axis as MM:SS elapsed."""
    def _fmt(v, _):
        m, s = divmod(int(v), 60)
        return f"{m}:{s:02d}"
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(_fmt))
    ax.set_xlabel("elapsed (mm:ss)", fontsize=7, color=_TEXT)


def build_qps_chart() -> plt.Figure:
    fig, ax = _fig()
    xs, ys = store.qps_series(300)
    if xs and any(y > 0 for y in ys):
        ax.plot(xs, ys, color=_C1, linewidth=1.5)
        ax.fill_between(xs, ys, alpha=0.15, color=_C1)
        ax.set_ylabel("req/s", fontsize=7, color=_TEXT)
        _fmt_elapsed(ax, xs)
    else:
        _no_data(ax, "No requests yet")
    ax.set_title("Request Rate (5 min)", fontsize=9, color=_WHITE, pad=6)
    return _tight(fig)


def build_latency_ts_chart() -> plt.Figure:
    fig, ax = _fig()
    has_data = False
    for step, color, label in [
        ("pipeline",   _C1,  "Pipeline"),
        ("generation", _C3,  "Generation"),
        ("retrieval",  _C5,  "Retrieval"),
    ]:
        xs, ys = store.latency_series(step, 300)
        if xs and any(y > 0 for y in ys):
            ax.plot(xs, ys, color=color, linewidth=1.5, label=label)
            has_data = True
    if has_data:
        ax.set_ylabel("mean ms", fontsize=7, color=_TEXT)
        _fmt_elapsed(ax, xs)  # type: ignore[possibly-undefined]
        ax.legend(fontsize=7, facecolor=_BG, labelcolor=_TEXT, edgecolor=_GRID)
    else:
        _no_data(ax, "No latency data yet")
    ax.set_title("Latency over Time (5 min)", fontsize=9, color=_WHITE, pad=6)
    return _tight(fig)


def build_indexing_ts_chart() -> plt.Figure:
    xs, chunks, thru = store.indexing_series(300)
    fig, ax1 = _fig()
    if xs and any(c > 0 for c in chunks):
        ax1.fill_between(xs, chunks, alpha=0.12, color=_C1)
        ax1.plot(xs, chunks, color=_C1, linewidth=1.5, label="Chunks")
        ax1.set_ylabel("cumulative chunks", fontsize=7, color=_C1)
        ax1.tick_params(axis="y", colors=_C1)
        _fmt_elapsed(ax1, xs)
        if any(t > 0 for t in thru):
            ax2 = ax1.twinx()
            ax2.set_facecolor(_BG)
            ax2.plot(xs, thru, color=_C3, linewidth=1.2, linestyle="--", alpha=0.8)
            ax2.set_ylabel("chunks/min", fontsize=7, color=_C3)
            ax2.tick_params(axis="y", colors=_C3)
            ax2.spines[["top", "right", "left", "bottom"]].set_color(_GRID)
    else:
        _no_data(ax1, "No indexing data yet")
    ax1.set_title("Indexing Throughput (5 min)", fontsize=9, color=_WHITE, pad=6)
    return _tight(fig)


def refresh_timeseries() -> tuple[plt.Figure, plt.Figure, plt.Figure]:
    return build_qps_chart(), build_latency_ts_chart(), build_indexing_ts_chart()


# ---------------------------------------------------------------------------
# Distribution charts (15s)
# ---------------------------------------------------------------------------

def _bar(ax: plt.Axes, x, heights, labels=None, color=_C1, width=0.6) -> None:
    bars = ax.bar(x, heights, width, color=color, alpha=0.85, edgecolor=_BG)
    if labels is not None:
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, fontsize=7)


def build_req_by_lang_chart() -> plt.Figure:
    data = _counter_by_two_labels("rag_requests", "lang", "endpoint")
    langs = sorted({k[0] for k in data})
    fig, ax = _fig()
    if langs:
        x = np.arange(len(langs))
        tv = [data.get((l, "text"),  0) for l in langs]
        vv = [data.get((l, "voice"), 0) for l in langs]
        ax.bar(x - 0.2, tv, 0.35, label="Text",  color=_C1, alpha=0.85)
        ax.bar(x + 0.2, vv, 0.35, label="Voice", color=_C2, alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(langs, fontsize=7)
        ax.legend(fontsize=7, facecolor=_BG, labelcolor=_TEXT, edgecolor=_GRID)
        ax.set_ylabel("requests", fontsize=7, color=_TEXT)
    else:
        _no_data(ax)
    ax.set_title("Requests by Language", fontsize=9, color=_WHITE, pad=6)
    return _tight(fig)


def build_latency_percentile_chart() -> plt.Figure:
    steps = {
        "STT": "rag_stt_latency_seconds",
        "Retrieval": "rag_retrieval_latency_seconds",
        "Generation": "rag_generation_latency_seconds",
        "Pipeline": "rag_pipeline_latency_seconds",
    }
    fig, ax = _fig()
    names = list(steps.keys())
    x = np.arange(len(names))
    w = 0.22
    has_data = False
    for i, (pk, color) in enumerate([("p50", _C5), ("p90", _C1), ("p99", _C4)]):
        vals = []
        for mname in steps.values():
            percs = _histogram_percentiles(mname, "lang")
            if percs:
                best = max((v.get(pk, 0) for v in percs.values()), default=0)
                vals.append(best * 1000)
                has_data = True
            else:
                vals.append(0)
        ax.bar(x + (i - 1) * w, vals, w, label=pk, color=color, alpha=0.85)
    if has_data:
        ax.set_xticks(x); ax.set_xticklabels(names, fontsize=7)
        ax.set_ylabel("ms", fontsize=7, color=_TEXT)
        ax.legend(fontsize=7, facecolor=_BG, labelcolor=_TEXT, edgecolor=_GRID)
    else:
        _no_data(ax)
    ax.set_title("Latency Percentiles per Step", fontsize=9, color=_WHITE, pad=6)
    return _tight(fig)


def build_guardrail_chart() -> plt.Figure:
    in_data  = _counter_by_label("rag_input_guardrail_blocked", "lang")
    gr_data  = _counter_by_label("rag_grounding_guardrail_blocked", "lang")
    langs    = sorted(set(in_data) | set(gr_data))
    fig, ax  = _fig()
    if langs:
        x  = np.arange(len(langs))
        iv = [in_data.get(l, 0) for l in langs]
        gv = [gr_data.get(l, 0) for l in langs]
        ax.bar(x, iv, 0.5, label="Input blocked",    color=_C4, alpha=0.85)
        ax.bar(x, gv, 0.5, bottom=iv, label="Grounding", color=_C3, alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(langs, fontsize=7)
        ax.set_ylabel("blocked", fontsize=7, color=_TEXT)
        ax.legend(fontsize=7, facecolor=_BG, labelcolor=_TEXT, edgecolor=_GRID)
    else:
        _no_data(ax, "No guardrail blocks yet")
    ax.set_title("Guardrail Blocks by Language", fontsize=9, color=_WHITE, pad=6)
    return _tight(fig)


def build_passages_chart() -> plt.Figure:
    les, counts = _histogram_per_bucket("rag_passages_retrieved")
    fig, ax = _fig()
    if les and any(c > 0 for c in counts):
        ax.bar([str(int(l)) for l in les], counts, color=_C1, alpha=0.85)
        ax.set_xlabel("passages returned", fontsize=7, color=_TEXT)
        ax.set_ylabel("queries", fontsize=7, color=_TEXT)
    else:
        _no_data(ax, "No retrieval data yet")
    ax.set_title("Passages Retrieved per Query", fontsize=9, color=_WHITE, pad=6)
    return _tight(fig)


def build_per_lang_latency_chart(metric_name: str, title: str) -> plt.Figure:
    per_lang = _histogram_percentiles(metric_name, "lang")
    fig, ax  = _fig()
    if per_lang:
        langs = sorted(per_lang)
        x = np.arange(len(langs))
        w = 0.25
        for i, (pk, color) in enumerate([("p50", _C5), ("p90", _C1), ("p99", _C4)]):
            vals = [per_lang[l].get(pk, 0) * 1000 for l in langs]
            ax.bar(x + (i - 1) * w, vals, w, label=pk, color=color, alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(langs, fontsize=7)
        ax.set_ylabel("ms", fontsize=7, color=_TEXT)
        ax.legend(fontsize=7, facecolor=_BG, labelcolor=_TEXT, edgecolor=_GRID)
    else:
        _no_data(ax)
    ax.set_title(title, fontsize=9, color=_WHITE, pad=6)
    return _tight(fig)


def refresh_distributions() -> tuple:
    return (
        build_req_by_lang_chart(),
        build_latency_percentile_chart(),
        build_guardrail_chart(),
        build_passages_chart(),
        build_per_lang_latency_chart("rag_stt_latency_seconds",        "STT Latency by Language"),
        build_per_lang_latency_chart("rag_retrieval_latency_seconds",  "Retrieval Latency by Language"),
        build_per_lang_latency_chart("rag_generation_latency_seconds", "Generation Latency by Language"),
        build_per_lang_latency_chart("rag_pipeline_latency_seconds",   "Pipeline Latency by Language"),
    )


# ---------------------------------------------------------------------------
# Tab builder
# ---------------------------------------------------------------------------

def build_metrics_tab() -> None:
    """Render the metrics tab. Call inside a `with gr.Tab(...)` block."""

    gr.Markdown(
        "Real-time Prometheus dashboard. "
        "KPIs refresh every **1 s**, time-series every **5 s**, "
        "distributions every **15 s**. "
        "Raw metrics at [`/metrics`](/metrics)."
    )

    # ── KPI row ───────────────────────────────────────────────────────────────
    with gr.Group():
        with gr.Row():
            serving_kpi  = gr.Markdown(value=_serving_kpi_md(),   min_height=0)
            indexing_kpi = gr.Markdown(value=_ingestion_kpi_md(), min_height=0)

    # ── Time-series ───────────────────────────────────────────────────────────
    gr.Markdown("### Time-series (5 s)")
    with gr.Group():
        with gr.Row():
            qps_plot     = gr.Plot(value=build_qps_chart(),        show_label=False)
            lat_ts_plot  = gr.Plot(value=build_latency_ts_chart(), show_label=False)
        with gr.Row():
            idx_ts_plot  = gr.Plot(value=build_indexing_ts_chart(), show_label=False)

    # ── Serving distributions ─────────────────────────────────────────────────
    gr.Markdown("### Serving Distributions (15 s)")
    with gr.Group():
        with gr.Row():
            req_plot  = gr.Plot(value=build_req_by_lang_chart(),        show_label=False)
            lat_plot  = gr.Plot(value=build_latency_percentile_chart(), show_label=False)
        with gr.Row():
            rail_plot = gr.Plot(value=build_guardrail_chart(),  show_label=False)
            pass_plot = gr.Plot(value=build_passages_chart(),   show_label=False)

    gr.Markdown("### Per-language Latency (15 s)")
    with gr.Group():
        with gr.Row():
            stt_plot  = gr.Plot(
                value=build_per_lang_latency_chart("rag_stt_latency_seconds", "STT Latency by Language"),
                show_label=False,
            )
            ret_plot  = gr.Plot(
                value=build_per_lang_latency_chart("rag_retrieval_latency_seconds", "Retrieval Latency by Language"),
                show_label=False,
            )
        with gr.Row():
            gen_plot  = gr.Plot(
                value=build_per_lang_latency_chart("rag_generation_latency_seconds", "Generation Latency by Language"),
                show_label=False,
            )
            pipe_plot = gr.Plot(
                value=build_per_lang_latency_chart("rag_pipeline_latency_seconds", "Pipeline Latency by Language"),
                show_label=False,
            )

    # ── Timers ────────────────────────────────────────────────────────────────

    timer_1s  = gr.Timer(value=1)
    timer_5s  = gr.Timer(value=5)
    timer_15s = gr.Timer(value=15)

    timer_1s.tick(
        fn=tick_kpis,
        outputs=[serving_kpi, indexing_kpi],
    )
    timer_5s.tick(
        fn=refresh_timeseries,
        outputs=[qps_plot, lat_ts_plot, idx_ts_plot],
    )
    timer_15s.tick(
        fn=refresh_distributions,
        outputs=[req_plot, lat_plot, rail_plot, pass_plot,
                 stt_plot, ret_plot, gen_plot, pipe_plot],
    )
