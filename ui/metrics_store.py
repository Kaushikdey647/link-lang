"""Rolling time-series store for the metrics dashboard.

Maintains a 5-minute circular buffer sampled at 1 Hz by the Gradio timer.
All reads are thread-safe. Designed to be imported as a singleton.

Usage:
    from ui.metrics_store import store

    # In the 1-second Gradio timer callback:
    store.tick()

    # Query helpers:
    store.qps(window_s=60)
    store.mean_latency_ms("pipeline", window_s=60)
    store.qps_series(window_s=300)   -> (elapsed_s[], qps[])
"""

from __future__ import annotations

import threading
import time
from collections import deque

from prometheus_client import REGISTRY

_WINDOW = 300  # 5 min at 1 sample/sec


# ---------------------------------------------------------------------------
# Low-level registry readers
# ---------------------------------------------------------------------------

def _samples(metric_name: str) -> list:
    for m in REGISTRY.collect():
        if m.name == metric_name:
            return m.samples
    return []


def _sum_counter(metric_name: str) -> float:
    """Sum all _total samples of a Counter across all label combinations."""
    return sum(
        s.value for s in _samples(metric_name)
        if s.name.endswith("_total")
    )


def _sum_histogram_field(metric_name: str, suffix: str) -> float:
    """Sum _sum or _count samples of a Histogram across all label combinations."""
    full = f"{metric_name}{suffix}"
    return sum(s.value for s in _samples(metric_name) if s.name == full)


def _sum_gauge(metric_name: str) -> float:
    """Sum all samples of a Gauge across all label combinations."""
    return sum(s.value for s in _samples(metric_name) if s.name == metric_name)


def _get_gauge_scalar(metric_name: str) -> float:
    """Return the single (unlabeled) value of a Gauge."""
    for s in _samples(metric_name):
        if s.name == metric_name:
            return s.value
    return 0.0


# ---------------------------------------------------------------------------
# Rolling store
# ---------------------------------------------------------------------------

class _RollingStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        # Shared timestamps
        self._times: deque[float] = deque(maxlen=_WINDOW)

        # --- Serving ---
        self._req:       deque[float] = deque(maxlen=_WINDOW)

        # Histogram sum + count for mean latency computation
        self._pipe_sum:  deque[float] = deque(maxlen=_WINDOW)
        self._pipe_cnt:  deque[float] = deque(maxlen=_WINDOW)
        self._ret_sum:   deque[float] = deque(maxlen=_WINDOW)
        self._ret_cnt:   deque[float] = deque(maxlen=_WINDOW)
        self._gen_sum:   deque[float] = deque(maxlen=_WINDOW)
        self._gen_cnt:   deque[float] = deque(maxlen=_WINDOW)
        self._stt_sum:   deque[float] = deque(maxlen=_WINDOW)
        self._stt_cnt:   deque[float] = deque(maxlen=_WINDOW)

        # --- Ingestion ---
        self._idx_chunks: deque[float] = deque(maxlen=_WINDOW)
        self._idx_target: deque[float] = deque(maxlen=_WINDOW)
        self._idx_thru:   deque[float] = deque(maxlen=_WINDOW)
        self._idx_run:    deque[float] = deque(maxlen=_WINDOW)  # 0 or 1

    # ── Public: sample ───────────────────────────────────────────────────────

    def tick(self) -> None:
        """Read current Prometheus values and append to buffers. Call every 1s."""
        now = time.time()

        req       = _sum_counter("rag_requests")
        pipe_sum  = _sum_histogram_field("rag_pipeline_latency_seconds",  "_sum")
        pipe_cnt  = _sum_histogram_field("rag_pipeline_latency_seconds",  "_count")
        ret_sum   = _sum_histogram_field("rag_retrieval_latency_seconds",  "_sum")
        ret_cnt   = _sum_histogram_field("rag_retrieval_latency_seconds",  "_count")
        gen_sum   = _sum_histogram_field("rag_generation_latency_seconds", "_sum")
        gen_cnt   = _sum_histogram_field("rag_generation_latency_seconds", "_count")
        stt_sum   = _sum_histogram_field("rag_stt_latency_seconds",        "_sum")
        stt_cnt   = _sum_histogram_field("rag_stt_latency_seconds",        "_count")

        idx_chunks = _sum_gauge("indexing_chunks_done")
        idx_target = _sum_gauge("indexing_chunks_target")
        idx_thru   = _sum_gauge("indexing_throughput_chunks_per_min")
        idx_run    = _get_gauge_scalar("indexing_running")

        with self._lock:
            self._times.append(now)
            self._req.append(req)
            self._pipe_sum.append(pipe_sum); self._pipe_cnt.append(pipe_cnt)
            self._ret_sum.append(ret_sum);   self._ret_cnt.append(ret_cnt)
            self._gen_sum.append(gen_sum);   self._gen_cnt.append(gen_cnt)
            self._stt_sum.append(stt_sum);   self._stt_cnt.append(stt_cnt)
            self._idx_chunks.append(idx_chunks)
            self._idx_target.append(idx_target)
            self._idx_thru.append(idx_thru)
            self._idx_run.append(idx_run)

    # ── Public: scalar queries ────────────────────────────────────────────────

    def qps(self, window_s: int = 60) -> float:
        """Request rate (req/s) over the last window_s seconds."""
        return self._counter_rate(self._req, window_s)

    def mean_latency_ms(self, step: str, window_s: int = 60) -> float:
        """Mean latency in ms for a pipeline step over the last window_s seconds."""
        _sums = {"pipeline": self._pipe_sum, "retrieval": self._ret_sum,
                 "generation": self._gen_sum, "stt": self._stt_sum}
        _cnts = {"pipeline": self._pipe_cnt, "retrieval": self._ret_cnt,
                 "generation": self._gen_cnt, "stt": self._stt_cnt}
        return self._histogram_mean(_sums[step], _cnts[step], window_s) * 1000

    def indexing_throughput(self) -> float:
        """Latest chunks/min reported by the indexing gauge."""
        with self._lock:
            return self._idx_thru[-1] if self._idx_thru else 0.0

    def indexing_running(self) -> bool:
        with self._lock:
            return bool(self._idx_run[-1] > 0.5) if self._idx_run else False

    # ── Public: time-series for charts ───────────────────────────────────────

    def qps_series(self, window_s: int = 300) -> tuple[list[float], list[float]]:
        """Instantaneous req/s between consecutive 1-sec samples."""
        return self._rate_series(self._req, window_s)

    def latency_series(
        self, step: str, window_s: int = 300
    ) -> tuple[list[float], list[float]]:
        """Mean latency in ms between consecutive samples for one pipeline step."""
        _sums = {"pipeline": self._pipe_sum, "retrieval": self._ret_sum,
                 "generation": self._gen_sum, "stt": self._stt_sum}
        _cnts = {"pipeline": self._pipe_cnt, "retrieval": self._ret_cnt,
                 "generation": self._gen_cnt, "stt": self._stt_cnt}
        return self._mean_series(_sums[step], _cnts[step], window_s, scale=1000.0)

    def indexing_series(
        self, window_s: int = 300
    ) -> tuple[list[float], list[float], list[float]]:
        """(elapsed_s, chunks_done, throughput_chunks_per_min) over the window."""
        with self._lock:
            times  = list(self._times)
            chunks = list(self._idx_chunks)
            thru   = list(self._idx_thru)
        if not times:
            return [], [], []
        now     = time.time()
        cutoff  = now - window_s
        triples = [(t, c, r) for t, c, r in zip(times, chunks, thru) if t >= cutoff]
        if not triples:
            return [], [], []
        t0 = triples[0][0]
        return (
            [x[0] - t0 for x in triples],
            [x[1] for x in triples],
            [x[2] for x in triples],
        )

    # ── Internals ────────────────────────────────────────────────────────────

    def _snapshot(self, series: deque, window_s: int) -> tuple[list, list]:
        with self._lock:
            times = list(self._times)
            vals  = list(series)
        cutoff = time.time() - window_s
        pairs = [(t, v) for t, v in zip(times, vals) if t >= cutoff]
        return pairs, (pairs[0][0] if pairs else 0.0)

    def _counter_rate(self, series: deque, window_s: int) -> float:
        pairs, _ = self._snapshot(series, window_s)
        if len(pairs) < 2:
            return 0.0
        dt = pairs[-1][0] - pairs[0][0]
        dv = pairs[-1][1] - pairs[0][1]
        return dv / dt if dt > 0 else 0.0

    def _histogram_mean(self, sums: deque, cnts: deque, window_s: int) -> float:
        with self._lock:
            times = list(self._times)
            sv    = list(sums)
            cv    = list(cnts)
        cutoff = time.time() - window_s
        pairs  = [(t, s, c) for t, s, c in zip(times, sv, cv) if t >= cutoff]
        if len(pairs) < 2:
            return 0.0
        d_sum = pairs[-1][1] - pairs[0][1]
        d_cnt = pairs[-1][2] - pairs[0][2]
        return d_sum / d_cnt if d_cnt > 0 else 0.0

    def _rate_series(self, series: deque, window_s: int) -> tuple[list, list]:
        pairs, t0 = self._snapshot(series, window_s)
        if len(pairs) < 2:
            return [], []
        xs, ys = [], []
        for i in range(1, len(pairs)):
            dt = pairs[i][0] - pairs[i - 1][0]
            dv = pairs[i][1] - pairs[i - 1][1]
            if dt > 0:
                xs.append(pairs[i][0] - t0)
                ys.append(max(0.0, dv / dt))
        return xs, ys

    def _mean_series(
        self, sums: deque, cnts: deque, window_s: int, scale: float = 1.0
    ) -> tuple[list, list]:
        with self._lock:
            times = list(self._times)
            sv    = list(sums)
            cv    = list(cnts)
        cutoff = time.time() - window_s
        triples = [(t, s, c) for t, s, c in zip(times, sv, cv) if t >= cutoff]
        if len(triples) < 2:
            return [], []
        t0 = triples[0][0]
        xs, ys = [], []
        for i in range(1, len(triples)):
            dt = triples[i][0] - triples[i - 1][0]
            ds = triples[i][1] - triples[i - 1][1]
            dc = triples[i][2] - triples[i - 1][2]
            if dt > 0 and dc > 0:
                xs.append(triples[i][0] - t0)
                ys.append((ds / dc) * scale)
        return xs, ys


# Module-level singleton — import and use everywhere
store = _RollingStore()
