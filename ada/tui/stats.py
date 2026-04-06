"""
Runtime stats collector — shared in-process metrics for the TUI dashboard.

The FastAPI middleware writes to this; the TUI reads from it.
Thread-safe via a lock since uvicorn runs in a background thread.
"""

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RequestRecord:
    """A single completed request."""
    timestamp: float
    tool: str
    model_id: str
    org_id: str
    duration_ms: float
    status: str  # "ok" or "error"


@dataclass
class ModelStats:
    """Snapshot of a loaded model's state."""
    model_id: str
    org_id: str
    glyph_count: int
    tool_count: int
    status: str  # "loaded", "loading", "error"
    last_query_at: Optional[float] = None


class StatsCollector:
    """Thread-safe metrics collector for the Ada Runtime.

    Stores a rolling window of recent requests and computes
    rates and percentiles on read.
    """

    def __init__(self, window_size: int = 1000):
        self._lock = threading.Lock()
        self._requests: deque[RequestRecord] = deque(maxlen=window_size)
        self._models: dict[str, ModelStats] = {}
        self._total_requests: int = 0
        self._total_errors: int = 0
        self._start_time: float = time.time()

        # Time-series buckets for graphs (1 bucket per second, last 5 min)
        self._rate_buckets: deque[tuple[float, int]] = deque(maxlen=300)
        self._latency_buckets: deque[tuple[float, float]] = deque(maxlen=300)
        self._current_bucket_time: float = 0
        self._current_bucket_count: int = 0
        self._current_bucket_latency_sum: float = 0
        self._current_bucket_latency_count: int = 0

    def record_request(self, record: RequestRecord) -> None:
        """Record a completed request."""
        with self._lock:
            self._requests.append(record)
            self._total_requests += 1
            if record.status == "error":
                self._total_errors += 1

            # Update model last_query_at
            key = f"{record.org_id}/{record.model_id}"
            if key in self._models:
                self._models[key].last_query_at = record.timestamp

            # Update time-series buckets
            bucket_time = int(record.timestamp)
            if bucket_time != self._current_bucket_time:
                # Flush previous bucket
                if self._current_bucket_time > 0:
                    self._rate_buckets.append(
                        (self._current_bucket_time, self._current_bucket_count)
                    )
                    if self._current_bucket_latency_count > 0:
                        avg = self._current_bucket_latency_sum / self._current_bucket_latency_count
                        self._latency_buckets.append(
                            (self._current_bucket_time, avg)
                        )
                self._current_bucket_time = bucket_time
                self._current_bucket_count = 0
                self._current_bucket_latency_sum = 0
                self._current_bucket_latency_count = 0

            self._current_bucket_count += 1
            self._current_bucket_latency_sum += record.duration_ms
            self._current_bucket_latency_count += 1

    def update_model(self, stats: ModelStats) -> None:
        """Update or register a model's stats."""
        with self._lock:
            key = f"{stats.org_id}/{stats.model_id}"
            self._models[key] = stats

    def remove_model(self, org_id: str, model_id: str) -> None:
        """Remove a model from tracking."""
        with self._lock:
            self._models.pop(f"{org_id}/{model_id}", None)

    # -- Read methods (called by TUI) --

    def get_recent_requests(self, limit: int = 20) -> list[RequestRecord]:
        """Get the most recent requests."""
        with self._lock:
            items = list(self._requests)
            return items[-limit:]

    def get_models(self) -> list[ModelStats]:
        """Get all tracked models."""
        with self._lock:
            return list(self._models.values())

    def get_rate_series(self, minutes: int = 5) -> list[float]:
        """Get requests-per-second time series for the last N minutes."""
        with self._lock:
            cutoff = time.time() - (minutes * 60)
            # Flush current bucket
            result = []
            for ts, count in self._rate_buckets:
                if ts >= cutoff:
                    result.append(float(count))
            return result

    def get_latency_series(self, minutes: int = 5) -> list[float]:
        """Get average latency (ms) time series for the last N minutes."""
        with self._lock:
            cutoff = time.time() - (minutes * 60)
            result = []
            for ts, avg_ms in self._latency_buckets:
                if ts >= cutoff:
                    result.append(avg_ms)
            return result

    def get_summary(self) -> dict:
        """Get summary stats."""
        with self._lock:
            now = time.time()
            uptime = now - self._start_time

            # Requests in last 60 seconds
            cutoff_60s = now - 60
            recent = [r for r in self._requests if r.timestamp >= cutoff_60s]
            rpm = len(recent)

            # Average latency of recent requests
            if recent:
                avg_latency = sum(r.duration_ms for r in recent) / len(recent)
            else:
                avg_latency = 0.0

            return {
                "total_requests": self._total_requests,
                "total_errors": self._total_errors,
                "requests_per_minute": rpm,
                "avg_latency_ms": avg_latency,
                "uptime_seconds": uptime,
                "models_loaded": len(self._models),
            }


# Module-level singleton
_collector: Optional[StatsCollector] = None


def get_stats() -> StatsCollector:
    """Get the global stats collector (lazy-initialized)."""
    global _collector
    if _collector is None:
        _collector = StatsCollector()
    return _collector
