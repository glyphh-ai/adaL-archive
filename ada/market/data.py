"""
Daily OHLC loader — genuine daily bars from Yahoo's public chart endpoint.

No API key and no extra dependency (uses httpx, already an Ada dependency).

IMPORTANT: the `range=max` shortcut silently downsamples to *monthly* bars for
long horizons. To always get true daily granularity we request an explicit
`period1`/`period2` unix-timestamp window instead. With period1=0 Yahoo
returns the full daily history back to listing (e.g. AAPL → 1980-12, KO →
1970-01).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import httpx

YAHOO_CHART = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
_HEADERS = {"User-Agent": "Mozilla/5.0"}


@dataclass
class Bar:
    """One daily OHLCV bar. `date` is an ISO 'YYYY-MM-DD' (UTC) string."""

    date: str
    open: float
    high: float
    low: float
    close: float
    volume: float


def fetch_daily_ohlc(
    symbol: str,
    *,
    start: str | None = None,
    end: str | None = None,
    timeout: float = 30.0,
) -> list[Bar]:
    """Fetch genuine daily OHLC bars for `symbol`.

    Args:
        symbol: Ticker, e.g. "AAPL".
        start:  ISO date 'YYYY-MM-DD' for the first bar. None → from listing.
        end:    ISO date 'YYYY-MM-DD' for the last bar. None → today.
        timeout: HTTP timeout in seconds.

    Returns:
        Bars in chronological order. Rows with missing open/close are dropped.

    Raises:
        httpx.HTTPError on network failure; ValueError on an empty response.
    """
    period1 = 0 if start is None else int(
        datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp()
    )
    period2 = (
        int(datetime.now(tz=timezone.utc).timestamp())
        if end is None
        else int(datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())
    )

    resp = httpx.get(
        YAHOO_CHART.format(symbol=symbol),
        params={"period1": period1, "period2": period2, "interval": "1d"},
        headers=_HEADERS,
        timeout=timeout,
    )
    resp.raise_for_status()

    chart = resp.json().get("chart", {})
    if chart.get("error"):
        raise ValueError(f"Yahoo error for {symbol!r}: {chart['error']}")
    results = chart.get("result")
    if not results:
        raise ValueError(f"No data returned for {symbol!r}")

    res = results[0]
    timestamps = res.get("timestamp") or []
    quote = res["indicators"]["quote"][0]

    bars: list[Bar] = []
    for i, ts in enumerate(timestamps):
        o, h, l, c = quote["open"][i], quote["high"][i], quote["low"][i], quote["close"][i]
        if o is None or c is None or h is None or l is None:
            continue  # Yahoo emits nulls for halted/illiquid sessions
        v = quote["volume"][i] or 0.0
        date = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
        bars.append(Bar(date, float(o), float(h), float(l), float(c), float(v)))

    return bars


def save_csv(bars: list[Bar], path: str | Path) -> None:
    """Cache bars to a CSV (date,open,high,low,close,volume)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "open", "high", "low", "close", "volume"])
        for b in bars:
            w.writerow([b.date, b.open, b.high, b.low, b.close, b.volume])


def load_csv(path: str | Path) -> list[Bar]:
    """Load bars from a CSV produced by `save_csv` (or any matching schema).

    Lets you backtest fully offline, network-independent and reproducible.
    """
    bars: list[Bar] = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            bars.append(
                Bar(
                    date=row["date"],
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row.get("volume", 0) or 0),
                )
            )
    return bars
