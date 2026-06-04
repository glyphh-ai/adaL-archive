#!/usr/bin/env python3
"""
Daily trend + magnitude forecast via HDC analog matching.

Pulls genuine daily OHLC from Yahoo, backtests the analog forecaster
walk-forward (with baselines + 95% CI), then prints the next-day forecast
and the historical analogs it is reasoning from.

This is a personal research tool, not trading advice. A directional edge in
the low-50s% — if it even survives the backtest — is not a license to trade;
it is a measurement.

Usage:
    PYTHONPATH=. python scripts/predict_trend.py --symbol AAPL
    PYTHONPATH=. python scripts/predict_trend.py --symbol SPY --start 2005-01-01 --window 7 --k 80
    PYTHONPATH=. python scripts/predict_trend.py --csv mydata.csv --symbol FOO
"""

from __future__ import annotations

import argparse

import numpy as np

from ada.market.data import fetch_daily_ohlc, load_csv
from ada.market.analog import AnalogForecaster, backtest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symbol", default="AAPL", help="ticker (default AAPL)")
    ap.add_argument("--start", default=None, help="ISO start date; default = listing")
    ap.add_argument("--end", default=None, help="ISO end date; default = today")
    ap.add_argument("--csv", default=None, help="load bars from CSV instead of Yahoo")
    ap.add_argument("--window", type=int, default=5, help="pattern length in days")
    ap.add_argument("--k", type=int, default=50, help="number of analogs to aggregate")
    ap.add_argument("--test-frac", type=float, default=0.3, help="out-of-sample fraction")
    args = ap.parse_args()

    if args.csv:
        bars = load_csv(args.csv)
        print(f"Loaded {len(bars)} bars from {args.csv}")
    else:
        bars = fetch_daily_ohlc(args.symbol, start=args.start, end=args.end)
        print(f"Fetched {len(bars)} daily bars for {args.symbol}: "
              f"{bars[0].date} .. {bars[-1].date}")

    closes = np.array([b.close for b in bars], dtype=np.float64)
    dates = [b.date for b in bars]

    report = backtest(
        args.symbol, closes, dates,
        window=args.window, k=args.k, test_frac=args.test_frac,
    )
    print("\n" + report.render())

    fc = AnalogForecaster(window=args.window, k=args.k).build(closes, dates)
    f = fc.forecast_latest()
    if f is None:
        print("Not enough history for a live forecast.")
        return

    print(f"  NEXT-DAY FORECAST (after {dates[-1]})")
    print(f"    trend ........... {f.direction.upper()}  "
          f"(P_up={f.p_up:.2%}, confidence={f.confidence:.2%})")
    print(f"    expected move ... {f.expected_return:+.3%}  "
          f"[p10 {f.ret_p10:+.2%} | p50 {f.ret_p50:+.2%} | p90 {f.ret_p90:+.2%}]")
    print(f"    based on {f.n_analogs} closest past setups. Nearest analogs:")
    for date, sim, ret in f.analogs:
        print(f"      {date}  sim={sim:+.3f}  next-day={ret:+.2%}")

    if not report.beats_chance:
        print("\n  ⚠ Backtest CI straddles 50% — treat the forecast as noise, "
              "not signal.")


if __name__ == "__main__":
    main()
