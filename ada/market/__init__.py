"""
Market — daily-direction trend forecasting on Ada's HDC substrate.

This package is a self-contained, honest experiment, NOT a trading product:

  - `data.py`    pulls genuine daily OHLC history (open/close) from Yahoo's
                 public chart endpoint — no API key, no extra data library.
  - `analog.py`  encodes recent price action as a quantized symbolic pattern,
                 binds/bundles it into a 2048-dim bipolar vector (the same
                 HDC algebra Ada uses everywhere), then forecasts the next
                 day's direction by aggregating the realized outcomes of the
                 most *similar past setups* (k-nearest analog forecasting).

The whole point is the backtest. A directional forecaster is only meaningful
relative to a baseline, so `analog.backtest()` reports accuracy alongside
persistence / majority-class baselines and a 95% confidence interval. If the
interval straddles 50%, the model is not distinguishable from a coin flip —
and it will say so.
"""

from ada.market.data import Bar, fetch_daily_ohlc, load_csv, save_csv
from ada.market.analog import (
    AnalogForecaster,
    Forecast,
    BacktestReport,
    backtest,
)

__all__ = [
    "Bar",
    "fetch_daily_ohlc",
    "load_csv",
    "save_csv",
    "AnalogForecaster",
    "Forecast",
    "BacktestReport",
    "backtest",
]
