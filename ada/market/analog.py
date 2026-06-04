"""
Analog forecasting on Ada's HDC substrate — direction *and* magnitude.

The idea (a.k.a. k-nearest-analog / non-parametric forecasting), expressed in
the exact bipolar HDC algebra Ada uses everywhere else:

  1. Turn the last `window` daily close-to-close returns into a *symbolic*
     pattern. Each return is bucketed by its size relative to recent
     volatility (causally — trailing std only), into one of five symbols:
     strong-down, down, flat, up, strong-up. Raw prices never enter the
     vector space; quantized *regimes* do, which is what HDC is good at.

  2. Encode that pattern as one 2048-dim bipolar vector:
        pattern = bundle_p( bind(pos_p, bucket_symbol_p) )
     i.e. bind each bucket symbol to its position symbol, then bundle. This
     is `ada.core.ops.bind` / `bundle` — the same role-filler composition the
     rest of Ada uses.

  3. To forecast day t, cosine-match today's pattern against every *earlier*
     pattern (strictly causal — no look-ahead), take the k most similar past
     setups, and aggregate the returns that actually followed them:
        - direction  ← similarity-weighted P(next return > 0)
        - magnitude  ← similarity-weighted mean + p10/p50/p90 of those returns

A forecast is only meaningful against a baseline, so `backtest()` reports
directional accuracy with a 95% CI next to persistence / majority baselines,
and magnitude skill (correlation + MAE vs a trailing-mean baseline). If the CI
straddles 50% it says so.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ada.core.ops import bind, bundle, generate_symbol

# ── Encoding constants ───────────────────────────────────────────────────────

DIM = 2048
SEED = 42
VOL_WINDOW = 20          # trailing window for volatility normalization
BUCKETS = ("sd", "d", "f", "u", "su")  # strong-down, down, flat, up, strong-up
# z-score (return / trailing-std) cutpoints between the five buckets:
_CUTS = (-1.0, -0.2, 0.2, 1.0)


def _bucket(z: float) -> str:
    """Quantize a volatility-normalized return into one of five regime symbols."""
    if z <= _CUTS[0]:
        return "sd"
    if z <= _CUTS[1]:
        return "d"
    if z < _CUTS[2]:
        return "f"
    if z < _CUTS[3]:
        return "u"
    return "su"


# Pre-generate the deterministic symbol vocabulary (positions + buckets).
_BUCKET_VEC = {b: generate_symbol(SEED, f"bucket__{b}", DIM) for b in BUCKETS}


def _pos_vec(p: int) -> np.ndarray:
    return generate_symbol(SEED, f"pos__{p}", DIM)


# ── Forecast / report containers ─────────────────────────────────────────────


@dataclass
class Forecast:
    """A single next-day forecast with provenance."""

    direction: str            # "up" | "down"
    p_up: float               # similarity-weighted P(next return > 0), in [0,1]
    confidence: float         # |p_up - 0.5| * 2, in [0,1]
    expected_return: float    # similarity-weighted mean next-day return
    ret_p10: float
    ret_p50: float
    ret_p90: float
    n_analogs: int
    analogs: list[tuple[str, float, float]] = field(default_factory=list)
    # ↑ (date_of_analog, similarity, realized_next_day_return) for the top matches


@dataclass
class BacktestReport:
    """Out-of-sample evaluation of both trend and magnitude."""

    symbol: str
    n_test: int
    train_span: tuple[str, str]
    test_span: tuple[str, str]
    # direction
    accuracy: float
    acc_ci95: tuple[float, float]
    baseline_persistence: float
    baseline_majority: float
    beats_chance: bool
    # magnitude
    return_corr: float          # Pearson corr(predicted, realized) on test
    mae: float                  # mean |predicted - realized| return
    mae_baseline: float         # same, predicting the trailing-mean return
    # config echo
    window: int
    k: int

    def render(self) -> str:
        lo, hi = self.acc_ci95
        verdict = (
            "distinguishable from chance"
            if self.beats_chance
            else "NOT distinguishable from a coin flip"
        )
        skill = "some" if self.return_corr > 0.03 else "no meaningful"
        return (
            f"Backtest — {self.symbol}  (window={self.window}, k={self.k})\n"
            f"  train: {self.train_span[0]} .. {self.train_span[1]}\n"
            f"  test:  {self.test_span[0]} .. {self.test_span[1]}  ({self.n_test} days)\n"
            f"\n  DIRECTION (trend)\n"
            f"    model accuracy ......... {self.accuracy:6.2%}  (95% CI {lo:.2%}–{hi:.2%})\n"
            f"    baseline: persistence .. {self.baseline_persistence:6.2%}\n"
            f"    baseline: majority ..... {self.baseline_majority:6.2%}\n"
            f"    verdict ................ {verdict}\n"
            f"\n  MAGNITUDE (size of move)\n"
            f"    corr(pred, realized) ... {self.return_corr:+.3f}  ({skill} magnitude skill)\n"
            f"    mean abs error ......... {self.mae:.4f}\n"
            f"    baseline (trailing mean) {self.mae_baseline:.4f}\n"
        )


# ── Core ──────────────────────────────────────────────────────────────────────


def _log_returns(closes: np.ndarray) -> np.ndarray:
    """Close-to-close log returns; r[i] corresponds to closes[i] vs closes[i-1]."""
    r = np.empty_like(closes)
    r[0] = 0.0
    r[1:] = np.log(closes[1:] / closes[:-1])
    return r


class AnalogForecaster:
    """Builds the analog library and forecasts next-day direction + magnitude.

    Strictly causal: the pattern at decision point (close of day i-1) is built
    only from returns up to i-1, and forecasts are aggregated only over
    *earlier* analogs.
    """

    def __init__(self, window: int = 5, k: int = 50):
        self.window = window
        self.k = k
        self._patterns: np.ndarray | None = None  # (n, DIM) int8
        self._returns: np.ndarray | None = None    # (n,) realized next-day return
        self._dates: list[str] = []                # (n,) date of the predicted day

    # -- encoding ---------------------------------------------------------------

    def _encode_window(self, buckets: list[str]) -> np.ndarray:
        """bundle_p bind(pos_p, bucket_symbol_p) -> one bipolar pattern vector."""
        bound = [bind(_pos_vec(p), _BUCKET_VEC[b]) for p, b in enumerate(buckets)]
        return bundle(bound)

    def build(self, closes: np.ndarray, dates: list[str]) -> "AnalogForecaster":
        """Encode every causal (pattern -> next-day-return) example in the series."""
        returns = _log_returns(closes)
        # causal trailing volatility, std of the prior VOL_WINDOW returns
        n = len(returns)
        patterns, outs, ds = [], [], []
        start = max(self.window, VOL_WINDOW) + 1
        for i in range(start, n):
            tstd = float(np.std(returns[i - VOL_WINDOW:i])) or 1e-9
            window_buckets = [
                _bucket(returns[j] / tstd) for j in range(i - self.window, i)
            ]
            patterns.append(self._encode_window(window_buckets))
            outs.append(returns[i])       # realized return of day i (the target)
            ds.append(dates[i])
        self._patterns = np.stack(patterns).astype(np.int8) if patterns else None
        self._returns = np.asarray(outs, dtype=np.float64)
        self._dates = ds
        return self

    # -- forecasting ------------------------------------------------------------

    def _aggregate(self, query: np.ndarray, upto: int) -> Forecast | None:
        """Forecast from the k nearest analogs among examples [0, upto)."""
        if self._patterns is None or upto < self.k:
            return None
        lib = self._patterns[:upto]
        sims = (lib.astype(np.int32) @ query.astype(np.int32)) / DIM  # cosine
        idx = np.argpartition(-sims, self.k - 1)[: self.k]
        idx = idx[np.argsort(-sims[idx])]

        w = np.clip(sims[idx], 0.0, None)
        if w.sum() <= 0:
            w = np.ones_like(w)
        w = w / w.sum()
        rets = self._returns[idx]

        p_up = float(w[rets > 0].sum())
        exp_ret = float(np.dot(w, rets))
        # weighted quantiles
        order = np.argsort(rets)
        cw = np.cumsum(w[order])
        q = lambda p: float(rets[order][np.searchsorted(cw, p)] if np.searchsorted(cw, p) < len(rets) else rets[order][-1])

        return Forecast(
            direction="up" if p_up >= 0.5 else "down",
            p_up=p_up,
            confidence=abs(p_up - 0.5) * 2,
            expected_return=exp_ret,
            ret_p10=q(0.10),
            ret_p50=q(0.50),
            ret_p90=q(0.90),
            n_analogs=self.k,
            analogs=[(self._dates[i], float(sims[i]), float(self._returns[i])) for i in idx[:8]],
        )

    def forecast_latest(self) -> Forecast | None:
        """Forecast the next day from the most recent pattern (full library)."""
        if self._patterns is None:
            return None
        return self._aggregate(self._patterns[-1], upto=len(self._patterns) - 1)


def backtest(
    symbol: str,
    closes: np.ndarray,
    dates: list[str],
    *,
    window: int = 5,
    k: int = 50,
    test_frac: float = 0.3,
) -> BacktestReport:
    """Walk-forward, strictly-causal evaluation of direction and magnitude.

    For each test day, the forecast uses ONLY analogs that occurred before it
    (expanding window), so there is no look-ahead. Reports model accuracy with
    a 95% CI beside persistence/majority baselines, plus magnitude skill.
    """
    fc = AnalogForecaster(window=window, k=k).build(closes, dates)
    if fc._patterns is None:
        raise ValueError("Not enough history to build any analog patterns")

    n = len(fc._patterns)
    test_start = max(int(n * (1 - test_frac)), k + 1)

    preds, actuals, pred_rets = [], [], []
    for i in range(test_start, n):
        f = fc._aggregate(fc._patterns[i], upto=i)
        if f is None:
            continue
        preds.append(1 if f.direction == "up" else 0)
        pred_rets.append(f.expected_return)
        actuals.append(1 if fc._returns[i] > 0 else 0)

    preds = np.asarray(preds)
    actuals = np.asarray(actuals)
    realized = fc._returns[test_start: test_start + len(actuals)]
    pred_rets = np.asarray(pred_rets)
    m = len(actuals)

    accuracy = float((preds == actuals).mean())
    # persistence: predict next direction == previous day's direction
    prev_dir = (fc._returns[test_start - 1: test_start - 1 + m] > 0).astype(int)
    baseline_persistence = float((prev_dir == actuals).mean())
    # majority class observed in the training portion
    train_dirs = (fc._returns[:test_start] > 0).astype(int)
    maj = int(round(train_dirs.mean()))
    baseline_majority = float((np.full(m, maj) == actuals).mean())

    # 95% Wald CI on accuracy; "beats chance" if the whole interval clears 50%
    se = math.sqrt(max(accuracy * (1 - accuracy), 1e-12) / m)
    ci = (accuracy - 1.96 * se, accuracy + 1.96 * se)
    beats_chance = ci[0] > 0.5

    # magnitude skill
    if np.std(pred_rets) > 0 and np.std(realized) > 0:
        return_corr = float(np.corrcoef(pred_rets, realized)[0, 1])
    else:
        return_corr = 0.0
    mae = float(np.mean(np.abs(pred_rets - realized)))
    trailing_mean = float(np.mean(fc._returns[:test_start]))
    mae_baseline = float(np.mean(np.abs(trailing_mean - realized)))

    return BacktestReport(
        symbol=symbol,
        n_test=m,
        train_span=(fc._dates[0], fc._dates[test_start - 1]),
        test_span=(fc._dates[test_start], fc._dates[-1]),
        accuracy=accuracy,
        acc_ci95=ci,
        baseline_persistence=baseline_persistence,
        baseline_majority=baseline_majority,
        beats_chance=beats_chance,
        return_corr=return_corr,
        mae=mae,
        mae_baseline=mae_baseline,
        window=window,
        k=k,
    )
