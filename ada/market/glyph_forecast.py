"""
Structured, glyph-native market forecasting — using Ada as Ada.

v1 (`ada.market.analog`) collapsed each day into a single bundled vector. That
threw away the whole point of a glyph. This module does it properly:

  * Each trading day becomes a real multi-role `Glyph` (ada.core.types.Glyph)
    whose primitives are mapped into the universal schema:

        quantitative.magnitude  body     (close vs open, vol-normalized)
        quantitative.ratio      gap      (open vs prev close)
        quantitative.count      volume   (vs trailing median)
        perceptual.size         range    (high-low, the volatility primitive)
        relational.predicate    trend    (short vs long moving average)
        temporal.frequency      weekday
        entity.subkind          return-regime of the day itself

    Every value is QUANTIZED into a small categorical alphabet, so atoms
    repeat across days and similarity / drift are meaningful (raw prices would
    be mutually orthogonal and carry no metric).

  * Days are versioned (v1, v2, ...). `TemporalEncoder.compute_temporal_delta`
    then yields a drift vector PER PRIMITIVE, not just for the cortex — the
    per-role capability that `BeamSearchPredictor._extract_vector` leaves
    stubbed.

  * `primitive_predictability()` reports, per primitive, how predictable its
    next regime is (a causal first-order Markov model vs. the marginal-majority
    baseline) plus its HDC drift-persistence. This is the real deliverable: it
    *shows* which primitives carry signal.

  * `backtest_target()` runs the structured analog forecaster against two
    targets — next-day DIRECTION and next-day HIGH-VOLATILITY — with baselines
    and a 95% CI, so the contrast is measured, not asserted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from ada.core.ops import bind, bundle, generate_symbol
from ada.core.types import Vector, Segment, Layer, Glyph
from ada.temporal.delta import TemporalEncoder

DIM = 2048
SEED = 42
SPACE = "market2048"
VOL_WINDOW = 20

# Which universal (layer, role) slot each market primitive occupies.
PRIMITIVES: dict[str, tuple[str, str]] = {
    "body":   ("quantitative", "magnitude"),
    "gap":    ("quantitative", "ratio"),
    "volume": ("quantitative", "count"),
    "range":  ("perceptual", "size"),
    "trend":  ("relational", "predicate"),
    "weekday": ("temporal", "frequency"),
    "retreg": ("entity", "subkind"),
}


# ── Quantization (all causal — thresholds use only trailing data) ────────────

_B5 = ("sd", "d", "f", "u", "su")
_L5 = ("vlow", "low", "mid", "high", "vhigh")


def _bucket5(z: float) -> str:
    if z <= -1.0:
        return "sd"
    if z <= -0.2:
        return "d"
    if z < 0.2:
        return "f"
    if z < 1.0:
        return "u"
    return "su"


def _level5(ratio: float) -> str:
    if ratio < 0.5:
        return "vlow"
    if ratio < 0.8:
        return "low"
    if ratio < 1.25:
        return "mid"
    if ratio < 2.0:
        return "high"
    return "vhigh"


def _trend3(slope: float) -> str:
    if slope < -0.01:
        return "down"
    if slope > 0.01:
        return "up"
    return "flat"


# ── Atom helpers ─────────────────────────────────────────────────────────────

def _role_atom(layer: str, role: str) -> np.ndarray:
    return generate_symbol(SEED, f"role::{layer}.{role}", DIM)


def _val_atom(layer: str, role: str, bucket: str) -> np.ndarray:
    return generate_symbol(SEED, f"val::{layer}.{role}={bucket}", DIM)


def _vec(data: np.ndarray) -> Vector:
    return Vector(data=data.astype(np.int8), dimension=DIM, space_id=SPACE)


# ── Day features + glyph ─────────────────────────────────────────────────────

@dataclass
class DayState:
    """A single day's quantized primitives plus the next-day targets."""

    date: str
    feats: dict[str, str]            # primitive name -> bucket
    next_dir: int | None = None      # 1 if next-day return > 0
    next_bigvol: int | None = None   # 1 if next-day range > trailing median


def build_states(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    volumes: np.ndarray,
    dates: list[str],
) -> list[DayState]:
    """Compute causal quantized primitives and next-day targets per day."""
    n = len(closes)
    ret = np.zeros(n)
    ret[1:] = closes[1:] / closes[:-1] - 1.0
    body = closes / opens - 1.0
    gap = np.zeros(n)
    gap[1:] = opens[1:] / closes[:-1] - 1.0
    rng = np.zeros(n)
    rng[1:] = (highs[1:] - lows[1:]) / closes[:-1]

    states: list[DayState] = []
    start = VOL_WINDOW + 1
    for i in range(start, n - 1):  # need i+1 for the target
        std = float(np.std(ret[i - VOL_WINDOW:i])) or 1e-9
        rng_med = float(np.median(rng[i - VOL_WINDOW:i])) or 1e-9
        vol_med = float(np.median(volumes[i - VOL_WINDOW:i])) or 1.0
        ma_s = float(np.mean(closes[i - 5:i]))
        ma_l = float(np.mean(closes[i - VOL_WINDOW:i]))
        slope = (ma_s - ma_l) / ma_l if ma_l else 0.0
        weekday = datetime.strptime(dates[i], "%Y-%m-%d").strftime("%a")

        feats = {
            "body":   _bucket5(body[i] / std),
            "gap":    _bucket5(gap[i] / std),
            "volume": _level5(volumes[i] / vol_med),
            "range":  _level5(rng[i] / rng_med),
            "trend":  _trend3(slope),
            "weekday": weekday.lower(),
            "retreg": _bucket5(ret[i] / std),
        }
        states.append(
            DayState(
                date=dates[i],
                feats=feats,
                next_dir=1 if ret[i + 1] > 0 else 0,
                next_bigvol=1 if rng[i + 1] > rng_med else 0,
            )
        )
    return states


def build_glyph(symbol: str, version: int, state: DayState) -> Glyph:
    """Assemble a real multi-role Glyph for one day (one segment per layer)."""
    by_layer: dict[str, list[tuple[str, str, np.ndarray]]] = {}
    for prim, bucket in state.feats.items():
        layer, role = PRIMITIVES[prim]
        binding = bind(_role_atom(layer, role), _val_atom(layer, role, bucket))
        by_layer.setdefault(layer, []).append((role, bucket, binding))

    layers: dict[str, Layer] = {}
    layer_cortices: list[np.ndarray] = []
    for layer, items in by_layer.items():
        seg_cortex = bundle([b for _, _, b in items])
        seg = Segment(
            name="main",
            cortex=_vec(seg_cortex),
            roles={role: _vec(b) for role, _, b in items},
            role_values={role: bucket for role, bucket, _ in items},
        )
        layers[layer] = Layer(name=layer, cortex=_vec(seg_cortex), segments={"main": seg})
        layer_cortices.append(seg_cortex)

    global_cortex = bundle(layer_cortices)
    return Glyph(
        identifier=f"{symbol}@{state.date}#v{version}",
        name=symbol,
        space_id=SPACE,
        global_cortex=_vec(global_cortex),
        layers=layers,
    )


# ── Per-primitive predictability (the real deliverable) ──────────────────────

@dataclass
class PrimitiveReport:
    primitive: str
    markov_acc: float       # causal first-order Markov next-regime accuracy
    majority_acc: float     # marginal-majority baseline
    lift: float             # markov - majority (signal above naive)
    drift_persistence: float  # mean HDC cosine(role_t, role_{t-1}) — regime stickiness


def primitive_predictability(
    glyphs: list[Glyph],
    states: list[DayState],
    test_frac: float = 0.3,
) -> list[PrimitiveReport]:
    """For each primitive: is its next regime predictable, and does it drift?

    - markov_acc: predict next bucket = most-likely successor of the current
      bucket, transition counts estimated ONLY on the training prefix (causal),
      evaluated on the held-out tail.
    - drift_persistence: average HDC delta-cosine of the primitive's role vector
      between consecutive days (1.0 = regime never changes, ~0 = changes every
      day). Uses TemporalEncoder.compute_temporal_delta, i.e. genuine per-role
      drift — the capability the beam predictor stubs out.
    """
    te = TemporalEncoder()
    n = len(states)
    split = max(int(n * (1 - test_frac)), VOL_WINDOW)
    reports: list[PrimitiveReport] = []

    for prim in PRIMITIVES:
        seq = [s.feats[prim] for s in states]

        # causal first-order Markov
        trans: dict[str, dict[str, int]] = {}
        for a, b in zip(seq[: split - 1], seq[1:split]):
            trans.setdefault(a, {}).setdefault(b, 0)
            trans[a][b] += 1
        best_next = {a: max(d, key=d.get) for a, d in trans.items()}
        majority = max(set(seq[:split]), key=seq[:split].count)

        correct = maj_correct = total = 0
        for t in range(split, n):
            actual = seq[t]
            pred = best_next.get(seq[t - 1], majority)
            correct += pred == actual
            maj_correct += majority == actual
            total += 1
        markov_acc = correct / total if total else 0.0
        majority_acc = maj_correct / total if total else 0.0

        # HDC per-primitive drift persistence
        layer, role = PRIMITIVES[prim]
        sims = []
        for t in range(1, len(glyphs)):
            prev = glyphs[t - 1].layers[layer].segments["main"].roles[role]
            cur = glyphs[t].layers[layer].segments["main"].roles[role]
            delta = te.compute_temporal_delta(prev, cur)  # bind(cur, prev)
            # cosine(prev, cur) == mean(delta) for bipolar vectors
            sims.append(float(np.mean(delta.data.astype(np.float32))))
        drift_persistence = float(np.mean(sims)) if sims else 0.0

        reports.append(
            PrimitiveReport(
                primitive=prim,
                markov_acc=markov_acc,
                majority_acc=majority_acc,
                lift=markov_acc - majority_acc,
                drift_persistence=drift_persistence,
            )
        )
    return reports


# ── Structured analog backtest (direction vs volatility) ─────────────────────

@dataclass
class TargetReport:
    target: str
    n_test: int
    accuracy: float
    ci95: tuple[float, float]
    baseline_majority: float
    beats_chance: bool
    window: int
    k: int


def _window_vectors(glyphs: list[Glyph], window: int) -> np.ndarray:
    """Bundle each day's cortex over a trailing `window` into one match vector."""
    cortices = np.stack([g.global_cortex.data for g in glyphs]).astype(np.int8)
    vecs = []
    for i in range(window - 1, len(glyphs)):
        vecs.append(bundle([cortices[j] for j in range(i - window + 1, i + 1)]))
    return np.stack(vecs).astype(np.int8)


def backtest_target(
    glyphs: list[Glyph],
    states: list[DayState],
    target: str,
    *,
    window: int = 5,
    k: int = 50,
    test_frac: float = 0.3,
) -> TargetReport:
    """Walk-forward structured analog forecast for 'direction' or 'bigvol'."""
    assert target in ("direction", "bigvol")
    labels = np.array(
        [s.next_dir if target == "direction" else s.next_bigvol for s in states]
    )
    wv = _window_vectors(glyphs, window)
    # align: window vector at row r corresponds to state index r + window - 1
    offset = window - 1
    y = labels[offset:]
    m = len(wv)
    test_start = max(int(m * (1 - test_frac)), k + 1)

    preds, actuals = [], []
    for i in range(test_start, m):
        sims = (wv[:i].astype(np.int32) @ wv[i].astype(np.int32)) / DIM
        idx = np.argpartition(-sims, k - 1)[:k]
        w = np.clip(sims[idx], 0.0, None)
        if w.sum() <= 0:
            w = np.ones_like(w)
        p1 = float(w[y[idx] == 1].sum() / w.sum())
        preds.append(1 if p1 >= 0.5 else 0)
        actuals.append(int(y[i]))

    preds = np.array(preds)
    actuals = np.array(actuals)
    nt = len(actuals)
    acc = float((preds == actuals).mean())
    maj = int(round(y[:test_start].mean()))
    base = float((maj == actuals).mean())
    se = (max(acc * (1 - acc), 1e-12) / nt) ** 0.5
    ci = (acc - 1.96 * se, acc + 1.96 * se)
    return TargetReport(
        target=target,
        n_test=nt,
        accuracy=acc,
        ci95=ci,
        baseline_majority=base,
        beats_chance=ci[0] > max(0.5, base),
        window=window,
        k=k,
    )
