"""
Encoder for the Trading Signal Engine.

Exports:
  ENCODER_CONFIG — EncoderConfig with 4 layers: price, volume, technical, sentiment
  encode_snapshot(data) — converts a market data snapshot to a Concept dict
  entry_to_record(entry) — converts a historical trade record to a build record
  score_signal(query_glyph, exemplar_glyphs) — returns signal with per-layer breakdown

Architecture:
  Data Extraction — intent.py (local, deterministic):
    - extract_features(snapshot) → features across 4 dimensions
    - Price action: trend, momentum, support/resistance levels
    - Volume: accumulation/distribution, relative volume, profile
    - Technical: RSI, MACD, Bollinger position, pattern recognition
    - Sentiment: news polarity, social volume, options flow direction

  Main model (seed=42):
  - Price layer (0.35): trend_direction (lexicon) + momentum_score (numeric) + price_signals (BoW)
  - Volume layer (0.20): volume_pattern (lexicon) + relative_volume (numeric) + volume_signals (BoW)
  - Technical layer (0.30): indicator_signal (lexicon) + pattern_type (lexicon) + tech_signals (BoW)
  - Sentiment layer (0.15): sentiment_polarity (lexicon) + sentiment_score (numeric) + sent_signals (BoW)

  Scoring: Differential — signal = max_bull_sim - max_bear_sim.
  Positive = bullish, negative = bearish, near-zero = neutral/no-trade.

  Temporal: Each snapshot is a time-windowed glyph. Dream loop (Ada) finds
  historical sequences that rhyme with the current pattern via Hebbian
  reinforcement across PathwayEncoder sequences.
"""

import hashlib
import time

from ada.core.config import (
    EncoderConfig,
    EncodingStrategy,
    Layer,
    NumericConfig,
    Role,
    Segment,
)

from intent import extract_features

# ---------------------------------------------------------------------------
# ENCODER_CONFIG — 4-layer trading signal detection
# ---------------------------------------------------------------------------

ENCODER_CONFIG = EncoderConfig(
    dimension=2000,
    seed=42,
    apply_weights_during_encoding=False,
    include_temporal=True,  # temporal encoding enabled for time-series
    layers=[
        Layer(
            name="price",
            similarity_weight=0.35,
            segments=[
                Segment(
                    name="classification",
                    roles=[
                        Role(
                            name="trend_direction",
                            similarity_weight=1.0,
                            lexicons=[
                                "strong_bull", "bull", "neutral",
                                "bear", "strong_bear", "reversal_up",
                                "reversal_down", "breakout", "breakdown",
                            ],
                        ),
                        Role(
                            name="momentum_score",
                            similarity_weight=0.7,
                            numeric_config=NumericConfig(
                                bin_width=10.0,
                                encoding_strategy=EncodingStrategy.THERMOMETER,
                                min_value=-100.0,
                                max_value=100.0,
                            ),
                        ),
                    ],
                ),
                Segment(
                    name="signals",
                    roles=[
                        Role(
                            name="price_signals",
                            similarity_weight=0.8,
                            text_encoding="bag_of_words",
                        ),
                    ],
                ),
            ],
        ),
        Layer(
            name="volume",
            similarity_weight=0.20,
            segments=[
                Segment(
                    name="classification",
                    roles=[
                        Role(
                            name="volume_pattern",
                            similarity_weight=1.0,
                            lexicons=[
                                "accumulation", "distribution", "climax",
                                "dry_up", "surge", "steady", "divergence",
                            ],
                        ),
                        Role(
                            name="relative_volume",
                            similarity_weight=0.6,
                            numeric_config=NumericConfig(
                                bin_width=0.5,
                                encoding_strategy=EncodingStrategy.THERMOMETER,
                                min_value=0.0,
                                max_value=5.0,
                            ),
                        ),
                    ],
                ),
                Segment(
                    name="signals",
                    roles=[
                        Role(
                            name="volume_signals",
                            similarity_weight=0.7,
                            text_encoding="bag_of_words",
                        ),
                    ],
                ),
            ],
        ),
        Layer(
            name="technical",
            similarity_weight=0.30,
            segments=[
                Segment(
                    name="classification",
                    roles=[
                        Role(
                            name="indicator_signal",
                            similarity_weight=1.0,
                            lexicons=[
                                "strong_buy", "buy", "neutral",
                                "sell", "strong_sell", "overbought",
                                "oversold", "divergence_bull", "divergence_bear",
                            ],
                        ),
                        Role(
                            name="pattern_type",
                            similarity_weight=0.8,
                            lexicons=[
                                "none", "double_bottom", "double_top",
                                "head_shoulders", "inv_head_shoulders",
                                "ascending_triangle", "descending_triangle",
                                "cup_handle", "flag_bull", "flag_bear",
                                "wedge_rising", "wedge_falling",
                                "channel_up", "channel_down",
                            ],
                        ),
                    ],
                ),
                Segment(
                    name="signals",
                    roles=[
                        Role(
                            name="tech_signals",
                            similarity_weight=0.7,
                            text_encoding="bag_of_words",
                        ),
                    ],
                ),
            ],
        ),
        Layer(
            name="sentiment",
            similarity_weight=0.15,
            segments=[
                Segment(
                    name="classification",
                    roles=[
                        Role(
                            name="sentiment_polarity",
                            similarity_weight=1.0,
                            lexicons=[
                                "very_bullish", "bullish", "neutral",
                                "bearish", "very_bearish", "mixed",
                                "fear", "greed", "panic",
                            ],
                        ),
                        Role(
                            name="sentiment_score",
                            similarity_weight=0.6,
                            numeric_config=NumericConfig(
                                bin_width=10.0,
                                encoding_strategy=EncodingStrategy.THERMOMETER,
                                min_value=-100.0,
                                max_value=100.0,
                            ),
                        ),
                    ],
                ),
                Segment(
                    name="signals",
                    roles=[
                        Role(
                            name="sent_signals",
                            similarity_weight=0.7,
                            text_encoding="bag_of_words",
                        ),
                    ],
                ),
            ],
        ),
    ],
)


# ---------------------------------------------------------------------------
# encode_snapshot — market data -> Concept dict
# ---------------------------------------------------------------------------

def encode_snapshot(data: dict) -> dict:
    """Convert a market data snapshot into a Concept-compatible dict.

    Args:
        data: Dict with OHLCV, indicators, sentiment, etc.

    Returns a dict with 'name' and 'attributes' keys matching ENCODER_CONFIG roles.
    """
    features = extract_features(data)
    ticker = data.get("ticker", "UNK")
    ts = data.get("timestamp", "")
    stable_id = int(hashlib.md5(f"{ticker}_{ts}".encode()).hexdigest()[:8], 16)
    return {
        "name": f"snap_{ticker}_{stable_id:08d}",
        "attributes": features,
    }


# ---------------------------------------------------------------------------
# entry_to_record — historical trade -> build record
# ---------------------------------------------------------------------------

def entry_to_record(entry: dict) -> dict:
    """Convert a historical trade record into a record for building/encoding.

    Expects entry keys: ticker, timestamp, outcome (bull/bear/neutral),
    plus OHLCV and indicator data.
    """
    if "trend_direction" in entry:
        # Pre-extracted features
        attributes = {
            "trend_direction": entry["trend_direction"],
            "momentum_score": entry.get("momentum_score", 0),
            "price_signals": entry.get("price_signals", ""),
            "volume_pattern": entry.get("volume_pattern", "steady"),
            "relative_volume": entry.get("relative_volume", 1.0),
            "volume_signals": entry.get("volume_signals", ""),
            "indicator_signal": entry.get("indicator_signal", "neutral"),
            "pattern_type": entry.get("pattern_type", "none"),
            "tech_signals": entry.get("tech_signals", ""),
            "sentiment_polarity": entry.get("sentiment_polarity", "neutral"),
            "sentiment_score": entry.get("sentiment_score", 0),
            "sent_signals": entry.get("sent_signals", ""),
        }
    else:
        attributes = extract_features(entry)

    outcome = entry.get("outcome", "neutral")
    ticker = entry.get("ticker", "UNK")
    ts = entry.get("timestamp", entry.get("date", ""))

    return {
        "concept_text": f"{ticker}_{outcome}_{ts}",
        "attributes": attributes,
        "metadata": {
            "ticker": ticker,
            "timestamp": ts,
            "outcome": outcome,
            "return_pct": entry.get("return_pct", 0.0),
        },
    }


# ---------------------------------------------------------------------------
# score_signal — differential scoring for trade signals
# ---------------------------------------------------------------------------

STRONG_SIGNAL = 0.30
WEAK_SIGNAL = 0.12

LAYER_WEIGHTS = {
    "price": 0.35,
    "volume": 0.20,
    "technical": 0.30,
    "sentiment": 0.15,
}


def _compute_layer_scores(query_glyph, exemplar_glyph, cos_sim) -> tuple[dict, float]:
    """Compute per-layer similarity and weighted score between two glyphs."""
    layer_scores = {}
    weighted_score = 0.0
    for layer_name, weight in LAYER_WEIGHTS.items():
        if layer_name in query_glyph.layers and layer_name in exemplar_glyph.layers:
            q_cortex = query_glyph.layers[layer_name].cortex.data
            e_cortex = exemplar_glyph.layers[layer_name].cortex.data
            sim = float(cos_sim(q_cortex, e_cortex))
            layer_scores[layer_name] = round(sim, 4)
            weighted_score += sim * weight
        else:
            layer_scores[layer_name] = 0.0
    return layer_scores, weighted_score


def score_signal(
    query_glyph,
    exemplar_glyphs: list,
    strong_threshold: float = STRONG_SIGNAL,
    weak_threshold: float = WEAK_SIGNAL,
) -> dict:
    """Score a market snapshot against historical exemplars using differential scoring.

    signal_score = max_bull_sim - max_bear_sim.
    Positive = bullish, negative = bearish, near-zero = no trade.

    Args:
        query_glyph: Encoded glyph of the current market snapshot
        exemplar_glyphs: List of (glyph, metadata) tuples for ALL exemplars
        strong_threshold: Score above this = strong signal
        weak_threshold: Score above this = weak signal

    Returns dict with:
        signal_score, direction, strength, matched_pattern,
        layer_scores, explanation
    """
    from ada.core.ops import cosine_similarity as cos_sim

    best_bull_score = 0.0
    best_bull_meta = {}
    best_bull_layers = {}

    best_bear_score = 0.0
    best_bear_meta = {}
    best_bear_layers = {}

    for exemplar_glyph, metadata in exemplar_glyphs:
        outcome = metadata.get("outcome", "neutral")
        layer_scores, weighted_score = _compute_layer_scores(
            query_glyph, exemplar_glyph, cos_sim,
        )

        if outcome in ("bull", "strong_bull") and weighted_score > best_bull_score:
            best_bull_score = weighted_score
            best_bull_meta = metadata
            best_bull_layers = layer_scores

        if outcome in ("bear", "strong_bear") and weighted_score > best_bear_score:
            best_bear_score = weighted_score
            best_bear_meta = metadata
            best_bear_layers = layer_scores

    signal_score = best_bull_score - best_bear_score

    abs_signal = abs(signal_score)
    if abs_signal >= strong_threshold:
        strength = "STRONG"
    elif abs_signal >= weak_threshold:
        strength = "WEAK"
    else:
        strength = "NONE"

    if signal_score > 0:
        direction = "BULL"
        best_layers = best_bull_layers
        best_meta = best_bull_meta
    elif signal_score < 0:
        direction = "BEAR"
        best_layers = best_bear_layers
        best_meta = best_bear_meta
    else:
        direction = "NEUTRAL"
        best_layers = {}
        best_meta = {}

    if strength == "NONE":
        explanation = "No significant pattern match. No trade signal."
    else:
        top_layer = max(best_layers, key=best_layers.get) if best_layers else "unknown"
        top_sim = best_layers.get(top_layer, 0)
        matched_ticker = best_meta.get("ticker", "?")
        matched_return = best_meta.get("return_pct", 0)
        explanation = (
            f"{strength} {direction}. {top_layer.capitalize()} layer {top_sim:.2f} match. "
            f"Historical rhyme: {matched_ticker} ({matched_return:+.1f}%)."
        )

    return {
        "signal_score": round(signal_score, 4),
        "direction": direction,
        "strength": strength,
        "bull_sim": round(best_bull_score, 4),
        "bear_sim": round(best_bear_score, 4),
        "matched_pattern": best_meta.get("timestamp", ""),
        "matched_ticker": best_meta.get("ticker", ""),
        "matched_return": best_meta.get("return_pct", 0.0),
        "layer_scores": best_layers,
        "explanation": explanation,
    }
