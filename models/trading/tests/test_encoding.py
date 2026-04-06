"""Test encoder config validation and basic encoding."""

from encoder import (
    ENCODER_CONFIG,
    encode_snapshot,
    entry_to_record,
)
from intent import extract_features
from ada.core.types import Concept
from ada.encoder import Encoder


# ── Sample market data ─────────────────────────────────────────────────────

BULLISH_SNAPSHOT = {
    "ticker": "AAPL",
    "timestamp": "2026-01-15T10:00:00",
    "close": 195.0,
    "open": 190.0,
    "high": 196.0,
    "low": 189.5,
    "prev_close": 188.0,
    "prev_high": 191.0,
    "prev_low": 187.0,
    "volume": 85_000_000,
    "avg_volume": 50_000_000,
    "sma_20": 185.0,
    "sma_50": 180.0,
    "sma_200": 170.0,
    "rsi": 62,
    "macd_histogram": 1.5,
    "macd_signal_cross": "bull_cross",
    "stochastic_k": 65,
    "bb_position": 0.7,
    "adx": 30,
    "atr_pct": 2.0,
    "fear_greed_index": 65,
    "sentiment_score": 30,
    "news_sentiment": 20,
}

BEARISH_SNAPSHOT = {
    "ticker": "AAPL",
    "timestamp": "2026-02-10T10:00:00",
    "close": 165.0,
    "open": 172.0,
    "high": 173.0,
    "low": 164.0,
    "prev_close": 174.0,
    "prev_high": 176.0,
    "prev_low": 171.0,
    "volume": 90_000_000,
    "avg_volume": 50_000_000,
    "sma_20": 175.0,
    "sma_50": 178.0,
    "sma_200": 172.0,
    "rsi": 28,
    "macd_histogram": -2.0,
    "macd_signal_cross": "bear_cross",
    "stochastic_k": 15,
    "bb_position": 0.05,
    "adx": 35,
    "atr_pct": 3.5,
    "fear_greed_index": 18,
    "sentiment_score": -45,
    "news_sentiment": -30,
}

NEUTRAL_SNAPSHOT = {
    "ticker": "MSFT",
    "timestamp": "2026-03-01T10:00:00",
    "close": 400.0,
    "open": 399.5,
    "high": 401.0,
    "low": 399.0,
    "prev_close": 400.0,
    "prev_high": 401.0,
    "prev_low": 399.0,
    "volume": 25_000_000,
    "avg_volume": 25_000_000,
    "sma_20": 400.0,
    "sma_50": 399.0,
    "sma_200": 395.0,
    "rsi": 50,
    "macd_histogram": 0.1,
    "stochastic_k": 50,
    "bb_position": 0.5,
    "adx": 12,
    "atr_pct": 0.8,
    "fear_greed_index": 50,
    "sentiment_score": 0,
    "news_sentiment": 5,
}


# ── Config tests ───────────────────────────────────────────────────────────

def test_config_has_four_layers():
    assert len(ENCODER_CONFIG.layers) == 4
    names = [l.name for l in ENCODER_CONFIG.layers]
    assert "price" in names
    assert "volume" in names
    assert "technical" in names
    assert "sentiment" in names


def test_layer_weights_sum_to_one():
    total = sum(l.similarity_weight for l in ENCODER_CONFIG.layers)
    assert abs(total - 1.0) < 0.001, f"Layer weights sum to {total}, expected 1.0"


def test_temporal_enabled():
    assert ENCODER_CONFIG.include_temporal is True


# ── Feature extraction tests ──────────────────────────────────────────────

def test_bullish_features():
    f = extract_features(BULLISH_SNAPSHOT)
    assert f["trend_direction"] in ("bull", "strong_bull", "breakout")
    assert f["momentum_score"] > 0
    assert f["volume_pattern"] in ("accumulation", "surge")
    assert f["relative_volume"] > 1.0
    assert f["indicator_signal"] in ("buy", "strong_buy")
    assert f["sentiment_polarity"] in ("bullish", "very_bullish", "greed")


def test_bearish_features():
    f = extract_features(BEARISH_SNAPSHOT)
    assert f["trend_direction"] in ("bear", "strong_bear", "breakdown")
    assert f["momentum_score"] < 0
    assert f["volume_pattern"] in ("distribution", "surge", "climax")
    # RSI 28 triggers oversold counter-signals that offset bearish momentum
    assert f["indicator_signal"] in ("sell", "strong_sell", "oversold", "neutral")
    assert f["sentiment_polarity"] in ("bearish", "very_bearish", "fear", "panic")


def test_neutral_features():
    f = extract_features(NEUTRAL_SNAPSHOT)
    assert f["trend_direction"] == "neutral"
    assert abs(f["momentum_score"]) < 30
    assert f["volume_pattern"] == "steady"
    assert f["indicator_signal"] == "neutral"
    assert f["sentiment_polarity"] == "neutral"


def test_all_feature_keys_present():
    f = extract_features(BULLISH_SNAPSHOT)
    expected = [
        "trend_direction", "momentum_score", "price_signals",
        "volume_pattern", "relative_volume", "volume_signals",
        "indicator_signal", "pattern_type", "tech_signals",
        "sentiment_polarity", "sentiment_score", "sent_signals",
    ]
    for key in expected:
        assert key in f, f"Missing feature key: {key}"


# ── Encoding tests ─────────────────────────────────────────────────────────

def test_encode_snapshot_returns_valid_concept():
    result = encode_snapshot(BULLISH_SNAPSHOT)
    assert "name" in result
    assert "attributes" in result
    assert result["name"].startswith("snap_AAPL_")
    attrs = result["attributes"]
    assert "trend_direction" in attrs
    assert "volume_pattern" in attrs
    assert "indicator_signal" in attrs
    assert "sentiment_polarity" in attrs


def test_entry_to_record_structure():
    entry = {
        "ticker": "AAPL",
        "timestamp": "2026-01-15",
        "outcome": "bull",
        "return_pct": 3.5,
        **BULLISH_SNAPSHOT,
    }
    record = entry_to_record(entry)
    assert "concept_text" in record
    assert "attributes" in record
    assert "metadata" in record
    assert record["metadata"]["outcome"] == "bull"
    assert record["metadata"]["return_pct"] == 3.5


def test_encoding_produces_glyph():
    encoder = Encoder(ENCODER_CONFIG)
    attrs = encode_snapshot(BULLISH_SNAPSHOT)["attributes"]
    concept = Concept(name="test", attributes=attrs)
    glyph = encoder.encode(concept)
    assert glyph is not None
    assert glyph.global_cortex is not None
    assert "price" in glyph.layers
    assert "volume" in glyph.layers
    assert "technical" in glyph.layers
    assert "sentiment" in glyph.layers


def test_glyph_dimension():
    encoder = Encoder(ENCODER_CONFIG)
    attrs = encode_snapshot(NEUTRAL_SNAPSHOT)["attributes"]
    concept = Concept(name="test", attributes=attrs)
    glyph = encoder.encode(concept)
    assert glyph.global_cortex.dimension == 2000


def test_bull_bear_glyphs_differ():
    """Bull and bear snapshots should produce meaningfully different glyphs."""
    from ada.core.ops import cosine_similarity

    encoder = Encoder(ENCODER_CONFIG)

    bull_attrs = encode_snapshot(BULLISH_SNAPSHOT)["attributes"]
    bear_attrs = encode_snapshot(BEARISH_SNAPSHOT)["attributes"]

    bull_glyph = encoder.encode(Concept(name="bull", attributes=bull_attrs))
    bear_glyph = encoder.encode(Concept(name="bear", attributes=bear_attrs))

    sim = float(cosine_similarity(
        bull_glyph.global_cortex.data,
        bear_glyph.global_cortex.data,
    ))
    # They should have some similarity (same stock, same structure) but not too much
    assert sim < 0.85, f"Bull/bear glyphs too similar: {sim:.4f}"


def test_similar_snapshots_high_similarity():
    """Slightly modified bullish snapshot should be very similar to original."""
    from ada.core.ops import cosine_similarity

    encoder = Encoder(ENCODER_CONFIG)

    attrs1 = encode_snapshot(BULLISH_SNAPSHOT)["attributes"]

    similar = {**BULLISH_SNAPSHOT, "close": 194.0, "rsi": 60}
    attrs2 = encode_snapshot(similar)["attributes"]

    glyph1 = encoder.encode(Concept(name="a", attributes=attrs1))
    glyph2 = encoder.encode(Concept(name="b", attributes=attrs2))

    sim = float(cosine_similarity(
        glyph1.global_cortex.data,
        glyph2.global_cortex.data,
    ))
    assert sim > 0.7, f"Similar snapshots not similar enough: {sim:.4f}"
