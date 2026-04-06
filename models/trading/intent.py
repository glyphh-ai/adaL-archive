"""
Feature extraction for the Trading Signal Engine.

Converts raw market data snapshots into features for HDC encoding.
All extraction is deterministic — no LLM, no API calls at runtime.

Input: dict with OHLCV bars, technical indicators, and optional sentiment.
Output: dict of features matching ENCODER_CONFIG roles.
"""

from __future__ import annotations

import math


# ---------------------------------------------------------------------------
# Price features
# ---------------------------------------------------------------------------

def _classify_trend(data: dict) -> str:
    """Classify trend direction from price data."""
    close = data.get("close", 0)
    open_ = data.get("open", close)
    sma_20 = data.get("sma_20", close)
    sma_50 = data.get("sma_50", close)
    high = data.get("high", close)
    low = data.get("low", close)
    prev_close = data.get("prev_close", close)

    if close == 0 or prev_close == 0:
        return "neutral"

    pct_change = (close - prev_close) / prev_close * 100

    # Breakout/breakdown: price crosses SMA 50 with momentum
    if prev_close < sma_50 and close > sma_50 and pct_change > 1.0:
        return "breakout"
    if prev_close > sma_50 and close < sma_50 and pct_change < -1.0:
        return "breakdown"

    # Reversal detection: wick > 2x body from extreme
    body = abs(close - open_)
    if body > 0:
        upper_wick = high - max(close, open_)
        lower_wick = min(close, open_) - low
        if lower_wick > body * 2 and pct_change > 0:
            return "reversal_up"
        if upper_wick > body * 2 and pct_change < 0:
            return "reversal_down"

    # Trend strength
    if pct_change > 3.0 and close > sma_20:
        return "strong_bull"
    if pct_change > 0.5 and close > sma_20:
        return "bull"
    if pct_change < -3.0 and close < sma_20:
        return "strong_bear"
    if pct_change < -0.5 and close < sma_20:
        return "bear"

    return "neutral"


def _momentum_score(data: dict) -> float:
    """Calculate momentum score (-100 to 100)."""
    rsi = data.get("rsi", 50)
    macd_hist = data.get("macd_histogram", 0)
    close = data.get("close", 0)
    sma_20 = data.get("sma_20", close)

    # RSI component: 0-100 mapped to -50 to +50
    rsi_component = (rsi - 50)

    # MACD histogram component
    macd_component = max(-30, min(30, macd_hist * 10))

    # Price vs SMA component
    if sma_20 and sma_20 > 0:
        sma_pct = (close - sma_20) / sma_20 * 100
        sma_component = max(-20, min(20, sma_pct * 5))
    else:
        sma_component = 0

    score = rsi_component + macd_component + sma_component
    return max(-100, min(100, round(score, 1)))


def _price_signals(data: dict) -> str:
    """Generate BoW price signals."""
    signals = []
    close = data.get("close", 0)
    sma_20 = data.get("sma_20", close)
    sma_50 = data.get("sma_50", close)
    sma_200 = data.get("sma_200", close)
    high = data.get("high", close)
    low = data.get("low", close)
    prev_high = data.get("prev_high", high)
    prev_low = data.get("prev_low", low)

    if close > sma_20:
        signals.append("above_sma20")
    else:
        signals.append("below_sma20")

    if close > sma_50:
        signals.append("above_sma50")
    else:
        signals.append("below_sma50")

    if sma_20 > sma_50:
        signals.append("golden_alignment")
    else:
        signals.append("death_alignment")

    if sma_50 > sma_200:
        signals.append("golden_cross")
    elif sma_50 < sma_200:
        signals.append("death_cross")

    if high > prev_high:
        signals.append("higher_high")
    if low > prev_low:
        signals.append("higher_low")
    if high < prev_high:
        signals.append("lower_high")
    if low < prev_low:
        signals.append("lower_low")

    # ATR-based volatility
    atr = data.get("atr", 0)
    atr_pct = data.get("atr_pct", 0)
    if atr_pct > 3.0:
        signals.append("high_volatility")
    elif atr_pct < 1.0:
        signals.append("low_volatility")

    # Gap detection
    open_ = data.get("open", close)
    prev_close = data.get("prev_close", close)
    if prev_close > 0:
        gap_pct = (open_ - prev_close) / prev_close * 100
        if gap_pct > 1.0:
            signals.append("gap_up")
        elif gap_pct < -1.0:
            signals.append("gap_down")

    return " ".join(signals)


# ---------------------------------------------------------------------------
# Volume features
# ---------------------------------------------------------------------------

def _classify_volume(data: dict) -> str:
    """Classify volume pattern."""
    volume = data.get("volume", 0)
    avg_volume = data.get("avg_volume", volume)
    close = data.get("close", 0)
    prev_close = data.get("prev_close", close)

    if avg_volume == 0:
        return "steady"

    rvol = volume / avg_volume if avg_volume > 0 else 1.0
    pct_change = (close - prev_close) / prev_close * 100 if prev_close > 0 else 0

    # Climax: huge volume with big move
    if rvol > 3.0 and abs(pct_change) > 3.0:
        return "climax"

    # Surge: notably above average
    if rvol > 2.0:
        return "surge"

    # Accumulation: rising price on rising volume
    if pct_change > 0.5 and rvol > 1.2:
        return "accumulation"

    # Distribution: falling price on rising volume
    if pct_change < -0.5 and rvol > 1.2:
        return "distribution"

    # Dry up: very low volume
    if rvol < 0.5:
        return "dry_up"

    # Divergence: price up but volume down (or vice versa)
    if pct_change > 1.0 and rvol < 0.8:
        return "divergence"
    if pct_change < -1.0 and rvol < 0.8:
        return "divergence"

    return "steady"


def _relative_volume(data: dict) -> float:
    """Calculate relative volume (current / average)."""
    volume = data.get("volume", 0)
    avg_volume = data.get("avg_volume", volume)
    if avg_volume == 0:
        return 1.0
    return round(min(5.0, volume / avg_volume), 2)


def _volume_signals(data: dict) -> str:
    """Generate BoW volume signals."""
    signals = []
    rvol = _relative_volume(data)

    if rvol > 2.0:
        signals.append("high_relative_volume")
    elif rvol < 0.5:
        signals.append("low_relative_volume")

    # OBV trend
    obv_trend = data.get("obv_trend", "flat")
    if obv_trend == "up":
        signals.append("obv_rising")
    elif obv_trend == "down":
        signals.append("obv_falling")

    # VWAP position
    close = data.get("close", 0)
    vwap = data.get("vwap", close)
    if close > vwap:
        signals.append("above_vwap")
    elif close < vwap:
        signals.append("below_vwap")

    return " ".join(signals)


# ---------------------------------------------------------------------------
# Technical features
# ---------------------------------------------------------------------------

def _classify_indicators(data: dict) -> str:
    """Classify combined indicator signal."""
    rsi = data.get("rsi", 50)
    macd_hist = data.get("macd_histogram", 0)
    macd_signal = data.get("macd_signal_cross", "none")
    stoch_k = data.get("stochastic_k", 50)
    bb_position = data.get("bb_position", 0.5)  # 0=lower, 0.5=mid, 1=upper

    buy_count = 0
    sell_count = 0

    # RSI
    if rsi < 30:
        buy_count += 2  # oversold
    elif rsi < 40:
        buy_count += 1
    elif rsi > 70:
        sell_count += 2  # overbought
    elif rsi > 60:
        sell_count += 1

    # MACD
    if macd_hist > 0:
        buy_count += 1
    elif macd_hist < 0:
        sell_count += 1
    if macd_signal == "bull_cross":
        buy_count += 2
    elif macd_signal == "bear_cross":
        sell_count += 2

    # Stochastic
    if stoch_k < 20:
        buy_count += 1
    elif stoch_k > 80:
        sell_count += 1

    # Bollinger
    if bb_position < 0.1:
        buy_count += 1  # at lower band
    elif bb_position > 0.9:
        sell_count += 1  # at upper band

    # RSI divergence
    rsi_div = data.get("rsi_divergence", "none")
    if rsi_div == "bull":
        return "divergence_bull"
    elif rsi_div == "bear":
        return "divergence_bear"

    # Overbought/oversold
    if rsi > 75 and stoch_k > 80:
        return "overbought"
    if rsi < 25 and stoch_k < 20:
        return "oversold"

    # Net signal
    net = buy_count - sell_count
    if net >= 4:
        return "strong_buy"
    elif net >= 2:
        return "buy"
    elif net <= -4:
        return "strong_sell"
    elif net <= -2:
        return "sell"

    return "neutral"


def _classify_pattern(data: dict) -> str:
    """Classify chart pattern if detected."""
    return data.get("chart_pattern", "none")


def _tech_signals(data: dict) -> str:
    """Generate BoW technical signals."""
    signals = []
    rsi = data.get("rsi", 50)

    if rsi < 30:
        signals.append("rsi_oversold")
    elif rsi > 70:
        signals.append("rsi_overbought")

    macd_hist = data.get("macd_histogram", 0)
    if macd_hist > 0:
        signals.append("macd_positive")
    else:
        signals.append("macd_negative")

    macd_signal = data.get("macd_signal_cross", "none")
    if macd_signal == "bull_cross":
        signals.append("macd_bull_cross")
    elif macd_signal == "bear_cross":
        signals.append("macd_bear_cross")

    bb_position = data.get("bb_position", 0.5)
    if bb_position < 0.2:
        signals.append("bollinger_low")
    elif bb_position > 0.8:
        signals.append("bollinger_high")

    # Squeeze
    bb_width = data.get("bb_width", 0)
    if bb_width and bb_width < data.get("bb_width_avg", bb_width) * 0.75:
        signals.append("bollinger_squeeze")

    adx = data.get("adx", 0)
    if adx > 25:
        signals.append("strong_trend")
    elif adx < 15:
        signals.append("weak_trend")

    return " ".join(signals)


# ---------------------------------------------------------------------------
# Sentiment features
# ---------------------------------------------------------------------------

def _classify_sentiment(data: dict) -> str:
    """Classify sentiment polarity from available data."""
    score = data.get("sentiment_score", 0)
    fear_greed = data.get("fear_greed_index", 50)
    news_sentiment = data.get("news_sentiment", 0)

    # Fear/greed extremes override
    if fear_greed < 15:
        return "panic"
    if fear_greed < 25:
        return "fear"
    if fear_greed > 80:
        return "greed"

    # Composite score
    composite = score * 0.5 + news_sentiment * 0.3 + (fear_greed - 50) * 0.2

    if composite > 40:
        return "very_bullish"
    if composite > 15:
        return "bullish"
    if composite < -40:
        return "very_bearish"
    if composite < -15:
        return "bearish"

    # Mixed signals
    if abs(score) > 20 and abs(news_sentiment) > 20 and (score * news_sentiment < 0):
        return "mixed"

    return "neutral"


def _sentiment_score(data: dict) -> float:
    """Calculate sentiment score (-100 to 100)."""
    score = data.get("sentiment_score", 0)
    return max(-100, min(100, round(float(score), 1)))


def _sent_signals(data: dict) -> str:
    """Generate BoW sentiment signals."""
    signals = []

    fear_greed = data.get("fear_greed_index", 50)
    if fear_greed < 25:
        signals.append("extreme_fear")
    elif fear_greed < 40:
        signals.append("fear")
    elif fear_greed > 75:
        signals.append("extreme_greed")
    elif fear_greed > 60:
        signals.append("greed")

    social_volume = data.get("social_volume", "normal")
    if social_volume == "high":
        signals.append("social_buzz")
    elif social_volume == "spike":
        signals.append("social_spike")

    options_flow = data.get("options_flow", "neutral")
    if options_flow == "bullish":
        signals.append("options_bullish")
    elif options_flow == "bearish":
        signals.append("options_bearish")

    # Insider activity
    insider = data.get("insider_activity", "none")
    if insider == "buying":
        signals.append("insider_buying")
    elif insider == "selling":
        signals.append("insider_selling")

    # Earnings proximity
    days_to_earnings = data.get("days_to_earnings")
    if days_to_earnings is not None and days_to_earnings <= 7:
        signals.append("earnings_imminent")

    return " ".join(signals)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_features(data: dict) -> dict:
    """Extract all features from a market data snapshot.

    Args:
        data: Dict with keys like close, open, high, low, volume,
              sma_20, sma_50, rsi, macd_histogram, etc.

    Returns:
        Dict of features matching ENCODER_CONFIG role names.
    """
    return {
        # Price layer
        "trend_direction": _classify_trend(data),
        "momentum_score": _momentum_score(data),
        "price_signals": _price_signals(data),
        # Volume layer
        "volume_pattern": _classify_volume(data),
        "relative_volume": _relative_volume(data),
        "volume_signals": _volume_signals(data),
        # Technical layer
        "indicator_signal": _classify_indicators(data),
        "pattern_type": _classify_pattern(data),
        "tech_signals": _tech_signals(data),
        # Sentiment layer
        "sentiment_polarity": _classify_sentiment(data),
        "sentiment_score": _sentiment_score(data),
        "sent_signals": _sent_signals(data),
    }
