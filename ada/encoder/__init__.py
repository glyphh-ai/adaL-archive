"""
Encoder module for the Ada SDK.

This module provides:
- Base Encoder class for structured concept encoding
- IntentEncoder for NL query matching using HDC similarity
- IntentPattern and IntentMatch data structures
- Default intent patterns for common Ada operations
- Numeric binning encoder for similarity-preserving numeric encoding

The SDK focuses on structured input (Concept objects) for encoding, and provides
IntentEncoder for deterministic rules-based NL query matching.
"""

from ada.encoder.base import Encoder
from ada.encoder.intent import IntentEncoder, IntentPattern, IntentMatch
from ada.encoder.default_intents import (
    DEFAULT_INTENT_PATTERNS,
    get_default_patterns,
    get_pattern_by_type,
)
from ada.encoder.numeric import (
    compute_bin_number,
    clamp_value,
    thermometer_encode,
    binary_encode,
    gray_encode,
    encode_numeric_value,
    compute_similarity,
)
from ada.encoder.projection import ContinuousProjector

__all__ = [
    "Encoder",
    "IntentEncoder",
    "IntentPattern",
    "IntentMatch",
    "DEFAULT_INTENT_PATTERNS",
    "get_default_patterns",
    "get_pattern_by_type",
    # Numeric binning
    "compute_bin_number",
    "clamp_value",
    "thermometer_encode",
    "binary_encode",
    "gray_encode",
    "encode_numeric_value",
    "compute_similarity",
    # Continuous projection
    "ContinuousProjector",
]
