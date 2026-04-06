"""
Model packaging and deployment module.

This module provides functionality for packaging Ada models for deployment:
- AdaModel: Packaged model with glyphs, configuration, and metadata
- Model serialization/deserialization to .ada files
- Model validation for completeness and consistency
"""

from ada.model.package import AdaModel

__all__ = ["AdaModel"]
