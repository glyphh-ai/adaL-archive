"""
Ada Assistant module.

The assistant is a Ada model that routes natural language queries
to actions (commands, knowledge responses, quick actions, query help).

Primary API:
- Assistant: thin client (online -> runtime, offline -> local .ada)
- AssistantConfig: configuration (runtime_url, threshold, etc.)
- AssistantResponse: state-based response (DONE, ASK, ERROR)

Build tooling:
- encoder_config: ASSISTANT_ENCODER_CONFIG (router-style EncoderConfig)
- concept_converter: JSONL entry -> Concept conversion
"""

from ada.assistant.core import (
    Assistant,
    AssistantConfig,
    AssistantResponse,
)

__all__ = [
    "Assistant",
    "AssistantConfig",
    "AssistantResponse",
]
