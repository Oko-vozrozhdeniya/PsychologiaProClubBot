"""Translate common extraction requirements into provider API parameters."""
from memory.ai_contract import EXTRACTION_MAX_TOKENS


def extraction_settings(provider):
    if provider == "Gemini":
        return {"temperature": 0, "maxOutputTokens": EXTRACTION_MAX_TOKENS,
                "responseMimeType": "application/json"}
    if provider == "Groq":
        return {"temperature": 0, "max_completion_tokens": EXTRACTION_MAX_TOKENS,
                "response_format": {"type": "json_object"}}
    raise ValueError("No structured-output adapter for provider")


