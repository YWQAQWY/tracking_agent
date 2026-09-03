"""Small shared helpers for extracting one JSON object from LLM output."""

import json
import re
from json import JSONDecodeError


def parse_json_object(raw: str) -> dict[str, object]:
    """Accept clean JSON or recover an object from common model wrappers."""
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    cleaned = re.sub(
        r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE
    )
    try:
        payload = json.loads(cleaned)
    except JSONDecodeError:
        payload = _extract_json_object(cleaned)
    if not isinstance(payload, dict):
        raise TypeError("Structured LLM output 必须是 JSON 对象")
    return payload


def _extract_json_object(text: str) -> object:
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except JSONDecodeError:
            continue
        return value
    raise JSONDecodeError("未找到 JSON 对象", text, 0)
