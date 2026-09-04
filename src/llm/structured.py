"""Small shared helpers for extracting one JSON object from LLM output."""

import json
import re
from json import JSONDecodeError
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from src.llm.client import LLMClient, LLMError


StructuredModel = TypeVar("StructuredModel", bound=BaseModel)


class StructuredOutputError(RuntimeError):
    """Raised after one bounded repair still cannot produce the requested model."""


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


def request_structured(
    llm: LLMClient,
    user_prompt: str,
    system_prompt: str,
    model_type: type[StructuredModel],
    component: str,
) -> StructuredModel:
    """Request and validate one JSON object, with exactly one repair attempt."""
    raw = ""
    try:
        raw = llm.chat(user_prompt, system_prompt)
        return model_type.model_validate(parse_json_object(raw))
    except LLMError:
        raise
    except (JSONDecodeError, ValidationError, TypeError, ValueError):
        repair_prompt = (
            "The prior response did not match the required JSON schema. "
            "Return only one corrected JSON object; no Markdown or commentary.\n\n"
            f"Original task:\n{user_prompt}\n\nPrior response:\n{raw}"
        )
        try:
            repaired = llm.chat(repair_prompt, system_prompt)
            return model_type.model_validate(parse_json_object(repaired))
        except LLMError:
            raise
        except (JSONDecodeError, ValidationError, TypeError, ValueError) as exc:
            raise StructuredOutputError(
                f"{component} 连续两次未生成合法 structured output。"
            ) from exc


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
