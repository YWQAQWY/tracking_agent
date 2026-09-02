"""Convert a user question into a minimal validated SearchPlan."""

from __future__ import annotations

import json
import re
from json import JSONDecodeError

from pydantic import ValidationError

from src.llm.client import LLMClient, LLMError
from src.models.search_plan import SearchPlan


class PlannerError(RuntimeError):
    """Raised when the local LLM cannot produce a valid plan."""


SYSTEM_PROMPT = """你是搜索规划器。根据用户问题生成适合搜索引擎的一条搜索关键词。
只输出一个符合以下结构的 JSON 对象：
{"query": "搜索关键词", "max_results": 5}
不要回答问题，不要输出思考过程、Markdown、代码围栏或解释。
query 必须是非空字符串；max_results 必须是 1 到 10 的整数。"""


class SearchPlanner:
    """Build one search plan with at most one repair attempt."""

    def __init__(self, llm: LLMClient, default_max_results: int = 5) -> None:
        self.llm = llm
        self.default_max_results = default_max_results

    def plan(self, question: str) -> SearchPlan:
        clean_question = question.strip()
        if not clean_question:
            raise PlannerError("用户问题不能为空。")

        prompt = (
            f"用户问题：{clean_question}\n"
            f"请将 max_results 设为 {self.default_max_results}。"
        )
        try:
            raw = self.llm.chat(prompt, SYSTEM_PROMPT)
            return self._parse(raw)
        except (JSONDecodeError, ValidationError, TypeError, ValueError) as first_error:
            repair_prompt = (
                "上一次输出无法通过 JSON 校验。请重新输出且只输出 JSON 对象。\n"
                f"用户问题：{clean_question}\n"
                f"max_results：{self.default_max_results}\n"
                f"上一次输出：{raw}"
            )
            try:
                repaired = self.llm.chat(repair_prompt, SYSTEM_PROMPT)
                return self._parse(repaired)
            except LLMError:
                raise
            except (JSONDecodeError, ValidationError, TypeError, ValueError) as exc:
                raise PlannerError(
                    "本地模型连续两次未生成合法 SearchPlan JSON。"
                ) from exc
        except LLMError:
            raise

    @classmethod
    def _parse(cls, raw: str) -> SearchPlan:
        cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)
        try:
            payload = json.loads(cleaned)
        except JSONDecodeError:
            payload = cls._extract_json_object(cleaned)
        if not isinstance(payload, dict):
            raise TypeError("SearchPlan 必须是 JSON 对象")
        return SearchPlan.model_validate(payload)

    @staticmethod
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

