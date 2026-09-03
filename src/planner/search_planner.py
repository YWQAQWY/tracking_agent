"""Convert a user question into complementary web search queries."""

from __future__ import annotations

import json
import re
from json import JSONDecodeError

from pydantic import ValidationError

from src.llm.client import LLMClient, LLMError
from src.models.search_plan import SearchPlan


class PlannerError(RuntimeError):
    """Raised when the local LLM cannot produce a valid plan."""


SYSTEM_PROMPT = """You are a search query planner.

Convert the user's question into multiple complementary web search queries.
The queries should explore different aspects, terminology, and formulations of
the same information need.

Rules:
1. Generate concise search-engine-friendly queries.
2. Do not produce queries that are only simple paraphrases of each other.
3. Use different terminology when useful.
4. Preserve important technical keywords from the user's question.
5. Prefer complementary queries that increase search coverage.
6. Do not answer the user's question.
7. Output only one valid JSON object shaped as {"queries": ["query 1"]}.
Do not output reasoning, Markdown, code fences, or explanations."""


class SearchPlanner:
    """Build a bounded multi-query plan with at most one repair attempt."""

    def __init__(self, llm: LLMClient, max_queries: int = 3) -> None:
        if not 1 <= max_queries <= 10:
            raise ValueError("max_queries 必须在 1 到 10 之间")
        self.llm = llm
        self.max_queries = max_queries

    def plan(self, question: str) -> SearchPlan:
        clean_question = question.strip()
        if not clean_question:
            raise PlannerError("用户问题不能为空。")

        prompt = (
            f"User question:\n{clean_question}\n\n"
            f"Generate at most {self.max_queries} complementary queries."
        )
        try:
            raw = self.llm.chat(prompt, SYSTEM_PROMPT)
            return self._parse(raw, self.max_queries)
        except (JSONDecodeError, ValidationError, TypeError, ValueError):
            repair_prompt = (
                "上一次输出无法通过 JSON 校验。请重新输出且只输出 JSON 对象。\n"
                f"用户问题：{clean_question}\n"
                f"最多 queries 数量：{self.max_queries}\n"
                f"上一次输出：{raw}"
            )
            try:
                repaired = self.llm.chat(repair_prompt, SYSTEM_PROMPT)
                return self._parse(repaired, self.max_queries)
            except LLMError:
                raise
            except (JSONDecodeError, ValidationError, TypeError, ValueError) as exc:
                raise PlannerError(
                    "本地模型连续两次未生成合法 SearchPlan JSON。"
                ) from exc
        except LLMError:
            raise

    @classmethod
    def _parse(cls, raw: str, max_queries: int) -> SearchPlan:
        cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)
        try:
            payload = json.loads(cleaned)
        except JSONDecodeError:
            payload = cls._extract_json_object(cleaned)
        if not isinstance(payload, dict):
            raise TypeError("SearchPlan 必须是 JSON 对象")
        plan = SearchPlan.model_validate(payload)
        return SearchPlan(queries=plan.queries[:max_queries])

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
