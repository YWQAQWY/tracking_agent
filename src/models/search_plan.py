"""Validated multi-query search plan produced by the local LLM."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SearchPlan(BaseModel):
    """Complementary search expressions for one user information need."""

    model_config = ConfigDict(extra="forbid")

    queries: list[str] = Field(min_length=1)

    @field_validator("queries", mode="before")
    @classmethod
    def normalize_queries(cls, value: Any) -> list[str]:
        if not isinstance(value, list):
            raise ValueError("queries 必须是数组")

        # V0.3 only removes exact duplicates after whitespace trimming.
        # Semantic query deduplication belongs to a later ranking stage.
        queries: list[str] = []
        seen: set[str] = set()
        for item in value:
            if not isinstance(item, str):
                raise ValueError("每个 query 必须是字符串")
            query = item.strip()
            if query and query not in seen:
                seen.add(query)
                queries.append(query)
        if not queries:
            raise ValueError("queries 不能为空")
        return queries
