"""Validated search plan produced by the local LLM."""

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SearchPlan(BaseModel):
    """The single query needed by Tracker V0.1."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1)
    max_results: int = Field(default=5, ge=1, le=10)

    @field_validator("query")
    @classmethod
    def normalize_query(cls, value: str) -> str:
        query = value.strip()
        if not query:
            raise ValueError("query 不能为空")
        return query

