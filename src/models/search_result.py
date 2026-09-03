"""Provider-independent search result model."""

from pydantic import AnyHttpUrl, BaseModel, ConfigDict, field_validator


class SearchResult(BaseModel):
    """A result with enough provenance to trace its query and provider."""

    model_config = ConfigDict(extra="forbid")

    title: str
    url: AnyHttpUrl
    snippet: str
    provider: str
    query: str

    @field_validator("title", "snippet")
    @classmethod
    def normalize_content(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("搜索结果的标题和摘要不能为空")
        return text

    @field_validator("provider", "query")
    @classmethod
    def normalize_provenance(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("搜索结果的 provider 和 query 不能为空")
        return text
