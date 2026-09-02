"""Provider-independent search result model."""

from pydantic import AnyHttpUrl, BaseModel, ConfigDict, field_validator


class SearchResult(BaseModel):
    """A normalized search result passed to the answer generator."""

    model_config = ConfigDict(extra="forbid")

    title: str
    url: AnyHttpUrl
    snippet: str

    @field_validator("title", "snippet")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("搜索结果的标题和摘要不能为空")
        return text

