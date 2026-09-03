"""Data returned by the HTTP crawler before content extraction."""

from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field


class FetchResult(BaseModel):
    """Successful HTML response with normalized metadata."""

    model_config = ConfigDict(extra="forbid")

    url: AnyHttpUrl
    html: str
    status_code: int = Field(ge=200, lt=400)
    content_type: str | None = None

