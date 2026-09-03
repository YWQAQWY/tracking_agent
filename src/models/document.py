"""A web document that Tracker actually fetched and read."""

from pydantic import AnyHttpUrl, BaseModel, ConfigDict, field_validator


class Document(BaseModel):
    """Clean article text extracted from one fetched web page."""

    model_config = ConfigDict(extra="forbid")

    url: AnyHttpUrl
    title: str | None = None
    text: str

    @field_validator("title")
    @classmethod
    def normalize_title(cls, value: str | None) -> str | None:
        if value is None:
            return None
        title = value.strip()
        return title or None

    @field_validator("text")
    @classmethod
    def require_text(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("Document text 不能为空")
        return text

