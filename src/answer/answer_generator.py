"""Generate an answer from normalized search snippets."""

from src.llm.client import LLMClient
from src.models.search_result import SearchResult


class AnswerGenerationError(RuntimeError):
    """Raised when an answer cannot be generated from search results."""


SYSTEM_PROMPT = """你是一个基于联网检索摘要回答问题的助手。
优先且只能依据提供的搜索结果中的具体事实回答。
如果结果不足以支持结论，明确说明信息不足，不要虚构。
用 [1]、[2] 等标注事实来源。回答使用与用户问题相同的语言。
当前上下文只有搜索引擎返回的标题、URL 和摘要，不包含网页正文。"""


class AnswerGenerator:
    """Format snippets and ask the local model for the final answer."""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def generate(self, question: str, results: list[SearchResult]) -> str:
        clean_question = question.strip()
        if not clean_question:
            raise AnswerGenerationError("用户问题不能为空。")
        if not results:
            raise AnswerGenerationError("搜索没有返回可用结果，无法生成可靠回答。")

        context = self._format_context(results)
        prompt = f"用户问题：\n{clean_question}\n\n搜索结果：\n{context}\n\n请生成最终回答。"
        return self.llm.chat(prompt, SYSTEM_PROMPT)

    @staticmethod
    def _format_context(results: list[SearchResult]) -> str:
        blocks = []
        for index, result in enumerate(results, start=1):
            blocks.append(
                f"[{index}]\n"
                f"Title: {result.title}\n"
                f"URL: {result.url}\n"
                f"Snippet: {result.snippet}"
            )
        return "\n\n".join(blocks)

