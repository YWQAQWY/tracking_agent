"""Generate an answer from extracted, source-aware web context."""

from src.llm.client import LLMClient


class AnswerGenerationError(RuntimeError):
    """Raised when an answer cannot be generated from extracted web context."""


SYSTEM_PROMPT = """You are a research assistant.
Answer the user's question using the provided web sources.
Rules:
1. Base factual claims on the provided sources.
2. Do not invent information that is not supported by the sources.
3. If the sources are insufficient, explicitly say so.
4. Cite supporting sources as [Source 1], [Source 2], and so on.
5. Keep the answer clear and concise, using the user's language."""


class AnswerGenerator:
    """Ask the local model to answer from prepared Document context."""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def generate(self, question: str, context: str) -> str:
        clean_question = question.strip()
        if not clean_question:
            raise AnswerGenerationError("用户问题不能为空。")
        if not context.strip():
            raise AnswerGenerationError("网页正文上下文为空，无法生成可靠回答。")

        prompt = (
            f"Question:\n{clean_question}\n\n"
            f"Web Sources:\n{context}\n\n"
            "Answer using only the web sources above."
        )
        return self.llm.chat(prompt, SYSTEM_PROMPT)
