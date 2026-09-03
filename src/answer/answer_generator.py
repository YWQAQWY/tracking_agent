"""Generate an answer from retrieval-selected, source-aware Evidence."""

from src.llm.client import LLMClient


class AnswerGenerationError(RuntimeError):
    """Raised when an answer cannot be generated from selected Evidence."""


SYSTEM_PROMPT = """You are a research assistant.
Answer the user's question using only the provided selected evidence.
Rules:
1. Ground factual claims in the supplied evidence.
2. Do not invent unsupported details.
3. If the evidence is insufficient, explicitly say so.
4. Prefer synthesizing multiple sources instead of relying on one source.
5. Cite supporting evidence as [Source 1], [Source 2], and so on.
6. Keep the answer clear and concise, using the user's language.
7. Answer the original user question, not intermediate search queries.
8. If research stopped with incomplete evidence, state the remaining limitation."""


class AnswerGenerator:
    """Ask the local model to answer only from prepared Evidence context."""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def generate(self, question: str, context: str) -> str:
        clean_question = question.strip()
        if not clean_question:
            raise AnswerGenerationError("用户问题不能为空。")
        if not context.strip():
            raise AnswerGenerationError("Evidence 上下文为空，无法生成可靠回答。")

        prompt = (
            f"Question:\n{clean_question}\n\n"
            f"Selected Evidence:\n{context}\n\n"
            "Answer using only the selected evidence above."
        )
        return self.llm.chat(prompt, SYSTEM_PROMPT)
