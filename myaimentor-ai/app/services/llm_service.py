"""LLM 답변 생성 서비스.

provider에 따라 OpenAI ChatCompletion 또는 Gemini generate_content 호출.
system_prompt + 검색된 지식 + 사용자 질문을 조합해서 전달한다.
"""

from typing import List

from app.config import settings


def _build_context(retrieved: List[str]) -> str:
    if not retrieved:
        return "(관련 지식 없음)"
    return "\n".join(f"- {content}" for content in retrieved)


def _build_system_prompt(system_prompt: str | None, context: str) -> str:
    base = system_prompt or "You are a helpful assistant."
    return (
        f"{base}\n\n"
        f"아래 참고 자료를 근거로 답변하세요. 자료가 부족하면 모른다고 답하세요.\n\n"
        f"참고:\n{context}"
    )


def _openai_chat(question: str, system_prompt: str | None, retrieved_contents: List[str]) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=settings.openai_api_key)
    final_system = _build_system_prompt(system_prompt, _build_context(retrieved_contents))
    response = client.chat.completions.create(
        model=settings.openai_chat_model,
        messages=[
            {"role": "system", "content": final_system},
            {"role": "user", "content": question},
        ],
    )
    return response.choices[0].message.content or ""


def _gemini_chat(question: str, system_prompt: str | None, retrieved_contents: List[str]) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=settings.gemini_api_key)
    final_system = _build_system_prompt(system_prompt, _build_context(retrieved_contents))

    response = client.models.generate_content(
        model=settings.gemini_chat_model,
        contents=question,
        config=types.GenerateContentConfig(system_instruction=final_system),
    )
    return response.text or ""


def _dummy_chat(question: str, system_prompt: str | None, retrieved_contents: List[str]) -> str:
    ctx = _build_context(retrieved_contents)
    return (
        f"[DUMMY MODE]\n"
        f"질문: {question}\n"
        f"시스템 프롬프트: {system_prompt or '(없음)'}\n"
        f"검색된 지식:\n{ctx}"
    )


def generate_answer(
    provider: str,
    question: str,
    system_prompt: str | None,
    retrieved_contents: List[str],
) -> str:
    if settings.use_dummy_embedding:
        return _dummy_chat(question, system_prompt, retrieved_contents)

    provider = provider.lower()
    if provider == "openai":
        return _openai_chat(question, system_prompt, retrieved_contents)
    if provider == "gemini":
        return _gemini_chat(question, system_prompt, retrieved_contents)
    raise ValueError(f"Unsupported provider: {provider}")