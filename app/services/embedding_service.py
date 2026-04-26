"""임베딩 생성 서비스.

provider 파라미터에 따라 OpenAI, Gemini, 또는 더미 벡터를 반환한다.
Gemini의 gemini-embedding-001은 기본 3072차원이지만 output_dimensionality로
축소 가능 (MRL: Matryoshka Representation Learning).
우리 테이블과 맞추기 위해 768차원으로 고정해서 요청한다.
"""

import hashlib
from typing import List

from app.config import settings

DIM_BY_PROVIDER = {
    "openai": 1536,
    "gemini": 768,
}


def _dummy_embedding(text: str, dim: int) -> List[float]:
    """결정론적 더미 벡터 (테스트용)."""
    h = hashlib.sha256(text.encode("utf-8")).digest()
    raw = (h * ((dim // len(h)) + 1))[:dim]
    return [(b - 127.5) / 127.5 for b in raw]


def _openai_embedding(text: str) -> List[float]:
    """OpenAI 임베딩 (1536차원)."""
    from openai import OpenAI

    client = OpenAI(api_key=settings.openai_api_key)
    response = client.embeddings.create(
        model=settings.openai_embedding_model,
        input=text,
    )
    return response.data[0].embedding


def _gemini_embedding(text: str) -> List[float]:
    """Gemini 임베딩 (768차원으로 축소)."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=settings.gemini_api_key)
    response = client.models.embed_content(
        model=settings.gemini_embedding_model,
        contents=text,
        config=types.EmbedContentConfig(
            output_dimensionality=DIM_BY_PROVIDER["gemini"],
        ),
    )
    return list(response.embeddings[0].values)


def create_embedding(text: str, provider: str) -> List[float]:
    """외부 공개 함수. provider에 따라 분기."""
    if not text or not text.strip():
        raise ValueError("text must not be empty")

    provider = provider.lower()
    if provider not in DIM_BY_PROVIDER:
        raise ValueError(f"Unsupported provider: {provider}")

    if settings.use_dummy_embedding:
        return _dummy_embedding(text, DIM_BY_PROVIDER[provider])

    if provider == "openai":
        return _openai_embedding(text)
    if provider == "gemini":
        return _gemini_embedding(text)

    raise ValueError(f"Unhandled provider: {provider}")