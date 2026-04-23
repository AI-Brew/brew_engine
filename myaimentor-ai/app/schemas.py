"""요청/응답 스키마.

모든 요청에 provider 필드가 추가되어 OpenAI/Gemini를 선택할 수 있다.
"""

from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


# provider 타입. 여기에 새 provider 추가하면 전체 시스템이 지원.
Provider = Literal["openai", "gemini"]


# ===== Knowledge =====

class KnowledgeCreate(BaseModel):
    bot_id: int = Field(..., gt=0, description="Bot ID")
    content: str = Field(..., min_length=1, description="Knowledge content")
    provider: Provider = Field(..., description="LLM provider for embedding")


class KnowledgeResponse(BaseModel):
    id: int
    bot_id: int
    content: str
    provider: Provider
    has_embedding: bool
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


# ===== Bot Vectors =====

class BotVectorCreate(BaseModel):
    bot_id: int = Field(..., gt=0, description="Bot ID (unique per provider)")
    description: str = Field(..., min_length=1, description="Bot description for routing")
    provider: Provider = Field(..., description="LLM provider for embedding")


class BotVectorResponse(BaseModel):
    bot_id: int
    description: str
    provider: Provider
    has_embedding: bool
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


# ===== Generate (RAG) =====

class GenerateRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User question")
    provider: Provider = Field(..., description="LLM provider to use")
    bot_id: Optional[int] = Field(
        None,
        description="If provided, skip routing and use this bot directly",
    )
    system_prompt: Optional[str] = Field(
        None,
        description="Bot personality/rules prompt (passed to LLM)",
    )
    top_k: int = Field(3, ge=1, le=10, description="Number of knowledge chunks to retrieve")


class RetrievedKnowledge(BaseModel):
    id: int
    content: str
    distance: float


class GenerateResponse(BaseModel):
    question: str
    provider: Provider
    selected_bot_id: int
    retrieved: List[RetrievedKnowledge]
    answer: str
    mode: str = Field(..., description="'dummy' or provider name")