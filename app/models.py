"""SQLAlchemy ORM 모델.

LLM provider별로 임베딩 차원이 달라서(OpenAI=1536, Gemini=768)
테이블을 provider별로 분리한다.

- bot_vectors_openai / bot_vectors_gemini: 봇 라우팅 벡터
- bot_knowledge_openai / bot_knowledge_gemini: 봇 지식 저장소
"""

from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, DateTime, Integer, Text
from sqlalchemy.sql import func

from app.db import Base


# ===== OpenAI (1536-dim) =====

class BotVectorOpenAI(Base):
    __tablename__ = "bot_vectors_openai"

    id = Column(Integer, primary_key=True, index=True)
    bot_id = Column(Integer, nullable=False, unique=True, index=True)
    description = Column(Text, nullable=False)
    description_embedding = Column(Vector(1536), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class BotKnowledgeOpenAI(Base):
    __tablename__ = "bot_knowledge_openai"

    id = Column(Integer, primary_key=True, index=True)
    bot_id = Column(Integer, nullable=False, index=True)
    content = Column(Text, nullable=False)
    embedding = Column(Vector(1536), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


# ===== Gemini (768-dim) =====

class BotVectorGemini(Base):
    __tablename__ = "bot_vectors_gemini"

    id = Column(Integer, primary_key=True, index=True)
    bot_id = Column(Integer, nullable=False, unique=True, index=True)
    description = Column(Text, nullable=False)
    description_embedding = Column(Vector(768), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class BotKnowledgeGemini(Base):
    __tablename__ = "bot_knowledge_gemini"

    id = Column(Integer, primary_key=True, index=True)
    bot_id = Column(Integer, nullable=False, index=True)
    content = Column(Text, nullable=False)
    embedding = Column(Vector(768), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


# provider 이름 -> (VectorModel, KnowledgeModel) 매핑
PROVIDER_MODELS = {
    "openai": (BotVectorOpenAI, BotKnowledgeOpenAI),
    "gemini": (BotVectorGemini, BotKnowledgeGemini),
}