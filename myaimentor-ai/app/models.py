from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, DateTime, Integer, String, Text
from sqlalchemy.sql import func

from app.db import Base


class BotKnowledge(Base):
    """Bot knowledge base. Retrieved in RAG."""
    __tablename__ = "bot_knowledge"

    id = Column(Integer, primary_key=True, index=True)
    bot_id = Column(Integer, nullable=False, index=True)
    content = Column(Text, nullable=False)
    embedding = Column(Vector(1536), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class BotEmbedding(Base):
    """Bot routing. Match user question to the right bot."""
    __tablename__ = "bot_embeddings"

    id = Column(Integer, primary_key=True, index=True)
    bot_id = Column(Integer, nullable=False, unique=True, index=True)
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=False)
    description_embedding = Column(Vector(1536), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())