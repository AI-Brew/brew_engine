from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class KnowledgeCreate(BaseModel):
    """POST /knowledge request body."""
    bot_id: int = Field(..., description="Bot ID", gt=0)
    content: str = Field(..., description="Knowledge content", min_length=1)


class KnowledgeResponse(BaseModel):
    """Knowledge response (embedding excluded)."""
    id: int
    bot_id: int
    content: str
    has_embedding: bool
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)