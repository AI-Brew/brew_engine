from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import BotKnowledge
from app.schemas import KnowledgeCreate, KnowledgeResponse

router = APIRouter(prefix="/knowledge", tags=["knowledge"])


def _to_response(k: BotKnowledge) -> KnowledgeResponse:
    return KnowledgeResponse(
        id=k.id,
        bot_id=k.bot_id,
        content=k.content,
        has_embedding=k.embedding is not None,
        created_at=k.created_at,
    )


@router.post("", response_model=KnowledgeResponse, status_code=201)
def create_knowledge(payload: KnowledgeCreate, db: Session = Depends(get_db)):
    """Register knowledge for a bot. Embedding is skipped for now (filled later)."""
    knowledge = BotKnowledge(
        bot_id=payload.bot_id,
        content=payload.content,
        embedding=None,
    )
    db.add(knowledge)
    db.commit()
    db.refresh(knowledge)
    return _to_response(knowledge)


@router.get("", response_model=List[KnowledgeResponse])
def list_knowledge(
    bot_id: Optional[int] = Query(None, description="Filter by bot_id"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    """List knowledge entries. Optionally filter by bot_id."""
    query = db.query(BotKnowledge)
    if bot_id is not None:
        query = query.filter(BotKnowledge.bot_id == bot_id)
    items = query.order_by(BotKnowledge.id.desc()).offset(offset).limit(limit).all()
    return [_to_response(i) for i in items]


@router.get("/{knowledge_id}", response_model=KnowledgeResponse)
def get_knowledge(knowledge_id: int, db: Session = Depends(get_db)):
    """Get a single knowledge entry by ID."""
    knowledge = db.query(BotKnowledge).filter(BotKnowledge.id == knowledge_id).first()
    if knowledge is None:
        raise HTTPException(status_code=404, detail="Knowledge not found")
    return _to_response(knowledge)