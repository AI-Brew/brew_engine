"""지식 관리 라우터.

provider에 따라 bot_knowledge_openai / bot_knowledge_gemini 테이블로 분기.
"""

from typing import List

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import PROVIDER_MODELS
from app.schemas import KnowledgeCreate, KnowledgeResponse, Provider
from app.services.embedding_service import create_embedding

router = APIRouter(tags=["knowledge"])


def _get_knowledge_model(provider: str):
    """provider 이름으로 Knowledge 모델 클래스를 가져온다."""
    if provider not in PROVIDER_MODELS:
        raise HTTPException(status_code=400, detail=f"Unsupported provider: {provider}")
    _, KnowledgeModel = PROVIDER_MODELS[provider]
    return KnowledgeModel


def _to_response(k, provider: str) -> KnowledgeResponse:
    return KnowledgeResponse(
        id=k.id,
        bot_id=k.bot_id,
        content=k.content,
        provider=provider,  # type: ignore[arg-type]
        has_embedding=k.embedding is not None,
        created_at=k.created_at,
    )


@router.post(
    "/knowledge",
    response_model=KnowledgeResponse,
    status_code=201,
    summary="지식 등록 (+ 벡터 자동 생성)",
    description=(
        "봇의 지식을 등록한다. provider에 따라 임베딩 차원이 다르고,\n"
        "별도 테이블에 저장된다 (openai=1536차원, gemini=768차원).\n"
        "같은 bot_id라도 provider가 다르면 서로 다른 레코드로 관리된다."
    ),
)
def create_knowledge(payload: KnowledgeCreate, db: Session = Depends(get_db)):
    KnowledgeModel = _get_knowledge_model(payload.provider)
    vector = create_embedding(payload.content, payload.provider)
    knowledge = KnowledgeModel(
        bot_id=payload.bot_id,
        content=payload.content,
        embedding=vector,
    )
    db.add(knowledge)
    db.commit()
    db.refresh(knowledge)
    return _to_response(knowledge, payload.provider)


@router.get(
    "/bots/{bot_id}/knowledge",
    response_model=List[KnowledgeResponse],
    summary="특정 봇의 지식 목록",
    description=(
        "provider 쿼리 파라미터로 어느 테이블에서 찾을지 지정한다.\n"
        "기본값: gemini (config의 default_llm_provider)."
    ),
)
def list_knowledge_by_bot(
    bot_id: int,
    provider: Provider = Query(..., description="openai | gemini"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    KnowledgeModel = _get_knowledge_model(provider)
    items = (
        db.query(KnowledgeModel)
        .filter(KnowledgeModel.bot_id == bot_id)
        .order_by(KnowledgeModel.id.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )
    return [_to_response(i, provider) for i in items]


@router.delete(
    "/knowledge/{knowledge_id}",
    status_code=204,
    summary="지식 1건 삭제",
    description="provider를 쿼리 파라미터로 받아 해당 테이블에서 삭제한다.",
)
def delete_knowledge(
    knowledge_id: int,
    provider: Provider = Query(..., description="openai | gemini"),
    db: Session = Depends(get_db),
):
    KnowledgeModel = _get_knowledge_model(provider)
    knowledge = db.query(KnowledgeModel).filter(KnowledgeModel.id == knowledge_id).first()
    if knowledge is None:
        raise HTTPException(status_code=404, detail="Knowledge not found")
    db.delete(knowledge)
    db.commit()
    return None