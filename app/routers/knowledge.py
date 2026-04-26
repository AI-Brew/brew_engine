"""지식 관리 라우터.

provider에 따라 bot_knowledge_openai / bot_knowledge_gemini 테이블로 분기.
등록 시 봇 설정의 청킹 파라미터로 텍스트를 분할해 청크별 1 row 로 저장한다.
"""

from typing import List

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import PROVIDER_MODELS
from app.schemas import (
    ChunkOut,
    KnowledgeCreate,
    KnowledgePreviewRequest,
    KnowledgePreviewResponse,
    KnowledgeResponse,
    Provider,
)
from app.services.chunker import chunk_text
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
    response_model=List[KnowledgeResponse],
    status_code=201,
    summary="지식 등록 (청크별 분할 + 벡터 자동 생성)",
    description=(
        "받은 content 를 봇 설정의 청킹 파라미터로 분할 후, 청크별로 1 row 저장한다.\n"
        "응답은 생성된 청크 리스트 (각 청크가 별도의 KnowledgeResponse)."
    ),
)
def create_knowledge(payload: KnowledgeCreate, db: Session = Depends(get_db)):
    KnowledgeModel = _get_knowledge_model(payload.provider)

    # 1) 청킹
    chunks = chunk_text(
        payload.content,
        chunk_size=payload.chunk_size or 500,
        chunk_overlap=payload.chunk_overlap or 100,
        splitter=payload.chunk_splitter or "recursive",
    )
    if not chunks:
        raise HTTPException(status_code=400, detail="No chunks produced from content")

    # 2) 청크별 임베딩 + 저장
    saved = []
    for chunk in chunks:
        vector = create_embedding(chunk, payload.provider)
        knowledge = KnowledgeModel(
            bot_id=payload.bot_id,
            content=chunk,
            embedding=vector,
        )
        db.add(knowledge)
        saved.append(knowledge)
    db.commit()
    for k in saved:
        db.refresh(k)
    return [_to_response(k, payload.provider) for k in saved]


@router.post(
    "/knowledge/preview",
    response_model=KnowledgePreviewResponse,
    summary="청킹 미리보기 (저장 없음)",
    description=(
        "등록 시 사용되는 동일한 청킹 함수를 호출해 분할 결과만 반환한다. DB 저장 X.\n"
        "프론트가 라이브 미리보기에 활용."
    ),
)
def preview_knowledge(payload: KnowledgePreviewRequest):
    size = payload.chunk_size or 500
    overlap = payload.chunk_overlap or 100
    splitter = payload.chunk_splitter or "recursive"
    chunks = chunk_text(payload.content, size, overlap, splitter)
    return KnowledgePreviewResponse(
        chunks=[ChunkOut(index=i, text=t) for i, t in enumerate(chunks)],
        chunk_size=size,
        chunk_overlap=overlap,
        splitter=splitter,
    )


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