"""봇 라우팅 벡터 라우터.

provider에 따라 bot_vectors_openai / bot_vectors_gemini 테이블에 저장.
동일 bot_id라도 provider가 다르면 별개 엔트리.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import PROVIDER_MODELS
from app.schemas import BotVectorCreate, BotVectorResponse, Provider
from app.services.embedding_service import create_embedding

router = APIRouter(tags=["bot-vectors"])


def _get_vector_model(provider: str):
    if provider not in PROVIDER_MODELS:
        raise HTTPException(status_code=400, detail=f"Unsupported provider: {provider}")
    VectorModel, _ = PROVIDER_MODELS[provider]
    return VectorModel


def _to_response(b, provider: str) -> BotVectorResponse:
    return BotVectorResponse(
        bot_id=b.bot_id,
        description=b.description,
        provider=provider,  # type: ignore[arg-type]
        has_embedding=b.description_embedding is not None,
        created_at=b.created_at,
    )


@router.post(
    "/bot-vectors",
    response_model=BotVectorResponse,
    summary="봇 라우팅 벡터 저장 (upsert)",
    description=(
        "봇의 description을 provider 기준으로 임베딩해 저장.\n"
        "이미 같은 bot_id+provider 조합이 있으면 덮어쓴다 (upsert).\n"
        "다른 provider에는 영향 없음."
    ),
)
def upsert_bot_vector(payload: BotVectorCreate, db: Session = Depends(get_db)):
    VectorModel = _get_vector_model(payload.provider)
    vector = create_embedding(payload.description, payload.provider)

    bot = db.query(VectorModel).filter(VectorModel.bot_id == payload.bot_id).first()
    if bot is None:
        bot = VectorModel(
            bot_id=payload.bot_id,
            description=payload.description,
            description_embedding=vector,
        )
        db.add(bot)
    else:
        bot.description = payload.description
        bot.description_embedding = vector

    db.commit()
    db.refresh(bot)
    return _to_response(bot, payload.provider)


@router.delete(
    "/bot-vectors/{bot_id}",
    status_code=204,
    summary="봇 라우팅 벡터 삭제",
    description="provider 쿼리 파라미터로 지정한 테이블에서만 삭제한다.",
)
def delete_bot_vector(
    bot_id: int,
    provider: Provider = Query(..., description="openai | gemini"),
    db: Session = Depends(get_db),
):
    VectorModel = _get_vector_model(provider)
    bot = db.query(VectorModel).filter(VectorModel.bot_id == bot_id).first()
    if bot is None:
        raise HTTPException(status_code=404, detail="Bot vector not found")
    db.delete(bot)
    db.commit()
    return None