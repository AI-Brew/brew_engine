"""RAG 답변 생성 라우터.

동작:
  1. provider 결정 (요청 파라미터)
  2. 질문을 해당 provider로 임베딩
  3. 라우팅: bot_id 없으면 같은 provider 봇 중 가장 가까운 봇 선택
  4. 검색: 해당 봇의 지식 중 top_k개 찾기
  5. 생성: 같은 provider의 LLM으로 답변 생성
"""

from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.config import settings
from app.db import get_db
from app.models import PROVIDER_MODELS
from app.schemas import GenerateRequest, GenerateResponse, RetrievedKnowledge
from app.services.embedding_service import create_embedding
from app.services.llm_service import generate_answer

router = APIRouter(tags=["generate"])


def _route_bot(db: Session, VectorModel, question_vec):
    bot = (
        db.query(VectorModel)
        .filter(VectorModel.description_embedding.isnot(None))
        .order_by(VectorModel.description_embedding.cosine_distance(question_vec))
        .first()
    )
    if bot is None:
        raise HTTPException(
            status_code=404,
            detail="No routable bot found for this provider. Register via POST /bot-vectors first.",
        )
    return bot


def _retrieve(db: Session, KnowledgeModel, bot_id: int, question_vec, top_k: int) -> List[RetrievedKnowledge]:
    rows = (
        db.query(
            KnowledgeModel.id,
            KnowledgeModel.content,
            KnowledgeModel.embedding.cosine_distance(question_vec).label("distance"),
        )
        .filter(KnowledgeModel.bot_id == bot_id)
        .filter(KnowledgeModel.embedding.isnot(None))
        .order_by("distance")
        .limit(top_k)
        .all()
    )
    return [
        RetrievedKnowledge(id=r.id, content=r.content, distance=float(r.distance))
        for r in rows
    ]


@router.post(
    "/generate",
    response_model=GenerateResponse,
    summary="RAG 답변 생성 (핵심)",
    description=(
        "사용자 질문 → 선택된 provider의 임베딩 → 관련 봇/지식 검색 → 같은 provider의 LLM 답변.\n\n"
        "파라미터:\n"
        "- provider: openai | gemini (필수)\n"
        "- bot_id: 생략하면 자동 라우팅\n"
        "- system_prompt: 봇 성격/규칙 (Spring에서 전달)\n"
        "- top_k: 검색할 지식 개수"
    ),
)
def generate(payload: GenerateRequest, db: Session = Depends(get_db)):
    if payload.provider not in PROVIDER_MODELS:
        raise HTTPException(status_code=400, detail=f"Unsupported provider: {payload.provider}")

    VectorModel, KnowledgeModel = PROVIDER_MODELS[payload.provider]

    # 1) 질문 임베딩
    question_vec = create_embedding(payload.question, payload.provider)

    # 2) 라우팅 또는 bot_id 지정
    if payload.bot_id is not None:
        bot = db.query(VectorModel).filter(VectorModel.bot_id == payload.bot_id).first()
        if bot is None:
            raise HTTPException(
                status_code=404,
                detail=f"Bot {payload.bot_id} not found for provider {payload.provider}",
            )
    else:
        bot = _route_bot(db, VectorModel, question_vec)

    # 3) 지식 검색
    retrieved = _retrieve(db, KnowledgeModel, bot.bot_id, question_vec, payload.top_k)

    # 4) 답변 생성
    answer = generate_answer(
        provider=payload.provider,
        question=payload.question,
        system_prompt=payload.system_prompt,
        retrieved_contents=[r.content for r in retrieved],
    )

    return GenerateResponse(
        question=payload.question,
        provider=payload.provider,
        selected_bot_id=bot.bot_id,
        retrieved=retrieved,
        answer=answer,
        mode="dummy" if settings.use_dummy_embedding else payload.provider,
    )