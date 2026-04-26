"""FastAPI 앱 진입점.

- 앱 시작 시 DB 테이블 자동 생성 (없을 경우에만).
- 3개 라우터(knowledge, bot-vectors, generate)를 등록.
- 루트(/)만 간단한 확인용 응답.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.db import Base, engine
from app import models  # noqa: F401  메타데이터 등록용
from app.routers import bot_vectors, generate, knowledge


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(
    title="My AI Mentor Studio - AI Service",
    description=(
        "벡터 임베딩, 유사도 검색, LLM 호출을 담당하는 마이크로서비스.\n"
        "봇 비즈니스 로직은 Spring 서비스가 담당하고, 여기는 AI 추론 전담."
    ),
    lifespan=lifespan,
)

app.include_router(knowledge.router)
app.include_router(bot_vectors.router)
app.include_router(generate.router)


@app.get("/", include_in_schema=False)
def read_root():
    return {"message": "My AI Mentor AI Service is running"}