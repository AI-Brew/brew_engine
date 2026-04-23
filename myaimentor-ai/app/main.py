from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.db import Base, engine, get_db
from app import models  # noqa: F401
from app.routers import knowledge


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(title="My AI Mentor Studio - AI Service", lifespan=lifespan)

app.include_router(knowledge.router)


@app.get("/")
def read_root():
    return {"message": "My AI Mentor AI Service is running"}


@app.get("/health")
def health_check(db: Session = Depends(get_db)):
    result = {"status": "ok", "db": "unknown", "pgvector": "unknown"}
    try:
        db.execute(text("SELECT 1"))
        result["db"] = "ok"
    except Exception as e:
        result["db"] = f"error: {str(e)}"
        result["status"] = "degraded"

    try:
        row = db.execute(
            text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
        ).fetchone()
        result["pgvector"] = row[0] if row else "not installed"
    except Exception as e:
        result["pgvector"] = f"error: {str(e)}"
        result["status"] = "degraded"

    return result