"""환경별 설정 로더.

APP_ENV 환경변수로 .env.<env> 파일을 선택한다.
- local / dev / prod

LLM provider는 봇마다 독립적으로 지정 가능하므로
OpenAI / Gemini 키와 모델명을 각각 별도로 관리한다.
"""

import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

APP_ENV = os.getenv("APP_ENV", "local").lower()
_ALLOWED = {"local", "dev", "prod"}
if APP_ENV not in _ALLOWED:
    raise ValueError(f"Invalid APP_ENV='{APP_ENV}'. Must be one of {_ALLOWED}.")

ENV_FILE = Path(__file__).resolve().parent.parent / f".env.{APP_ENV}"


class Settings(BaseSettings):
    # DB
    database_url: str
    mongo_url: str

    # Provider 기본값
    default_llm_provider: str = "gemini"

    # OpenAI
    openai_api_key: str = ""
    openai_embedding_model: str = "text-embedding-3-small"
    openai_chat_model: str = "gpt-4o-mini"

    # Gemini
    gemini_api_key: str = ""
    gemini_embedding_model: str = "text-embedding-004"
    gemini_chat_model: str = "gemini-2.0-flash"

    # 더미 모드 (키 없이 테스트)
    use_dummy_embedding: bool = False

    # 현재 환경 참조용
    app_env: str = APP_ENV

    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE),
        env_file_encoding="utf-8-sig",
        extra="ignore",
    )


settings = Settings()

print(f"[config] APP_ENV={settings.app_env}, env_file={ENV_FILE.name}")