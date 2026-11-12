"""Настройки PostgreSQL и создание таблиц приложения."""

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import URL, create_engine
from sqlalchemy.engine import Engine

from app.database.models import Base


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Создаёт движок на основе .env и переменных окружения."""
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    password = os.getenv("POSTGRES_PASSWORD")
    if not password:
        raise ValueError("Задайте POSTGRES_PASSWORD в .env или в окружении.")

    url = URL.create(
        drivername="postgresql+psycopg2",
        username=os.getenv("POSTGRES_USER", "sales_analytics"),
        password=password,
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        database=os.getenv("POSTGRES_DB", "sales_analytics"),
    )
    return create_engine(url, pool_pre_ping=True)


def init_db() -> None:
    """Создаёт отсутствующие таблицы при старте приложения."""
    Base.metadata.create_all(get_engine())
