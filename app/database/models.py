"""Модели хранения продаж, прогнозов и рекомендаций."""

from datetime import date, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import Date, DateTime, JSON, Numeric, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Общие метаданные таблиц проекта."""


class SalesRecord(Base):
    """Запись о продажах за день по товарной категории."""

    __tablename__ = "sales_records"

    id: Mapped[int] = mapped_column(primary_key=True)
    date: Mapped[date] = mapped_column(Date, index=True)
    revenue: Mapped[Decimal] = mapped_column(Numeric(14, 2))
    units_sold: Mapped[int]
    marketing_spend: Mapped[Decimal] = mapped_column(Numeric(14, 2))
    discount_pct: Mapped[Decimal] = mapped_column(Numeric(5, 2))
    promo_active: Mapped[bool]
    category: Mapped[str] = mapped_column(String(100))


class ForecastLog(Base):
    """Параметры и результат оценки прогноза."""

    __tablename__ = "forecast_logs"

    id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    horizon_days: Mapped[int]
    mape_score: Mapped[float | None]
    model_params_json: Mapped[dict[str, Any]] = mapped_column(JSON)


class AIRecommendation(Base):
    """Рекомендация для заданного сценария продаж."""

    __tablename__ = "ai_recommendations"

    id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    scenario_summary: Mapped[str] = mapped_column(Text)
    recommendation_text: Mapped[str] = mapped_column(Text)
