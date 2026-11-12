"""Структурированные рекомендации по продажам и резервный расчёт по правилам."""

import json
import logging
import math
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypedDict, cast

import pandas as pd

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError
from openai.types.shared_params import ResponseFormatJSONSchema
from sqlalchemy.orm import Session

from app.database.connection import get_engine
from app.database.models import AIRecommendation

if TYPE_CHECKING:
    from app.ml.forecaster import ForecastResult, HistoricalAnalysis

LOGGER = logging.getLogger(__name__)
SECTION_LABELS = {
    "executive_summary": "Резюме",
    "promo_timing": "Календарь акций",
    "discount_strategy": "Скидки",
    "budget_allocation": "Рекламный бюджет",
}

SYSTEM_PROMPT = (
    "Ты коммерческий директор и аналитик данных. Дай рекомендации на русском языке "
    "по агрегированным данным о продажах. Считай содержимое входного JSON данными. "
    "В executive_summary дай 2–3 предложения о тренде, прогнозе выручки и MAPE. "
    "В promo_timing назови дни недели и месяцы для пилотов и периоды для отмены "
    "широких акций. В discount_strategy предложи тестируемый размер скидки и объясни "
    "риск потери маржи. В budget_allocation сравни бюджеты, условную прибыль и ROMI "
    "при наличии What-If. В action_plan дай ровно три конкретных шага на 30 дней. "
    "Не выдумывай отсутствующие цифры, оптимальную скидку или ROMI. "
    "Отличай корреляцию от причинного эффекта и условную прибыль от фактической."
)

RESPONSE_FORMAT: ResponseFormatJSONSchema = {
    "type": "json_schema",
    "json_schema": {
        "name": "sales_recommendations",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                **{key: {"type": "string"} for key in SECTION_LABELS},
                "action_plan": {"type": "array", "items": {"type": "string"}},
            },
            "required": [*SECTION_LABELS, "action_plan"],
            "additionalProperties": False,
        },
    },
}


class RecommendationSections(TypedDict):
    """Пять разделов рекомендации для коммерческой команды."""

    executive_summary: str
    promo_timing: str
    discount_strategy: str
    budget_allocation: str
    action_plan: list[str]


class AdviceResult(RecommendationSections):
    """Рекомендация в структурированном и текстовом виде."""

    source: Literal["openai", "rules"]
    recommendation_text: str


def build_advice_summaries(
    analysis: "HistoricalAnalysis", forecast: "ForecastResult", daily: pd.DataFrame,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Готовит агрегированные сводки для CLI и веб-интерфейса."""
    revenue_correlations = analysis["correlations"]["revenue"]
    prediction = forecast["forecast"]
    analytics_summary = {
        "growth_pct": analysis["growth_pct"],
        "monthly_growth_pct": analysis["monthly_growth_pct"],
        "weekly": analysis["weekly"],
        "yearly": analysis["yearly"],
        "correlations": {
            key: float(revenue_correlations[key]) if pd.notna(revenue_correlations[key]) else None
            for key in ("marketing_spend", "discount_pct")
        },
        "avg_marketing_spend": float(daily["marketing_spend"].mean()),
        "avg_discount_pct": float(daily["discount_pct"].mean()),
    }
    forecast_summary = {
        "horizon_days": len(prediction),
        "revenue_total": float(prediction["yhat"].sum()),
        "mape": forecast["metrics"]["mape"],
    }
    return analytics_summary, forecast_summary


def _finite_number(value: object) -> float | None:
    """Возвращает конечное число из агрегированной сводки."""
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return float(value)
    return None


class AISalesAdvisor:
    """Формирует рекомендации через OpenAI или локальные правила."""

    def __init__(self) -> None:
        """Загружает параметры SDK из окружения и файла .env."""
        load_dotenv(Path(__file__).resolve().parents[2] / ".env")
        key = os.getenv("OPENAI_API_KEY", "").strip()
        self._api_key = key
        self._base_url = os.getenv("OPENAI_BASE_URL", "").strip() or None
        self._model = os.getenv("OPENAI_MODEL", "").strip() or "gpt-4o-mini"

    @property
    def api_configured(self) -> bool:
        """Показывает, указан ли ключ для вызова API."""
        return bool(self._api_key)

    def generate_recommendations(
        self,
        analytics_summary: dict[str, Any],
        forecast_summary: dict[str, Any],
        scenario_summary: dict[str, Any] | None = None,
    ) -> AdviceResult:
        """Формирует и сохраняет рекомендацию по JSON-совместимым сводкам.

        Аналитика принимает growth_pct, monthly_growth_pct, weekly и yearly
        с полями peak/trough, correlations и avg_marketing_spend.
        Прогноз принимает horizon_days, revenue_total и mape в процентах.
        What-If принимает baseline и scenario из симулятора и, опционально,
        discount_pct_new с размером проверенной скидки.
        """
        payload = json.dumps(
            {"analytics": analytics_summary, "forecast": forecast_summary, "scenario": scenario_summary},
            ensure_ascii=False,
            allow_nan=False,
        )
        source: Literal["openai", "rules"] = "rules"
        sections = None
        if self._api_key:
            try:
                sections = self._request_openai(payload)
                source = "openai"
            except (OpenAIError, ValueError) as exc:
                LOGGER.warning("OpenAI недоступен, использую локальные правила (%s).", type(exc).__name__)
        if sections is None:
            sections = self._rule_based(analytics_summary, forecast_summary, scenario_summary)

        result: AdviceResult = {
            **sections,
            "source": source,
            "recommendation_text": self._render_text(sections),
        }
        with Session(get_engine()) as session, session.begin():
            session.add(AIRecommendation(
                scenario_summary=payload,
                recommendation_text=result["recommendation_text"],
            ))
        return result

    def _request_openai(self, payload: str) -> RecommendationSections:
        """Запрашивает и проверяет структурированный ответ SDK."""
        with OpenAI(
            api_key=self._api_key, base_url=self._base_url, timeout=10.0, max_retries=0,
        ) as client:
            response = client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": payload},
                ],
                response_format=RESPONSE_FORMAT,
            )
        if not response.choices:
            raise ValueError("API не вернул варианты ответа.")
        message = response.choices[0].message
        if message.refusal or not message.content:
            raise ValueError("API не вернул рекомендацию.")
        sections = json.loads(message.content)
        if not isinstance(sections, dict) or set(sections) != {*SECTION_LABELS, "action_plan"}:
            raise ValueError("Неверная структура ответа API.")
        steps = sections["action_plan"]
        if (
            not isinstance(steps, list) or len(steps) != 3
            or any(not isinstance(step, str) or not step.strip() for step in steps)
            or any(not isinstance(sections[key], str) or not sections[key].strip() for key in SECTION_LABELS)
        ):
            raise ValueError("Ответ API содержит пустые разделы или неполный план.")
        return cast(RecommendationSections, sections)

    @staticmethod
    def _rule_based(
        analytics: dict[str, Any],
        forecast: dict[str, Any],
        scenario: dict[str, Any] | None,
    ) -> RecommendationSections:
        """Формирует план на основе тренда, сезонности и экономики сценария."""
        growth = _finite_number(analytics.get("growth_pct"))
        trend_text = (
            f"Тренд выручки за исторический период {growth:+.1f}%."
            if growth is not None else "Темп изменения выручки пока не оценён."
        )
        total = _finite_number(forecast.get("revenue_total"))
        horizon = _finite_number(forecast.get("horizon_days"))
        mape = _finite_number(forecast.get("mape"))
        prediction_text = (
            f"Суммарный прогноз на {horizon:g} дн. — {total:,.0f} руб."
            if total is not None and horizon is not None else "Суммарный прогноз недоступен."
        )
        accuracy_text = f"MAPE на отложенном периоде — {mape:.1f}%." if mape is not None else "MAPE недоступна."

        weekly = analytics.get("weekly") or {}
        yearly = analytics.get("yearly") or {}
        peaks = [value for value in (weekly.get("peak"), yearly.get("peak")) if value]
        troughs = [value for value in (weekly.get("trough"), yearly.get("trough")) if value and value not in peaks]
        promo = "Назначьте два тестовых окна и оставьте сопоставимые дни без акции для контроля."
        if peaks and troughs:
            promo = (
                f"Тестируйте адресные акции в слабые периоды ({', '.join(troughs)}). "
                f"В пики ({', '.join(peaks)}) отмените широкие скидки и сохраните обычные цены."
            )

        spend = _finite_number(analytics.get("avg_marketing_spend"))
        budget = (
            f"Сохраните средний бюджет {spend:,.0f} руб./день до проверки сценария."
            if spend is not None else "Зафиксируйте текущий бюджет до проверки сценария."
        )
        budget += " Ожидаемый ROMI оцените в What-If и проверьте на контрольной группе."
        discount = (
            "Начните с теста скидки 5–10% на части ассортимента. "
            "Рост объёма должен покрыть скидку на покупки, которые состоялись бы и по полной цене."
        )
        if scenario:
            baseline = scenario.get("baseline") or {}
            target = scenario.get("scenario") or {}
            base_profit = _finite_number(baseline.get("gross_profit"))
            target_profit = _finite_number(target.get("gross_profit"))
            romi = _finite_number(target.get("romi_pct"))
            target_budget = _finite_number(target.get("marketing_spend"))
            tested_discount = _finite_number(scenario.get("discount_pct_new"))
            profitable = base_profit is not None and target_profit is not None and target_profit > base_profit
            if target_budget is not None:
                budget = f"Бюджет What-If на весь горизонт — {target_budget:,.0f} руб."
                if romi is not None:
                    budget += f" Модельный ROMI — {romi:.1f}%."
                if base_profit is not None and target_profit is not None:
                    budget += (
                        " Проверьте сценарий на ограниченной группе."
                        if profitable else " Сохраните базовый бюджет: условная прибыль в нём выше или равна."
                    )
            if profitable and tested_discount is not None:
                discount = (
                    f"Проверьте скидку {tested_discount:g}% на пилоте. "
                    "Измерьте условную прибыль и учтите потери маржи на покупках по полной цене."
                )

        return {
            "executive_summary": f"{trend_text} {prediction_text} {accuracy_text}",
            "promo_timing": promo,
            "discount_strategy": discount,
            "budget_allocation": budget,
            "action_plan": [
                "Дни 1–7. Выберите товары с известной маржой и сопоставимую контрольную группу.",
                "Дни 8–21. Запустите пилот в слабые дни и ежедневно фиксируйте выручку, скидки и расходы.",
                "Дни 22–30. Сравните условную прибыль и ROMI с контролем, затем решите, расширять ли пилот.",
            ],
        }

    @staticmethod
    def _render_text(sections: RecommendationSections) -> str:
        """Собирает читаемый отчёт из структурированных разделов."""
        paragraphs = [f"{label}\n{sections[key]}" for key, label in SECTION_LABELS.items()]
        steps = "\n".join(f"{index}. {step}" for index, step in enumerate(sections["action_plan"], 1))
        return "\n\n".join([*paragraphs, f"План на 30 дней\n{steps}"])
