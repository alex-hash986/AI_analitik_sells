"""Сценарный анализ выручки, затрат и окупаемости маркетинга."""

from typing import TypedDict

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.ml.data_processor import MARKETING_RESPONSE_SCALE
from app.ml.forecaster import aggregate_daily


class ScenarioTotals(TypedDict):
    """Суммы по горизонту и окупаемость маркетинга в процентах."""

    revenue: float
    marketing_spend: float
    gross_profit: float
    romi_pct: float | None


class ScenarioComparison(TypedDict):
    """Финансовые показатели двух сценариев и изменение выручки."""

    baseline: ScenarioTotals
    scenario: ScenarioTotals
    revenue_delta: float
    revenue_delta_pct: float | None


class ScenarioSimulator:
    """Выбирает регрессию по R² на последовательных временных выборках."""

    def __init__(self) -> None:
        """Создаёт симулятор, готовый к обучению."""
        self._model: Pipeline | None = None
        self.cv_scores: dict[str, float] = {}
        self.model_name: str | None = None
        self.avg_marketing_spend = 0.0
        self.avg_discount_pct = 0.0
        self.promo_days_ratio = 0.0

    def fit(self, df: pd.DataFrame) -> "ScenarioSimulator":
        """Обучает лучшую из Ridge и RandomForest на дневных продажах."""
        history = aggregate_daily(df)
        if "promo_active" not in history or history["promo_active"].isna().any():
            raise ValueError("Для симуляции нужен признак promo_active за каждый день.")
        if len(history) < 14:
            raise ValueError("Для симуляции нужно минимум 14 дней с продажами.")

        features = self._features(
            history["date"], history["marketing_spend"],
            history["discount_pct"], history["promo_active"].astype(int),
        )
        target = history["revenue"]
        preprocessor = ColumnTransformer([
            ("numeric", StandardScaler(), ["log_marketing", "discount_pct", "promo_active"]),
            ("calendar", OneHotEncoder(handle_unknown="ignore"), ["weekday", "month"]),
        ])
        candidates = {
            "Ridge": Ridge(alpha=1.0),
            "RandomForest": RandomForestRegressor(
                n_estimators=150, min_samples_leaf=5, random_state=42,
            ),
        }
        splitter = TimeSeriesSplit(n_splits=3)
        scores: dict[str, float] = {}
        for name, estimator in candidates.items():
            pipeline = make_pipeline(preprocessor, estimator)
            scores[name] = float(cross_val_score(
                pipeline, features, target, cv=splitter, scoring="r2", error_score="raise",
            ).mean())

        model_name = max(scores, key=scores.__getitem__)
        self._model = make_pipeline(preprocessor, candidates[model_name]).fit(features, target)
        self.model_name = model_name
        self.cv_scores = scores
        self.avg_marketing_spend = float(history["marketing_spend"].mean())
        self.avg_discount_pct = float(history["discount_pct"].mean())
        self.promo_days_ratio = float(history["promo_active"].mean())
        return self

    def simulate_scenario(
        self,
        base_forecast_df: pd.DataFrame,
        avg_marketing_spend: float,
        avg_discount_pct: float,
        marketing_change_pct: float,
        discount_pct_new: float,
        promo_days_ratio: float,
    ) -> ScenarioComparison:
        """Сравнивает прогноз Prophet с прогнозом при изменении условий.

        Доли акционных дней задаются от 0 до 1. Для каждой даты
        модель усредняет прогнозы с акцией и без акции по заданной доле.
        Условная прибыль равна выручке минус расчётная скидка и реклама.
        ROMI сравнивает вклад в прибыль с тем же сценарием без рекламы.
        """
        model = self._model
        if model is None:
            raise ValueError("Перед симуляцией вызовите fit с историей продаж.")
        numbers = [
            avg_marketing_spend, avg_discount_pct, marketing_change_pct,
            discount_pct_new, promo_days_ratio,
        ]
        if not np.isfinite(numbers).all():
            raise ValueError("Параметры сценария должны быть конечными числами.")
        if avg_marketing_spend < 0 or marketing_change_pct < -100:
            raise ValueError("Маркетинговый бюджет должен быть неотрицательным.")
        if not (0 <= avg_discount_pct <= 100 and 0 <= discount_pct_new <= 100):
            raise ValueError("Скидки должны быть в пределах от 0 до 100%.")
        if not 0 <= promo_days_ratio <= 1:
            raise ValueError("Доля акционных дней должна быть в пределах от 0 до 1.")
        if base_forecast_df.empty or not {"ds", "yhat"}.issubset(base_forecast_df.columns):
            raise ValueError("Для сценария нужен непустой прогноз с колонками ds и yhat.")

        dates = pd.to_datetime(base_forecast_df["ds"], errors="raise").dt.normalize()
        revenue = base_forecast_df["yhat"].to_numpy(dtype=float)
        if dates.isna().any() or dates.duplicated().any() or not np.isfinite(revenue).all():
            raise ValueError("Прогноз должен содержать уникальные даты и конечную выручку.")

        baseline_revenue = np.maximum(revenue, 0)
        target_spend = avg_marketing_spend * (1 + marketing_change_pct / 100)
        if not np.isfinite(target_spend):
            raise ValueError("Целевой бюджет выходит за пределы числового диапазона.")

        model_baseline = self._predict(model, dates, avg_marketing_spend, avg_discount_pct, self.promo_days_ratio)
        model_target = self._predict(model, dates, target_spend, discount_pct_new, promo_days_ratio)
        model_baseline_no_ads = self._predict(model, dates, 0, avg_discount_pct, self.promo_days_ratio)
        model_target_no_ads = self._predict(model, dates, 0, discount_pct_new, promo_days_ratio)

        baseline_no_ads = np.maximum(baseline_revenue + (model_baseline_no_ads - model_baseline), 0)
        target_revenue = np.maximum(baseline_revenue + (model_target - model_baseline), 0)
        target_no_ads = np.maximum(baseline_revenue + (model_target_no_ads - model_baseline), 0)
        baseline = self._totals(baseline_revenue, baseline_no_ads, avg_marketing_spend, avg_discount_pct)
        scenario = self._totals(target_revenue, target_no_ads, target_spend, discount_pct_new)
        delta = scenario["revenue"] - baseline["revenue"]
        return {
            "baseline": baseline,
            "scenario": scenario,
            "revenue_delta": delta,
            "revenue_delta_pct": delta / baseline["revenue"] * 100 if baseline["revenue"] else None,
        }

    def estimate_average_romi(self, daily: pd.DataFrame) -> float | None:
        """Усредняет модельный ROMI дней с положительными расходами на рекламу."""
        model = self._model
        paid = daily.loc[daily["marketing_spend"] > 0]
        if model is None or paid.empty:
            return None
        dates = paid["date"]
        spend = paid["marketing_spend"]
        discount = paid["discount_pct"]
        promo = paid["promo_active"].astype(int)
        estimated = model.predict(self._features(dates, spend, discount, promo))
        no_ads = model.predict(self._features(dates, 0, discount, promo))
        observed = paid["revenue"].to_numpy()
        counterfactual = np.maximum(observed + no_ads - estimated, 0)
        lift_after_discount = (observed - counterfactual) * (1 - discount / 100)
        return float((lift_after_discount / spend - 1).mean() * 100)

    @staticmethod
    def _features(
        dates: pd.Series,
        spend: pd.Series | float,
        discount: pd.Series | float,
        promo: pd.Series | int,
    ) -> pd.DataFrame:
        """Логарифмирует бюджет и выделяет календарные признаки."""
        return pd.DataFrame({
            "log_marketing": np.log1p(spend / MARKETING_RESPONSE_SCALE),
            "discount_pct": discount,
            "promo_active": promo,
            "weekday": dates.dt.dayofweek,
            "month": dates.dt.month,
        })

    @classmethod
    def _predict(
        cls, model: Pipeline, dates: pd.Series, spend: float, discount: float, ratio: float,
    ) -> np.ndarray:
        """Возвращает ожидаемую выручку при заданной доле акций."""
        no_promo = model.predict(cls._features(dates, spend, discount, 0))
        promo = model.predict(cls._features(dates, spend, discount, 1))
        return no_promo * (1 - ratio) + promo * ratio

    @staticmethod
    def _totals(
        revenue: np.ndarray, no_ads: np.ndarray, spend: float, discount: float,
    ) -> ScenarioTotals:
        """Рассчитывает прибыль после скидок и ROMI по контрфакту без рекламы."""
        revenue_total = float(revenue.sum())
        marketing_total = spend * len(revenue)
        margin_factor = 1 - discount / 100
        incremental_margin = (revenue_total - float(no_ads.sum())) * margin_factor
        return {
            "revenue": revenue_total,
            "marketing_spend": marketing_total,
            "gross_profit": revenue_total * margin_factor - marketing_total,
            "romi_pct": (incremental_margin / marketing_total - 1) * 100 if marketing_total else None,
        }
