"""Историческая аналитика, прогноз Prophet и журналирование запусков."""

from typing import TypedDict

import numpy as np
import pandas as pd
from prophet import Prophet
from sqlalchemy.orm import Session

from app.database.connection import get_engine
from app.database.models import ForecastLog

REGRESSORS = ("marketing_spend", "discount_pct")
NUMERIC_COLUMNS = ("revenue", *REGRESSORS, "units_sold")
WEEKDAYS = ("понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье")
MONTHS = (
    "январь", "февраль", "март", "апрель", "май", "июнь",
    "июль", "август", "сентябрь", "октябрь", "ноябрь", "декабрь",
)


class Seasonality(TypedDict):
    """Календарные пики, минимумы и их размах в процентах средней выручки."""

    peak: str
    trough: str
    strength_pct: float


class HistoricalAnalysis(TypedDict):
    """Тренд, сезонность, корреляции и даты аномалий."""

    growth_pct: float | None
    monthly_growth_pct: float | None
    weekly: Seasonality
    yearly: Seasonality
    correlations: pd.DataFrame
    anomaly_dates: list[str]


class ForecastMetrics(TypedDict):
    """Ошибки прогноза на отложенной выборке."""

    mape: float | None
    mae: float
    rmse: float


class ForecastResult(TypedDict):
    """Прогноз будущих дней, ошибки и резюме сезонности."""

    forecast: pd.DataFrame
    metrics: ForecastMetrics
    seasonality_summary: str


def aggregate_daily(df: pd.DataFrame, category: str | None = None) -> pd.DataFrame:
    """Суммирует выручку, продажи и расходы, усредняет скидку за день.

    День считается акционным, если акция была хотя бы в одной категории.
    Даты без записей не заполняются нулевой выручкой.
    """
    required = {"date", *NUMERIC_COLUMNS}
    if category is not None:
        required.add("category")
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"В истории нет колонок: {', '.join(sorted(missing))}.")

    sales = (df if category is None else df.loc[df["category"].eq(category)]).copy()
    if sales.empty:
        raise ValueError("Нет истории продаж для выбранной выборки.")

    sales["date"] = pd.to_datetime(sales["date"], errors="raise").dt.normalize()
    if sales["date"].isna().any():
        raise ValueError("В истории есть пустые даты.")
    sales[list(NUMERIC_COLUMNS)] = sales[list(NUMERIC_COLUMNS)].astype(float)
    if not np.isfinite(sales[list(NUMERIC_COLUMNS)].to_numpy()).all():
        raise ValueError("История содержит пропуски или бесконечные числовые значения.")
    daily = sales.groupby("date", sort=True)
    result = daily.agg(
        revenue=("revenue", "sum"),
        marketing_spend=("marketing_spend", "sum"),
        discount_pct=("discount_pct", "mean"),
        units_sold=("units_sold", "sum"),
    )
    if "promo_active" in sales:
        result["promo_active"] = daily["promo_active"].max()
    return result.reset_index()


def _seasonality(values: pd.Series, labels: tuple[str, ...], average: float) -> Seasonality:
    """Определяет размах календарных средних относительно средней выручки."""
    return {
        "peak": labels[int(values.idxmax())],
        "trough": labels[int(values.idxmin())],
        "strength_pct": float((values.max() - values.min()) / average * 100) if average else 0.0,
    }


class SalesForecaster:
    """Анализирует дневную выручку и прогнозирует её моделью Prophet."""

    def analyze_historical(self, df: pd.DataFrame) -> HistoricalAnalysis:
        """Оценивает линейный тренд, сезонность и IQR-выбросы остатков."""
        history = aggregate_daily(df)
        if len(history) < 2:
            raise ValueError("Для анализа нужны продажи хотя бы за два дня.")

        dates, revenue = history["date"], history["revenue"]
        elapsed = (dates - dates.iloc[0]).dt.days
        slope, start = np.polyfit(elapsed, revenue, 1)
        duration = int(elapsed.iloc[-1])
        end = start + slope * duration
        growth = float((end / start - 1) * 100) if start > 0 else None
        monthly = (
            float(((end / start) ** (30.4375 / duration) - 1) * 100)
            if start > 0 and end >= 0 else None
        )

        detrended = revenue - slope * elapsed
        weekday = dates.dt.dayofweek
        month = dates.dt.month - 1
        weekday_means = detrended.groupby(weekday).mean()
        without_weekly = detrended - detrended.groupby(weekday).transform("mean")
        monthly_means = without_weekly.groupby(month).mean()

        residual = without_weekly - without_weekly.groupby(month).transform("median")
        q1, q3 = residual.quantile([0.25, 0.75])
        spread = q3 - q1
        outliers = (residual < q1 - 1.5 * spread) | (residual > q3 + 1.5 * spread)
        return {
            "growth_pct": growth,
            "monthly_growth_pct": monthly,
            "weekly": _seasonality(weekday_means, WEEKDAYS, float(revenue.mean())),
            "yearly": _seasonality(monthly_means, MONTHS, float(revenue.mean())),
            "correlations": history[list(NUMERIC_COLUMNS)].corr(method="pearson"),
            "anomaly_dates": dates[outliers].dt.strftime("%Y-%m-%d").tolist(),
        }

    def train_and_forecast(
        self,
        df: pd.DataFrame,
        horizon_days: int = 30,
        category: str | None = None,
        *,
        future_regressors: pd.DataFrame | None = None,
    ) -> ForecastResult:
        """Обучает Prophet, проверяет на последних 14–30 днях и пишет лог.

        Будущие регрессоры передаются с колонкой ds и одним или двумя
        столбцами из REGRESSORS. Остальные значения берутся из средних
        по истории. При валидации используются средние только обучающей части.
        Выручка ограничивается квантилями отдельно для каждого обучения.
        Метрики сравниваются с исходной выручкой отложенного периода.
        MAPE рассчитывается для дней с ненулевой фактической выручкой.
        """
        if horizon_days < 1:
            raise ValueError("Горизонт прогноза должен быть положительным.")
        history = aggregate_daily(df, category)
        model_data = history.rename(columns={"date": "ds", "revenue": "y"})
        span = (model_data["ds"].iloc[-1] - model_data["ds"].iloc[0]).days + 1
        holdout_days = min(30, max(14, span // 5))
        cutoff = model_data["ds"].iloc[-1] - pd.Timedelta(days=holdout_days)
        train = model_data.loc[model_data["ds"] <= cutoff]
        test = model_data.loc[model_data["ds"] > cutoff]
        if len(train) < 14 or len(test) < 2:
            raise ValueError("Для прогноза нужно минимум 14 наблюдений до отложенного периода и два в нём.")

        means = {name: float(model_data[name].mean()) for name in REGRESSORS}
        future = pd.DataFrame({"ds": pd.date_range(
            start=model_data["ds"].iloc[-1] + pd.Timedelta(days=1), periods=horizon_days,
        )}).assign(**means)
        if future_regressors is not None:
            future = self._apply_future_regressors(future, future_regressors)

        validation = self._new_model(uncertainty_samples=0).fit(self._winsorize(train))
        validation_input = test[["ds"]].assign(
            **{name: float(train[name].mean()) for name in REGRESSORS}
        )
        validation_prediction = validation.predict(validation_input)["yhat"].clip(lower=0)
        actual = test["y"].to_numpy()
        error = validation_prediction.to_numpy() - actual
        nonzero = actual != 0
        metrics: ForecastMetrics = {
            "mape": float(np.mean(np.abs(error[nonzero] / actual[nonzero])) * 100) if nonzero.any() else None,
            "mae": float(np.abs(error).mean()),
            "rmse": float(np.sqrt(np.square(error).mean())),
        }

        model = self._new_model(uncertainty_samples=1000).fit(self._winsorize(model_data))
        predicted = model.predict(future)
        columns = ["ds", "yhat", "yhat_lower", "yhat_upper", "trend", "weekly", "yearly"]
        forecast = predicted.loc[:, columns].copy()
        bounds = ["yhat", "yhat_lower", "yhat_upper"]
        forecast[bounds] = forecast[bounds].clip(lower=0)

        analysis = self.analyze_historical(history)
        weekly, yearly = analysis["weekly"], analysis["yearly"]
        summary = (
            f"Пик недели — {weekly['peak']}, минимум — {weekly['trough']} "
            f"(размах {weekly['strength_pct']:.1f}%). "
            f"Пик по месяцам — {yearly['peak']}, минимум — {yearly['trough']} "
            f"(размах {yearly['strength_pct']:.1f}%)."
        )
        if span < 365:
            summary += " Годовая сезонность оценена по неполному календарному циклу."

        with Session(get_engine()) as session, session.begin():
            session.add(ForecastLog(
                horizon_days=horizon_days,
                mape_score=metrics["mape"],
                model_params_json={
                    "model": "Prophet",
                    "category": category,
                    "weekly_seasonality": True,
                    "yearly_seasonality": True,
                    "seasonality_mode": model.seasonality_mode,
                    "changepoint_prior_scale": model.changepoint_prior_scale,
                    "interval_width": model.interval_width,
                    "regressors": list(REGRESSORS),
                    "historical_regressor_means": means,
                    "overridden_regressors": [
                        name for name in REGRESSORS
                        if future_regressors is not None and name in future_regressors
                    ],
                    "holdout_days": holdout_days,
                    "training_observations": len(train),
                    "holdout_observations": len(test),
                    "winsorization_quantiles": [0.02, 0.98],
                    "validation_regressors": "train_means",
                    "metrics": dict(metrics),
                },
            ))
        return {"forecast": forecast, "metrics": metrics, "seasonality_summary": summary}

    @staticmethod
    def _winsorize(training_data: pd.DataFrame) -> pd.DataFrame:
        """Ограничивает целевую выручку по квантилям обучающей выборки."""
        lower, upper = training_data["y"].quantile([0.02, 0.98])
        return training_data.assign(y=training_data["y"].clip(lower, upper))

    @staticmethod
    def _new_model(uncertainty_samples: int) -> Prophet:
        """Создаёт модель с недельной и годовой сезонностью и регрессорами."""
        model = Prophet(
            weekly_seasonality=True,
            yearly_seasonality=True,
            daily_seasonality=False,
            uncertainty_samples=uncertainty_samples,
        )
        for name in REGRESSORS:
            model.add_regressor(name)
        return model

    @staticmethod
    def _apply_future_regressors(future: pd.DataFrame, overrides: pd.DataFrame) -> pd.DataFrame:
        """Сопоставляет пользовательские значения с датами будущего горизонта."""
        names = [name for name in REGRESSORS if name in overrides]
        if "ds" not in overrides or not names:
            raise ValueError("Будущие регрессоры требуют ds и marketing_spend или discount_pct.")
        provided = overrides.loc[:, ["ds", *names]].copy()
        provided["ds"] = pd.to_datetime(provided["ds"], errors="raise").dt.normalize()
        if provided["ds"].duplicated().any() or set(provided["ds"]) != set(future["ds"]):
            raise ValueError("Будущие регрессоры должны покрывать каждую дату горизонта ровно один раз.")
        provided = provided.set_index("ds").loc[future["ds"]]
        for name in names:
            values = pd.to_numeric(provided[name], errors="raise").to_numpy(dtype=float)
            if not np.isfinite(values).all() or (values < 0).any():
                raise ValueError(f"Будущий регрессор {name} должен содержать конечные неотрицательные значения.")
            if name == "discount_pct" and (values > 100).any():
                raise ValueError("Скидка должна быть в пределах от 0 до 100%.")
            future[name] = values
        return future
