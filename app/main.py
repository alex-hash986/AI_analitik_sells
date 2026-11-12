"""Команды работы с продажами, прогнозом и сценариями."""

import argparse
from pathlib import Path

from app.database.connection import init_db
from app.ml.data_processor import (
    DEFAULT_SAMPLE_PATH, SAMPLE_DAYS, generate_sales_data, import_sales_csv,
)
from app.ml.data_processor import load_sales_history as _load_history


def _format_pct(value: float | None) -> str:
    """Форматирует процент или обозначает отсутствие оценки."""
    return f"{value:.2f}%" if value is not None else "н/д"


def _run_forecast(days: int, category: str | None) -> None:
    """Строит прогноз и выводит метрики отложенного периода."""
    from app.ml.forecaster import SalesForecaster

    result = SalesForecaster().train_and_forecast(_load_history(), days, category)
    metrics = result["metrics"]
    print(f"Прогноз на {days} дней")
    print(f"MAPE: {_format_pct(metrics['mape'])}")
    print(f"MAE: {metrics['mae']:,.2f} руб.")
    print(f"RMSE: {metrics['rmse']:,.2f} руб.")
    print(result["seasonality_summary"])


def _run_simulation(marketing_change: float, discount: float, promo_ratio: float | None) -> None:
    """Обучает сценарную модель и выводит финансовое сравнение на 30 дней."""
    from app.ml.forecaster import SalesForecaster
    from app.ml.simulator import ScenarioSimulator

    history = _load_history()
    simulator = ScenarioSimulator().fit(history)
    result = SalesForecaster().train_and_forecast(history)
    ratio = simulator.promo_days_ratio if promo_ratio is None else promo_ratio
    comparison = simulator.simulate_scenario(
        result["forecast"], simulator.avg_marketing_spend, simulator.avg_discount_pct,
        marketing_change, discount, ratio,
    )
    baseline, scenario = comparison["baseline"], comparison["scenario"]
    print(f"Сценарий на 30 дней · реклама {marketing_change:+g}% · скидка {discount:g}% · акции {ratio:.1%} дней")
    scores = ", ".join(f"{name} {score:.3f}" for name, score in simulator.cv_scores.items())
    print(f"Модель: {simulator.model_name} (R² по временной кросс-валидации: {scores})")
    print(f"{'Показатель':<26} {'Baseline':>16} {'What-If':>16}")
    for label, key in (
        ("Выручка, руб.", "revenue"),
        ("Маркетинг, руб.", "marketing_spend"),
        ("Условная прибыль, руб.", "gross_profit"),
    ):
        print(f"{label:<26} {baseline[key]:>16,.2f} {scenario[key]:>16,.2f}")
    print(f"{'ROMI':<26} {_format_pct(baseline['romi_pct']):>16} {_format_pct(scenario['romi_pct']):>16}")
    print(f"Дельта выручки: {comparison['revenue_delta']:+,.2f} руб. ({_format_pct(comparison['revenue_delta_pct'])})")


def _run_advice(days: int) -> None:
    """Передаёт агрегированную аналитику и прогноз AI-советнику."""
    from app.ai.advisor import AISalesAdvisor, build_advice_summaries
    from app.ml.forecaster import SalesForecaster, aggregate_daily

    history = _load_history()
    forecaster = SalesForecaster()
    analysis = forecaster.analyze_historical(history)
    forecast = forecaster.train_and_forecast(history, days)
    analytics_summary, forecast_summary = build_advice_summaries(
        analysis, forecast, aggregate_daily(history)
    )
    advice = AISalesAdvisor().generate_recommendations(analytics_summary, forecast_summary)
    source = "OpenAI" if advice["source"] == "openai" else "локальные правила"
    print(f"Источник рекомендаций: {source}")
    print(advice["recommendation_text"])


def main() -> None:
    """Запускает выбранную команду приложения."""
    parser = argparse.ArgumentParser(description="Предиктивная аналитика продаж")
    commands = parser.add_subparsers(dest="command")
    commands.add_parser("init-db", help="Создать таблицы PostgreSQL")
    generator = commands.add_parser("generate", help=f"Сгенерировать CSV за {SAMPLE_DAYS} дней")
    generator.add_argument("--output", type=Path, default=DEFAULT_SAMPLE_PATH)
    generator.add_argument("--seed", type=int, default=42)
    importer = commands.add_parser("import-csv", help="Загрузить CSV в PostgreSQL")
    importer.add_argument("path", type=Path)
    forecast = commands.add_parser("forecast", help="Спрогнозировать выручку")
    forecast.add_argument("--days", type=int, default=30)
    forecast.add_argument("--category", help="Спрогнозировать одну категорию")
    simulation = commands.add_parser("simulate", help="Сравнить сценарий с базовым прогнозом")
    simulation.add_argument("--mkt-change", type=float, required=True, help="Изменение бюджета в процентах")
    simulation.add_argument("--discount", type=float, required=True, help="Новая скидка в процентах")
    simulation.add_argument("--promo-ratio", type=float, help="Доля акционных дней от 0 до 1")
    advisor = commands.add_parser("advise", help="Подготовить рекомендации по продажам")
    advisor.add_argument("--days", type=int, default=30)
    args = parser.parse_args()

    if args.command == "generate":
        sales = generate_sales_data(output_path=args.output, seed=args.seed)
        print(f"Сохранено строк: {len(sales)}. Файл: {args.output}")
    elif args.command == "import-csv":
        count = import_sales_csv(args.path)
        print(f"Импортировано строк: {count}")
    elif args.command == "forecast":
        _run_forecast(args.days, args.category)
    elif args.command == "simulate":
        _run_simulation(args.mkt_change, args.discount, args.promo_ratio)
    elif args.command == "advise":
        _run_advice(args.days)
    else:
        init_db()
        print("Таблицы PostgreSQL готовы.")


if __name__ == "__main__":
    main()
