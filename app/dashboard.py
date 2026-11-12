"""Интерактивная аналитика продаж, прогнозирование и сценарии в Streamlit."""

from datetime import datetime
from hashlib import sha256
from io import BytesIO
import logging
from pathlib import Path
import sys

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
import streamlit as st

# При запуске через Streamlit путь скрипта может быть единственным путём проекта.
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ai.advisor import AISalesAdvisor, AdviceResult, build_advice_summaries
from app.database.connection import get_engine
from app.database.models import AIRecommendation
from app.ml.data_processor import CSV_COLUMNS, DEFAULT_SAMPLE_PATH, import_sales_csv, load_sales_history
from app.ml.forecaster import ForecastResult, HistoricalAnalysis, MONTHS, WEEKDAYS, SalesForecaster, aggregate_daily
from app.ml.simulator import ScenarioComparison, ScenarioSimulator

TITLE = "AI-система предиктивной аналитики продаж (Магазин электроники)"
TAB_NAMES = (
    "Обзор и сезонность", "Прогноз продаж (ML)", "Сценарный симулятор (What-If)",
    "AI-советник", "Загрузка данных",
)
TEAL, BLUE, AMBER = "#007E80", "#365CC5", "#C25D20"
FACTOR_LABELS = {
    "revenue": "Выручка", "marketing_spend": "Реклама",
    "discount_pct": "Скидки", "units_sold": "Продажи, шт.",
}
WEEKDAY_LABELS = ("Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс")
MONTH_LABELS = ("Янв", "Фев", "Мар", "Апр", "Май", "Июн", "Июл", "Авг", "Сен", "Окт", "Ноя", "Дек")
LOGGER = logging.getLogger(__name__)


@st.cache_data(ttl=30, show_spinner=False)
def _history() -> pd.DataFrame:
    """Читает историю продаж и обновляет снимок каждые 30 секунд."""
    return load_sales_history()


@st.cache_data(ttl=30, show_spinner=False)
def _recommendation_history() -> list[tuple[datetime, str]]:
    """Возвращает пять последних отчётов из PostgreSQL."""
    query = select(AIRecommendation.created_at, AIRecommendation.recommendation_text).order_by(
        AIRecommendation.created_at.desc(), AIRecommendation.id.desc()
    ).limit(5)
    with get_engine().connect() as connection:
        return [(date, report) for date, report in connection.execute(query)]


@st.cache_data(max_entries=32, show_spinner=False)
def _forecast(snapshot: str, category: str | None, days: int, _sales: pd.DataFrame) -> ForecastResult:
    """Кэширует обучение Prophet и прогноз для снимка, категории и горизонта."""
    return SalesForecaster().train_and_forecast(_sales, days, category)


@st.cache_resource(max_entries=16, show_spinner=False)
def _simulator(snapshot: str, _sales: pd.DataFrame) -> ScenarioSimulator:
    """Обучает сценарную модель один раз для выбранной истории продаж."""
    return ScenarioSimulator().fit(_sales)


def _snapshot(sales: pd.DataFrame) -> str:
    """Учитывает все строки истории при расчёте ключа кэша."""
    return sha256(pd.util.hash_pandas_object(sales, index=False).to_numpy().tobytes()).hexdigest()


def _money(value: float, *, signed: bool = False) -> str:
    """Форматирует сумму в рублях для карточки."""
    return format(value, "+,.0f" if signed else ",.0f").replace(",", " ") + " ₽"


def _percent(value: float | None) -> str:
    """Форматирует проценты для карточки."""
    return "—" if value is None else f"{value:.1f}%".replace(".", ",")


def _plot(fig: go.Figure, *, key: str, height: int = 340) -> None:
    """Применяет единый вид и выводит интерактивный график."""
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=16, b=8),
        legend=dict(orientation="h", y=-0.20), separators=", ",
    )
    st.plotly_chart(fig, width="stretch", key=key, config={"displaylogo": False})


def _daily_line(daily: pd.DataFrame, name: str = "Факт") -> go.Scatter:
    """Создаёт линию фактической выручки."""
    return go.Scatter(
        x=daily["date"], y=daily["revenue"], name=name, mode="lines",
        line=dict(color=TEAL, width=2),
        hovertemplate="%{x|%d.%m.%Y}<br>%{y:,.0f} ₽",
    )


def _season_chart(
    daily: pd.DataFrame, period: pd.Series, labels: tuple[str, ...],
    full_labels: tuple[str, ...], peak: str, key: str,
) -> None:
    """Показывает среднюю дневную выручку по календарным периодам."""
    values = daily.groupby(period)["revenue"].mean().reindex(range(len(labels)))
    colors = [BLUE if label == peak else TEAL for label in full_labels]
    fig = go.Figure(go.Bar(
        x=labels, y=values, marker_color=colors,
        hovertemplate="%{x}<br>%{y:,.0f} ₽<extra></extra>",
    ))
    fig.update_yaxes(ticksuffix=" ₽")
    _plot(fig, key=key, height=270)


def _show_overview(
    daily: pd.DataFrame, analysis: HistoricalAnalysis | None, simulator: ScenarioSimulator | None,
) -> None:
    """Показывает KPI, аномалии, сезонность и корреляции."""
    revenue, spend = daily["revenue"], daily["marketing_spend"]
    romi = simulator.estimate_average_romi(daily) if simulator else None
    cards = st.columns(4)
    cards[0].metric("Общая выручка", _money(revenue.sum()), border=True)
    cards[1].metric("Средний дневной оборот", _money(revenue.mean()), border=True)
    cards[2].metric("Средний бюджет на маркетинг", _money(spend.mean()), border=True)
    cards[3].metric(
        "Средний ROMI", _percent(romi), border=True,
        help="Среднее по дням с рекламными расходами. Оценка модели относительно сценария без рекламы.",
    )
    st.subheader("Динамика выручки")
    figure = go.Figure(_daily_line(daily))
    if analysis:
        anomalies = daily[daily["date"].dt.strftime("%Y-%m-%d").isin(analysis["anomaly_dates"])]
        figure.add_scatter(
            x=anomalies["date"], y=anomalies["revenue"], mode="markers", name="Аномалии",
            marker=dict(color=AMBER, size=8),
            hovertemplate="%{x|%d.%m.%Y}<br>%{y:,.0f} ₽",
        )
    figure.update_yaxes(ticksuffix=" ₽")
    _plot(figure, key="history_chart", height=390)

    if analysis is None:
        st.info("Для анализа сезонности нужны продажи хотя бы за два дня.")
        return

    left, right = st.columns(2)
    with left:
        st.subheader("Недельная сезонность")
        _season_chart(
            daily, daily["date"].dt.day_of_week, WEEKDAY_LABELS, WEEKDAYS,
            analysis["weekly"]["peak"], "weekday_chart",
        )
    with right:
        st.subheader("Годовая сезонность")
        _season_chart(
            daily, daily["date"].dt.month - 1, MONTH_LABELS, MONTHS,
            analysis["yearly"]["peak"], "month_chart",
        )
    st.caption("Столбцы показывают среднюю выручку. Синим отмечены пики после учёта тренда.")
    st.subheader("Корреляции факторов")
    matrix = analysis["correlations"].rename(index=FACTOR_LABELS, columns=FACTOR_LABELS)
    heatmap = go.Figure(go.Heatmap(
        z=matrix.to_numpy(), x=matrix.columns, y=matrix.index,
        zmin=-1, zmax=1, colorscale=[[0, AMBER], [0.5, "#F3F6FC"], [1, BLUE]],
        text=matrix.map(lambda value: f"{value:.2f}" if pd.notna(value) else "—").to_numpy(),
        texttemplate="%{text}", hovertemplate="%{y} × %{x}<br>%{text}<extra></extra>",
    ))
    heatmap.update_yaxes(autorange="reversed")
    _plot(heatmap, key="correlation_chart", height=320)


def _show_forecast(daily: pd.DataFrame, result: ForecastResult) -> None:
    """Показывает качество, интервал и ежедневный прогноз для скачивания."""
    prediction, metrics = result["forecast"], result["metrics"]
    cards = st.columns(4)
    cards[0].metric("MAPE", _percent(metrics["mape"]), border=True)
    cards[1].metric("MAE", _money(metrics["mae"]), border=True)
    cards[2].metric("RMSE", _money(metrics["rmse"]), border=True)
    cards[3].metric("Выручка за период", _money(prediction["yhat"].sum()), border=True)
    st.caption(result["seasonality_summary"])

    recent = daily[daily["date"] >= daily["date"].max() - pd.Timedelta(days=119)]
    chart = go.Figure(_daily_line(recent))
    chart.add_scatter(
        x=prediction["ds"], y=prediction["yhat_lower"], mode="lines",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    )
    chart.add_scatter(
        x=prediction["ds"], y=prediction["yhat_upper"], mode="lines",
        line=dict(width=0), fill="tonexty", fillcolor="rgba(54, 92, 197, 0.18)",
        name="80% интервал", hoverinfo="skip",
    )
    chart.add_scatter(
        x=prediction["ds"], y=prediction["yhat"], mode="lines", name="Прогноз",
        line=dict(color=BLUE, width=3),
        customdata=prediction[["yhat_lower", "yhat_upper"]].to_numpy(),
        hovertemplate=(
            "%{x|%d.%m.%Y}<br>%{y:,.0f} ₽"
            "<br>Интервал: %{customdata[0]:,.0f}–%{customdata[1]:,.0f} ₽"
        ),
    )
    chart.update_yaxes(ticksuffix=" ₽")
    _plot(chart, key="forecast_chart", height=410)

    st.subheader("Прогноз по дням")
    table = prediction[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    table["ds"] = table["ds"].dt.strftime("%d.%m.%Y")
    st.dataframe(
        table, hide_index=True, width="stretch",
        column_config={
            "ds": "Дата",
            "yhat": st.column_config.NumberColumn("Прогноз, ₽", format="%.0f"),
            "yhat_lower": st.column_config.NumberColumn("Нижняя граница, ₽", format="%.0f"),
            "yhat_upper": st.column_config.NumberColumn("Верхняя граница, ₽", format="%.0f"),
        },
    )
    st.download_button(
        "Скачать прогноз (CSV)", prediction.to_csv(index=False, date_format="%Y-%m-%d").encode("utf-8-sig"),
        "sales_forecast.csv", "text/csv", on_click="ignore",
    )


def _scenario_controls() -> tuple[int, int, int]:
    """Читает параметры сценария из ползунков."""
    columns = st.columns(3)
    budget = columns[0].slider("Изменение рекламного бюджета", -50, 100, 20, format="%d%%")
    discount = columns[1].slider("Целевая скидка", 0, 30, 10, format="%d%%")
    promo = columns[2].slider("Доля акционных дней", 0, 100, 25, format="%d%%")
    return budget, discount, promo


def _show_scenario(result: ScenarioComparison, simulator: ScenarioSimulator) -> None:
    """Сравнивает базовый прогноз и сценарий по финансовым показателям."""
    baseline, scenario = result["baseline"], result["scenario"]
    labels = ("Выручка", "Маркетинг", "Условная прибыль")
    keys = ("revenue", "marketing_spend", "gross_profit")
    cards = st.columns(4)
    for card, label, key in zip(cards, labels, keys):
        base, target = baseline[key], scenario[key]
        card.metric(
            label, _money(target), _money(target - base, signed=True), border=True,
            delta_color="off" if key == "marketing_spend" else "normal",
        )
        card.caption(f"Baseline · {_money(base)}")
    base_romi, target_romi = baseline["romi_pct"], scenario["romi_pct"]
    delta = f"{target_romi - base_romi:+.1f} п. п." if base_romi is not None and target_romi is not None else None
    cards[3].metric("ROMI", _percent(target_romi), delta, border=True)
    cards[3].caption(f"Baseline · {_percent(base_romi)}")

    fig = make_subplots(rows=1, cols=2, column_widths=[0.77, 0.23], horizontal_spacing=0.15)
    for name, totals, color in (("Baseline", baseline, TEAL), ("What-If", scenario, BLUE)):
        fig.add_bar(
            x=labels, y=[totals[key] for key in keys], name=name, legendgroup=name,
            marker_color=color, hovertemplate="%{x}<br>%{y:,.0f} ₽", row=1, col=1,
        )
        fig.add_bar(
            x=["ROMI"], y=[totals["romi_pct"]], name=name, legendgroup=name, showlegend=False,
            marker_color=color, hovertemplate="%{y:.1f}%", row=1, col=2,
        )
    fig.update_layout(barmode="group")
    fig.update_yaxes(ticksuffix=" ₽", row=1, col=1)
    fig.update_yaxes(ticksuffix="%", row=1, col=2)
    _plot(fig, key="scenario_chart", height=370)
    st.caption(
        f"Модель · {simulator.model_name}   |   Изменение выручки · {_percent(result['revenue_delta_pct'])}. "
        "Условная прибыль учитывает скидку и расходы на рекламу. ROMI оценивается относительно нулевого бюджета."
    )


def _report_markdown(advice: AdviceResult, context: str) -> str:
    """Формирует Markdown с контекстом текущего расчёта."""
    steps = "\n".join(f"{i}. {step}" for i, step in enumerate(advice["action_plan"], start=1))
    return "\n\n".join((
        "# Управленческий отчёт", context,
        f"## Резюме\n\n{advice['executive_summary']}",
        f"## Календарь акций\n\n{advice['promo_timing']}",
        f"## Скидки\n\n{advice['discount_strategy']}",
        f"## Рекламный бюджет\n\n{advice['budget_allocation']}",
        f"## План на 30 дней\n\n{steps}",
    )) + "\n"


def _show_advice(advice: AdviceResult, context: str) -> None:
    """Показывает пять блоков отчёта и предлагает скачать Markdown."""
    st.caption(context)
    with st.container(border=True):
        st.subheader("Резюме")
        st.write(advice["executive_summary"])
    columns = st.columns(3)
    for column, title, content in (
        (columns[0], "Календарь акций", advice["promo_timing"]),
        (columns[1], "Скидки", advice["discount_strategy"]),
        (columns[2], "Рекламный бюджет", advice["budget_allocation"]),
    ):
        with column.container(border=True):
            st.markdown(f"**{title}**")
            st.write(content)
    with st.container(border=True):
        st.subheader("План на 30 дней")
        for i, step in enumerate(advice["action_plan"], start=1):
            st.write(f"{i}. {step}")
    st.download_button(
        "Скачать управленческий отчёт (.md)", _report_markdown(advice, context).encode(),
        "sales_management_report.md", "text/markdown", on_click="ignore",
    )


def _show_advice_history() -> None:
    """Отображает сохранённые в базе рекомендации."""
    st.subheader("Последние рекомендации")
    try:
        reports = _recommendation_history()
    except SQLAlchemyError:
        LOGGER.exception("Не удалось получить рекомендации.")
        st.warning("История рекомендаций временно недоступна.")
        return
    if not reports:
        st.caption("История пока пуста.")
    for created_at, report in reports:
        with st.expander(created_at.strftime("%d.%m.%Y · %H:%M %Z").strip()):
            st.text(report)


def _imported(message: str) -> None:
    """Сбрасывает снимок БД после импорта и сохраняет отчёт для следующего запуска."""
    _history.clear()
    st.session_state["import_report"] = message
    st.rerun()


def _show_upload(db_available: bool) -> None:
    """Показывает шаблон CSV и форму импорта."""
    st.subheader("Импорт продаж в PostgreSQL")
    st.markdown(
        "Загрузите CSV в UTF-8 с запятыми. Дата — `ГГГГ-ММ-ДД` или `ДД.ММ.ГГГГ`. "
        "Выручку и рекламу укажите в рублях, скидку в процентах от 0 до 100, "
        "признак акции как `true` или `false`. Количество продаж — целое число."
    )
    st.code(", ".join(CSV_COLUMNS), language=None)
    st.caption(
        "Пустые бюджет и скидка заполняются нулями. Пустой признак акции определяется по скидке. "
        "Импорт добавляет строки к истории."
    )
    template = ",".join(CSV_COLUMNS) + "\n2026-01-01,450000,220,68000,10,true,Электроника\n"
    st.download_button(
        "Скачать шаблон CSV", template.encode("utf-8-sig"), "sales_template.csv",
        "text/csv", on_click="ignore",
    )
    with st.form("sales_upload"):
        uploaded = st.file_uploader("Файл с продажами", type=["csv"])
        submitted = st.form_submit_button("Проверить и импортировать", disabled=not db_available, type="primary")
    if submitted:
        if uploaded is None:
            st.warning("Выберите CSV для импорта.")
            return
        try:
            count = import_sales_csv(BytesIO(uploaded.getvalue()))
        except ValueError as exc:
            st.error(str(exc))
        except SQLAlchemyError:
            LOGGER.exception("Не удалось импортировать CSV.")
            st.error("PostgreSQL не сохранила файл. Проверьте подключение и повторите импорт.")
        else:
            _imported(f"Импортировано строк: {count:,}. Файл: {uploaded.name}.")


def main() -> None:
    """Запускает дашборд и связывает его вкладки общими фильтрами."""
    st.set_page_config(page_title=TITLE, layout="wide")
    st.markdown(
        "<style>[data-testid='stMetricValue'] {font-size: clamp(1.25rem, 2.2vw, 2rem);}"
        "[data-testid='stMetricLabel'] p {white-space: normal;}</style>",
        unsafe_allow_html=True,
    )
    st.title(TITLE)
    advisor = AISalesAdvisor()
    db_available = True
    try:
        sales = _history()
    except (SQLAlchemyError, ValueError):
        LOGGER.exception("Не удалось загрузить историю продаж.")
        db_available = False
        sales = pd.DataFrame(columns=CSV_COLUMNS)

    categories = sorted(sales["category"].unique())
    with st.sidebar:
        st.header("Фильтры")
        category = st.selectbox(
            "Товарная категория", [None, *categories],
            format_func=lambda name: "Все категории" if name is None else name,
        )
        days = st.slider("Горизонт прогноза, дней", 7, 90, 30)
        demo = st.button(
            "Загрузить демо-данные (760 дней)", disabled=not db_available or not sales.empty,
            width="stretch",
        )
        st.divider()
        if db_available:
            st.success("PostgreSQL · подключена")
        else:
            st.error("PostgreSQL · нет подключения")
        llm_status = st.empty()

    if demo:
        try:
            count = import_sales_csv(DEFAULT_SAMPLE_PATH, only_if_empty=True)
        except ValueError as exc:
            _history.clear()
            st.warning(str(exc))
        except (OSError, SQLAlchemyError):
            LOGGER.exception("Не удалось загрузить демо-данные.")
            st.error("Не удалось загрузить демо-данные. Проверьте подключение к PostgreSQL.")
        else:
            _imported(f"Демо-данные загружены. Дней: {count}.")

    report = st.session_state.pop("import_report", None)
    if report:
        st.success(report)
    if not db_available:
        st.error("Запустите PostgreSQL и проверьте параметры подключения в .env.")
    elif sales.empty:
        st.info("Загрузите демо-данные в боковой панели или импортируйте CSV на вкладке «Загрузка данных».")

    selected = sales if category is None else sales.loc[sales["category"].eq(category)]
    daily = aggregate_daily(selected) if not selected.empty else None
    analysis = SalesForecaster().analyze_historical(daily) if daily is not None and len(daily) >= 2 else None
    simulator = None
    result = None
    forecast_error = "Загрузите данные для расчёта прогноза."
    snapshot = _snapshot(selected)
    if daily is not None:
        period = f"{daily['date'].min():%d.%m.%Y} — {daily['date'].max():%d.%m.%Y}"
        st.caption(f"{category or 'Все категории'}  ·  {period}  ·  Дней с продажами: {len(daily)}")
        if len(daily) >= 14:
            with st.spinner("Обучение сценарной модели…"):
                simulator = _simulator(snapshot, selected)
        if len(daily) >= 16:
            try:
                with st.spinner("Расчёт прогноза Prophet…"):
                    result = _forecast(snapshot, category, days, selected)
            except ValueError as exc:
                forecast_error = str(exc)
            except SQLAlchemyError:
                LOGGER.exception("Не удалось сохранить прогноз.")
                forecast_error = "Не удалось сохранить прогноз в PostgreSQL."
        else:
            forecast_error = "Недостаточно истории. Для прогноза нужно минимум 14 обучающих наблюдений и два отложенных."

    overview_tab, forecast_tab, scenario_tab, advice_tab, upload_tab = st.tabs(TAB_NAMES)
    with overview_tab:
        if daily is not None:
            _show_overview(daily, analysis, simulator)
        else:
            st.caption("После импорта здесь появится обзор продаж.")
    with forecast_tab:
        if result is not None and daily is not None:
            _show_forecast(daily, result)
        else:
            st.info(forecast_error)
    comparison = None
    with scenario_tab:
        budget, discount, promo = _scenario_controls()
        if result is not None and simulator is not None:
            comparison = simulator.simulate_scenario(
                result["forecast"], simulator.avg_marketing_spend, simulator.avg_discount_pct,
                budget, discount, promo / 100,
            )
            _show_scenario(comparison, simulator)
        else:
            st.info("Сравнение появится после расчёта базового прогноза.")
    with advice_tab:
        context = (
            f"{category or 'Все категории'} · {days} дн. · реклама {budget:+d}% · "
            f"скидка {discount}% · акционные дни {promo}%"
        )
        st.caption(f"Текущий сценарий · {context}")
        advice_key = (snapshot, category, days, budget, discount, promo)
        ready = all(item is not None for item in (daily, analysis, result, comparison))
        clicked = st.button("Сформировать AI-рекомендации", type="primary", disabled=not ready)
        if clicked and daily is not None and analysis is not None and result is not None and comparison is not None:
            analytics_summary, forecast_summary = build_advice_summaries(analysis, result, daily)
            analytics_summary["category"] = category
            scenario_summary = {
                **comparison, "discount_pct_new": discount,
                "marketing_change_pct": budget, "promo_days_ratio": promo / 100,
            }
            try:
                with st.spinner("Подготовка рекомендаций…"):
                    advice = advisor.generate_recommendations(analytics_summary, forecast_summary, scenario_summary)
            except SQLAlchemyError:
                LOGGER.exception("Не удалось сохранить рекомендации.")
                st.error("Не удалось сохранить рекомендации в PostgreSQL.")
            else:
                st.session_state["advice"] = (advice_key, advice)
                st.session_state["advisor_source"] = advice["source"]
                _recommendation_history.clear()
        saved = st.session_state.get("advice")
        if saved and saved[0] == advice_key:
            _show_advice(saved[1], context)
        if db_available:
            _show_advice_history()
    with upload_tab:
        _show_upload(db_available)

    source = st.session_state.get("advisor_source")
    if not advisor.api_configured:
        llm_status.info("OpenAI · локальный аналитик")
    elif source == "openai":
        llm_status.success("OpenAI API · ответ получен")
    elif source == "rules":
        llm_status.warning("OpenAI API · последний отчёт подготовил локальный аналитик")
    else:
        llm_status.info("OpenAI API · ключ настроен, ожидается первый запрос")


if __name__ == "__main__":
    main()
