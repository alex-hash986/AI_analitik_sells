"""Генерация синтетических продаж и импорт пользовательских CSV."""

from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from typing import BinaryIO, TextIO

import numpy as np
import pandas as pd
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from app.database.connection import get_engine, init_db
from app.database.models import SalesRecord

DEFAULT_SAMPLE_PATH = Path(__file__).resolve().parents[2] / "data" / "sample_sales.csv"
SAMPLE_DAYS = 760
MARKETING_RESPONSE_SCALE = 30_000
CSV_COLUMNS = (
    "date", "revenue", "units_sold", "marketing_spend",
    "discount_pct", "promo_active", "category",
)
NUMERIC_LIMITS = {
    "revenue": 999_999_999_999.99,
    "units_sold": 2_147_483_647,
    "marketing_spend": 999_999_999_999.99,
    "discount_pct": 100,
}
PROMO_VALUES = {
    "true": True, "false": False, "1": True, "0": False,
    "yes": True, "no": False, "да": True, "нет": False,
}


def generate_sales_data(
    output_path: Path = DEFAULT_SAMPLE_PATH,
    *,
    seed: int = 42,
    end_date: date | None = None,
) -> pd.DataFrame:
    """Сохраняет 760 завершённых дней продаж и возвращает таблицу.

    Конечная дата включается в выборку. По умолчанию это вчера.
    """
    rng = np.random.default_rng(seed)
    days = SAMPLE_DAYS
    dates = pd.date_range(end=end_date or date.today() - timedelta(days=1), periods=days)
    trend = np.linspace(0.88, 1.12, days)
    weekday = np.array([0.87, 0.89, 0.94, 1.00, 1.08, 1.17, 1.05])[dates.dayofweek]
    annual = 1 + 0.18 * np.cos(2 * np.pi * (dates.dayofyear.to_numpy() - 350) / 365.25)
    organic_demand = trend * weekday * annual
    category = rng.choice(["Электроника", "Дом и быт", "Аксессуары"], size=days)
    unit_price = pd.Series(category).map(
        {"Электроника": 2200, "Дом и быт": 1250, "Аксессуары": 780}
    ).to_numpy()
    promo = rng.random(days) < 0.24
    discount = np.where(promo, rng.integers(5, 26, days), 0)
    marketing = np.round(70_000 * organic_demand**1.15 * rng.lognormal(0, 0.12, days), 2)
    marketing_lift = 1 + 0.60 * np.log1p(marketing / MARKETING_RESPONSE_SCALE)
    volume_lift = 1 + 0.022 * discount - 0.0003 * discount**2 + 0.035 * promo
    gross_sales = (
        255_000 * organic_demand * marketing_lift * volume_lift
        * rng.lognormal(0, 0.06, days)
    )
    outliers = rng.choice(days, size=round(days * 0.02), replace=False)
    gross_sales[outliers] *= 1 + rng.choice([-1, 1], len(outliers)) * rng.uniform(
        0.25, 0.35, len(outliers)
    )
    units = np.maximum(1, np.rint(gross_sales / unit_price)).astype(int)
    revenue = np.round(units * unit_price * (1 - discount / 100), 2)
    sales = pd.DataFrame({
        "date": dates,
        "revenue": revenue,
        "units_sold": units,
        "marketing_spend": marketing,
        "discount_pct": discount,
        "promo_active": promo,
        "category": category,
    })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sales.to_csv(output_path, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
    return sales


def validate_sales_csv(source: str | Path | BinaryIO | TextIO) -> pd.DataFrame:
    """Читает CSV с запятыми в UTF-8 и проверяет строки перед импортом.

    Поддерживает даты ГГГГ-ММ-ДД и ДД.ММ.ГГГГ. Пустые расходы и скидки
    заменяет нулями, пустой признак акции выводит из наличия скидки.
    Ошибки в обязательных значениях отклоняют весь файл.
    """
    try:
        raw = pd.read_csv(source, dtype="string", encoding="utf-8-sig", keep_default_na=False)
    except (ValueError, UnicodeError) as exc:
        raise ValueError(f"Не удалось прочитать CSV: {exc}") from exc

    raw.columns = raw.columns.str.strip()
    if not raw.columns.is_unique:
        raise ValueError("Названия колонок CSV должны быть уникальными.")
    if not isinstance(raw.index, pd.RangeIndex):
        raise ValueError("В CSV есть строки с лишними полями.")
    missing = set(CSV_COLUMNS).difference(raw.columns)
    if missing:
        raise ValueError(f"Отсутствуют колонки: {', '.join(sorted(missing))}.")
    if raw.empty:
        raise ValueError("CSV не содержит записей о продажах.")

    sales = raw.loc[:, list(CSV_COLUMNS)].apply(lambda column: column.str.strip())
    sales = sales.replace("", pd.NA)
    dates = sales["date"]
    iso_dates = dates.where(dates.str.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", na=False))
    local_dates = dates.where(dates.str.fullmatch(r"[0-9]{2}\.[0-9]{2}\.[0-9]{4}", na=False))
    parsed_dates = pd.to_datetime(iso_dates, format="%Y-%m-%d", errors="coerce").fillna(
        pd.to_datetime(local_dates, format="%d.%m.%Y", errors="coerce")
    )
    _check_rows(parsed_dates.isna(), "date")
    sales["date"] = parsed_dates.dt.date

    for name, limit in NUMERIC_LIMITS.items():
        values = sales[name]
        if name in ("marketing_spend", "discount_pct"):
            values = values.fillna("0")
        numbers = pd.to_numeric(values, errors="coerce")
        invalid = numbers.isna() | ~numbers.between(0, limit)
        if name == "units_sold":
            _check_rows(invalid | (numbers % 1 != 0), name)
            sales[name] = numbers.astype(int)
        else:
            _check_rows(invalid, name)
            sales[name] = numbers.round(2)

    promo = sales["promo_active"].str.lower()
    parsed_promo = promo.map(PROMO_VALUES)
    _check_rows(promo.notna() & parsed_promo.isna(), "promo_active")
    sales["promo_active"] = parsed_promo.where(promo.notna(), sales["discount_pct"] > 0)
    category = sales["category"]
    _check_rows(
        category.isna() | (category.str.len() > 100) | category.str.contains("\x00"),
        "category",
    )
    return sales


def _check_rows(invalid: pd.Series, column: str) -> None:
    """Сообщает номера первых некорректных записей, считая с единицы."""
    if invalid.any():
        positions = np.flatnonzero(invalid.to_numpy(dtype=bool))[:5] + 1
        records = ", ".join(map(str, positions))
        raise ValueError(f"Некорректное значение в колонке {column}: записи {records}.")


def load_sales_history() -> pd.DataFrame:
    """Загружает продажи из PostgreSQL и создаёт таблицы при первом запуске."""
    init_db()
    query = select(*(getattr(SalesRecord, name) for name in CSV_COLUMNS)).order_by(
        SalesRecord.date, SalesRecord.id
    )
    with get_engine().connect() as connection:
        return pd.read_sql(query, connection)


def import_sales_csv(
    source: str | Path | BinaryIO | TextIO, *, only_if_empty: bool = False,
) -> int:
    """Проверяет CSV и сохраняет все записи одной транзакцией PostgreSQL."""
    sales = validate_sales_csv(source)
    for column in ("revenue", "marketing_spend", "discount_pct"):
        sales[column] = sales[column].map(lambda value: Decimal(str(value)))

    init_db()
    with Session(get_engine()) as session, session.begin():
        if only_if_empty:
            session.execute(text("LOCK TABLE sales_records IN SHARE ROW EXCLUSIVE MODE"))
            if session.scalar(select(SalesRecord.id).limit(1)) is not None:
                raise ValueError("В базе уже есть продажи. Демо-данные можно загрузить в пустую базу.")
        session.add_all(SalesRecord(**record) for record in sales.to_dict("records"))
    return len(sales)
