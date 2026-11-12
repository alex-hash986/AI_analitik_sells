# AI-система предиктивной аналитики продаж

Платформа для магазина электроники и бытовой техники. Она превращает историю продаж в прогноз выручки, сравнивает варианты рекламного бюджета и скидок и готовит управленческий отчёт с планом действий. Проект ориентирован на коммерческих директоров, маркетологов и категорийных менеджеров среднего e-commerce и омниканального ритейла.

## Что можно сделать

| Задача | Что показывает система |
| --- | --- |
| Подготовиться к пикам спроса | Недельная и годовая сезонность, дневная выручка и аномалии по магазину или категории. |
| Оценить ожидаемую выручку | Прогноз Prophet на 7–90 дней, 80% прогнозный интервал и ошибки MAPE, MAE и RMSE на отложенных днях. |
| Сравнить промо-сценарии до запуска | Выручка, рекламные расходы, условная прибыль и ROMI при изменении бюджета, скидки и доли акционных дней. |
| Подготовить коммерческий план | AI-советник объединяет историческую аналитику, прогноз и выбранный What-If в отчёт с планом на 30 дней. |
| Начать с собственной выгрузки | CSV-шаблон, проверка формата, импорт в PostgreSQL и экспорт прогноза и отчёта. |

В репозитории есть демонстрационный набор на 760 дней с сезонностью, акциями и выбросами. На нём 30-дневный прогноз показал MAPE около 12% в проверочном запуске. Для собственной истории оценка появится после обучения. ROMI — модельная оценка относительно сценария с нулевым рекламным бюджетом, а условная прибыль учитывает скидку и рекламу.

## Архитектура

```text
app/
├── dashboard.py          # Веб-интерфейс Streamlit и графики Plotly
├── main.py               # CLI для импорта, прогноза, симуляции и рекомендаций
├── database/
│   ├── connection.py     # Подключение и создание таблиц
│   └── models.py         # SalesRecord, ForecastLog, AIRecommendation
├── ml/
│   ├── data_processor.py # Генерация, валидация, импорт и чтение продаж
│   ├── forecaster.py     # Анализ сезонности и прогноз Prophet
│   └── simulator.py      # Сценарная модель Ridge или Random Forest
└── ai/
    └── advisor.py        # OpenAI Structured Outputs и локальные правила
data/sample_sales.csv     # Демонстрационные продажи
docs/                     # Техническое задание и каталог для скриншотов
scripts/generate_tz.py     # Сборка ТЗ в Word
```

Streamlit и CLI используют общие модули. SQLAlchemy хранит продажи, параметры и метрики прогонов и отчёты советника в PostgreSQL 15. Prophet строит прогноз с недельной и годовой сезонностью и регрессорами рекламы и скидок. Для симуляции scikit-learn сравнивает Ridge и Random Forest на временных выборках. Интерфейс кэширует чтение истории, прогнозы и обученную сценарную модель, чтобы повторный выбор фильтров не запускал обучение заново. OpenAI получает агрегированные сводки. Без ключа или при ошибке API отчёт формирует локальный аналитик.

**Стек** · Python 3.12, pandas, NumPy, Prophet, scikit-learn, Streamlit, Plotly, PostgreSQL 15, SQLAlchemy, OpenAI SDK, Docker Compose.

## Документация и бизнес-кейс

- [Техническое задание (DOCX)](docs/TZ_AI_Sales_Analytics.docx). Цели проекта, портрет целевой аудитории и её задачи, функциональные требования к шести модулям, стек и матрица рисков внедрения.
- [Бизнес-кейс и расчет экономического эффекта (Markdown)](docs/BUSINESS_CASE.md). Бизнес-проблема ритейла электроники, логика работы ML- и AI-модулей, сравнение Baseline и What-If с приростом прогнозной выручки на 1,74 млн ₽ и готовый B2B-оффер.

## Запуск

Команды выполняются из корня репозитория. Для обоих вариантов создайте `.env` из шаблона:

```powershell
Copy-Item .env.example .env
```

В Bash используйте `cp .env.example .env`. В `.env` задайте `POSTGRES_PASSWORD`. Остальные параметры базы можно оставить как в примере. Для советника добавьте `OPENAI_API_KEY`, если хотите использовать API. Модель по умолчанию — `gpt-4o-mini`. При необходимости её и адрес API можно изменить через `OPENAI_MODEL` и `OPENAI_BASE_URL`. Файл `.env` предназначен для локальной конфигурации.

### Весь стек через Docker Compose

1. Установите Docker с плагином Compose.
2. Соберите и запустите PostgreSQL и Streamlit:

   ```bash
   docker compose --profile dashboard up -d --build --wait
   ```

3. Откройте [http://localhost:8501](http://localhost:8501). На пустой базе нажмите **Загрузить демо-данные (760 дней)** или откройте вкладку **Загрузка данных** для импорта своего CSV.
4. CLI можно запустить внутри контейнера:

   ```bash
   docker compose --profile dashboard exec dashboard python -m app.main forecast --days 30
   docker compose --profile dashboard exec dashboard python -m app.main simulate --mkt-change 20 --discount 10
   docker compose --profile dashboard exec dashboard python -m app.main advise --days 30
   ```

Остановка сервисов: `docker compose --profile dashboard down`. Данные PostgreSQL хранятся в Docker-томе.

### Локальный Python и PostgreSQL в Docker

1. Запустите PostgreSQL:

   ```bash
   docker compose up -d --wait postgres
   ```

   При использовании своей PostgreSQL укажите её адрес, порт, имя базы и пользователя в `.env`. Пустые таблицы создаются автоматически при запуске.

2. Создайте виртуальное окружение, активируйте его и установите зависимости:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install -r requirements.txt
   ```

   В Bash используйте `source .venv/bin/activate`.

3. Подготовьте таблицы и импортируйте пример. На пустой базе можно вместо этого загрузить пример кнопкой в интерфейсе.

   ```bash
   python -m app.main init-db
   python -m app.main import-csv data/sample_sales.csv
   ```

4. Запустите дашборд и откройте [http://localhost:8501](http://localhost:8501):

   ```bash
   streamlit run app/dashboard.py
   ```

### Команды CLI

```bash
python -m app.main --help
python -m app.main generate --output data/sample_sales.csv
python -m app.main import-csv data/sample_sales.csv
python -m app.main forecast --days 30
python -m app.main forecast --days 30 --category "Электроника"
python -m app.main simulate --mkt-change 20 --discount 10 --promo-ratio 0.25
python -m app.main advise --days 30
```

`generate` создаёт новый CSV за 760 завершённых дней. `import-csv` добавляет записи, поэтому повторный импорт одного файла повторит его строки. В CLI `--promo-ratio` задаётся долей от 0 до 1, а в интерфейсе — в процентах.

### Формат данных

CSV в UTF-8 с запятыми и строкой заголовков:

```csv
date,revenue,units_sold,marketing_spend,discount_pct,promo_active,category
2026-01-01,450000,220,68000,10,true,Электроника
```

`date` принимает `ГГГГ-ММ-ДД` или `ДД.ММ.ГГГГ`. Выручка и расходы на рекламу указываются в рублях, продажи — целым числом, скидка — в процентах от 0 до 100. Пустые расходы и скидки заменяются нулями. Пустой признак акции определяется по наличию скидки. Файл с ошибкой в обязательном поле отклоняется целиком. Шаблон доступен на вкладке **Загрузка данных**.

## Демонстрация интерфейса

| Экран | Место для скриншота |
| --- | --- |
| Обзор и сезонность | `docs/screenshots/01-overview.png` |
| Прогноз продаж | `docs/screenshots/02-forecast.png` |
| Сценарный симулятор | `docs/screenshots/03-what-if.png` |
| AI-советник | `docs/screenshots/04-advisor.png` |
| Загрузка данных | `docs/screenshots/05-import.png` |
