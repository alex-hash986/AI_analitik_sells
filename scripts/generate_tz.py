"""Генерирует техническое задание по AI-системе аналитики продаж."""

from pathlib import Path

from docx import Document
from docx.document import Document as DocumentType
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

PROJECT_TITLE = "AI-система предиктивной аналитики продаж"
AUTHOR = "Алексей Ханипов"

OVERVIEW = (
    "1. Название и общее описание проекта",
    "Название продукта: AI-система предиктивной аналитики продаж (AI Sales Predictive Analytics Platform).",
    "Назначение и суть решения:",
    "Программный комплекс представляет собой веб-ориентированную аналитическую платформу для автоматизированного прогнозирования спроса, выручки и оптимизации маркетинговых затрат в ритейле электроники. Система объединяет классические методы машинного обучения для анализа временных рядов (Prophet, scikit-learn) с генеративным искусственным интеллектом (OpenAI API) для интерпретации сложных статистических закономерностей и формирования готовых управленческих рекомендаций на естественном языке.",
    "Решаемые бизнес-задачи:",
)
BUSINESS_TASKS = (
    "Точное планирование закупок и товарных остатков: прогнозирование объема продаж в единицах и деньгах на горизонт от 7 до 90 дней для предотвращения дефицита ходовых позиций и затоваривания склада.",
    "Достоверный прогноз выручки и денежного потока: построение базового, оптимистичного и пессимистичного коридоров выручки с оценкой точности модели (MAPE, MAE, RMSE).",
    "Выявление скрытой сезонности и аномалий: автоматическая декомпозиция продаж на долгосрочный тренд, недельную цикличность, годовую сезонность (высокий сезон Q4, гендерные праздники, «Черная пятница») и детектирование нетипичных всплесков/провалов.",
    "Оптимизация маркетингового бюджета и промо-акций: сценарное моделирование (What-if анализ) эластичности спроса от рекламных расходов и глубины скидок для максимизации валовой прибыли и ROMI.",
)

AUDIENCE_HEADING = "2. Целевая аудитория и её боли"
AUDIENCE = (
    (
        "Роль в компании",
        "Ключевые задачи",
        "Текущие боли и проблемы",
        "Как система решает боль",
    ),
    (
        "Коммерческий директор / Собственник",
        "Выполнение плана по выручке и маржинальности",
        "Решения о скидках и бюджетах принимаются «по интуиции» или по устаревшим Excel-таблицам; высокая цена ошибки при запуске акций.",
        "Дает прозрачный прогноз выручки с доверительным интервалом и готовый план действий от AI-советника.",
    ),
    (
        "Руководитель отдела маркетинга (CMO)",
        "Эффективное распределение рекламного бюджета и рост ROMI",
        "Непонятно, где наступает точка насыщения рекламы и какая скидка начинает «съедать» маржу вместо привлечения нового спроса.",
        "Интерактивный What-if симулятор показывает прогнозную выручку, маржу и ROMI до запуска рекламной кампании.",
    ),
    (
        "Руководитель отдела закупок / Категорийный менеджер",
        "Планирование складских запасов по категориям",
        "Либо заморозка оборотных средств в неликвиде, либо упущенная выручка из-за пустых полок в пиковые дни недели и сезоны.",
        "Выявляет точные пики недельной и годовой сезонности по категориям товаров с горизонтом до 3 месяцев.",
    ),
    (
        "Бизнес-аналитик",
        "Подготовка регулярной отчетности и поиск инсайтов",
        "До 70% рабочего времени уходит на ручную очистку выгрузок из CRM/1С, свод таблиц и рутинный пересчет моделей.",
        "Автоматическая валидация CSV-выгрузок, хранение истории в PostgreSQL и мгновенная пересборка графиков в один клик.",
    ),
)

FEATURES_HEADING = "3. Функциональные требования"
FEATURES = (
    ("3.1. Модуль загрузки, валидации и хранения данных", (
        "Импорт исторических данных о продажах в формате CSV через веб-интерфейс и CLI-утилиту.",
        "Автоматическая валидация структуры файла: проверка обязательных полей (date, revenue, units_sold, marketing_spend, discount_pct, promo_active, category), приведение форматов дат, обработка пропусков и отклонение поврежденных файлов с информативным сообщением об ошибке.",
        "Генератор реалистичных анонимизированных/синтетических данных (не менее 2 лет истории) для демонстрации и тестирования системы без риска утечки коммерческой тайны.",
        "Сохранение очищенных записей, логов обучения моделей и сгенерированных AI-рекомендаций в реляционную СУБД PostgreSQL.",
    )),
    ("3.2. Модуль интеллектуального анализа исторических данных (EDA & ML)", (
        "Расчет интегрального тренда продаж (темп прироста в % за период и среднемесячная динамика).",
        "Декомпозиция временного ряда: выделение недельного профиля (пиковые и провальные дни недели) и годовой сезонности (пиковые и низкие месяцы).",
        "Корреляционный анализ (коэффициент Пирсона): количественная оценка влияния рекламного бюджета, размера скидки и промо-активности на итоговую выручку и количество проданных единиц.",
        "Автоматическое выявление исторических аномалий (резких скачков и падений спроса) с фиксацией дат для последующего разбора причин.",
    )),
    ("3.3. Модуль предиктивного прогнозирования", (
        "Построение прогноза выручки и спроса на задаваемый пользователем горизонт (от 7 до 90 дней) с использованием библиотеки Prophet и учетом внешних регрессоров (рекламный бюджет, скидки).",
        "Фильтрация прогноза как по всему магазину целиком, так и в разрезе отдельных товарных категорий.",
        "Обязательная кросс-валидация на отложенной выборке с расчетом метрик качества: MAPE (средняя абсолютная процентная ошибка), MAE и RMSE, а также визуализация верхней и нижней границ доверительного интервала (80% / 95%).",
    )),
    ("3.4. Модуль сценарного моделирования (What-If Simulator)", (
        "Обучение регрессионных моделей (Ridge, RandomForestRegressor из scikit-learn) с временной кросс-валидацией (TimeSeriesSplit) и автоматическим выбором лучшей модели по метрике R².",
        "Интерактивное управление параметрами будущего периода: изменение рекламного бюджета (в % от базового уровня), целевой размер средней скидки (%) и доля акционных дней в периоде.",
        "Расчет и сравнение показателей Baseline (базовый прогноз) против What-If (моделируемый сценарий): прогнозируемая выручка, дельта выручки (в рублях и %), расходы на маркетинг, условная валовая прибыль и показатель возврата маркетинговых инвестиций (ROMI).",
    )),
    ("3.5. Модуль AI-рекомендаций (AI Sales Advisor)", (
        "Интеграция с OpenAI API с передачей агрегированных статистических выводов, параметров сезонности, метрик точности ML-модели и результатов What-if симуляции.",
        "Генерация структурированного управленческого отчета по 4 блокам:",
        "Резюме состояния и прогноза выручки.",
        "Рекомендации по таймингу акций (в какие дни недели и месяцы запускать промо, а когда поддерживать спрос нецелесообразно).",
        "Стратегия скидок (оптимальный коридор дисконта без каннибализации маржи).",
        "Аллокация рекламного бюджета и план конкретных действий на ближайшие 30 дней.",
        "Наличие резервного алгоритмического генератора инсайтов (fallback-режим) на случай отсутствия подключения к внешнему LLM API.",
    )),
    ("3.6. Интерактивный веб-дашборд", (
        "Реализация пользовательского интерфейса на базе Streamlit и интерактивной графики Plotly.",
        "Основные экраны/вкладки: «Обзор и исторический анализ», «Прогноз продаж (ML)», «Сценарный симулятор (What-If)», «AI-советник и экспорт отчета», «Загрузка данных».",
        "Возможность скачивания таблицы прогноза в CSV и текстового отчета с AI-рекомендациями.",
    )),
)

STACK_HEADING = "4. Технические требования и стек технологий"
STACK = (
    (
        "Язык программирования",
        "Python 3.11+",
    ),
    (
        "Анализ данных и ML",
        "pandas, numpy, prophet, scikit-learn",
    ),
    (
        "Генеративный AI",
        "openai (Python SDK)",
    ),
    (
        "Веб-интерфейс и визуализация",
        "streamlit, plotly",
    ),
    (
        "База данных и ORM",
        "PostgreSQL 15, SQLAlchemy, psycopg2-binary",
    ),
    (
        "Контейнеризация и развертывание",
        "Docker, Docker Compose",
    ),
    (
        "Безопасность данных",
        "использование исключительно анонимизированных или синтетических датасетов; хранение API-ключей и параметров подключения к БД в переменных окружения (.env).",
    ),
)

RISKS_HEADING = "5. Ограничения и риски проекта"
RISKS = (
    (
        "Категория риска",
        "Описание риска / ограничения",
        "Вероятность",
        "Влияние на проект",
        "Меры по минимизации (план реагирования)",
    ),
    (
        "Качество и полнота исторических данных",
        "У заказчика может быть менее 12 месяцев истории продаж, пропуски в данных о рекламных расходах или нефиксированные ручные скидки.",
        "Высокая",
        "Высокое",
        "Реализован строгий валидатор при импорте; при истории менее 2 полных лет система автоматически отключает/сглаживает годовую сезонность и предупреждает пользователя.",
    ),
    (
        "Внешние шоки («Черные лебеди»)",
        "Резкие колебания курсов валют, логистические сбои, выход новых флагманов или агрессивный демпинг конкурентов не отражены в прошлых данных.",
        "Средняя",
        "Высокое",
        "Использование доверительных интервалов прогноза (yhat_lower / yhat_upper), регулярное дообучение модели на свежих данных (раз в неделю) и корректировка через What-if сценарии.",
    ),
    (
        "Вычислительные ресурсы и скорость отклика",
        "Обучение моделей Prophet и RandomForest при большом количестве товарных категорий может занимать 10–30 секунд в UI.",
        "Средняя",
        "Среднее",
        "Кэширование обученных моделей в сессии Streamlit (@st.cache_data / @st.cache_resource) и логирование результатов в PostgreSQL.",
    ),
    (
        "Доступность и стоимость LLM API",
        "Сетевые ограничения доступа к OpenAI API, превышение лимитов токенов или нестабильный формат ответа языковой модели.",
        "Средняя",
        "Среднее",
        "В LLM передаются только агрегированные метрики (экономия токенов >95%), используется строгий JSON-формат ответа и встроен локальный rule-based fallback-генератор рекомендаций.",
    ),
    (
        "Бюджет и сроки MVP",
        "Ограниченный бюджет пилотного внедрения требует фокуса на ключевом функционале без сложной интеграции с 1С/ERP на первом этапе.",
        "Низкая",
        "Среднее",
        "Стандартизированный обмен через универсальный CSV-шаблон и контейнеризация через Docker Compose позволяют развернуть MVP за 15 минут без затрат на инфраструктуру.",
    ),
    (
        "Конфиденциаль­ность данных",
        "Риск утечки коммерческой информации о выручке и марже при передаче данных в облачные AI-сервисы.",
        "Низкая",
        "Высокое",
        "На этапе пилота используются синтетические или обезличенные данные; в LLM не передаются персональные данные клиентов и закупочные цены.",
    ),
)


def configure(doc: DocumentType) -> None:
    """Настраивает формат A4, типографику и нумерацию страниц."""
    section = doc.sections[0]
    section.page_width = Cm(21)
    section.page_height = Cm(29.7)
    section.left_margin = section.right_margin = Cm(1.8)
    section.top_margin = Cm(1.8)
    section.bottom_margin = Cm(1.65)
    section.footer_distance = Cm(0.8)
    section.different_first_page_header_footer = True

    for name, size, color in (
        ("Normal", 11, "202831"),
        ("Title", 30, "000000"),
        ("Heading 1", 17, "172D47"),
        ("Heading 2", 13, "172D47"),
        ("Footer", 9, "536275"),
    ):
        style = doc.styles[name]
        style.font.name = "Arial"
        style.font.size = Pt(size)
        style.font.color.rgb = RGBColor.from_string(color)
        fonts = style.element.get_or_add_rPr().rFonts
        for attribute in ("asciiTheme", "hAnsiTheme", "eastAsiaTheme", "cstheme"):
            fonts.attrib.pop(qn(f"w:{attribute}"), None)
        language = OxmlElement("w:lang")
        language.set(qn("w:val"), "ru-RU")
        style.element.get_or_add_rPr().append(language)

    normal = doc.styles["Normal"].paragraph_format
    title_properties = doc.styles["Title"].element.get_or_add_pPr()
    title_border = title_properties.find(qn("w:pBdr"))
    if title_border is not None:
        title_properties.remove(title_border)
    normal.space_after = Pt(6)
    normal.line_spacing = 1.12
    for name in ("Heading 1", "Heading 2"):
        heading = doc.styles[name].paragraph_format
        heading.space_before = Pt(12)
        heading.space_after = Pt(6)
        heading.keep_with_next = True
    for name, indent in (("List Bullet", 0.50), ("List Bullet 2", 1.10)):
        bullet = doc.styles[name]
        bullet.base_style = doc.styles["Normal"]
        fmt = bullet.paragraph_format
        fmt.left_indent = Cm(indent)
        fmt.first_line_indent = Cm(-0.40)
        fmt.space_after = Pt(5)

    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    field = OxmlElement("w:fldSimple")
    field.set(qn("w:instr"), "PAGE")
    footer._p.append(field)


def cover(doc: DocumentType) -> None:
    """Создаёт отдельный титульный лист."""
    title = doc.add_paragraph("Техническое задание", style="Title")
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_before = Pt(126)
    title.paragraph_format.space_after = Pt(18)

    subtitle = doc.add_paragraph("на разработку программного комплекса")
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle.runs[0].font.size = Pt(13)

    project = doc.add_paragraph(PROJECT_TITLE)
    project.alignment = WD_ALIGN_PARAGRAPH.CENTER
    project.paragraph_format.space_after = Pt(40)
    project.runs[0].bold = True
    project.runs[0].font.size = Pt(22)

    for label, value in (
        ("Целевой сегмент", "Средний e-commerce и омниканальный ритейл"),
        ("Ниша", "Магазин электроники и бытовой техники"),
        ("Автор и разработчик", AUTHOR),
        ("Версия документа", "1.0"),
    ):
        paragraph = doc.add_paragraph()
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        paragraph.add_run(f"{label}  ·  ").bold = True
        paragraph.add_run(value)

    year = doc.add_paragraph("2026")
    year.alignment = WD_ALIGN_PARAGRAPH.CENTER
    year.paragraph_format.space_before = Pt(70)


def table(doc: DocumentType, rows: tuple[tuple[str, ...], ...], widths: tuple[float, ...]) -> None:
    """Добавляет таблицу Word с повторяемой шапкой и цельными строками."""
    grid = doc.add_table(rows=len(rows), cols=len(widths))
    grid.style = "Table Grid"
    grid.autofit = False
    for column, width in zip(grid.columns, widths):
        column.width = Cm(width)

    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        border = OxmlElement(f"w:{edge}")
        border.set(qn("w:val"), "single")
        border.set(qn("w:sz"), "4")
        border.set(qn("w:color"), "CFD8E3")
        borders.append(border)
    grid._tbl.tblPr.append(borders)

    margins = OxmlElement("w:tblCellMar")
    for edge in ("top", "bottom", "left", "right"):
        margin = OxmlElement(f"w:{edge}")
        margin.set(qn("w:w"), "85")
        margin.set(qn("w:type"), "dxa")
        margins.append(margin)
    grid._tbl.tblPr.append(margins)

    for index, values in enumerate(rows):
        row = grid.rows[index]
        properties = row._tr.get_or_add_trPr()
        properties.append(OxmlElement("w:cantSplit"))
        if index == 0:
            properties.append(OxmlElement("w:tblHeader"))
        for col, (cell, value, width) in enumerate(zip(row.cells, values, widths)):
            cell.width = Cm(width)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            fill = OxmlElement("w:shd")
            fill.set(qn("w:fill"), "233B56" if index == 0 else ("F0F4F8" if index % 2 == 0 else "FFFFFF"))
            cell._tc.get_or_add_tcPr().append(fill)
            paragraph = cell.paragraphs[0]
            paragraph.paragraph_format.space_after = Pt(0)
            paragraph.paragraph_format.line_spacing = 1.06
            run = paragraph.add_run(value)
            run.font.size = Pt(10)
            run.bold = index == 0 or col == 0
            if index == 0:
                run.font.color.rgb = RGBColor.from_string("FFFFFF")


def label(doc: DocumentType, text: str) -> None:
    """Добавляет вводную строку и удерживает её рядом с основным текстом."""
    paragraph = doc.add_paragraph()
    paragraph.paragraph_format.keep_with_next = True
    paragraph.add_run(text).bold = True


def build_document() -> DocumentType:
    """Собирает титульный лист, все разделы и таблицы ТЗ."""
    doc = Document()
    doc.core_properties.title = f"Техническое задание. {PROJECT_TITLE}"
    doc.core_properties.author = AUTHOR
    doc.core_properties.subject = "Магазин электроники и бытовой техники"
    doc.core_properties.version = "1.0"
    configure(doc)
    cover(doc)

    heading, name, purpose, description, tasks = OVERVIEW
    doc.add_heading(heading, 1).paragraph_format.page_break_before = True
    doc.add_paragraph(name)
    label(doc, purpose)
    doc.add_paragraph(description)
    label(doc, tasks)
    for item in BUSINESS_TASKS:
        lead, rest = item.split(": ", 1)
        paragraph = doc.add_paragraph(style="List Bullet")
        paragraph.add_run(f"{lead}: ").bold = True
        paragraph.add_run(rest)

    doc.add_heading(AUDIENCE_HEADING, 1)
    table(doc, AUDIENCE, (3.25, 3.15, 5.55, 5.45))

    doc.add_heading(FEATURES_HEADING, 1)
    for heading, items in FEATURES:
        paragraph = doc.add_heading(heading, 2)
        if heading.startswith("3.5."):
            paragraph.paragraph_format.page_break_before = True
            integration, intro, *blocks, fallback = items
            doc.add_paragraph(integration, style="List Bullet")
            label(doc, intro)
            for block in blocks:
                doc.add_paragraph(block, style="List Bullet 2")
            doc.add_paragraph(fallback, style="List Bullet")
        else:
            for item in items:
                doc.add_paragraph(item, style="List Bullet")

    doc.add_heading(STACK_HEADING, 1)
    table(doc, (("Область", "Требования"), *STACK), (5.2, 12.2))

    doc.add_heading(RISKS_HEADING, 1).paragraph_format.page_break_before = True
    table(doc, RISKS, (3.55, 3.75, 2.70, 2.00, 5.40))
    return doc


def main() -> None:
    """Записывает документ в каталог docs независимо от рабочей директории."""
    output = Path(__file__).resolve().parents[1] / "docs" / "TZ_AI_Sales_Analytics.docx"
    output.parent.mkdir(parents=True, exist_ok=True)
    build_document().save(output)
    print(output)


if __name__ == "__main__":
    main()
