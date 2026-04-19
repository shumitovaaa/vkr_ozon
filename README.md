# Программный комплекс для прогнозирования доходности акций OZON (MOEX)

**Назначение.** Репозиторий содержит реализацию вычислительного эксперимента для выпускной квалификационной работы: обработка дневных котировок, формирование признаков технического анализа (ТА), эконометрический baseline ARIMA–GARCH, сравнение моделей машинного обучения с **walk-forward** валидацией, опционально — признаки на основе новостного потока и тональности. Результаты сохраняются в виде таблиц метрик, графиков и конфигурационных сводок для включения в пояснительную записку.

**Тематика исследования (кратко).** Оценка применимости классических и современных методов прогнозирования **лог-доходностей** на горизонтах \(h \in \{1, 5, 10\}\) торговых дней (настраивается в `config.py`) при соблюдении временной структуры данных и снижении риска утечки информации из будущего.

---

## Соответствие структуре ВКР

| Раздел записки (типовой) | Содержание в репозитории |
|--------------------------|---------------------------|
| Исходные данные и предобработка | `data/loader.py`, `data/validator.py`, расчёт `log_ret` в `pipeline.py` |
| Признаковое описание ряда | `features/indicators.py`, `features/builder.py` |
| Модели и оценка качества | `models/walk_forward.py`, `models/stacking.py`, `evaluation/` |
| Визуализация и численные приложения | `visualization/`, артефакты в каталоге результатов |
| Новости и NLP (при включении в работу) | `news/`, параметры `USE_NEWS_SENTIMENT`, `NEWS_*` в `config.py` |

Теоретическое обоснование выбора метрик, ограничения моделей и интерпретация результатов излагаются в **тексте ВКР**; в README приведена только **инженерная** схема воспроизведения расчётов.

---

## Методологическая схема (что реализовано в коде)

1. **Эконометрика.** Rolling-прогноз по ряду лог-доходностей: ARIMA; для шага \(h=1\) — гибрид с GARCH на остатках ARIMA (условная дисперсия для доверительных интервалов). Для \(h>1\) — многошаговый прогноз в постановке, согласованной с кодом `run_hybrid_arima_garch` в `pipeline.py`.
2. **Машинное обучение.** Ridge, Random Forest, LightGBM: регрессия кумулятивной лог-доходности на горизонт \(h\) (**direct multi-step**) и бинарная классификация знака доходности. Сравнение моделей — по **walk-forward** с окном обучения и **embargo** между обучающей и тестовой частями фолда.
3. **Стекинг (опционально).** Мета-модель на **хвосте** ряда (`TEST_FRACTION` последних наблюдений) — отдельно от фолдов WF; см. комментарии в `pipeline.py` (`_holdout_split`, `fit_stacking`).
4. **Новости.** При `USE_NEWS_SENTIMENT=True` — слияние дневных агрегатов и признаков тональности с рядом котировок (`news/merge.py` и связанные модули).

Подробности по реализации моделей и комментарии к выбору библиотек — в docstring-ах модулей `models/walk_forward.py`, `pipeline.py`.

---

## Структура каталогов

| Путь | Назначение |
|------|------------|
| `main.py` | Точка входа: полный эксперимент или режим `--smoke` |
| `pipeline.py` | Оркестратор `run_experiment`: загрузка → валидация → индикаторы → EDA → ARIMA–GARCH → ML → опционально ablation новостей |
| `config.py` | Единый словарь настроек `CFG` (воспроизводимость эксперимента) |
| `data/` | Загрузка и валидация CSV |
| `features/` | Лог-доходности, индикаторы ТА, сборка матрицы признаков и целей |
| `news/` | Новости, фильтрация, тональность, слияние с котировками |
| `models/` | Walk-forward, стекинг, масштабирование признаков |
| `evaluation/` | Метрики регрессии, классификации, торговые метрики |
| `visualization/` | Построение графиков для отчёта |
| `scripts/` | Вспомогательные утилиты (выгрузка новостей, загрузка модели с Hugging Face и т.д.) |

---

## Воспроизводимость эксперимента

1. Зафиксировать версию Python и установить зависимости из `requirements.txt` (желательно в виртуальном окружении).
2. Указать путь к файлу котировок в `FILE_PATH` (`config.py`) или передать `--data` при запуске.
3. Сохранить или приложить к работе копию используемого фрагмента `CFG` (или весь `config.py`), чтобы читатель мог восстановить параметры горизонтов, WF, моделей и новостей.
4. Каталог `OUT_DIR` после прогона содержит метрики и графики; его можно архивировать как **приложение к ВКР** или указать путь к нему в записке.

Команда проверки окружения без полного ML (первые 50 строк данных):

```bash
python main.py --smoke
```

Полный прогон:

```bash
python main.py
python main.py --data путь\к\OZON_combined.csv --out .\results
```

---

## Конфигурация (`CFG` в `config.py`)

| Группа | Примеры ключей | Назначение |
|--------|----------------|------------|
| Данные | `FILE_PATH`, `OUT_DIR` | Входной CSV и каталог результатов |
| Горизонты | `HORIZONS` | Список \(h\) для таргетов `reg_h`, `clf_h` |
| Walk-forward | `WF_SPLITS`, `TRAIN_WINDOW`, `EMBARGO_DAYS`, `TEST_FRACTION` | Фолды, окно обучения, зазор, доля хвоста для стекинга |
| ARIMA–GARCH | `ARIMA_ORDER`, `GARCH_P`, `GARCH_Q`, `ARIMA_GARCH_MIN_TRAIN`, `ARIMA_FORECAST_HORIZONS`, `CI_ALPHA` | Параметры baseline и горизонты сравнения |
| ML | `RIDGE_*`, `RF_*`, `LGB_*`, `LGBCLF_*` | Гиперпараметры моделей |
| Выбор модели | `BEST_MODEL_METRIC`, `ARIMA_BEST_HORIZON_METRIC` | Критерии сравнения на WF |
| Новости | `USE_NEWS_SENTIMENT`, `NEWS_CSV_PATH`, `NEWS_MODEL`, `NEWS_BATCH_SIZE`, … | Включение и параметры NLP-признаков |

---

## Результаты (каталог `OUT_DIR`, по умолчанию `./results`)

- **EDA:** графики, `eda_adf_log_ret.json`, файлы сезонности и корреляций; при новостях — сводки по тональности.
- **ARIMA–GARCH:** `arima_horizons_metrics.csv`, `arima_best_horizon.json`, `fig_forecast_arima_garch_best.png`.
- **ML:** `wf_reg_mean_h*.csv`, `wf_clf_mean_h*.csv`, `best_models_wf.json`, графики лучших моделей по горизонту, `stacking_h*.csv`, при необходимости `stacking_summary.csv`.
- **Ablation новостей** (если `RUN_NEWS_ABLATION=True`): дополнительные CSV сравнения сценариев с признаками новостей и без них.

---

## Установка зависимостей

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Переменные окружения (новости / Hugging Face)

- `HF_TOKEN` — при необходимости доступа к моделям на Hugging Face; токен не публикуйте и при компрометации отзовите на сайте HF.

### Вспомогательные скрипты

- `scripts/fetch_news.py` — подготовка CSV новостей.
- `scripts/download_news_model.py` — локальная загрузка модели тональности (см. раздел ниже при ошибках сети).

---

## Ограничения и дисклеймер

Результаты носят **исследовательский** характер. Программный комплекс не является средством инвестиционных рекомендаций. Рыночные данные и новости могут содержать ошибки; юридические и этические аспекты использования источников при оформлении ВКР следует отразить в записке отдельно.

---

## Технические примечания

### macOS: падение при расчёте тональности (PyTorch / transformers)

Возможны конфликты с **Python 3.13** и многопоточностью BLAS. В `news/sentiment.py` для Darwin ограничено число потоков. Рекомендуется Python **3.11–3.12**, при сбоях — `"NEWS_DEVICE": "cpu"`, уменьшение `"NEWS_BATCH_SIZE": 1`.

### Проблемы загрузки модели с Hugging Face (401, timeout)

1. Выполнить `huggingface-cli logout` и убрать устаревший `HF_TOKEN` из окружения. При **404** сбросить зеркало: `unset HF_ENDPOINT` (Linux/macOS) или `set HF_ENDPOINT=` (Windows CMD).
2. Скачать модель: `python scripts/download_news_model.py --out data/models/sentiment`, в `config.py` указать `"NEWS_MODEL"` как полный путь к каталогу.
3. Временно отключить тональность: `"USE_NEWS_SENTIMENT": False`.

### Модель тональности (`NEWS_MODEL`)

По умолчанию используется модель с Hugging Face Hub (`seara/rubert-tiny2-russian-sentiment`); допустим путь к локальному каталогу после `save_pretrained`. Детали — в `news/sentiment.py`.

---

## Лицензия

**Released under MIT License**

Copyright (c) 2013 Mark Otto.

Copyright (c) 2017 Andrew Fong.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

---

## Сведения о работе (заполнить при сдаче)

| Поле | Значение |
|------|----------|
| Автор | shumitovaaa |
| Наименование ВКР | Анализ и прогнозирование динамики фондового рынка |
| Год | 2026 |
