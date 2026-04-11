# Прогнозирование доходности OZON (MOEX)

Дневные котировки OZON, технические индикаторы, гибрид **ARIMA–GARCH**, ML с **walk-forward** валидацией (Ridge, Random Forest, LightGBM), опционально признаки по **новостям и тональности**.

## Структура проекта

| Путь | Назначение |
|------|------------|
| `main.py` | CLI: полный прогон, `--data`, `--out`, `--smoke` |
| `pipeline.py` | Сборка: загрузка → валидация → индикаторы → новости → EDA → ARIMA–GARCH → ML |
| `config.py` | Словарь `CFG`: горизонты, окна WF, гиперпараметры, пути, `NEWS_*`, `VK_*` |
| `data/loader.py` | Чтение CSV (OHLCV), парсинг дат |
| `data/validator.py` | Дубликаты, пропуски, логирование выбросов (без удаления шоков) |
| `features/indicators.py` | RSI, BB, MACD, HAR-RV, OBV, календарные признаки |
| `features/builder.py` | Матрица `X` и таргеты `reg_h` / `clf_h` |
| `models/walk_forward.py` | Purged WF: Ridge, RF, LightGBM; rolling ARIMA–GARCH |
| `models/stacking.py` | Стекинг (например RF + LGB → мета-модель) |
| `models/scaling.py` | Общий `RobustScaler` на train/test (без утечки) |
| `models/optuna_search.py` | Optuna для LGB (не вызывается из `pipeline` по умолчанию) |
| `evaluation/metrics.py` | MAE, RMSE, MDA, IC, торговые метрики, диагностика остатков |
| `evaluation/wf_selection.py` | Агрегация WF-метрик и выбор лучшей модели по горизонту |
| `visualization/plots.py` | EDA, WF-метрики, прогнозы, ARIMA–GARCH, важности признаков |
| `news/` | VK/CSV, препроцесс, фильтр Ozon, тональность, merge в дневной ряд |
| `scripts/fetch_news.py` | Выгрузка новостей из VK → CSV |
| `scripts/compute_sentiment.py` | Фильтр Ozon + колонка `sentiment` |
| `scripts/download_news_model.py` | Скачивание модели HF в локальный каталог |
| `scripts/compare_news_runs.py` | Сравнение прогонов по сохранённым метрикам |
| `tests/` | Юнит-тесты (`pytest`, см. `requirements-dev.txt`) |
| `notebooks/vkr_analysis.ipynb` | Дополнительный анализ (по необходимости) |

Подробности по новостям и `NEWS_CSV_PATH`: [`NEWS_INTEGRATION.md`](NEWS_INTEGRATION.md).

## Установка и запуск

### Клонирование

```bash
git clone <url-репозитория>
cd vkr-master
```

### Виртуальное окружение

**venv (Windows):**

```bat
python -m venv .venv
.venv\Scripts\activate
```

**venv (Linux/macOS):**

```bash
python -m venv .venv
source .venv/bin/activate
```

**conda:**

```bash
conda create -n ozon-ts python=3.11
conda activate ozon-ts
```

### Зависимости

```bash
pip install -r requirements.txt
```

Опционально для тестов:

```bash
pip install -r requirements-dev.txt
```

Положите файл котировок (например `OZON_combined.csv`) в корень или укажите путь в `config.py` (`FILE_PATH`) или через `--data`.

### Запуск пайплайна

```bash
python main.py
python main.py --data path/to/OZON_combined.csv --out ./results
```

Эквивалентно (берёт `CFG` из `config.py`):

```bash
python pipeline.py
```

Быстрая проверка загрузки, валидации и признаков **без обучения ML**:

```bash
python main.py --smoke
```

## Модели

| Модель | Роль в проекте | Зачем |
|--------|----------------|-------|
| **ARIMA–GARCH** | Одномерный baseline на лог-доходностях | Сравнение с ML на одной шкале; ДИ из условной дисперсии |
| **Ridge** | Линейная регрессия с L2 | Быстрый baseline с признаками; устойчивость к мультиколлинеарности |
| **Random Forest** | Нелинейности, устойчивость к выбросам | Ансамбль деревьев для reg/clf |
| **LightGBM** | Градиентный бустинг | Сильные нелинейные эффекты по табличным признакам |

> В литературе часто упоминают XGBoost; в этом репозитории вместо него используется **LightGBM** (сходная ниша — бустинг по табличным признакам).

Детали сильных/слабых сторон для ряда OZON — в docstring модуля `models/walk_forward.py`.

## Результаты

После прогона в каталоге `results/` (или `--out`) появляются, в частности:

- `fig_price_volume.png`, `fig_returns.png`, `fig_seasonality.png`, `fig_correlation.png` — EDA
- `fig_news_sentiment.png` — при включённых новостных признаках
- `fig_wf_metrics_h*.png` — метрики walk-forward по горизонтам
- `fig_forecast_<model>_h*.png`, `fig_<slug>_h*_forecast.png` / `*_residuals.png` — лучшие модели по горизонту
- `fig_forecast_arima_garch_h1.png` — ARIMA–GARCH
- `best_models_wf.json` — выбранные модели

Отдельного модуля «квантильный LightGBM» в репозитории нет: интервалы для baseline задаёт **ARIMA–GARCH**; у ML — классические point-прогнозы и метрики по фолдам.

**Ориентировочные порядки величин** (зависят от выборки и периода; не фиксируются кодом):

| Горизонт | MAE (лог-ret, порядок) | R² (часто ≤ 0) |
|----------|-------------------------|----------------|
| h = 1 | ~0.015–0.02 | около нуля или отрицательный |
| h = 5, 10 | выше | сильнее шум |

Точные числа — в логах WF и на графиках после локального запуска.

## Новости и тональность

Кратко: `USE_NEWS_SENTIMENT`, `NEWS_CSV_PATH`, модель HF или локальный путь — в [`NEWS_INTEGRATION.md`](NEWS_INTEGRATION.md).

### macOS: segmentation fault при тональности (PyTorch + transformers)

Часто связано с **Python 3.13** и многопоточностью BLAS. В `news/sentiment.py` уже ограничены потоки на Darwin. Если падает: venv на **Python 3.11 или 3.12**, в `config` задайте `"NEWS_DEVICE": "cpu"`, уменьшите `"NEWS_BATCH_SIZE": 1`. Не публикуйте `HF_TOKEN` — при утечке отзовите токен на Hugging Face.

### Если не качается модель с Hugging Face (401, timeout)

1. Удалите устаревший токен: `huggingface-cli logout` и уберите `HF_TOKEN` из окружения. При **404** сбросьте зеркало: `unset HF_ENDPOINT` (Linux/macOS) или `set HF_ENDPOINT=` (Windows CMD).
2. Скачайте модель: `python scripts/download_news_model.py --out data/models/sentiment`, затем в `config.py` укажите `"NEWS_MODEL": r"полный\путь\к\data\models\sentiment"`.
3. Временно отключите тональность: `"USE_NEWS_SENTIMENT": False`.

## Лицензия

MIT License

Copyright (c) 2026 shumitovaaa

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## Авторы

- **shumitovaaa**
