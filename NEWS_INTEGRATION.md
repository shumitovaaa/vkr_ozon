# Новости и тональность: фильтрация и интеграция в ML

При `NEWS_POST_PARSE_OZON_FILTER=True` в `config.py` те же правила релевантности Ozon применяются сразу после парсинга VK (`fetch_vk_news`) и при загрузке новостей из CSV (`load_news_csv`).

## 1. Очистка и (опционально) тональность

```bash
python scripts/compute_sentiment.py --input data/news.csv
```

Создаётся:

- `data/news_filtered.csv` — только строки про Ozon (по ключевым словам в `title` и `text`).
- `data/news_filtered_sentiment.csv` — то же + колонка `sentiment` (если не задан `--no-inference`).

Статистика (сколько удалено) пишется в лог.

Только фильтр, без PyTorch (например, тональность посчитаете отдельно и вставите в CSV вручную):

```bash
python scripts/compute_sentiment.py --input data/news.csv --no-inference
```

Потом допишите `sentiment` и пересохраните файл.

## 2. Настройка `config.py`

Укажите путь к файлу **с колонкой `sentiment`** и включите новости:

```python
"USE_NEWS_SENTIMENT": True,
"NEWS_CSV_PATH": "data/news_filtered_sentiment.csv",
```

Параметры **`NEWS_MODEL`**, **`NEWS_BATCH_SIZE`**, **`NEWS_DEVICE`**, **`NEWS_HF_TOKEN`**, **`NEWS_HF_REVISION`** при **уже готовой** колонке `sentiment` в CSV **не используются** при загрузке: `load_news_csv` переносит `sentiment` из файла, а `ensure_sentiment_column` сразу выходит, если колонка есть (`news/sentiment.py`).

Их можно оставить в конфиге для запуска `compute_sentiment.py` без `--no-inference` (инференс там вызывается через тот же `ensure_sentiment_column`).

## 3. Как признаки попадают в матрицу `X`

1. `pipeline.run_experiment` → `attach_news_sentiment_features` добавляет к дневному ряду котировок колонки  
   `news_count`, `has_news`, `sentiment_score`, `sentiment_trend_3`, `sentiment_volatility`.
2. `FeatureBuilder` при `USE_NEWS_SENTIMENT=True` дописывает к базовому списку признаков имена из `NEWS_FEATURE_COLS` (`features/builder.py`).
3. `build_features` берёт пересечение доступных колонок с этим списком и строит `X`.

Вручную список менять не нужно, если названия совпадают. Исключить новости можно только выставив `USE_NEWS_SENTIMENT=False`.

## 4. Запуск эксперимента

После обновления `NEWS_CSV_PATH`:

```bash
python main.py
```

Каждый запуск заново строит признаки и **переобучает** walk-forward модели (Ridge, RF, LGB и т.д.). Ранее сохранённые веса без новостных колонок **не подмешиваются автоматически** — для сравнения «с новостями / без» делайте два прогона с разным `USE_NEWS_SENTIMENT` или разными CSV.

## 5. Примечание

Если модели уже обучались **без** новостных признаков, для использования тональности нужен **новый** полный прогон пайплайна с `USE_NEWS_SENTIMENT=True` и корректным `news_filtered_sentiment.csv`.
