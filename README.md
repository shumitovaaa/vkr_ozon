# Прогнозирование доходности OZON (MOEX)

Дневные котировки OZON, признаки ТА, ARIMA–GARCH, ML с walk-forward валидацией.

## Структура

| Путь | Назначение |
|------|------------|
| `main.py` | CLI, `--smoke` |
| `pipeline.py` | Загрузка → валидация → индикаторы → EDA → ARIMA–GARCH → ML |
| `config.py` | Единый `CFG` (данные, горизонты, WF, признаки, новости, параметры моделей) |
| `data/` | Загрузка и валидация CSV |
| `features/` | Индикаторы и сборка `X` |
| `models/` | WF, стекинг, scaling |
| `evaluation/` | Метрики |
| `visualization/` | Графики |
| `news/` | Новости, тональность, merge в признаки |
| `scripts/fetch_news.py` | Выгрузка новостей → CSV |

## Запуск

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python main.py
python main.py --data path/to/OZON_combined.csv --out ./results
python main.py --smoke
```

Котировки: корень или `FILE_PATH` в `config.py`.

## Модели

ARIMA–GARCH (baseline), Ridge, Random Forest, LightGBM (регрессия и классификация направления). Все настройки — в `config.py` (`CFG`).

## Результаты

В `results/`: EDA, метрики WF, графики лучших моделей по горизонту, `best_models_wf.json`, ARIMA–GARCH. Обзор типов моделей и пайплайна — `docs/MODEL_OVERVIEW.md`.

## macOS: segmentation fault при тональности (PyTorch + transformers)

Часто связано с **Python 3.13** и многопоточностью BLAS. В `news/sentiment.py` уже ограничены потоки на Darwin. Если всё ещё падает: venv на **Python 3.11 или 3.12**, в `config` поставьте `"NEWS_DEVICE": "cpu"`, уменьшите `"NEWS_BATCH_SIZE": 1`. Не публикуйте `HF_TOKEN` в чатах — при утечке отзовите токен на Hugging Face.

## Если не качается модель с Hugging Face (401, timeout)

1. Удалите устаревший токен: `huggingface-cli logout` и уберите `HF_TOKEN` из окружения. При **404** сбросьте зеркало: `unset HF_ENDPOINT` (Linux/macOS) или `set HF_ENDPOINT=` (Windows CMD).
2. Скачайте модель: `python scripts/download_news_model.py --out data/models/sentiment`, затем в `config.py` укажите `"NEWS_MODEL": r"полный\путь\к\data\models\sentiment"`.
3. Временно отключите тональность: `"USE_NEWS_SENTIMENT": False`.

## Тональность новостей (`NEWS_MODEL`)

Укажите модель с Hugging Face Hub (по умолчанию `seara/rubert-tiny2-russian-sentiment`; альтернатива — `cointegrated/rubert-tiny-sentiment-balanced`), **или** путь к локальному каталогу после `save_pretrained`. Опционально: `NEWS_HF_REVISION` (ветка, по умолчанию через код передаётся `main`). Подробности — в `news/sentiment.py`.

## Авторы

shumitovaaa
